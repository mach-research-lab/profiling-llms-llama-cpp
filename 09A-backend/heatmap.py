#!/usr/bin/env python3
"""
Create profiler heatmaps for prefill/decode analysis.

Supported heatmaps:
    1. time
       Runtime heatmap
    2. memory
       Memory pressure heatmap
    3. ipc
       IPC heatmap
    4. op-share
       Phase-compare runtime heatmap
    5. op-share-ipc
       Phase-compare IPC heatmap
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sqlite3
import sys
from pathlib import Path
from typing import Mapping

try:
    import db_queries
except ImportError:  # pragma: no cover
    db_queries = None

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    matplotlib = None
    plt = None

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None

try:
    import psycopg
except ImportError:  # pragma: no cover
    psycopg = None


SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = SCRIPT_DIR.parent / "run_every_view_results"
DEFAULT_JSON_DIR = RESULTS_DIR
DEFAULT_SQLITE_DB = RESULTS_DIR / "tensor_op_view.db"
DEFAULT_OUTPUT = SCRIPT_DIR / "plots" / "heatmap.png"

CANONICAL_COLUMNS = {
    "phase": "phase",
    "event_phase": "phase",
    "token_index": "token_index",
    "event_token_index": "token_index",
    "tensor_name": "tensor_name",
    "event_tensor_name": "tensor_name",
    "op_type": "op_type",
    "event_operation_type": "op_type",
    "time_ns": "time_ns",
    "time_us": "time_us",
    "event_time_microseconds": "time_us",
    "size_bytes": "size_bytes",
    "event_size_bytes": "size_bytes",
    "n_elements": "n_elements",
    "event_n_elements": "n_elements",
    "papi_l1_dcm": "papi_l1_dcm",
    "papi_l1_icm": "papi_l1_icm",
    "papi_l2_icm": "papi_l2_icm",
    "papi_l2_tcm": "papi_l2_tcm",
    "papi_l3_tcm": "papi_l3_tcm",
    "papi_tot_ins": "tot_ins",
    "papi_tot_cyc": "tot_cyc",
    "tot_ins": "tot_ins",
    "tot_cyc": "tot_cyc",
    "instructions": "tot_ins",
    "cycles": "tot_cyc",
}

LAYER_PATTERNS = (
    re.compile(r"_l(\d+)\b"),
    re.compile(r"-(\d+)\b"),
)


def load_measurements_from_csv(csv_path: str | os.PathLike[str]) -> pd.DataFrame:
    return pd.read_csv(csv_path)


def load_measurements_from_sqlite_db(
    db_path: str | os.PathLike[str],
    sql: str | None = None,
) -> pd.DataFrame:
    if db_queries is not None and sql is None:
        rows = db_queries.get_events_with_papi(db_path=str(db_path))
        if rows:
            raw = pd.DataFrame.from_records(rows)
        else:
            raw = pd.DataFrame(
                columns=[
                    "event_phase",
                    "event_token_index",
                    "event_tensor_name",
                    "event_operation_type",
                    "event_time_microseconds",
                    "event_size_bytes",
                    "event_n_elements",
                    "papi_event_name",
                    "papi_value",
                ]
            )
    else:
        query = sql
        if query is None:
            query = """
                SELECT
                    event_phase,
                    event_token_index,
                    event_tensor_name,
                    event_operation_type,
                    event_time_microseconds,
                    event_size_bytes,
                    event_n_elements,
                    papi_event_name,
                    papi_value
                FROM event_item
                LEFT JOIN event_papi_counter USING (event_item_id)
                ORDER BY event_item_id
            """

        with sqlite3.connect(db_path) as conn:
            raw = pd.read_sql_query(query, conn)

    if "papi_event_name" not in raw.columns or "papi_value" not in raw.columns:
        return raw

    base_columns = [
        "event_phase",
        "event_token_index",
        "event_tensor_name",
        "event_operation_type",
        "event_time_microseconds",
        "event_size_bytes",
        "event_n_elements",
    ]

    base = raw[base_columns].drop_duplicates().reset_index(drop=True)
    event_id_columns = [
        "event_phase",
        "event_token_index",
        "event_tensor_name",
        "event_operation_type",
        "event_time_microseconds",
        "event_size_bytes",
        "event_n_elements",
        "papi_event_name",
    ]
    grouped = (
        raw.groupby(event_id_columns, dropna=False, as_index=False)["papi_value"]
        .sum()
    )
    counters = grouped.pivot_table(
        index=base_columns,
        columns="papi_event_name",
        values="papi_value",
        aggfunc="sum",
        fill_value=0,
    ).reset_index()
    counters.columns.name = None

    return base.merge(counters, on=base_columns, how="left")


def load_measurements_from_db(
    dsn: str,
    run_id: int | None = None,
    sql: str | None = None,
) -> pd.DataFrame:
    if psycopg is None:
        raise RuntimeError(
            "psycopg is not installed. Install it with `pip install psycopg[binary]`."
        )

    query = sql
    params: tuple[object, ...] = ()

    if query is None:
        query = """
            SELECT
                event_phase,
                event_token_index,
                event_tensor_name,
                event_operation_type,
                event_time_microseconds,
                event_size_bytes,
                event_n_elements
            FROM event_item
        """
        if run_id is not None:
            query += " WHERE run_id = %s"
            params = (run_id,)
        query += " ORDER BY event_item_id"

    with psycopg.connect(dsn) as conn:
        return pd.read_sql_query(query, conn, params=params)


def normalize_measurements(df: pd.DataFrame) -> pd.DataFrame:
    renamed = df.rename(
        columns={
            col: CANONICAL_COLUMNS[col.strip().lower()]
            for col in df.columns
            if col.strip().lower() in CANONICAL_COLUMNS
        }
    ).copy()

    required = {"phase", "op_type"}
    missing = required - set(renamed.columns)
    if missing:
        raise ValueError(
            "Missing required columns after normalization: "
            + ", ".join(sorted(missing))
        )

    renamed["phase"] = renamed["phase"].astype(str).str.strip().str.lower()
    renamed["op_type"] = renamed["op_type"].astype(str).str.strip()
    if "tensor_name" in renamed.columns:
        renamed["tensor_name"] = renamed["tensor_name"].astype(str).str.strip()
    else:
        renamed["tensor_name"] = ""

    if "token_index" in renamed.columns:
        renamed["token_index"] = pd.to_numeric(
            renamed["token_index"], errors="coerce"
        )
        renamed = renamed.dropna(subset=["token_index"]).copy()
        renamed["token_index"] = renamed["token_index"].astype(int)

    for numeric_col in (
        "time_ns",
        "time_us",
        "size_bytes",
        "n_elements",
        "papi_l1_dcm",
        "papi_l1_icm",
        "papi_l2_icm",
        "papi_l2_tcm",
        "papi_l3_tcm",
        "tot_ins",
        "tot_cyc",
    ):
        if numeric_col in renamed.columns:
            renamed[numeric_col] = (
                pd.to_numeric(renamed[numeric_col], errors="coerce").fillna(0)
            )

    renamed = renamed[renamed["op_type"] != ""].copy()
    return renamed


def load_json_file(path: str | os.PathLike[str]) -> object:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def select_top_matrix_rows(matrix: pd.DataFrame, top_n: int | None) -> pd.DataFrame:
    if top_n is None or top_n <= 0 or matrix.empty:
        return matrix
    order = matrix.sum(axis=1).sort_values(ascending=False)
    return matrix.loc[order.head(top_n).index]


def resolve_phase_view_path(results_dir: str | os.PathLike[str]) -> Path | None:
    path = Path(results_dir) / "phase-view.json"
    return path if path.is_file() else None


def resolve_decoder_block_view_path(results_dir: str | os.PathLike[str]) -> Path | None:
    path = Path(results_dir) / "decoder-block-view.json"
    return path if path.is_file() else None


def build_operation_share_matrix_from_phase_json(
    phase_view_path: str | os.PathLike[str],
    metric_name: str = "total_time_us",
    phases: list[str] | None = None,
    top_n: int | None = 20,
) -> tuple[pd.DataFrame, str]:
    raw = load_json_file(phase_view_path)
    if not isinstance(raw, dict):
        raise ValueError("phase-view.json must contain a JSON object.")

    selected_phases = phases if phases else ["prefill", "decode"]
    records: list[dict[str, object]] = []
    for phase in selected_phases:
        phase_payload = raw.get(phase)
        if not isinstance(phase_payload, dict):
            continue

        op_type_share = phase_payload.get("op_type_share", {})
        if not isinstance(op_type_share, dict):
            continue

        for op_type, metrics in op_type_share.items():
            if not isinstance(metrics, dict):
                continue
            records.append(
                {
                    "phase": phase,
                    "op_type": op_type,
                    metric_name: float(metrics.get(metric_name, 0)),
                }
            )

    if not records:
        raise ValueError("phase-view.json does not contain op_type_share data.")

    frame = pd.DataFrame.from_records(records)
    matrix = frame.pivot_table(
        index="op_type",
        columns="phase",
        values=metric_name,
        aggfunc="sum",
        fill_value=0.0,
    )

    if "prefill" not in matrix.columns:
        matrix["prefill"] = 0.0
    if "decode" not in matrix.columns:
        matrix["decode"] = 0.0
    delta_column = "delta_ipc" if metric_name == "IPC" else "delta_time"
    matrix[delta_column] = matrix["decode"] - matrix["prefill"]
    matrix = matrix[["prefill", "decode", delta_column]]
    matrix = select_top_matrix_rows(matrix, top_n)
    op_order = (matrix["prefill"] + matrix["decode"]).sort_values(ascending=False).index
    value_label = "ipc" if metric_name == "IPC" else "time_us"
    return matrix.reindex(op_order), value_label


def block_label(block: Mapping[str, object]) -> str:
    block_type = str(block.get("block_type", "block")).strip().lower()
    block_id = int(block.get("block_id", 0))
    prefix = "P" if block_type.startswith("prefill") else "D" if block_type.startswith("decode") else "B"
    return f"{prefix}{block_id:02d}"


def extract_decoder_metric(subcomponent: Mapping[str, object], heatmap_kind: str) -> float | None:
    kind = heatmap_kind.strip().lower()
    if kind == "time":
        if "runtime_us" in subcomponent:
            return float(subcomponent["runtime_us"])
        if "runtime_ms" in subcomponent:
            return float(subcomponent["runtime_ms"]) * 1000.0
        if "runtime_ns" in subcomponent:
            return float(subcomponent["runtime_ns"]) / 1000.0
        return None

    if kind in {"memory", "memory-pressure", "llc", "cache-misses"}:
        cache_behavior = subcomponent.get("cache_behavior", {})
        papi = subcomponent.get("papi", {})
        if isinstance(cache_behavior, dict):
            for key in ("L3_misses", "L2_misses", "L1_misses"):
                if key in cache_behavior:
                    return float(cache_behavior[key])
        if isinstance(papi, dict):
            for key in ("PAPI_L3_TCM", "PAPI_L2_TCM", "PAPI_L1_DCM"):
                if key in papi:
                    return float(papi[key])
        return None

    if kind == "ipc":
        if "IPC" in subcomponent:
            return float(subcomponent["IPC"])
        papi = subcomponent.get("papi", {})
        if isinstance(papi, dict):
            cycles = float(papi.get("PAPI_TOT_CYC", 0))
            instructions = float(papi.get("PAPI_TOT_INS", 0))
            if cycles != 0:
                return instructions / cycles
        return None

    raise ValueError(f"Unsupported JSON decoder metric for heatmap kind: {heatmap_kind}")


def build_decoder_block_matrix_from_json(
    decoder_block_path: str | os.PathLike[str],
    heatmap_kind: str,
    phases: list[str] | None = None,
    top_n: int | None = 20,
) -> tuple[pd.DataFrame, str]:
    raw = load_json_file(decoder_block_path)
    if not isinstance(raw, list):
        raise ValueError("decoder-block-view.json must contain a JSON array.")

    selected_phases = {phase.strip().lower() for phase in phases} if phases else None
    records: list[dict[str, object]] = []

    for block in raw:
        if not isinstance(block, dict):
            continue
        block_type = str(block.get("block_type", "")).strip().lower()
        if selected_phases is not None and block_type not in selected_phases:
            continue

        subcomponents = block.get("subcomponents", {})
        if not isinstance(subcomponents, dict) or not subcomponents:
            subcomponents = {"block_total": block}

        for name, subcomponent in subcomponents.items():
            if not isinstance(subcomponent, dict):
                continue
            value = extract_decoder_metric(subcomponent, heatmap_kind)
            if value is None:
                continue
            records.append(
                {
                    "component": str(name),
                    "block": block_label(block),
                    "value": value,
                }
            )

    if not records:
        raise ValueError("decoder-block-view.json does not contain the requested metric.")

    frame = pd.DataFrame.from_records(records)
    matrix = frame.pivot_table(
        index="component",
        columns="block",
        values="value",
        aggfunc="sum",
        fill_value=0.0,
    )
    matrix = select_top_matrix_rows(matrix, top_n)
    column_order = sorted(matrix.columns)
    return matrix.reindex(columns=column_order), "ipc" if heatmap_kind == "ipc" else "time_us" if heatmap_kind == "time" else "papi_l3_tcm"


def load_group_mapping(group_file: str | os.PathLike[str] | None) -> dict[str, str]:
    if group_file is None:
        return {}

    with open(group_file, "r", encoding="utf-8") as handle:
        raw_mapping = json.load(handle)

    group_map: dict[str, str] = {}
    for key, value in raw_mapping.items():
        if isinstance(value, str):
            group_map[str(key)] = value
        elif isinstance(value, list):
            for op_name in value:
                group_map[str(op_name)] = str(key)
        else:
            raise ValueError("Group file values must be a string or list of strings.")

    return group_map


def apply_operation_groups(
    df: pd.DataFrame,
    group_map: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    if not group_map:
        return df.copy()

    grouped = df.copy()
    grouped["op_type"] = grouped["op_type"].map(lambda op: group_map.get(op, op))
    return grouped


def infer_layer_label(tensor_name: object) -> str | None:
    if tensor_name is None:
        return None
    if isinstance(tensor_name, float) and math.isnan(tensor_name):
        return None

    name = str(tensor_name).strip()
    if not name or name.lower() in {"n/a", "na"}:
        return None

    lowered = name.lower()
    for pattern in LAYER_PATTERNS:
        match = pattern.search(lowered)
        if match:
            return f"L{int(match.group(1)):02d}"

    if lowered == "embd":
        return "embd"
    if lowered.startswith("result_") or lowered.startswith("node_"):
        return "final"
    return None


def add_layer_column(
    df: pd.DataFrame,
    include_special_layers: bool = False,
) -> pd.DataFrame:
    layered = df.copy()
    tensor_series = layered["tensor_name"]
    if tensor_series.dtype != object:
        tensor_series = tensor_series.astype(object)
    layered["layer"] = tensor_series.map(infer_layer_label)
    if include_special_layers:
        layered["layer"] = layered["layer"].fillna("other")
        return layered
    return layered.dropna(subset=["layer"]).copy()


def filter_phases(df: pd.DataFrame, phases: list[str] | None) -> pd.DataFrame:
    if not phases:
        return df.copy()
    allowed = {phase.strip().lower() for phase in phases}
    filtered = df[df["phase"].isin(allowed)].copy()
    if filtered.empty:
        raise ValueError(f"No rows found for phases: {sorted(allowed)}")
    return filtered


def select_top_operations(
    df: pd.DataFrame,
    value_column: str,
    top_n: int | None,
) -> pd.DataFrame:
    if top_n is None or top_n <= 0:
        return df.copy()

    top_ops = (
        df.groupby("op_type", as_index=False)[value_column]
        .sum()
        .sort_values(value_column, ascending=False)
        .head(top_n)["op_type"]
    )
    return df[df["op_type"].isin(set(top_ops))].copy()


def resolve_time_column(df: pd.DataFrame) -> str:
    if "time_ns" in df.columns:
        return "time_ns"
    if "time_us" in df.columns:
        return "time_us"
    raise ValueError("No time column found. Expected time_ns or time_us.")


def resolve_memory_column(df: pd.DataFrame) -> str:
    for candidate in ("papi_l3_tcm", "papi_l2_tcm", "papi_l1_dcm"):
        if candidate in df.columns:
            return candidate
    raise ValueError(
        "No cache-miss column found. Expected one of papi_l3_tcm, papi_l2_tcm or papi_l1_dcm."
    )


def compute_ipc(df: pd.DataFrame) -> pd.DataFrame:
    if "tot_ins" not in df.columns or "tot_cyc" not in df.columns:
        raise ValueError(
            "IPC requires tot_ins and tot_cyc columns. Add them in the DB/PAPI collection first."
        )

    ipc_df = df.copy()
    ipc_df["ipc"] = 0.0
    non_zero_cycles = ipc_df["tot_cyc"] != 0
    ipc_df.loc[non_zero_cycles, "ipc"] = (
        ipc_df.loc[non_zero_cycles, "tot_ins"] / ipc_df.loc[non_zero_cycles, "tot_cyc"]
    )
    return ipc_df


def build_layer_metric_matrix(
    df: pd.DataFrame,
    value_column: str,
    top_n: int | None = 20,
    include_special_layers: bool = False,
) -> pd.DataFrame:
    layered = add_layer_column(df, include_special_layers=include_special_layers)
    if layered.empty:
        raise ValueError("No rows with layer information were found.")

    aggregated = (
        layered.groupby(["op_type", "layer"], as_index=False)[value_column]
        .sum()
        .sort_values(["op_type", "layer"])
    )
    aggregated = select_top_operations(aggregated, value_column, top_n=top_n)

    matrix = aggregated.pivot_table(
        index="op_type",
        columns="layer",
        values=value_column,
        aggfunc="sum",
        fill_value=0.0,
    )

    sorted_columns = sorted(
        matrix.columns,
        key=lambda layer: (not str(layer).startswith("L"), str(layer)),
    )
    matrix = matrix.reindex(columns=sorted_columns)
    op_order = matrix.sum(axis=1).sort_values(ascending=False).index
    return matrix.reindex(op_order)


def build_operation_share_matrix(
    df: pd.DataFrame,
    phases: list[str] | None = None,
    top_n: int | None = 20,
) -> pd.DataFrame:
    time_column = resolve_time_column(df)
    selected_phases = phases if phases else ["prefill", "decode"]
    filtered = filter_phases(df, selected_phases)
    aggregated = (
        filtered.groupby(["phase", "op_type"], as_index=False)[time_column]
        .sum()
        .sort_values(["phase", "op_type"])
    )
    aggregated = select_top_operations(aggregated, time_column, top_n=top_n)

    matrix = aggregated.pivot_table(
        index="op_type",
        columns="phase",
        values=time_column,
        aggfunc="sum",
        fill_value=0.0,
    )

    column_by_lower = {str(col).strip().lower(): col for col in matrix.columns}
    decode_col = column_by_lower.get("decode")
    prefill_col = column_by_lower.get("prefill")

    if prefill_col is not None:
        matrix["prefill"] = matrix[prefill_col]
    else:
        matrix["prefill"] = 0.0
    if decode_col is not None:
        matrix["decode"] = matrix[decode_col]
    else:
        matrix["decode"] = 0.0
    matrix["delta_time"] = matrix["decode"] - matrix["prefill"]

    matrix = matrix[["prefill", "decode", "delta_time"]]
    op_order = (matrix["prefill"] + matrix["decode"]).sort_values(ascending=False).index
    return matrix.reindex(op_order)


def build_heatmap_matrix(
    df: pd.DataFrame,
    heatmap_kind: str,
    phases: list[str] | None = None,
    top_n: int | None = 20,
    include_special_layers: bool = False,
) -> tuple[pd.DataFrame, str, str, str]:
    kind = heatmap_kind.strip().lower()
    filtered = filter_phases(df, phases)

    if kind == "time":
        value_column = resolve_time_column(filtered)
        matrix = build_layer_metric_matrix(
            filtered,
            value_column=value_column,
            top_n=top_n,
            include_special_layers=include_special_layers,
        )
        return matrix, "Time Heatmap", "Layer", value_column

    if kind in {"memory", "memory-pressure", "llc", "cache-misses"}:
        value_column = resolve_memory_column(filtered)
        matrix = build_layer_metric_matrix(
            filtered,
            value_column=value_column,
            top_n=top_n,
            include_special_layers=include_special_layers,
        )
        return matrix, "Memory Pressure Heatmap", "Layer", value_column

    if kind == "ipc":
        ipc_df = compute_ipc(filtered)
        matrix = build_layer_metric_matrix(
            ipc_df,
            value_column="ipc",
            top_n=top_n,
            include_special_layers=include_special_layers,
        )
        return matrix, "IPC Heatmap", "Layer", "ipc"

    if kind in {"op-share", "operation-share", "share"}:
        matrix = build_operation_share_matrix(filtered, phases=phases, top_n=top_n)
        time_column = resolve_time_column(filtered)
        return matrix, "Phase Time Comparison Heatmap", "Metric", time_column

    raise ValueError(
        "Unknown heatmap kind. Expected one of: time, memory, ipc, op-share."
    )


def plot_heatmap(
    matrix: pd.DataFrame,
    output_path: str | os.PathLike[str],
    title: str,
    xlabel: str,
    value_label: str,
    annotate: bool = False,
) -> Path:
    if matrix.empty:
        raise ValueError("Heatmap matrix is empty, nothing to plot.")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig_width = max(8, 1.1 * max(1, len(matrix.columns)))
    fig_height = max(8, 0.4 * max(1, len(matrix.index)))

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(matrix.values, cmap="YlOrRd", aspect="auto")

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Operation type")
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_yticks(range(len(matrix.index)))
    ax.set_xticklabels(matrix.columns, rotation=45, ha="right")
    ax.set_yticklabels(matrix.index)

    phase_compare_cols = {str(col).strip().lower(): idx for idx, col in enumerate(matrix.columns)}
    prefill_col_idx = phase_compare_cols.get("prefill")
    decode_col_idx = phase_compare_cols.get("decode")
    is_phase_compare = prefill_col_idx is not None and decode_col_idx is not None

    if annotate:
        for row_index, row_name in enumerate(matrix.index):
            for col_index, col_name in enumerate(matrix.columns):
                value = matrix.loc[row_name, col_name]
                if is_phase_compare:
                    lowered_col = str(col_name).strip().lower()
                    if lowered_col in {"delta_time", "delta_ipc"}:
                        label = f"{value:+.2f}" if "ipc" in lowered_col else f"{value:+.0f}"
                    else:
                        label = f"{value:.2f}" if "ipc" in lowered_col or value_label == "ipc" else f"{value:.0f}"
                else:
                    label = f"{value:.2f}" if value_label == "ipc" else f"{value:.0f}"
                ax.text(
                    col_index,
                    row_index,
                    label,
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=8,
                )

    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label(value_label)

    fig.tight_layout()
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output


def create_heatmap(
    *,
    results_dir: str | os.PathLike[str] = DEFAULT_JSON_DIR,
    db_path: str | os.PathLike[str] | None = DEFAULT_SQLITE_DB,
    sql: str | None = None,
    heatmap_kind: str = "time",
    phases: list[str] | None = None,
    top_n: int | None = 20,
    group_file: str | os.PathLike[str] | None = None,
    include_special_layers: bool = False,
    output_path: str | os.PathLike[str] = DEFAULT_OUTPUT,
    matrix_output_path: str | os.PathLike[str] | None = None,
    annotate: bool = False,
) -> tuple[pd.DataFrame, Path]:
    if pd is None:
        raise RuntimeError(
            "pandas is not installed. Install it with `pip install pandas`."
        )
    if plt is None:
        raise RuntimeError(
            "matplotlib is not installed. Install it with `pip install matplotlib`."
        )

    kind = heatmap_kind.strip().lower()
    phase_view_path = resolve_phase_view_path(results_dir)
    decoder_block_path = resolve_decoder_block_view_path(results_dir)

    matrix: pd.DataFrame
    title: str
    xlabel: str
    value_label: str

    if kind in {"op-share", "operation-share", "share"} and phase_view_path is not None:
        matrix, value_label = build_operation_share_matrix_from_phase_json(
            phase_view_path,
            metric_name="total_time_us",
            phases=phases,
            top_n=top_n,
        )
        title = "Phase Time Comparison Heatmap"
        xlabel = "Metric"
    elif kind in {"op-share-ipc", "ipc-share"} and phase_view_path is not None:
        matrix, value_label = build_operation_share_matrix_from_phase_json(
            phase_view_path,
            metric_name="IPC",
            phases=phases,
            top_n=top_n,
        )
        title = "Phase IPC Comparison Heatmap"
        xlabel = "Metric"
    else:
        resolved_db_path = Path(db_path) if db_path is not None else None
        if resolved_db_path is None or not resolved_db_path.is_file():
            raise ValueError(
                "tensor_op_view.db is required for runtime, memory pressure and IPC heatmaps."
            )

        measurements = load_measurements_from_sqlite_db(resolved_db_path, sql=sql)
        normalized = normalize_measurements(measurements)
        grouped = apply_operation_groups(normalized, load_group_mapping(group_file))
        matrix, title, xlabel, value_label = build_heatmap_matrix(
            grouped,
            heatmap_kind=heatmap_kind,
            phases=phases,
            top_n=top_n,
            include_special_layers=include_special_layers,
        )

    if matrix_output_path is not None:
        matrix_output = Path(matrix_output_path)
        matrix_output.parent.mkdir(parents=True, exist_ok=True)
        matrix.to_csv(matrix_output)

    plot_path = plot_heatmap(
        matrix,
        output_path=output_path,
        title=title,
        xlabel=xlabel,
        value_label=value_label,
        annotate=annotate,
    )
    return matrix, plot_path


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create analysis heatmaps from run_every_view_results JSON outputs, with SQLite fallback."
    )
    parser.add_argument(
        "--results-dir",
        default=str(DEFAULT_JSON_DIR),
        help="Directory containing phase-view.json, decoder-block-view.json and tensor_op_view.db",
    )
    parser.add_argument(
        "--db-path",
        default=str(DEFAULT_SQLITE_DB),
        help="Optional SQLite fallback database path, typically run_every_view_results/tensor_op_view.db",
    )
    parser.add_argument("--sql", help="Custom SQL query to use when falling back to SQLite")
    parser.add_argument(
        "--kind",
        default="time",
        choices=["time", "memory", "ipc", "op-share", "op-share-ipc"],
        help="Which heatmap to build (op-share compares time, op-share-ipc compares IPC between prefill and decode)",
    )
    parser.add_argument(
        "--phase",
        dest="phases",
        action="append",
        help="Phase(s) to include. Repeat the flag for multiple phases, e.g. --phase prefill --phase decode",
    )
    parser.add_argument("--top-n", type=int, default=20, help="Keep the top N operation types")
    parser.add_argument("--group-file", help="JSON file that maps detailed op_type names into broader groups")
    parser.add_argument(
        "--include-special-layers",
        action="store_true",
        help="Keep non-numbered tensor names as synthetic layers like other/final",
    )
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT), help="Where to save the heatmap image")
    parser.add_argument("--matrix-out", help="Optional path for saving the heatmap matrix as CSV")
    parser.add_argument("--annotate", action="store_true", help="Write values inside each heatmap cell")
    return parser


def _prompt_text(
    prompt: str,
    *,
    default: str | None = None,
    required: bool = False,
) -> str | None:
    while True:
        suffix = f" [{default}]" if default is not None else ""
        raw = input(f"{prompt}{suffix}: ").strip()

        if raw:
            return raw
        if default is not None:
            return default
        if not required:
            return None
        print("This value is required.")


def _prompt_int(
    prompt: str,
    *,
    default: int | None = None,
    minimum: int | None = None,
    allow_empty: bool = True,
) -> int | None:
    while True:
        default_text = f" [{default}]" if default is not None else ""
        raw = input(f"{prompt}{default_text}: ").strip()

        if not raw:
            if default is not None:
                return default
            if allow_empty:
                return None
            print("Please enter a number.")
            continue

        try:
            value = int(raw)
        except ValueError:
            print("Please enter a valid integer.")
            continue

        if minimum is not None and value < minimum:
            print(f"Please enter a value >= {minimum}.")
            continue
        return value


def _prompt_choice(
    prompt: str,
    *,
    options: list[str],
    default_index: int | None = 1,
) -> int:
    print(prompt)
    for index, option in enumerate(options, start=1):
        print(f"{index}. {option}")

    min_index = 1
    max_index = len(options)
    while True:
        default_text = f" [{default_index}]" if default_index is not None else ""
        raw = input(f"Choose number{default_text}: ").strip()
        if not raw and default_index is not None:
            return default_index
        if not raw.isdigit():
            print("Please enter a valid number.")
            continue
        selected = int(raw)
        if selected < min_index or selected > max_index:
            print(f"Please choose a number between {min_index} and {max_index}.")
            continue
        return selected


def prompt_interactive_args() -> argparse.Namespace:
    print("Interactive heatmap quick setup")
    print("Answer with numbers.")
    print()

    kind_options = ["time", "memory", "ipc", "op-share", "op-share-ipc"]
    kind_labels = {
        "time": "Runtime heatmap",
        "memory": "Memory pressure heatmap",
        "ipc": "IPC heatmap",
        "op-share": "Phase-compare runtime heatmap",
        "op-share-ipc": "Phase-compare IPC heatmap",
    }
    kind_choice = _prompt_choice(
        "Choose heatmap kind:",
        options=[kind_labels[name] for name in kind_options],
        default_index=1,
    )
    kind = kind_options[kind_choice - 1]

    phases = None
    if kind in {"op-share", "op-share-ipc"}:
        phases = ["prefill", "decode"]

    return argparse.Namespace(
        results_dir=str(DEFAULT_JSON_DIR),
        db_path=str(DEFAULT_SQLITE_DB),
        sql=None,
        kind=kind,
        phases=phases,
        top_n=20,
        group_file=None,
        include_special_layers=False,
        output=str(DEFAULT_OUTPUT),
        matrix_out=None,
        annotate=True,
    )


def main() -> None:
    parser = build_argument_parser()
    if len(sys.argv) == 1:
        args = prompt_interactive_args()
    else:
        args = parser.parse_args()

    matrix, plot_path = create_heatmap(
        results_dir=args.results_dir,
        db_path=args.db_path,
        sql=args.sql,
        heatmap_kind=args.kind,
        phases=args.phases,
        top_n=args.top_n,
        group_file=args.group_file,
        include_special_layers=args.include_special_layers,
        output_path=args.output,
        matrix_output_path=args.matrix_out,
        annotate=args.annotate,
    )

    print(f"Saved heatmap to: {plot_path}")
    print(f"Matrix shape: {matrix.shape[0]} x {matrix.shape[1]}")


if __name__ == "__main__":
    main()
