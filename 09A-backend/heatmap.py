#!/usr/bin/env python3
"""
Create profiler heatmaps for prefill/decode analysis.

Supported heatmaps:
    1. time
       Runtime heatmap
    2. memory
       Memory heatmap
    3. ipc
       IPC heatmap
    4. op-share
       Phase-compare runtime heatmap
    5. op-share-ipc
       Phase-compare IPC heatmap
    6. op-share-memory
       Phase-compare memory heatmap
"""

from __future__ import annotations

import argparse
import json
import math
import numpy as np
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
    from matplotlib import colors
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover
    matplotlib = None
    colors = None
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
DEFAULT_JSON_OUTPUT = SCRIPT_DIR / "plots" / "heatmap.json"

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


def build_operation_share_matrix_from_phase_json(
    phase_view_path: str | os.PathLike[str],
    metric_name: str = "total_time_us",
    metric_candidates: list[str] | None = None,
    phases: list[str] | None = None,
    top_n: int | None = 20,
) -> tuple[pd.DataFrame, str]:
    raw = load_json_file(phase_view_path)
    if not isinstance(raw, dict):
        raise ValueError("phase-view.json must contain a JSON object.")

    selected_phases = phases if phases else ["prefill", "decode"]
    candidate_names = metric_candidates if metric_candidates else [metric_name]
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
            metric_value = None
            for candidate_name in candidate_names:
                if candidate_name in metrics and metrics[candidate_name] is not None:
                    metric_value = metrics[candidate_name]
                    break
            if metric_value is None and metric_candidates is not None:
                continue
            if metric_value is None:
                metric_value = 0
            records.append(
                {
                    "phase": phase,
                    "op_type": op_type,
                    metric_name: float(metric_value),
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
    delta_column_by_metric = {
        "IPC": "delta_ipc",
        "bytes_moved": "delta_memory",
        "total_bytes": "delta_memory",
        "memory_bytes": "delta_memory",
        "papi_l3_tcm": "delta_memory",
    }
    delta_column = delta_column_by_metric.get(metric_name, "delta_time")
    matrix[delta_column] = matrix["decode"] - matrix["prefill"]
    matrix = matrix[["prefill", "decode", delta_column]]
    matrix = select_top_matrix_rows(matrix, top_n)
    op_order = (matrix["prefill"] + matrix["decode"]).sort_values(ascending=False).index
    value_label_by_metric = {
        "IPC": "ipc",
        "bytes_moved": "bytes_moved",
        "total_bytes": "total_bytes",
        "memory_bytes": "memory_bytes",
        "papi_l3_tcm": "papi_l3_tcm",
    }
    value_label = value_label_by_metric.get(metric_name, "time_us")
    return matrix.reindex(op_order), value_label


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
    if "papi_l3_tcm" in df.columns:
        return "papi_l3_tcm"
    raise ValueError(
        "L3 cache miss heatmap requires papi_l3_tcm. L2/L1 fallback is disabled."
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


def build_layer_ipc_matrix(
    df: pd.DataFrame,
    top_n: int | None = 20,
    include_special_layers: bool = False,
) -> pd.DataFrame:
    if "tot_ins" not in df.columns or "tot_cyc" not in df.columns:
        raise ValueError(
            "IPC requires tot_ins and tot_cyc columns. Add them in the DB/PAPI collection first."
        )

    layered = add_layer_column(df, include_special_layers=include_special_layers)
    if layered.empty:
        raise ValueError("No rows with layer information were found.")

    aggregated = (
        layered.groupby(["op_type", "layer"], as_index=False)[["tot_ins", "tot_cyc"]]
        .sum()
        .sort_values(["op_type", "layer"])
    )
    aggregated["ipc"] = 0.0
    non_zero_cycles = aggregated["tot_cyc"] != 0
    aggregated.loc[non_zero_cycles, "ipc"] = (
        aggregated.loc[non_zero_cycles, "tot_ins"]
        / aggregated.loc[non_zero_cycles, "tot_cyc"]
    )

    if top_n is not None and top_n > 0:
        top_ops = (
            aggregated.groupby("op_type", as_index=False)["tot_cyc"]
            .sum()
            .sort_values("tot_cyc", ascending=False)
            .head(top_n)["op_type"]
        )
        aggregated = aggregated[aggregated["op_type"].isin(set(top_ops))].copy()

    matrix = aggregated.pivot_table(
        index="op_type",
        columns="layer",
        values="ipc",
        aggfunc="first",
        fill_value=0.0,
    )

    sorted_columns = sorted(
        matrix.columns,
        key=lambda layer: (not str(layer).startswith("L"), str(layer)),
    )
    matrix = matrix.reindex(columns=sorted_columns)
    op_order = matrix.mean(axis=1).sort_values(ascending=False).index
    return matrix.reindex(op_order)


def build_operation_share_matrix(
    df: pd.DataFrame,
    phases: list[str] | None = None,
    top_n: int | None = 20,
    value_column: str | None = None,
    delta_column: str = "delta_time",
) -> pd.DataFrame:
    metric_column = value_column if value_column is not None else resolve_time_column(df)
    selected_phases = phases if phases else ["prefill", "decode"]
    filtered = filter_phases(df, selected_phases)
    aggregated = (
        filtered.groupby(["phase", "op_type"], as_index=False)[metric_column]
        .sum()
        .sort_values(["phase", "op_type"])
    )
    aggregated = select_top_operations(aggregated, metric_column, top_n=top_n)

    matrix = aggregated.pivot_table(
        index="op_type",
        columns="phase",
        values=metric_column,
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
    matrix[delta_column] = matrix["decode"] - matrix["prefill"]

    matrix = matrix[["prefill", "decode", delta_column]]
    op_order = (matrix["prefill"] + matrix["decode"]).sort_values(ascending=False).index
    return matrix.reindex(op_order)


def build_operation_share_ipc_matrix(
    df: pd.DataFrame,
    phases: list[str] | None = None,
    top_n: int | None = 20,
) -> pd.DataFrame:
    selected_phases = phases if phases else ["prefill", "decode"]
    filtered = filter_phases(df, selected_phases)
    if "tot_ins" not in filtered.columns or "tot_cyc" not in filtered.columns:
        raise ValueError(
            "Phase IPC requires tot_ins and tot_cyc columns. Add them in the DB/PAPI collection first."
        )

    aggregated = (
        filtered.groupby(["phase", "op_type"], as_index=False)[["tot_ins", "tot_cyc"]]
        .sum()
        .sort_values(["phase", "op_type"])
    )
    aggregated["ipc"] = 0.0
    non_zero_cycles = aggregated["tot_cyc"] != 0
    aggregated.loc[non_zero_cycles, "ipc"] = (
        aggregated.loc[non_zero_cycles, "tot_ins"]
        / aggregated.loc[non_zero_cycles, "tot_cyc"]
    )

    if top_n is not None and top_n > 0:
        top_ops = (
            aggregated.groupby("op_type", as_index=False)["tot_cyc"]
            .sum()
            .sort_values("tot_cyc", ascending=False)
            .head(top_n)["op_type"]
        )
        aggregated = aggregated[aggregated["op_type"].isin(set(top_ops))].copy()

    matrix = aggregated.pivot_table(
        index="op_type",
        columns="phase",
        values="ipc",
        aggfunc="first",
        fill_value=0.0,
    )

    column_by_lower = {str(col).strip().lower(): col for col in matrix.columns}
    decode_col = column_by_lower.get("decode")
    prefill_col = column_by_lower.get("prefill")
    matrix["prefill"] = matrix[prefill_col] if prefill_col is not None else 0.0
    matrix["decode"] = matrix[decode_col] if decode_col is not None else 0.0
    matrix["delta_ipc"] = matrix["decode"] - matrix["prefill"]
    matrix = matrix[["prefill", "decode", "delta_ipc"]]
    op_order = matrix[["prefill", "decode"]].mean(axis=1).sort_values(ascending=False).index
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
        return matrix, "Memory Heatmap", "Layer", value_column

    if kind == "ipc":
        matrix = build_layer_ipc_matrix(
            filtered,
            top_n=top_n,
            include_special_layers=include_special_layers,
        )
        return matrix, "IPC Heatmap", "Layer", "ipc"

    if kind in {"op-share", "operation-share", "share"}:
        matrix = build_operation_share_matrix(filtered, phases=phases, top_n=top_n)
        time_column = resolve_time_column(filtered)
        return matrix, "Phase Runtime Heatmap", "Metric", time_column

    if kind in {"op-share-memory", "memory-share"}:
        value_column = resolve_memory_column(filtered)
        matrix = build_operation_share_matrix(
            filtered,
            phases=phases,
            top_n=top_n,
            value_column=value_column,
            delta_column="delta_memory",
        )
        return matrix, "Phase Memory Heatmap", "Metric", value_column

    if kind in {"op-share-ipc", "ipc-share"}:
        matrix = build_operation_share_ipc_matrix(filtered, phases=phases, top_n=top_n)
        return matrix, "Phase IPC Heatmap", "Metric", "ipc"

    raise ValueError(
        "Unknown heatmap kind. Expected one of: time, memory, ipc, op-share, op-share-ipc, op-share-memory."
    )


def plot_heatmap(
    matrix: pd.DataFrame,
    output_path: str | os.PathLike[str],
    title: str,
    xlabel: str,
    value_label: str,
    display_mode: str = "default",
    annotate: bool = False,
) -> Path:
    if matrix.empty:
        raise ValueError("Heatmap matrix is empty, nothing to plot.")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig_width = max(8, 1.1 * max(1, len(matrix.columns)))
    fig_height = max(8, 0.4 * max(1, len(matrix.index)))

    display_matrix = matrix.astype(float).copy()
    colorbar_label = value_label
    color_norm = None
    cmap = plt.get_cmap("YlOrRd").copy()
    image_values = display_matrix.values

    if display_mode in {"time-log", "time-symlog"}:
        positive_values = display_matrix.values[display_matrix.values > 0]
        if positive_values.size > 0:
            vmax = float(np.percentile(positive_values, 98))
            if display_mode == "time-symlog":
                max_abs = float(np.percentile(np.abs(display_matrix.values), 98))
                if max_abs > 0:
                    linear_width = max(1.0, float(np.percentile(positive_values, 10)))
                    color_norm = colors.SymLogNorm(
                        linthresh=linear_width,
                        vmin=-max_abs,
                        vmax=max_abs,
                        clip=True,
                    )
                    cmap = plt.get_cmap("RdBu_r").copy()
                    colorbar_label = f"sym-log {value_label} (clipped at p98)"
            else:
                vmin = float(np.min(positive_values))
                if vmax <= vmin:
                    vmax = float(np.max(positive_values))
                if vmax > vmin:
                    color_norm = colors.LogNorm(vmin=vmin, vmax=vmax, clip=True)
                    image_values = np.ma.masked_less_equal(display_matrix.values, 0)
                    cmap.set_bad("#fffde7")
                    colorbar_label = f"log-scaled {value_label} (clipped at p98)"

    if display_mode in {"memory-log", "memory-symlog"}:
        positive_values = display_matrix.values[display_matrix.values > 0]
        if positive_values.size > 0:
            vmax = float(np.percentile(positive_values, 98))
            if vmax <= 0:
                vmax = float(np.max(positive_values))
            if display_mode == "memory-symlog":
                max_abs = float(np.percentile(np.abs(display_matrix.values), 98))
                if max_abs > 0:
                    linear_width = max(1.0, float(np.percentile(positive_values, 10)))
                    color_norm = colors.SymLogNorm(
                        linthresh=linear_width,
                        vmin=-max_abs,
                        vmax=max_abs,
                        clip=True,
                    )
                    cmap = plt.get_cmap("RdBu_r").copy()
                    colorbar_label = f"sym-log {value_label} (clipped at p98)"
            else:
                vmin = float(np.min(positive_values))
                if vmax <= vmin:
                    vmax = float(np.max(positive_values))
                if vmax > vmin:
                    color_norm = colors.LogNorm(vmin=vmin, vmax=vmax, clip=True)
                    image_values = np.ma.masked_less_equal(display_matrix.values, 0)
                    cmap.set_bad("#fffde7")
                    colorbar_label = f"log-scaled {value_label} (clipped at p98)"

    if display_mode == "ipc-diverging":
        max_abs = float(np.max(np.abs(display_matrix.values)))
        if max_abs > 0:
            color_norm = colors.TwoSlopeNorm(
                vmin=-max_abs,
                vcenter=0.0,
                vmax=max_abs,
            )
            cmap = plt.get_cmap("RdBu_r").copy()
            colorbar_label = f"{value_label} delta-centered at 0"

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    image = ax.imshow(image_values, cmap=cmap, norm=color_norm, aspect="auto")

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
                    if lowered_col in {"delta_time", "delta_ipc", "delta_memory"}:
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
    colorbar.set_label(colorbar_label)

    fig.tight_layout()
    fig.savefig(output, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output


def _json_safe_number(value: object) -> float | int | None:
    if pd.isna(value):
        return None
    numeric = float(value)
    if numeric.is_integer():
        return int(numeric)
    return numeric


def _finite_matrix_values(matrix: pd.DataFrame) -> np.ndarray:
    values = matrix.astype(float).values
    return values[np.isfinite(values)]


def build_color_scale_payload(
    matrix: pd.DataFrame,
    *,
    display_mode: str,
    value_label: str,
) -> dict[str, object]:
    values = _finite_matrix_values(matrix)
    positive_values = values[values > 0]

    color_scale: dict[str, object] = {
        "palette": "YlOrRd",
        "scale": "linear",
        "label": value_label,
        "higher_is_hotter": True,
        "zero_centered": False,
    }

    if values.size == 0:
        return color_scale

    color_scale["data_min"] = _json_safe_number(np.min(values))
    color_scale["data_max"] = _json_safe_number(np.max(values))

    if display_mode in {"time-log", "memory-log"} and positive_values.size > 0:
        vmin = float(np.min(positive_values))
        vmax = float(np.percentile(positive_values, 98))
        if vmax <= vmin:
            vmax = float(np.max(positive_values))
        color_scale.update(
            {
                "palette": "YlOrRd",
                "scale": "log",
                "label": f"log-scaled {value_label} (clipped at p98)",
                "vmin": _json_safe_number(vmin),
                "vmax": _json_safe_number(vmax),
                "clip": True,
                "clip_percentile": 98,
                "masked_values": "<= 0",
                "masked_color": "#fffde7",
            }
        )
        return color_scale

    if display_mode in {"time-symlog", "memory-symlog"} and positive_values.size > 0:
        max_abs = float(np.percentile(np.abs(values), 98))
        if max_abs > 0:
            linthresh = max(1.0, float(np.percentile(positive_values, 10)))
            color_scale.update(
                {
                    "palette": "RdBu_r",
                    "scale": "symlog",
                    "label": f"sym-log {value_label} (clipped at p98)",
                    "vmin": _json_safe_number(-max_abs),
                    "vcenter": 0,
                    "vmax": _json_safe_number(max_abs),
                    "linthresh": _json_safe_number(linthresh),
                    "clip": True,
                    "clip_percentile": 98,
                    "zero_centered": True,
                    "higher_is_hotter": False,
                }
            )
            return color_scale

    if display_mode == "ipc-diverging":
        max_abs = float(np.max(np.abs(values)))
        if max_abs > 0:
            color_scale.update(
                {
                    "palette": "RdBu_r",
                    "scale": "diverging",
                    "label": f"{value_label} delta-centered at 0",
                    "vmin": _json_safe_number(-max_abs),
                    "vcenter": 0,
                    "vmax": _json_safe_number(max_abs),
                    "zero_centered": True,
                    "higher_is_hotter": False,
                }
            )
            return color_scale

    color_scale.update(
        {
            "vmin": _json_safe_number(np.min(values)),
            "vmax": _json_safe_number(np.max(values)),
        }
    )
    return color_scale


def build_heatmap_json_payload(
    matrix: pd.DataFrame,
    *,
    heatmap_kind: str,
    title: str,
    xlabel: str,
    value_label: str,
    display_mode: str,
) -> dict[str, object]:
    rows = [str(row) for row in matrix.index]
    columns = [str(column) for column in matrix.columns]
    values = [
        [
            _json_safe_number(matrix.iloc[row_index, column_index])
            for column_index in range(len(columns))
        ]
        for row_index in range(len(rows))
    ]

    return {
        "schema_version": 1,
        "kind": heatmap_kind,
        "title": title,
        "x_label": xlabel,
        "y_label": "Operation type",
        "value_label": value_label,
        "color_scale": build_color_scale_payload(
            matrix,
            display_mode=display_mode,
            value_label=value_label,
        ),
        "rows": rows,
        "columns": columns,
        "values": values,
    }


def write_heatmap_json(
    matrix: pd.DataFrame,
    output_path: str | os.PathLike[str],
    *,
    heatmap_kind: str,
    title: str,
    xlabel: str,
    value_label: str,
    display_mode: str,
) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = build_heatmap_json_payload(
        matrix,
        heatmap_kind=heatmap_kind,
        title=title,
        xlabel=xlabel,
        value_label=value_label,
        display_mode=display_mode,
    )
    with open(output, "w", encoding="utf-8") as handle:
        handle.write(format_json_for_ui(payload))
        handle.write("\n")
    return output


def is_json_scalar(value: object) -> bool:
    return value is None or isinstance(value, (str, int, float, bool))


def format_json_for_ui(value: object, indent: int = 0) -> str:
    space = " " * indent
    next_indent = indent + 2
    next_space = " " * next_indent

    if is_json_scalar(value):
        return json.dumps(value)

    if isinstance(value, list):
        if not value:
            return "[]"
        if all(is_json_scalar(item) for item in value):
            return "[" + ", ".join(json.dumps(item) for item in value) + "]"

        lines = ["["]
        for index, item in enumerate(value):
            suffix = "," if index < len(value) - 1 else ""
            lines.append(f"{next_space}{format_json_for_ui(item, next_indent)}{suffix}")
        lines.append(f"{space}]")
        return "\n".join(lines)

    if isinstance(value, dict):
        if not value:
            return "{}"

        items = list(value.items())
        lines = ["{"]
        for index, (key, item) in enumerate(items):
            suffix = "," if index < len(items) - 1 else ""
            formatted_item = format_json_for_ui(item, next_indent)
            lines.append(f"{next_space}{json.dumps(str(key))}: {formatted_item}{suffix}")
        lines.append(f"{space}}}")
        return "\n".join(lines)

    return json.dumps(value)


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
    json_output_path: str | os.PathLike[str] | None = DEFAULT_JSON_OUTPUT,
    annotate: bool = False,
) -> tuple[pd.DataFrame, Path, Path | None]:
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

    matrix: pd.DataFrame
    title: str
    xlabel: str
    value_label: str
    display_mode = "default"
    handled = False

    if kind in {"op-share", "operation-share", "share"} and phase_view_path is not None:
        try:
            matrix, value_label = build_operation_share_matrix_from_phase_json(
                phase_view_path,
                metric_name="total_time_us",
                phases=phases,
                top_n=top_n,
            )
            title = "Phase Runtime Heatmap"
            xlabel = "Metric"
            display_mode = "time-symlog"
            handled = True
        except ValueError:
            handled = False
    elif kind in {"op-share-ipc", "ipc-share"} and phase_view_path is not None:
        try:
            matrix, value_label = build_operation_share_matrix_from_phase_json(
                phase_view_path,
                metric_name="IPC",
                phases=phases,
                top_n=top_n,
            )
            title = "Phase IPC Heatmap"
            xlabel = "Metric"
            display_mode = "ipc-diverging"
            handled = True
        except ValueError:
            handled = False
    elif kind in {"op-share-memory", "memory-share"} and phase_view_path is not None:
        try:
            matrix, value_label = build_operation_share_matrix_from_phase_json(
                phase_view_path,
                metric_name="memory_bytes",
                metric_candidates=["bytes_moved", "total_bytes"],
                phases=phases,
                top_n=top_n,
            )
            title = "Phase Memory Heatmap"
            xlabel = "Metric"
            display_mode = "memory-symlog"
            handled = True
        except ValueError:
            handled = False

    if not handled:
        resolved_db_path = Path(db_path) if db_path is not None else None
        if resolved_db_path is None or not resolved_db_path.is_file():
            raise ValueError(
                "tensor_op_view.db is required for runtime, L3 cache miss and IPC heatmaps."
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
        if kind == "time":
            display_mode = "time-log"
        elif kind in {"op-share", "operation-share", "share"}:
            display_mode = "time-symlog"
        elif kind in {"op-share-memory", "memory-share"}:
            display_mode = "memory-symlog"
        elif kind in {"memory", "memory-pressure", "llc", "cache-misses"}:
            display_mode = "memory-log"
        elif kind in {"op-share-ipc", "ipc-share"}:
            display_mode = "ipc-diverging"

    if matrix_output_path is not None:
        matrix_output = Path(matrix_output_path)
        matrix_output.parent.mkdir(parents=True, exist_ok=True)
        matrix.to_csv(matrix_output)

    json_path = None
    if json_output_path is not None:
        json_path = write_heatmap_json(
            matrix,
            json_output_path,
            heatmap_kind=kind,
            title=title,
            xlabel=xlabel,
            value_label=value_label,
            display_mode=display_mode,
        )

    plot_path = plot_heatmap(
        matrix,
        output_path=output_path,
        title=title,
        xlabel=xlabel,
        value_label=value_label,
        display_mode=display_mode,
        annotate=annotate,
    )
    return matrix, plot_path, json_path


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create analysis heatmaps from run_every_view_results JSON outputs, with SQLite fallback."
    )
    parser.add_argument(
        "--results-dir",
        default=str(DEFAULT_JSON_DIR),
        help="Directory containing phase-view.json and tensor_op_view.db",
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
        choices=["time", "memory", "ipc", "op-share", "op-share-ipc", "op-share-memory"],
        help="Which heatmap to build (op-share compares time, op-share-ipc compares IPC, op-share-memory compares memory between prefill and decode)",
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
    parser.add_argument(
        "--json-out",
        default=str(DEFAULT_JSON_OUTPUT),
        help="Where to save the heatmap data as JSON for the UI",
    )
    parser.add_argument(
        "--no-json-out",
        action="store_true",
        help="Do not write the heatmap data JSON file",
    )
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

    view_options = ["regular", "phase"]
    view_choice = _prompt_choice(
        "Choose heatmap view:",
        options=["Regular layer heatmap", "Phase comparison heatmap"],
        default_index=1,
    )
    selected_view = view_options[view_choice - 1]

    metric_options_by_view = {
        "regular": [
            ("time", "Runtime heatmap"),
            ("memory", "Memory heatmap"),
            ("ipc", "IPC heatmap"),
        ],
        "phase": [
            ("op-share", "Phase runtime heatmap"),
            ("op-share-memory", "Phase memory heatmap"),
            ("op-share-ipc", "Phase IPC heatmap"),
        ],
    }
    metric_options = metric_options_by_view[selected_view]
    metric_choice = _prompt_choice(
        "Choose metric:",
        options=[label for _, label in metric_options],
        default_index=1,
    )
    kind = metric_options[metric_choice - 1][0]

    phases = None
    if selected_view == "phase":
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
        json_out=str(DEFAULT_JSON_OUTPUT),
        no_json_out=False,
        annotate=True,
    )


def main() -> None:
    parser = build_argument_parser()
    if len(sys.argv) == 1:
        args = prompt_interactive_args()
    else:
        args = parser.parse_args()

    json_output_path = None if args.no_json_out else args.json_out
    matrix, plot_path, json_path = create_heatmap(
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
        json_output_path=json_output_path,
        annotate=args.annotate,
    )

    print(f"Saved heatmap to: {plot_path}")
    if json_path is not None:
        print(f"Saved heatmap data JSON to: {json_path}")
    print(f"Matrix shape: {matrix.shape[0]} x {matrix.shape[1]}")


if __name__ == "__main__":
    main()
