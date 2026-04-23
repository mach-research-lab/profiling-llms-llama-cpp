#!/usr/bin/env python3
"""
Create profiler heatmaps for prefill/decode analysis.

Supported heatmaps:
    1. time
       x = layer
       y = op_type
       value = time spent
    2. memory
       x = layer
       y = op_type
       value = cache-miss pressure (prefers LLC/L3, then L2)
    3. ipc
       x = layer
       y = op_type
       value = TOT_INS / TOT_CYC
    4. op-share
       x = phase
       y = op_type
       value = share of total time in each phase
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Mapping

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
    filtered = filter_phases(df, phases)
    aggregated = (
        filtered.groupby(["phase", "op_type"], as_index=False)[time_column]
        .sum()
        .sort_values(["phase", "op_type"])
    )
    aggregated = select_top_operations(aggregated, time_column, top_n=top_n)

    phase_totals = aggregated.groupby("phase", as_index=False)[time_column].sum()
    phase_totals = phase_totals.rename(columns={time_column: "phase_total"})
    shares = aggregated.merge(phase_totals, on="phase", how="left")
    shares["share"] = shares[time_column] / shares["phase_total"]

    matrix = shares.pivot_table(
        index="op_type",
        columns="phase",
        values="share",
        aggfunc="sum",
        fill_value=0.0,
    )

    column_by_lower = {str(col).strip().lower(): col for col in matrix.columns}
    decode_col = column_by_lower.get("decode")
    prefill_col = column_by_lower.get("prefill")
    if decode_col is not None and prefill_col is not None:
        matrix["delta_share"] = matrix[decode_col] - matrix[prefill_col]

    op_order = matrix.sum(axis=1).sort_values(ascending=False).index
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
        return matrix, "Operation Share Heatmap", "Phase", "share"

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

    if annotate:
        is_share = value_label == "share"
        for row_index, row_name in enumerate(matrix.index):
            for col_index, col_name in enumerate(matrix.columns):
                value = matrix.loc[row_name, col_name]
                if is_share:
                    if str(col_name).strip().lower() == "delta_share":
                        label = f"{value * 100.0:+.1f}pp"
                    else:
                        label = f"{value:.1%}"
                else:
                    label = f"{value:.0f}"
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
    csv_path: str | os.PathLike[str] | None = None,
    dsn: str | None = None,
    run_id: int | None = None,
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

    if csv_path is None and dsn is None:
        raise ValueError("Provide either csv_path or dsn.")

    if csv_path is not None:
        measurements = load_measurements_from_csv(csv_path)
    else:
        measurements = load_measurements_from_db(dsn=dsn, run_id=run_id, sql=sql)

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
        annotate=annotate or value_label == "share",
    )
    return matrix, plot_path


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create analysis heatmaps for time, memory pressure, IPC and operation share."
    )
    parser.add_argument("--csv", dest="csv_path", help="Path to measurements.csv")
    parser.add_argument(
        "--dsn",
        help="PostgreSQL DSN, for example postgresql://user:pass@localhost:5434/toolDB",
    )
    parser.add_argument("--run-id", type=int, help="Optional run_id filter when reading from DB")
    parser.add_argument("--sql", help="Custom SQL query to use instead of the default event_item query")
    parser.add_argument(
        "--kind",
        default="time",
        choices=["time", "memory", "ipc", "op-share"],
        help="Which heatmap to build",
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


def _csv_looks_like_measurements(csv_path: Path) -> bool:
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader, None)
    except (OSError, UnicodeDecodeError):
        return False

    if not header:
        return False

    normalized = {column.strip().lower() for column in header}
    has_phase = "phase" in normalized or "event_phase" in normalized
    has_op_type = "op_type" in normalized or "event_operation_type" in normalized
    return has_phase and has_op_type


def _discover_csv_paths(limit: int = 12) -> list[str]:
    roots = [Path.cwd(), SCRIPT_DIR, SCRIPT_DIR.parent]
    seen: set[Path] = set()
    targeted: list[Path] = []
    preferred: list[Path] = []
    fallback: list[Path] = []

    def add_candidate(path: Path, bucket: list[Path]) -> None:
        resolved = path.resolve()
        if resolved in seen or not path.is_file():
            return
        seen.add(resolved)
        bucket.append(path)

    for root in roots:
        if not root.exists():
            continue

        # First priority: explicit project CSVs users typically care about.
        for path in sorted(root.rglob("measurements.csv")):
            add_candidate(path, targeted)

        run_all_results_dir = root / "run_all_results"
        if run_all_results_dir.exists():
            for path in sorted(run_all_results_dir.rglob("*.csv")):
                add_candidate(path, targeted)

    if targeted:
        return [str(path) for path in targeted[:limit]]

    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*.csv")):
            if _csv_looks_like_measurements(path):
                add_candidate(path, preferred)
            else:
                add_candidate(path, fallback)

    combined = preferred + fallback
    return [str(path) for path in combined[:limit]]


def prompt_interactive_args() -> argparse.Namespace:
    print("Interactive heatmap quick setup")
    print("Answer with numbers.")
    print()

    csv_candidates = _discover_csv_paths()
    names = [Path(path).name for path in csv_candidates]
    name_counts = Counter(names)
    seen_name_counts: dict[str, int] = {}
    csv_options: list[str] = []
    for name in names:
        if name_counts[name] == 1:
            csv_options.append(f"Use {name}")
            continue
        seen_name_counts[name] = seen_name_counts.get(name, 0) + 1
        csv_options.append(f"Use {name} ({seen_name_counts[name]})")
    csv_options.append("Type a custom CSV path")
    csv_choice = _prompt_choice("Choose CSV source:", options=csv_options, default_index=1)
    if csv_choice == len(csv_options):
        csv_path = _prompt_text("CSV path", required=True)
        assert csv_path is not None
    else:
        csv_path = csv_candidates[csv_choice - 1]

    kind_options = ["time", "memory", "ipc", "op-share"]
    kind_choice = _prompt_choice(
        "Choose heatmap kind:",
        options=[f"{name} heatmap" for name in kind_options],
        default_index=1,
    )
    kind = kind_options[kind_choice - 1]

    phases = None
    if kind == "op-share":
        phase_choice = _prompt_choice(
            "Choose phase filter:",
            options=[
                "All phases",
                "Prefill only",
                "Decode only",
            ],
            default_index=1,
        )
        if phase_choice == 2:
            phases = ["prefill"]
        elif phase_choice == 3:
            phases = ["decode"]

    return argparse.Namespace(
        csv_path=csv_path,
        dsn=None,
        run_id=None,
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

    if not args.csv_path and not args.dsn:
        parser.error("one of --csv or --dsn is required")

    matrix, plot_path = create_heatmap(
        csv_path=args.csv_path,
        dsn=args.dsn,
        run_id=args.run_id,
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
