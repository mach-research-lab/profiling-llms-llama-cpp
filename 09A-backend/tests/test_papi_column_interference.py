#!/usr/bin/env python3
"""
test_papi_column_interference.py — Does activating more PAPI events change
ALL tensor metrics, not just the PAPI counters?

Three runs of llama-papi with increasing event counts:
  Run 1 (1 event):  PAPI_L1_DCM
  Run 2 (2 events): PAPI_L1_DCM, PAPI_L1_ICM
  Run 3 (3 events): PAPI_L1_DCM, PAPI_L1_ICM, PAPI_L2_DCM

For every numeric column present in all three runs (time_ns, size_bytes,
n_elements, papi_l1_dcm), row values are summed and compared across runs,
grouped two ways:
  1. By phase         — tokenization / prefill / decode
  2. By decode level  — per token_index within the decode phase

An 8-panel plot (2 rows × 4 columns) shows phase-level totals on the top row
and decode-level totals on the bottom row, with runs shown as side-by-side bars
so any interference is immediately visible.

Run directly:
    python tests/test_papi_column_interference.py
Or via pytest:
    pytest tests/test_papi_column_interference.py -v
"""

import csv
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import pytest

# ── Paths ──────────────────────────────────────────────────────────────────────

LLAMA_ROOT = Path(__file__).parents[2]
BIN_DIR    = LLAMA_ROOT / "build/bin"
MODEL_PATH = (
    LLAMA_ROOT
    / "models/models"
    / "bartowski_Qwen2.5-1.5B-Instruct-GGUF"
    / "Qwen2.5-1.5B-Instruct-Q4_K_M.gguf"
)
OUT_DIR = Path(__file__).parent / "test_outputs" / "papi_column_interference"

# ── Run parameters ─────────────────────────────────────────────────────────────

PROMPT    = "Hello"
N_PREDICT = "5"
TEMP      = "0"
BINARY    = "llama-papi"

EVENT_SETS = [
    ["PAPI_L1_DCM"],
    ["PAPI_L1_DCM", "PAPI_L1_ICM"],
    ["PAPI_L1_DCM", "PAPI_L1_ICM", "PAPI_L2_DCM"],
]

# Columns present in every run — used for cross-run comparison.
# size_bytes / n_elements are tensor properties (should be stable, good control).
# time_ns measures wall-clock overhead. papi_l1_dcm is the main counter of interest.
NUMERIC_COLS = ["time_ns", "size_bytes", "n_elements", "papi_l1_dcm"]

TOLERANCE = 0.15


# ── CSV parsing & aggregation ──────────────────────────────────────────────────

def parse_csv(path: Path) -> list[dict]:
    """Return every data row as a dict with lower-cased keys."""
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        return [{k.lower(): v for k, v in row.items()} for row in reader]


def _numeric(raw: str) -> float:
    try:
        return float(raw.strip())
    except (ValueError, AttributeError):
        return 0.0


def aggregate_by_phase(rows: list[dict]) -> dict[str, dict[str, float]]:
    """Return {phase -> {col -> total}} summing NUMERIC_COLS per phase."""
    totals: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for row in rows:
        phase = row.get("phase", "unknown").strip() or "unknown"
        for col in NUMERIC_COLS:
            totals[phase][col] += _numeric(row.get(col, ""))
    return {p: dict(cols) for p, cols in totals.items()}


def aggregate_by_decode_level(rows: list[dict]) -> dict[int, dict[str, float]]:
    """Return {token_index -> {col -> total}} for decode-phase rows only."""
    totals: dict[int, dict[str, float]] = defaultdict(lambda: defaultdict(float))
    for row in rows:
        if row.get("phase", "").strip().lower() != "decode":
            continue
        try:
            tok = int(row.get("token_index", "0").strip())
        except ValueError:
            tok = 0
        for col in NUMERIC_COLS:
            totals[tok][col] += _numeric(row.get(col, ""))
    return {t: dict(cols) for t, cols in totals.items()}


# ── Binary runner ──────────────────────────────────────────────────────────────

def run_llama_papi(events: list[str], out_path: Path, timeout: int = 300) -> None:
    binary = BIN_DIR / BINARY
    if not binary.is_file():
        pytest.skip(f"{BINARY} not compiled — build it first.\nExpected: {binary}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        str(binary),
        "--papi-events", ",".join(events),
        "--result-path", str(out_path),
        "-m", str(MODEL_PATH),
        "-p", PROMPT,
        "-n", N_PREDICT,
        "--temp", TEMP,
        "--log-disable",
    ]
    print(f"\n[col-interference] running: {' '.join(cmd)}")
    result = subprocess.run(
        cmd, cwd=str(LLAMA_ROOT), capture_output=True, text=True, timeout=timeout
    )
    if result.returncode != 0:
        pytest.fail(f"llama-papi failed (exit {result.returncode}):\n{result.stderr}")
    if not out_path.exists():
        pytest.fail(f"Binary did not create output file: {out_path}")


# ── Module fixture: three runs ─────────────────────────────────────────────────

@pytest.fixture(scope="module")
def run_aggregates():
    """
    Execute the three event-set runs once per module.
    Returns (phase_data, decode_data):
      phase_data[i]  = {phase        -> {col -> total}}  for run i
      decode_data[i] = {token_index  -> {col -> total}}  for run i
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    phase_data:  list[dict] = []
    decode_data: list[dict] = []

    for i, events in enumerate(EVENT_SETS, start=1):
        csv_path = OUT_DIR / f"run_{i}_events.csv"
        run_llama_papi(events, csv_path)
        rows = parse_csv(csv_path)
        pd = aggregate_by_phase(rows)
        dd = aggregate_by_decode_level(rows)
        phase_data.append(pd)
        decode_data.append(dd)
        print(
            f"[col-interference] run {i} — phases: {list(pd.keys())}, "
            f"decode tokens: {sorted(dd.keys())}"
        )

    return phase_data, decode_data


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_phase_totals_stable(run_aggregates):
    """
    For every phase and numeric column, the total must not deviate by more than
    TOLERANCE from run 1 when additional PAPI events are added.
    """
    phase_data, _ = run_aggregates
    baseline = phase_data[0]
    failures = []

    for run_i, run in enumerate(phase_data[1:], start=2):
        for phase, cols in baseline.items():
            for col, base_val in cols.items():
                if base_val == 0:
                    continue
                cmp_val = run.get(phase, {}).get(col, 0.0)
                delta = abs(cmp_val - base_val) / base_val
                if delta > TOLERANCE:
                    failures.append(
                        f"  run {run_i}, phase={phase}, col={col}: "
                        f"run1={base_val:,.0f}  run{run_i}={cmp_val:,.0f}  "
                        f"delta={delta * 100:.1f}%"
                    )

    assert not failures, (
        f"Phase-level totals deviate >{TOLERANCE * 100:.0f}% from run 1:\n"
        + "\n".join(failures[:30])
        + ("\n  ... (truncated)" if len(failures) > 30 else "")
    )


def test_decode_level_totals_stable(run_aggregates):
    """
    For every decode token index and numeric column, the per-token total must
    not deviate by more than TOLERANCE from run 1.
    """
    _, decode_data = run_aggregates
    baseline = decode_data[0]
    failures = []

    for run_i, run in enumerate(decode_data[1:], start=2):
        for tok, cols in sorted(baseline.items()):
            for col, base_val in cols.items():
                if base_val == 0:
                    continue
                cmp_val = run.get(tok, {}).get(col, 0.0)
                delta = abs(cmp_val - base_val) / base_val
                if delta > TOLERANCE:
                    failures.append(
                        f"  run {run_i}, token={tok}, col={col}: "
                        f"run1={base_val:,.0f}  run{run_i}={cmp_val:,.0f}  "
                        f"delta={delta * 100:.1f}%"
                    )

    assert not failures, (
        f"Decode-level totals deviate >{TOLERANCE * 100:.0f}% from run 1:\n"
        + "\n".join(failures[:30])
        + ("\n  ... (truncated)" if len(failures) > 30 else "")
    )


def test_control_columns_invariant(run_aggregates):
    """
    size_bytes and n_elements are tensor properties — they must be identical
    across all three runs (within floating-point rounding).
    """
    phase_data, _ = run_aggregates
    baseline = phase_data[0]
    failures = []
    control_cols = ["size_bytes", "n_elements"]

    for run_i, run in enumerate(phase_data[1:], start=2):
        for phase, cols in baseline.items():
            for col in control_cols:
                base_val = cols.get(col, 0.0)
                cmp_val  = run.get(phase, {}).get(col, 0.0)
                if base_val != cmp_val:
                    failures.append(
                        f"  run {run_i}, phase={phase}, col={col}: "
                        f"run1={base_val:,.0f}  run{run_i}={cmp_val:,.0f}"
                    )

    assert not failures, (
        "Tensor shape columns (size_bytes, n_elements) differ across runs — "
        "the binary may be instrumenting different operations:\n"
        + "\n".join(failures)
    )


# ── Visualization ──────────────────────────────────────────────────────────────

def _fmt(v: float) -> str:
    """Compact human-readable number label."""
    if v >= 1_000_000_000:
        return f"{v / 1_000_000_000:.1f}B"
    if v >= 1_000_000:
        return f"{v / 1_000_000:.1f}M"
    if v >= 1_000:
        return f"{v / 1_000:.1f}K"
    return f"{int(v)}"


def _build_plot(
    phase_data:  list[dict[str, dict[str, float]]],
    decode_data: list[dict[int,  dict[str, float]]],
) -> Path | None:
    """
    8-panel figure (2 rows × 4 columns):
      Row 0 — phase-level totals   for each numeric column
      Row 1 — decode-level totals  for each numeric column
    Each panel shows three side-by-side bars (one per run).
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
        import numpy as np
    except ImportError:
        print("[col-interference] matplotlib/numpy not installed — skipping plot")
        return None

    colors     = ["#2196F3", "#FF9800", "#4CAF50"]
    run_labels = ["Run 1\n(1 event)", "Run 2\n(+L1_ICM)", "Run 3\n(+L2_DCM)"]
    col_titles = {
        "time_ns":    "Wall-clock time (ns)",
        "size_bytes": "Tensor size (bytes)",
        "n_elements": "Tensor elements",
        "papi_l1_dcm": "L1 D-cache misses",
    }
    bar_w = 0.22

    # Stable phase order
    all_phases = []
    for run in phase_data:
        for p in run:
            if p not in all_phases:
                all_phases.append(p)

    all_tokens = sorted({tok for run in decode_data for tok in run})

    fig, axes = plt.subplots(
        2, len(NUMERIC_COLS),
        figsize=(5.5 * len(NUMERIC_COLS), 11),
        gridspec_kw={"hspace": 0.55, "wspace": 0.42},
    )
    fig.suptitle(
        "All column totals across 1 / 2 / 3 active PAPI counters\n"
        "Top: grouped by phase — Bottom: decode phase by token index",
        fontsize=13, fontweight="bold", y=0.995,
    )

    # ── Row 0: phase-level totals ──────────────────────────────────────────────
    for ci, col in enumerate(NUMERIC_COLS):
        ax = axes[0, ci]
        x  = np.arange(len(all_phases))

        for ri, (run, lbl, clr) in enumerate(zip(phase_data, run_labels, colors)):
            vals = [run.get(ph, {}).get(col, 0.0) for ph in all_phases]
            bars = ax.bar(
                x + ri * bar_w, vals, width=bar_w,
                color=clr, alpha=0.85, label=lbl, edgecolor="white", linewidth=0.4,
            )
            for bar, v in zip(bars, vals):
                if v > 0:
                    ax.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height() * 1.015,
                        _fmt(v), ha="center", va="bottom", fontsize=6.5, rotation=45,
                    )

        ax.set_title(col_titles[col], fontsize=10, fontweight="bold")
        ax.set_xticks(x + bar_w)
        ax.set_xticklabels(all_phases, fontsize=9)
        ax.set_ylabel("Phase total")
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: _fmt(v)))
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_axisbelow(True)

    # ── Row 1: decode-level totals ─────────────────────────────────────────────
    for ci, col in enumerate(NUMERIC_COLS):
        ax = axes[1, ci]
        x  = np.arange(len(all_tokens))

        for ri, (run, lbl, clr) in enumerate(zip(decode_data, run_labels, colors)):
            vals = [run.get(tok, {}).get(col, 0.0) for tok in all_tokens]
            ax.bar(
                x + ri * bar_w, vals, width=bar_w,
                color=clr, alpha=0.85, label=lbl, edgecolor="white", linewidth=0.4,
            )

        ax.set_title(col_titles[col], fontsize=10, fontweight="bold")
        ax.set_xticks(x + bar_w)
        ax.set_xticklabels([str(t) for t in all_tokens], fontsize=8)
        ax.set_xlabel("Decode token index")
        ax.set_ylabel("Per-token total")
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: _fmt(v)))
        ax.legend(fontsize=7, loc="upper right")
        ax.grid(True, alpha=0.3, axis="y")
        ax.set_axisbelow(True)

    # Row labels
    for row, label in enumerate(["by phase", "decode by token index"]):
        fig.text(
            0.005, 0.75 - row * 0.5, label,
            va="center", ha="left", fontsize=9, color="#555",
            rotation=90, style="italic",
        )

    out_path = OUT_DIR / "papi_column_interference_plot.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[col-interference] plot saved to {out_path}")
    return out_path


def test_generate_plot(run_aggregates):
    """Generate and save the 8-panel comparison plot — always passes, informational only."""
    phase_data, decode_data = run_aggregates
    out_path = _build_plot(phase_data, decode_data)
    if out_path is not None:
        assert out_path.exists(), f"Plot was not written to {out_path}"


# ── Standalone entry point ─────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Running PAPI column interference check (standalone mode)")
    print(f"Binary:  {BIN_DIR / BINARY}")
    print(f"Model:   {MODEL_PATH}")
    print(f"Output:  {OUT_DIR}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    phase_data_all:  list[dict] = []
    decode_data_all: list[dict] = []

    for i, events in enumerate(EVENT_SETS, start=1):
        csv_path = OUT_DIR / f"run_{i}_events.csv"
        binary   = BIN_DIR / BINARY
        if not binary.is_file():
            print(f"ERROR: {BINARY} not compiled — expected at {binary}", file=sys.stderr)
            sys.exit(1)

        cmd = [
            str(binary),
            "--papi-events", ",".join(events),
            "--result-path", str(csv_path),
            "-m", str(MODEL_PATH),
            "-p", PROMPT,
            "-n", N_PREDICT,
            "--temp", TEMP,
            "--log-disable",
        ]
        print(f"\n[run {i}/3]  events: {events}")
        print(f"  cmd: {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=str(LLAMA_ROOT))
        if result.returncode != 0:
            print(f"ERROR: run {i} failed (exit {result.returncode})", file=sys.stderr)
            sys.exit(1)

        rows = parse_csv(csv_path)
        pd   = aggregate_by_phase(rows)
        dd   = aggregate_by_decode_level(rows)
        phase_data_all.append(pd)
        decode_data_all.append(dd)

        print(f"  → phases: {list(pd.keys())},  decode tokens: {sorted(dd.keys())}")
        for phase, cols in pd.items():
            print(f"    {phase}:")
            for col, total in cols.items():
                print(f"      {col:20s}: {total:>18,.0f}")

    print("\n── Phase totals comparison ──────────────────────────────────────────")
    for phase in list(phase_data_all[0].keys()):
        print(f"\n  {phase}:")
        for col in NUMERIC_COLS:
            vals = [run.get(phase, {}).get(col, 0.0) for run in phase_data_all]
            base = vals[0]
            pcts = [
                f"{(v - base) / max(base, 1) * 100:+.1f}%" if base else "n/a"
                for v in vals[1:]
            ]
            print(
                f"    {col:20s}: {vals[0]:>15,.0f}"
                f"  → {vals[1]:>15,.0f} ({pcts[0]})"
                f"  → {vals[2]:>15,.0f} ({pcts[1]})"
            )

    _build_plot(phase_data_all, decode_data_all)
    print("\nDone — check test_outputs/papi_column_interference/ for CSVs and the plot.")
