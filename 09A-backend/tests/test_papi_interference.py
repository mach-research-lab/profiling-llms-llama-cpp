#!/usr/bin/env python3
"""
test_papi_interference.py — Does measuring more PAPI events change the results?

Three runs of llama-papi with the same model/prompt but increasing event counts:
  Run 1 (1 event):  PAPI_L1_DCM
  Run 2 (2 events): PAPI_L1_DCM, PAPI_L1_ICM
  Run 3 (3 events): PAPI_L1_DCM, PAPI_L1_ICM, PAPI_L2_DCM

After each run, PAPI_L1_DCM is extracted per operation and compared across
all three runs. If adding more counters introduces interference the values
should diverge; if PAPI is well-isolated they should remain close.

Run directly:
    python tests/test_papi_interference.py
Or via pytest (slow, needs compiled binary):
    pytest tests/test_papi_interference.py -v
"""

import csv
import subprocess
import sys
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
OUT_DIR = Path(__file__).parent / "test_outputs" / "papi_interference"

# ── Run parameters ─────────────────────────────────────────────────────────────

PROMPT    = "Hello"
N_PREDICT = "5"
TEMP      = "0"
BINARY    = "llama-papi"

# Three event sets — each is a superset of the previous one
EVENT_SETS = [
    ["PAPI_L1_DCM"],
    ["PAPI_L1_DCM", "PAPI_L1_ICM"],
    ["PAPI_L1_DCM", "PAPI_L1_ICM", "PAPI_L2_DCM"],
]

# Column we care about across all three runs
TARGET_EVENT = "papi_l1_dcm"

# Tolerance for the pytest comparison (15 % relative deviation)
TOLERANCE = 0.15


# ── CSV parsing ────────────────────────────────────────────────────────────────

def parse_csv(path: Path) -> list[dict]:
    """Return every data row as a dict; keys are lower-cased column names."""
    rows = []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({k.lower(): v for k, v in row.items()})
    return rows


def extract_l1_dcm_per_op(rows: list[dict]) -> list[tuple[str, int]]:
    """
    Return [(label, value), ...] for every row that has a non-empty
    papi_l1_dcm column.  label = "phase/tensor_name/op_type".
    """
    results = []
    for row in rows:
        raw = row.get(TARGET_EVENT, "").strip()
        if not raw:
            continue
        try:
            val = int(raw)
        except ValueError:
            continue
        label = f"{row.get('phase','?')}/{row.get('tensor_name','?')}/{row.get('op_type','?')}"
        results.append((label, val))
    return results


# ── Binary runner ──────────────────────────────────────────────────────────────

def run_llama_papi(events: list[str], out_path: Path, timeout: int = 300) -> None:
    binary = BIN_DIR / BINARY
    if not binary.is_file():
        pytest.skip(
            f"{BINARY} not compiled — build it first.\nExpected: {binary}"
        )
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
    print(f"\n[interference] running: {' '.join(cmd)}")
    result = subprocess.run(
        cmd, cwd=str(LLAMA_ROOT), capture_output=True, text=True, timeout=timeout
    )
    if result.returncode != 0:
        pytest.fail(
            f"llama-papi failed (exit {result.returncode}):\n{result.stderr}"
        )
    if not out_path.exists():
        pytest.fail(f"Binary did not create output file: {out_path}")


# ── Session fixture: three runs ────────────────────────────────────────────────

@pytest.fixture(scope="module")
def three_run_data():
    """
    Execute the three event-set runs once per module.
    Returns a list of three lists of (label, l1_dcm_value) tuples.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_runs: list[list[tuple[str, int]]] = []
    for i, events in enumerate(EVENT_SETS, start=1):
        csv_path = OUT_DIR / f"run_{i}_events.csv"
        run_llama_papi(events, csv_path)
        rows = parse_csv(csv_path)
        per_op = extract_l1_dcm_per_op(rows)
        if not per_op:
            pytest.fail(
                f"Run {i} ({events}) produced no {TARGET_EVENT} data.\n"
                f"CSV: {csv_path}"
            )
        all_runs.append(per_op)
        print(f"[interference] run {i} — {len(per_op)} operations with {TARGET_EVENT}")

    return all_runs


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_runs_produce_data(three_run_data):
    """All three runs must yield at least one PAPI_L1_DCM data point."""
    for i, run in enumerate(three_run_data, start=1):
        assert len(run) > 0, f"Run {i} has no {TARGET_EVENT} rows"


def test_operation_count_matches(three_run_data):
    """All three runs should instrument the same number of operations."""
    counts = [len(r) for r in three_run_data]
    assert counts[0] == counts[1] == counts[2], (
        f"Operation counts differ across runs: {counts}\n"
        "This suggests the binary is skipping operations when more events are active."
    )


def test_totals_within_tolerance(three_run_data):
    """
    The total PAPI_L1_DCM across all operations must not change by more than
    TOLERANCE (15 %) when additional events are added.
    """
    totals = [sum(v for _, v in run) for run in three_run_data]
    baseline = totals[0]
    print(f"\n[interference] totals: run1={totals[0]:,}  run2={totals[1]:,}  run3={totals[2]:,}")
    for i, total in enumerate(totals[1:], start=2):
        if baseline == 0:
            pytest.skip("Baseline total is zero — counter may be unsupported")
        delta = abs(total - baseline) / baseline
        assert delta <= TOLERANCE, (
            f"Run {i} total diverges from run 1 by {delta*100:.1f}% "
            f"(limit {TOLERANCE*100:.0f}%):\n"
            f"  run 1 total = {baseline:,}\n"
            f"  run {i} total = {total:,}\n"
            "Adding more PAPI events may be introducing measurement interference."
        )


def test_per_operation_within_tolerance(three_run_data):
    """
    For each operation, the PAPI_L1_DCM value must stay within TOLERANCE of
    run 1 across runs 2 and 3.  Small deviations are normal; large systematic
    shifts indicate interference.
    """
    run1 = three_run_data[0]
    failures = []
    for i, run in enumerate(three_run_data[1:], start=2):
        for idx, ((label1, val1), (label2, val2)) in enumerate(zip(run1, run)):
            if val1 == 0:
                continue
            delta = abs(val2 - val1) / val1
            if delta > TOLERANCE:
                failures.append(
                    f"  run {i}, op {idx} ({label1}): "
                    f"val1={val1:,} val{i}={val2:,} delta={delta*100:.1f}%"
                )
    assert not failures, (
        f"Per-operation {TARGET_EVENT} deviations exceed {TOLERANCE*100:.0f}%:\n"
        + "\n".join(failures[:20])
        + ("\n  ... (truncated)" if len(failures) > 20 else "")
    )


# ── Visualization ──────────────────────────────────────────────────────────────

def _build_plot(three_run_data: list[list[tuple[str, int]]]) -> Path:
    """
    Generate a four-panel comparison figure and save it to OUT_DIR.
    Returns the path to the saved PNG.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker
        import numpy as np
    except ImportError:
        print("[interference] matplotlib / numpy not installed — skipping plot")
        return None

    labels  = [label for label, _ in three_run_data[0]]
    values  = [np.array([v for _, v in run], dtype=float) for run in three_run_data]
    totals  = [v.sum() for v in values]
    n_ops   = len(labels)
    x       = np.arange(n_ops)
    colors  = ["#2196F3", "#FF9800", "#4CAF50"]  # blue, orange, green
    run_labels = [
        "Run 1 — 1 event\n(PAPI_L1_DCM)",
        "Run 2 — 2 events\n(+ PAPI_L1_ICM)",
        "Run 3 — 3 events\n(+ PAPI_L2_DCM)",
    ]

    fig, axes = plt.subplots(
        2, 2, figsize=(16, 10),
        gridspec_kw={"hspace": 0.45, "wspace": 0.35},
    )
    fig.suptitle(
        "PAPI Interference Check — PAPI_L1_DCM across 1 / 2 / 3 active counters",
        fontsize=13, fontweight="bold", y=0.98,
    )

    # ── Panel 1: per-operation line chart ──────────────────────────────────────
    ax = axes[0, 0]
    for i, (vals, col, lbl) in enumerate(zip(values, colors, run_labels)):
        ax.plot(x, vals, color=col, linewidth=1.2, alpha=0.85, label=lbl)
    ax.set_title("Per-operation PAPI_L1_DCM", fontsize=11)
    ax.set_xlabel("Operation index")
    ax.set_ylabel("L1 D-cache misses")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, alpha=0.3)

    # ── Panel 2: grouped bar chart (first 30 ops for legibility) ──────────────
    ax = axes[0, 1]
    show = min(n_ops, 30)
    bar_w = 0.25
    for i, (vals, col, lbl) in enumerate(zip(values, colors, run_labels)):
        ax.bar(x[:show] + i * bar_w, vals[:show], width=bar_w, color=col, alpha=0.85, label=lbl)
    ax.set_title(f"Grouped bars — first {show} operations", fontsize=11)
    ax.set_xlabel("Operation index")
    ax.set_ylabel("L1 D-cache misses")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3, axis="y")

    # ── Panel 3: relative deviation vs run 1 ──────────────────────────────────
    ax = axes[1, 0]
    base = values[0].copy()
    base[base == 0] = 1  # avoid division by zero
    for i, (vals, col, lbl) in enumerate(zip(values[1:], colors[1:], run_labels[1:]), start=2):
        pct_diff = (vals - values[0]) / base * 100
        ax.plot(x, pct_diff, color=col, linewidth=1.2, alpha=0.85, label=lbl)
    ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
    ax.axhline(15,  color="red", linewidth=0.6, linestyle=":", alpha=0.7, label="+15% limit")
    ax.axhline(-15, color="red", linewidth=0.6, linestyle=":", alpha=0.7, label="-15% limit")
    ax.set_title("Relative deviation vs Run 1 (%)", fontsize=11)
    ax.set_xlabel("Operation index")
    ax.set_ylabel("Δ from run 1 (%)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── Panel 4: total bar chart ───────────────────────────────────────────────
    ax = axes[1, 1]
    bar_colors = colors
    bars = ax.bar(run_labels, totals, color=bar_colors, alpha=0.85, edgecolor="black", linewidth=0.5)
    for bar, total in zip(bars, totals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.01,
            f"{int(total):,}",
            ha="center", va="bottom", fontsize=9,
        )
    ax.set_title("Total PAPI_L1_DCM across all operations", fontsize=11)
    ax.set_ylabel("Total L1 D-cache misses")
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{int(v):,}"))
    ax.grid(True, alpha=0.3, axis="y")

    out_path = OUT_DIR / "papi_interference_plot.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[interference] plot saved to {out_path}")
    return out_path


def test_generate_plot(three_run_data):
    """Generate and save a comparison plot — always passes, plot is informational."""
    out_path = _build_plot(three_run_data)
    if out_path is not None:
        assert out_path.exists(), f"Plot was not written to {out_path}"


# ── Standalone entry point ─────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Running PAPI interference check (standalone mode)")
    print(f"Binary:  {BIN_DIR / BINARY}")
    print(f"Model:   {MODEL_PATH}")
    print(f"Output:  {OUT_DIR}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_runs: list[list[tuple[str, int]]] = []

    for i, events in enumerate(EVENT_SETS, start=1):
        csv_path = OUT_DIR / f"run_{i}_events.csv"
        binary = BIN_DIR / BINARY
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
            print(f"ERROR: run {i} failed with exit code {result.returncode}", file=sys.stderr)
            sys.exit(1)

        rows   = parse_csv(csv_path)
        per_op = extract_l1_dcm_per_op(rows)
        all_runs.append(per_op)
        total = sum(v for _, v in per_op)
        print(f"  → {len(per_op)} ops, total {TARGET_EVENT} = {total:,}")

    print("\n── Totals ──────────────────────────────────────────────────────────")
    for i, run in enumerate(all_runs, start=1):
        total = sum(v for _, v in run)
        pct = ""
        if i > 1 and all_runs[0]:
            base = sum(v for _, v in all_runs[0])
            pct = f"  ({(total-base)/max(base,1)*100:+.1f}% vs run 1)"
        print(f"  Run {i}: {total:,}{pct}")

    _build_plot(all_runs)
    print("\nDone — check test_outputs/papi_interference/ for CSVs and the plot.")
