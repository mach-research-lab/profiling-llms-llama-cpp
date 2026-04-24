"""
conftest.py — pytest fixtures for the LLM profiling test suite.

Fixtures are loaded automatically by pytest. Helper functions, constants, and
CSV parsers live in helpers.py and are imported directly by test modules.

Test flow:
  1. run_all_baseline (session fixture): runs the full multibatch suite (all PAPI
     event groups + energy + KV cache) — the same flow real users see — and parses
     baseline metrics: total runtime_ns, total energy_uj, total l1_dcm.
     Re-uses existing run_all_results/ unless --fresh-baseline is passed.

  2. Individual test files run stripped measurement binaries (top-view, phase-view,
     decoder-block-view) and compare against the baseline to verify the data is
     within reason and to quantify any measurement overhead.
"""

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

# Make the tests package importable (so helpers.py can be imported)
sys.path.insert(0, str(Path(__file__).parent))

from helpers import (
    BIN_DIR,
    LLAMA_ROOT,
    MODEL_PATH,
    N_PREDICT,
    K_CACHE,
    V_CACHE,
    TEST_PROMPT,
    parse_events_group_csv,
    parse_energy_csv,
)

# Also put 09A-backend on the path for event_retriever
sys.path.insert(0, str(Path(__file__).parent.parent))


# ── pytest CLI option ──────────────────────────────────────────────────────────

def pytest_addoption(parser):
    parser.addoption(
        "--fresh-baseline",
        action="store_true",
        default=False,
        help="Force a fresh run_all baseline (re-run all PAPI event groups + energy + KV).",
    )


# ── Session-scoped multibatch baseline fixture ─────────────────────────────────

@pytest.fixture(scope="session")
def run_all_baseline(request):
    """
    Run the full multibatch suite once per session and return baseline metrics.

    Returns a dict:
        {
          "runtime_ns": int,         # sum(time_ns) from events_group_1.csv
          "energy_uj":  float|None,  # sum(cpu_package_uj) from energy.csv; None if RAPL unavailable
          "l1_dcm":     int|None,    # sum(papi_l1_dcm) from best event group; None if not measured
        }

    By default, re-uses existing run_all_results/ to keep tests fast.
    Pass --fresh-baseline to force a new run.
    """
    output_dir = LLAMA_ROOT / "run_all_results"
    fresh = request.config.getoption("--fresh-baseline")
    needs_run = fresh or not _baseline_looks_complete(output_dir)

    if needs_run:
        _run_full_multibatch(output_dir)

    # ── Parse baseline ──────────────────────────────────────────────────────
    group1 = output_dir / "events_group_1.csv"
    if not group1.exists():
        pytest.fail(
            "run_all_results/events_group_1.csv not found.\n"
            "Run with --fresh-baseline to generate it."
        )

    group1_metrics = parse_events_group_csv(group1)

    # Look for L1_DCM in any group (may not be in group 1)
    l1_dcm = group1_metrics["l1_dcm"]
    if l1_dcm is None:
        for csv_file in sorted(output_dir.glob("events_group_*.csv")):
            m = parse_events_group_csv(csv_file)
            if m["l1_dcm"] is not None:
                l1_dcm = m["l1_dcm"]
                break

    energy_csv = output_dir / "energy.csv"
    energy_uj = parse_energy_csv(energy_csv) if energy_csv.exists() else None

    return {
        "runtime_ns": group1_metrics["runtime_ns"],
        "energy_uj": energy_uj,
        "l1_dcm": l1_dcm,
    }


_PARAMS_FILE = LLAMA_ROOT / "run_all_results" / "test_params.json"
_CURRENT_PARAMS = {
    "prompt":    TEST_PROMPT,
    "n_predict": N_PREDICT,
    "k_cache":   K_CACHE,
    "v_cache":   V_CACHE,
}


def _baseline_looks_complete(output_dir: Path) -> bool:
    """
    True when run_all_results/ exists, has at least one events group CSV,
    and was generated with the same test parameters as the current run.
    A mismatch triggers an automatic re-run to keep comparisons meaningful.
    """
    if not output_dir.is_dir() or not any(output_dir.glob("events_group_*.csv")):
        return False
    if not _PARAMS_FILE.exists():
        return False  # old baseline without params — regenerate
    try:
        import json as _json
        saved = _json.loads(_PARAMS_FILE.read_text())
        return saved == _CURRENT_PARAMS
    except Exception:
        return False


def _run_full_multibatch(output_dir: Path) -> None:
    """
    Replicate tui.py run_all() logic without the interactive prompts.
    Runs all PAPI event groups, energy, and KV cache measurements.
    """
    try:
        from event_retriever import get_valid_runs
    except Exception as exc:
        pytest.skip(
            f"Cannot import event_retriever (PAPI tools may not be available): {exc}"
        )

    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)

    llama_papi = BIN_DIR / "llama-papi"
    if not llama_papi.is_file():
        pytest.skip(f"llama-papi not compiled — expected at {llama_papi}")

    event_groups = get_valid_runs()
    for i, group in enumerate(event_groups, start=1):
        csv_path = output_dir / f"events_group_{i}.csv"
        cmd = [
            str(llama_papi),
            "--papi-events", ",".join(group),
            "--result-path", str(csv_path),
            "--papi-events-unrestricted",
            "-m", str(MODEL_PATH),
            "-p", TEST_PROMPT,
            "-n", N_PREDICT,
            "--temp", "0",
            "--log-disable",
        ]
        result = subprocess.run(
            cmd, cwd=str(LLAMA_ROOT), capture_output=True, text=True, timeout=300
        )
        if result.returncode != 0:
            pytest.fail(
                f"llama-papi group {i}/{len(event_groups)} failed "
                f"(exit {result.returncode}):\n{result.stderr}"
            )

    # Energy run
    llama_energy = BIN_DIR / "llama-energy"
    if llama_energy.is_file():
        cmd = [
            str(llama_energy),
            "--result-path", str(output_dir / "energy.csv"),
            "-m", str(MODEL_PATH),
            "-p", TEST_PROMPT,
            "-n", N_PREDICT,
            "--temp", "0",
            "--log-disable",
        ]
        subprocess.run(cmd, cwd=str(LLAMA_ROOT), capture_output=True, text=True, timeout=300)

    # KV cache run
    kv_measure = BIN_DIR / "kv-measure"
    if kv_measure.is_file():
        cmd = [
            str(kv_measure),
            "--result-path", str(output_dir / "kv_measurement.csv"),
            "-m", str(MODEL_PATH),
            "-p", TEST_PROMPT,
            "-n", N_PREDICT,
            "--cache-type-k", K_CACHE,
            "--cache-type-v", V_CACHE,
            "--temp", "0",
            "--log-disable",
        ]
        subprocess.run(cmd, cwd=str(LLAMA_ROOT), capture_output=True, text=True, timeout=300)

    # Record the parameters used so stale baselines are detected on the next run
    import json as _json
    _PARAMS_FILE.write_text(_json.dumps(_CURRENT_PARAMS, indent=2))
