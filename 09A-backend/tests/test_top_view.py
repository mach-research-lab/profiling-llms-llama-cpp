"""
test_top_view.py — Test 1: whole-LLM top-view measurement.

Runs llama-measurement-top-view with PAPI_L1_DCM and compares against the
multibatch baseline to verify the simplified measurement is within reason.

Metrics checked:
  - Structural:  required JSON fields, generated token count
  - vs baseline: cpu_package_uj (energy), PAPI_L1_DCM (cache misses)
  - Sanity:      runtime_ns > 0, throughput > 0
"""

from pathlib import Path

import pytest

from helpers import (
    PAPI_EVENTS,
    TEST_OUTPUTS,
    TOLERANCES,
    N_PREDICT,
    assert_within_tolerance,
    parse_top_view_file,
    run_binary,
)

_OUT = TEST_OUTPUTS / "top_view.json"


# ── Module-scoped fixture ──────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def top_view_result():
    """
    Run llama-measurement-top-view once per module and return the parsed JSON.
    Output is written to tests/test_outputs/top_view.json for inspection.
    """
    TEST_OUTPUTS.mkdir(exist_ok=True)
    _OUT.unlink(missing_ok=True)

    proc = run_binary(
        "llama-measurement-top-view",
        ["--papi-events", PAPI_EVENTS],
        _OUT,
    )
    assert proc.returncode == 0, (
        f"llama-measurement-top-view exited {proc.returncode}:\n{proc.stderr}"
    )
    assert _OUT.exists(), "top-view binary did not write output file"
    return parse_top_view_file(_OUT)


# ── Structure tests ────────────────────────────────────────────────────────────

def test_top_view_produces_valid_json(top_view_result):
    """Output is a non-empty JSON object."""
    assert isinstance(top_view_result, dict)
    assert len(top_view_result) > 0


def test_top_view_required_fields(top_view_result):
    """All expected metric keys are present."""
    required = [
        "runtime_ns",
        "runtime_s",
        "generated_tokens",
        "total_tokens",
        "token_throughput",
        "peak_rss_mb",
        "model_size_bytes",
        "kv_size_used_bytes",
    ]
    missing = [k for k in required if k not in top_view_result]
    assert not missing, f"Missing fields in top-view JSON: {missing}"


def test_top_view_papi_field_present(top_view_result):
    """PAPI_L1_DCM was recorded (key present and non-negative)."""
    assert "PAPI_L1_DCM" in top_view_result, (
        "PAPI_L1_DCM not found in top-view output — was --papi-events passed correctly?"
    )
    assert top_view_result["PAPI_L1_DCM"] >= 0


def test_top_view_tokens_generated(top_view_result):
    """Exactly N_PREDICT tokens were decoded."""
    got = top_view_result.get("generated_tokens")
    assert got == int(N_PREDICT), (
        f"Expected generated_tokens={N_PREDICT}, got {got}"
    )


# ── Sanity tests ───────────────────────────────────────────────────────────────

def test_top_view_runtime_positive(top_view_result):
    """Wall-clock runtime is a positive, finite value."""
    runtime = top_view_result.get("runtime_ns", 0)
    assert runtime > 0, f"runtime_ns should be > 0, got {runtime}"


def test_top_view_throughput_positive(top_view_result):
    """Token throughput (tokens/s) is greater than zero."""
    tput = top_view_result.get("token_throughput", 0)
    assert tput > 0, f"token_throughput should be > 0, got {tput}"


def test_top_view_memory_recorded(top_view_result):
    """Peak RSS memory usage was captured."""
    rss = top_view_result.get("peak_rss_mb", 0)
    assert rss > 0, f"peak_rss_mb should be > 0, got {rss}"


# ── Comparison vs. multibatch baseline ────────────────────────────────────────

def test_top_view_energy_vs_multibatch(top_view_result, run_all_baseline):
    """
    cpu_package_uj from top-view is within tolerance of the multibatch energy baseline.

    Both measurements use the same Linux RAPL domain for the same prompt, so they
    should agree within normal system-load variation.
    """
    baseline_energy = run_all_baseline.get("energy_uj")
    if baseline_energy is None:
        pytest.skip("Energy baseline unavailable (RAPL not accessible or all-zero)")

    actual_energy = top_view_result.get("cpu_package_uj")
    if actual_energy is None:
        pytest.skip("cpu_package_uj not in top-view output (RAPL not accessible)")

    assert_within_tolerance(
        "energy (cpu_package_uj)",
        float(actual_energy),
        float(baseline_energy),
        TOLERANCES["energy"],
    )


def test_top_view_cache_misses_vs_multibatch(top_view_result, run_all_baseline):
    """
    Compare the top-view global L1_DCM against the multibatch per-tensor sum.

    These two methods measure the same hardware counter differently:
    - Top-view: one PAPI counter running continuously across the whole forward pass.
    - Multibatch: PAPI counter started and stopped for EACH tensor operation, then summed.

    Per-tensor start/stop introduces its own cache pressure (PAPI library accesses),
    so the multibatch sum is expected to be HIGHER than the global top-view value.
    The test quantifies this overhead and catches regressions in either direction.

    Assertions:
      1. Top-view does not exceed multibatch (global <= per-tensor, directionally expected).
      2. The overhead ratio (per-tensor / global) is at most 4× — anything beyond
         that suggests excessive instrumentation cost in the multibatch path.
    """
    baseline_l1dcm = run_all_baseline.get("l1_dcm")
    if baseline_l1dcm is None:
        pytest.skip(
            "PAPI_L1_DCM not measured in any multibatch event group — "
            "run --fresh-baseline on a machine where PAPI_L1_DCM is available"
        )

    actual_l1dcm = top_view_result.get("PAPI_L1_DCM")
    if actual_l1dcm is None:
        pytest.skip("PAPI_L1_DCM not in top-view output")

    actual   = float(actual_l1dcm)
    baseline = float(baseline_l1dcm)
    ratio    = baseline / actual if actual > 0 else float("inf")

    # Global (top-view) should not exceed per-tensor sum by more than 15 %
    # (noise floor: at most 15 % higher due to measurement timing differences)
    assert actual <= baseline * 1.15, (
        f"Top-view global L1_DCM ({actual:.0f}) unexpectedly exceeds "
        f"multibatch per-tensor sum ({baseline:.0f}) — check measurement setup"
    )

    # Per-tensor overhead should not be excessive (≤ 4× global)
    assert ratio <= 4.0, (
        f"Multibatch per-tensor L1_DCM ({baseline:.0f}) is {ratio:.2f}× "
        f"the top-view global value ({actual:.0f}) — "
        f"per-tensor PAPI measurement overhead appears excessive (limit: 4×)"
    )

    # Informational: print the measured overhead for the test report
    print(
        f"\n  [cache-miss overhead] top-view={actual:.0f}  multibatch={baseline:.0f}  "
        f"ratio={ratio:.2f}× (per-tensor PAPI overhead)"
    )
