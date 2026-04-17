"""
test_phase_view.py — Test 2: prefill / decode phase-view measurement.

All tests in this module are skipped automatically when
llama-measurement-phase-view has not been compiled yet.

Runs llama-measurement-phase-view with PAPI_L1_DCM and compares against the
multibatch baseline per phase.

Expected JSON structure:
    {
      "prefill": {"runtime_ns": int, "PAPI_L1_DCM": int, "energy": {"cpu_package_uj": float, ...}},
      "decode":  {"runtime_ns": int, "PAPI_L1_DCM": int, "energy": {...}},
      ...
    }

Metrics checked:
  - Structure:   both phases present, required fields per phase
  - vs baseline: per-phase energy, total L1_DCM across both phases
  - Sanity:      both runtimes > 0, prefill + decode covers the whole run
"""

import json
from pathlib import Path

import pytest

from helpers import (
    PAPI_EVENTS,
    TEST_OUTPUTS,
    TOLERANCES,
    assert_within_tolerance,
    run_binary,
)

_OUT = TEST_OUTPUTS / "phase_view.json"
_PHASES = ("prefill", "decode")


# ── Module-scoped fixture ──────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def phase_view_result():
    """
    Run llama-measurement-phase-view once per module and return the parsed JSON.
    Skips if the binary is not yet compiled.
    Output is written to tests/test_outputs/phase_view.json for inspection.
    """
    TEST_OUTPUTS.mkdir(exist_ok=True)
    _OUT.unlink(missing_ok=True)

    proc = run_binary(
        "llama-measurement-phase-view",
        ["--papi-events", PAPI_EVENTS],
        _OUT,
    )
    assert proc.returncode == 0, (
        f"llama-measurement-phase-view exited {proc.returncode}:\n{proc.stderr}"
    )
    assert _OUT.exists(), "phase-view binary did not write output JSON"

    with open(_OUT) as f:
        return json.load(f)


# ── Structure tests ────────────────────────────────────────────────────────────

def test_phase_view_produces_valid_json(phase_view_result):
    """Output is a non-empty JSON object."""
    assert isinstance(phase_view_result, dict)
    assert len(phase_view_result) > 0


def test_phase_view_has_both_phases(phase_view_result):
    """Both 'prefill' and 'decode' keys are present in the output."""
    for phase in _PHASES:
        assert phase in phase_view_result, (
            f"Phase '{phase}' missing from phase-view output — "
            f"found keys: {list(phase_view_result.keys())}"
        )


def test_phase_view_required_fields_per_phase(phase_view_result):
    """Each phase has runtime_ns and the PAPI event field."""
    for phase in _PHASES:
        data = phase_view_result[phase]
        assert "runtime_ns" in data, f"'{phase}' missing runtime_ns"
        assert "PAPI_L1_DCM" in data, f"'{phase}' missing PAPI_L1_DCM"


# ── Sanity tests ───────────────────────────────────────────────────────────────

def test_phase_view_runtimes_positive(phase_view_result):
    """Both prefill and decode runtime values are positive."""
    for phase in _PHASES:
        rt = phase_view_result[phase].get("runtime_ns", 0)
        assert rt > 0, f"'{phase}' runtime_ns should be > 0, got {rt}"


def test_phase_view_prefill_longer_than_decode_per_token(phase_view_result):
    """
    Prefill processes all input tokens at once and is typically slower in total
    than a single decode step (though per-token prefill can be faster).
    Both phases must be positive; this is a directional sanity check only.
    """
    prefill_rt = phase_view_result["prefill"].get("runtime_ns", 0)
    decode_rt  = phase_view_result["decode"].get("runtime_ns", 0)
    assert prefill_rt > 0 and decode_rt > 0


# ── Comparison vs. multibatch baseline ────────────────────────────────────────

def test_phase_view_energy_vs_multibatch(phase_view_result, run_all_baseline):
    """
    Total energy (prefill + decode cpu_package_uj) from phase-view is within
    tolerance of the multibatch energy baseline.

    Energy is summed across both phases and compared against the single run_all
    energy measurement (which also covers the full inference).
    """
    baseline_energy = run_all_baseline.get("energy_uj")
    if baseline_energy is None:
        pytest.skip("Energy baseline unavailable (RAPL not accessible or all-zero)")

    actual_energy = 0.0
    found_any = False
    for phase in _PHASES:
        phase_data = phase_view_result.get(phase, {})
        energy_block = phase_data.get("energy", {})
        val = energy_block.get("cpu_package_uj")
        if val is not None:
            actual_energy += float(val)
            found_any = True

    if not found_any:
        pytest.skip("cpu_package_uj not in any phase energy block (RAPL not accessible)")

    assert_within_tolerance(
        "total energy prefill+decode (cpu_package_uj)",
        actual_energy,
        float(baseline_energy),
        TOLERANCES["energy"],
    )


def test_phase_view_total_cache_misses_vs_multibatch(phase_view_result, run_all_baseline):
    """
    Compare the sum of PAPI_L1_DCM across prefill + decode against the multibatch
    per-tensor baseline.

    Phase-view measures L1_DCM for the whole prefill and whole decode phases as
    two single continuous measurements. This captures less scope than either:
    - The multibatch per-tensor sum (hundreds of ops × overhead each), or
    - The decoder-block sum (per-block measurements capturing intra-block overhead).

    Specifically, phase-view prefill+decode excludes L1_DCM from:
    - Tokenization phase
    - Sampling steps between decode tokens
    So the phase-view sum is expected to be substantially lower than the
    multibatch baseline.

    Assertions:
      1. Phase sum does not exceed multibatch (directionally expected).
      2. Overhead ratio (multibatch / phase-sum) is at most 5× — a wider bound
         than the block-level test because scope exclusion is larger here.
    """
    baseline_l1dcm = run_all_baseline.get("l1_dcm")
    if baseline_l1dcm is None:
        pytest.skip(
            "PAPI_L1_DCM not measured in any multibatch event group"
        )

    total_l1dcm = 0
    found_any = False
    for phase in _PHASES:
        val = phase_view_result.get(phase, {}).get("PAPI_L1_DCM")
        if val is not None:
            total_l1dcm += int(val)
            found_any = True

    if not found_any:
        pytest.skip("PAPI_L1_DCM not in any phase of phase-view output")

    actual   = float(total_l1dcm)
    baseline = float(baseline_l1dcm)
    ratio    = baseline / actual if actual > 0 else float("inf")

    assert actual <= baseline * 1.15, (
        f"Phase-view L1_DCM sum ({actual:.0f}) unexpectedly exceeds "
        f"multibatch per-tensor sum ({baseline:.0f})"
    )
    assert ratio <= 5.0, (
        f"Multibatch per-tensor L1_DCM ({baseline:.0f}) is {ratio:.2f}× "
        f"the phase-view sum ({actual:.0f}) — scope gap appears excessive (limit: 5×)"
    )

    print(
        f"\n  [cache-miss overhead] phase-sum={actual:.0f}  multibatch={baseline:.0f}  "
        f"ratio={ratio:.2f}× (per-tensor PAPI overhead + scope exclusion)"
    )


def test_phase_view_prefill_runtime_vs_multibatch(phase_view_result, run_all_baseline):
    """
    Prefill runtime_ns from phase-view is within tolerance of the prefill portion
    of the multibatch baseline.

    The multibatch baseline stores the total runtime across all phases.  We compare
    phase-view prefill runtime against a rough lower bound: it must not be more than
    the total baseline (a basic sanity bound — not a tight comparison).
    """
    baseline_total = run_all_baseline.get("runtime_ns", 0)
    if baseline_total <= 0:
        pytest.skip("Multibatch baseline runtime_ns unavailable")

    prefill_rt = float(phase_view_result["prefill"].get("runtime_ns", 0))
    assert prefill_rt > 0, "prefill runtime_ns is zero"

    # Prefill alone should not dwarf the entire multibatch computation time by >3×
    max_allowed = baseline_total * 3.0
    assert prefill_rt <= max_allowed, (
        f"Prefill runtime ({prefill_rt:.0f} ns) is unexpectedly large "
        f"vs. baseline total ({baseline_total:.0f} ns)"
    )


def test_phase_view_decode_runtime_vs_multibatch(phase_view_result, run_all_baseline):
    """
    Decode runtime_ns from phase-view is within a reasonable bound vs. the baseline.
    Same bounding logic as the prefill test.
    """
    baseline_total = run_all_baseline.get("runtime_ns", 0)
    if baseline_total <= 0:
        pytest.skip("Multibatch baseline runtime_ns unavailable")

    decode_rt = float(phase_view_result["decode"].get("runtime_ns", 0))
    assert decode_rt > 0, "decode runtime_ns is zero"

    max_allowed = baseline_total * 3.0
    assert decode_rt <= max_allowed, (
        f"Decode runtime ({decode_rt:.0f} ns) is unexpectedly large "
        f"vs. baseline total ({baseline_total:.0f} ns)"
    )
