"""
test_decoder_blocks.py — Tests 3+: decoder-block-level measurement.

All tests in this module are skipped automatically when
llama-measurement-decoder-block-view has not been compiled yet.

Runs llama-measurement-decoder-block-view with PAPI_L1_DCM and validates
the structure and values of each decoder block's metrics.

Expected JSON structure (array):
    [
      {"block_type": "Prefill", "block_id": 0, "runtime_ns": int,
       "kv_cache_footprint_bytes": int, "PAPI_L1_DCM": int},
      ...
      {"block_type": "Decode",  "block_id": 0, ...},
      ...
    ]

Metrics checked per block type (Prefill / Decode):
  - Structure:   required fields on every block entry
  - Sanity:      all runtimes > 0, block_ids form contiguous sequences
  - vs baseline: total runtime sum within bound of baseline, total L1_DCM within tolerance
"""

import json
from pathlib import Path
from typing import List

import pytest

from helpers import (
    PAPI_EVENTS,
    TEST_OUTPUTS,
    TOLERANCES,
    assert_within_tolerance,
    run_binary,
)

_OUT    = TEST_OUTPUTS / "decoder_blocks.json"
_TYPES  = ("Prefill", "Decode")
_FIELDS = ("runtime_ns", "kv_cache_footprint_bytes", "PAPI_L1_DCM")


# ── Module-scoped fixture ──────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def decoder_blocks_result() -> List[dict]:
    """
    Run llama-measurement-decoder-block-view once per module and return the
    parsed JSON array.
    Skips if the binary is not yet compiled.
    Output is written to tests/test_outputs/decoder_blocks.json for inspection.
    """
    TEST_OUTPUTS.mkdir(exist_ok=True)
    _OUT.unlink(missing_ok=True)

    proc = run_binary(
        "llama-measurement-decoder-block-view",
        ["--papi-events", PAPI_EVENTS],
        _OUT,
    )
    assert proc.returncode == 0, (
        f"llama-measurement-decoder-block-view exited {proc.returncode}:\n{proc.stderr}"
    )
    assert _OUT.exists(), "decoder-block-view binary did not write output JSON"

    with open(_OUT) as f:
        data = json.load(f)

    assert isinstance(data, list), "Expected JSON array from decoder-block-view"
    return data


def _blocks_of_type(blocks: List[dict], block_type: str) -> List[dict]:
    return [b for b in blocks if b.get("block_type") == block_type]


# ── Structure tests ────────────────────────────────────────────────────────────

def test_decoder_blocks_is_nonempty_array(decoder_blocks_result):
    """Output is a non-empty JSON array."""
    assert len(decoder_blocks_result) > 0


def test_decoder_blocks_has_prefill_entries(decoder_blocks_result):
    """At least one Prefill block entry is present."""
    prefill = _blocks_of_type(decoder_blocks_result, "Prefill")
    assert len(prefill) > 0, "No Prefill blocks found in decoder-block-view output"


def test_decoder_blocks_has_decode_entries(decoder_blocks_result):
    """At least one Decode block entry is present."""
    decode = _blocks_of_type(decoder_blocks_result, "Decode")
    assert len(decode) > 0, "No Decode blocks found in decoder-block-view output"


def test_decoder_blocks_prefill_required_fields(decoder_blocks_result):
    """Every Prefill block has all required metric fields."""
    for block in _blocks_of_type(decoder_blocks_result, "Prefill"):
        missing = [f for f in _FIELDS if f not in block]
        assert not missing, (
            f"Prefill block {block.get('block_id')} missing fields: {missing}"
        )


def test_decoder_blocks_decode_required_fields(decoder_blocks_result):
    """Every Decode block has all required metric fields."""
    for block in _blocks_of_type(decoder_blocks_result, "Decode"):
        missing = [f for f in _FIELDS if f not in block]
        assert not missing, (
            f"Decode block {block.get('block_id')} missing fields: {missing}"
        )


# ── Sanity tests ───────────────────────────────────────────────────────────────

def test_decoder_blocks_prefill_runtimes_positive(decoder_blocks_result):
    """All Prefill block runtimes are positive."""
    for block in _blocks_of_type(decoder_blocks_result, "Prefill"):
        rt = block.get("runtime_ns", 0)
        assert rt > 0, (
            f"Prefill block {block.get('block_id')} has non-positive runtime_ns: {rt}"
        )


def test_decoder_blocks_decode_runtimes_positive(decoder_blocks_result):
    """All Decode block runtimes are positive."""
    for block in _blocks_of_type(decoder_blocks_result, "Decode"):
        rt = block.get("runtime_ns", 0)
        assert rt > 0, (
            f"Decode block {block.get('block_id')} has non-positive runtime_ns: {rt}"
        )


def test_decoder_blocks_prefill_ids_contiguous(decoder_blocks_result):
    """Prefill block_ids form a contiguous sequence starting from 0."""
    ids = sorted(b["block_id"] for b in _blocks_of_type(decoder_blocks_result, "Prefill"))
    expected = list(range(len(ids)))
    assert ids == expected, (
        f"Prefill block_ids not contiguous: got {ids}, expected {expected}"
    )


def test_decoder_blocks_decode_ids_contiguous(decoder_blocks_result):
    """Decode block_ids form a contiguous sequence starting from 0."""
    ids = sorted(b["block_id"] for b in _blocks_of_type(decoder_blocks_result, "Decode"))
    expected = list(range(len(ids)))
    assert ids == expected, (
        f"Decode block_ids not contiguous: got {ids}, expected {expected}"
    )


def test_decoder_blocks_kv_footprint_nonnegative(decoder_blocks_result):
    """KV-cache footprint is non-negative for all blocks."""
    for block in decoder_blocks_result:
        kv = block.get("kv_cache_footprint_bytes", 0)
        assert kv >= 0, (
            f"{block.get('block_type')} block {block.get('block_id')} "
            f"has negative kv_cache_footprint_bytes: {kv}"
        )


# ── Comparison vs. multibatch baseline ────────────────────────────────────────

def test_decoder_blocks_prefill_total_runtime_vs_multibatch(
    decoder_blocks_result, run_all_baseline
):
    """
    Sum of all Prefill block runtimes is within a reasonable bound of the
    multibatch baseline's total computation time.

    Decoder-block runtimes measure per-block wall time; the multibatch baseline
    stores the sum of per-tensor times across the whole run. We check that the
    prefill block sum is not excessively larger than the baseline (upper bound only).
    """
    baseline_total = run_all_baseline.get("runtime_ns", 0)
    if baseline_total <= 0:
        pytest.skip("Multibatch baseline runtime_ns unavailable")

    prefill_sum = sum(
        b.get("runtime_ns", 0)
        for b in _blocks_of_type(decoder_blocks_result, "Prefill")
    )
    assert prefill_sum > 0, "Sum of Prefill block runtimes is zero"

    max_allowed = baseline_total * 3.0
    assert prefill_sum <= max_allowed, (
        f"Prefill block runtime sum ({prefill_sum:.0f} ns) is unexpectedly large "
        f"vs. baseline total ({baseline_total:.0f} ns)"
    )


def test_decoder_blocks_decode_total_runtime_vs_multibatch(
    decoder_blocks_result, run_all_baseline
):
    """
    Sum of all Decode block runtimes is within a reasonable bound of the baseline.
    """
    baseline_total = run_all_baseline.get("runtime_ns", 0)
    if baseline_total <= 0:
        pytest.skip("Multibatch baseline runtime_ns unavailable")

    decode_sum = sum(
        b.get("runtime_ns", 0)
        for b in _blocks_of_type(decoder_blocks_result, "Decode")
    )
    assert decode_sum > 0, "Sum of Decode block runtimes is zero"

    max_allowed = baseline_total * 3.0
    assert decode_sum <= max_allowed, (
        f"Decode block runtime sum ({decode_sum:.0f} ns) is unexpectedly large "
        f"vs. baseline total ({baseline_total:.0f} ns)"
    )


def test_decoder_blocks_total_cache_misses_vs_multibatch(
    decoder_blocks_result, run_all_baseline
):
    """
    Compare the total PAPI_L1_DCM across all decoder blocks against the
    multibatch per-tensor sum.

    Measurement granularity and overhead differ:
    - Multibatch: PAPI started/stopped PER TENSOR (hundreds of measurements).
      Each start/stop causes cache pressure in the PAPI library itself, inflating
      the total count.
    - Decoder-block: PAPI started/stopped PER BLOCK (≈ N_layers × 2 measurements).
      Much less overhead → sum is typically ~1.5× lower than the per-tensor baseline.

    Assertions:
      1. Block sum does not exceed multibatch (directionally expected).
      2. Overhead ratio (multibatch / block-sum) is at most 4× — anything beyond
         that indicates an unexpected difference in measurement scope.
    """
    baseline_l1dcm = run_all_baseline.get("l1_dcm")
    if baseline_l1dcm is None:
        pytest.skip(
            "PAPI_L1_DCM not measured in any multibatch event group"
        )

    total_l1dcm = sum(b.get("PAPI_L1_DCM", 0) for b in decoder_blocks_result)
    if total_l1dcm == 0:
        pytest.skip("PAPI_L1_DCM is all-zero in decoder-block output")

    actual   = float(total_l1dcm)
    baseline = float(baseline_l1dcm)
    ratio    = baseline / actual if actual > 0 else float("inf")

    assert actual <= baseline * 1.15, (
        f"Decoder-block L1_DCM sum ({actual:.0f}) unexpectedly exceeds "
        f"multibatch per-tensor sum ({baseline:.0f})"
    )
    assert ratio <= 4.0, (
        f"Multibatch per-tensor L1_DCM ({baseline:.0f}) is {ratio:.2f}× "
        f"the decoder-block sum ({actual:.0f}) — overhead appears excessive (limit: 4×)"
    )

    print(
        f"\n  [cache-miss overhead] block-sum={actual:.0f}  multibatch={baseline:.0f}  "
        f"ratio={ratio:.2f}× (per-tensor PAPI overhead)"
    )
