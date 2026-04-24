"""
helpers.py — shared constants, CSV parsers, binary runner, and assertion helper
for the LLM profiling test suite.

Imported directly by test modules (unlike conftest.py, which is loaded by pytest
automatically and cannot be imported as a regular module).
"""

import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Optional

import pytest

# ── Path constants ─────────────────────────────────────────────────────────────

LLAMA_ROOT = Path(__file__).parents[2]          # profiling-llms-llama-cpp/
BIN_DIR    = LLAMA_ROOT / "build/bin"
MODEL_PATH = (
    LLAMA_ROOT
    / "models/models"
    / "bartowski_Qwen2.5-1.5B-Instruct-GGUF"
    / "Qwen2.5-1.5B-Instruct-Q4_K_M.gguf"
)
TEST_OUTPUTS = Path(__file__).parent / "test_outputs"

# ── Run parameters ─────────────────────────────────────────────────────────────

TEST_PROMPT = "Hello"
N_PREDICT   = "5"
K_CACHE     = "f16"
V_CACHE     = "f16"
PAPI_EVENTS = "PAPI_L1_DCM"

# ── Comparison tolerances (relative fraction) ──────────────────────────────────

TOLERANCES = {
    "runtime":      0.20,   # ±20%  — wall-clock vs. compute time naturally differ
    "energy":       0.25,   # ±25%  — RAPL readings vary with CPU state / thermals
    "cache_misses": 0.15,   # ±15%  — PAPI counters are stable but global vs. per-tensor differs
}


# ── Top-view output parser ────────────────────────────────────────────────────

# Text-format label → normalised JSON key name
_TOP_VIEW_KEY_MAP: dict[str, str] = {
    "runtime_ns":              "runtime_ns",
    "runtime_s":               "runtime_s",
    "model_size_mb":           "model_size_mb",
    "peak_rss_mb":             "peak_rss_mb",
    "avg_cpu_usage":           "avg_cpu_usage",
    "generated_tokens":        "generated_tokens",
    "total_tokens":            "total_tokens",
    "token_throughput":        "token_throughput",
    "kv_tokens_used":          "kv_tokens_used",
    "kv_tokens_capacity":      "kv_tokens_capacity",
    "kv_size_used_bytes":      "kv_size_used_bytes",
    "kv_size_estimated_bytes": "kv_size_estimated_bytes",
    "kv_size_capacity_bytes":  "kv_size_capacity_bytes",
    "cpu_package_uj":          "cpu_package_uj",
    "cpu_cores_uj":            "cpu_cores_uj",
    "full_system_uj":          "full_system_uj",
}

_INTEGER_KEYS = {
    "runtime_ns", "model_size_bytes", "generated_tokens", "total_tokens",
    "kv_tokens_used", "kv_tokens_capacity",
    "kv_size_used_bytes", "kv_size_estimated_bytes", "kv_size_capacity_bytes",
}


def parse_top_view_file(path: Path) -> dict:
    """
    Parse a top-view output file and return a normalised dict.

    Supports two formats produced by llama-measurement-top-view:
      - JSON (rebuilt binary): standard JSON object
      - Text (older binary):   "KEY: VALUE [unit]" lines with a "TOP_VIEW" header
    """
    text = path.read_text().strip()

    # ── JSON format ──
    if text.startswith("{"):
        return json.loads(text)

    # ── Text format ──
    result: dict = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line == "TOP_VIEW":
            continue
        if ":" not in line:
            continue

        raw_key, _, rest = line.partition(":")
        raw_key = raw_key.strip()
        rest    = rest.strip()

        # First whitespace-separated token is the numeric value;
        # strip any trailing non-numeric chars (e.g. '%')
        parts = rest.split()
        if not parts:
            continue
        try:
            raw_val = parts[0].rstrip("%")
            num_val = float(raw_val)
        except ValueError:
            continue

        key_lower = raw_key.lower()

        # model_size appears as "model_size: N bytes" in text format
        if key_lower == "model_size" and "bytes" in rest:
            result["model_size_bytes"] = int(num_val)
            continue

        # Normalise key
        json_key = _TOP_VIEW_KEY_MAP.get(key_lower)
        if json_key is None:
            # PAPI events (e.g. PAPI_L1_DCM) keep their original name
            if raw_key.startswith("PAPI_"):
                json_key = raw_key
            else:
                continue  # unknown field, skip

        result[json_key] = int(num_val) if json_key in _INTEGER_KEYS else num_val

    return result


# ── CSV parsers ────────────────────────────────────────────────────────────────

def parse_events_group_csv(path: Path) -> dict:
    """
    Read a per-tensor events CSV (events_group_N.csv) and return:
      {runtime_ns: int, l1_dcm: int|None}

    runtime_ns = sum of all time_ns rows.
    l1_dcm     = sum of papi_l1_dcm column if present, else None.
    """
    total_time_ns = 0
    total_l1_dcm: Optional[int] = None

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        has_l1_dcm = "papi_l1_dcm" in fieldnames
        if has_l1_dcm:
            total_l1_dcm = 0
        for row in reader:
            try:
                total_time_ns += int(row.get("time_ns") or 0)
            except (ValueError, TypeError):
                pass
            if has_l1_dcm:
                try:
                    total_l1_dcm += int(row.get("papi_l1_dcm") or 0)  # type: ignore[operator]
                except (ValueError, TypeError):
                    pass

    return {"runtime_ns": total_time_ns, "l1_dcm": total_l1_dcm}


def parse_energy_csv(path: Path) -> Optional[float]:
    """
    Sum cpu_package_uj from energy.csv.
    Returns None when the column is missing or all values are zero
    (indicating RAPL is unavailable on this machine).
    """
    total = 0.0
    has_nonzero = False

    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        if "cpu_package_uj" not in (reader.fieldnames or []):
            return None
        for row in reader:
            try:
                val = float(row.get("cpu_package_uj") or 0)
                total += val
                if val > 0:
                    has_nonzero = True
            except (ValueError, TypeError):
                pass

    return total if has_nonzero else None


# ── Binary runner ──────────────────────────────────────────────────────────────

def run_binary(
    binary_name: str,
    extra_args: list,
    out_path: Path,
    timeout: int = 300,
) -> subprocess.CompletedProcess:
    """
    Run a measurement binary and return the CompletedProcess.
    Calls pytest.skip() if the binary has not been compiled yet.
    """
    binary = BIN_DIR / binary_name
    if not binary.is_file():
        pytest.skip(
            f"{binary_name} not compiled — build it first to enable these tests.\n"
            f"Expected: {binary}"
        )

    cmd = [
        str(binary),
        *extra_args,
        "--result-path", str(out_path),
        "-m", str(MODEL_PATH),
        "-p", TEST_PROMPT,
        "-n", N_PREDICT,
        "--temp", "0",
        "--log-disable",
    ]
    return subprocess.run(
        cmd, cwd=str(LLAMA_ROOT), capture_output=True, text=True, timeout=timeout
    )


# ── Assertion helper ───────────────────────────────────────────────────────────

def assert_within_tolerance(
    label: str,
    actual: float,
    baseline: float,
    tol: float,
) -> None:
    """
    Assert that actual is within tol fraction of baseline.
    Skips when baseline is zero or None (metric unavailable).
    """
    if not baseline:
        pytest.skip(f"Baseline for '{label}' is zero or unavailable — skipping comparison")
    delta = abs(actual - baseline) / baseline
    assert delta <= tol, (
        f"{label} out of tolerance:\n"
        f"  actual   = {actual:.2f}\n"
        f"  baseline = {baseline:.2f}\n"
        f"  delta    = {delta * 100:.1f}%  (limit: {tol * 100:.0f}%)"
    )
