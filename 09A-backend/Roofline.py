#!/usr/bin/env python3
"""
Roofline.py — Interactive terminal roofline analyser for llama.cpp profiling data.

Peak FLOPS  : cores × ((base_ghz + boost_ghz) / 2) × flops_per_cycle
Mem BW      : STREAM benchmark (~/ stream)
OI          : PAPI_FP_OPS / (PAPI_L3_TCM × 64)

Sessions: when you run tui.py → "Run all", all event-group runs share a
session_id. Roofline automatically finds the run with PAPI_FP_OPS and the
run with PAPI_L3_TCM within the same session and joins them.
"""

import json
import sys
import os
import re
import sqlite3
import subprocess
from contextlib import contextmanager
from functools import lru_cache
from typing import Optional

# ── Configuration ─────────────────────────────────────────────────────────────
DB_PATH = os.path.join(
    os.path.expanduser("~"),
    "09A", "profiling-llms-llama-cpp",
    "profiling_data.db",
)
STREAM_PATH      = os.path.expanduser("~/stream")
CACHE_LINE_BYTES = 64
PAPI_FLOPS       = "PAPI_FP_OPS"
PAPI_L3_MISS     = "PAPI_L3_TCM"

# ── ANSI colours ──────────────────────────────────────────────────────────────
BOLD   = "\033[1m"
CYAN   = "\033[96m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
RESET  = "\033[0m"


# ══════════════════════════════════════════════════════════════════════════════
#  Database helpers
# ══════════════════════════════════════════════════════════════════════════════

@contextmanager
def get_conn(db_path: str = DB_PATH):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def fetchone(sql: str, params: tuple = (), db_path: str = DB_PATH) -> Optional[dict]:
    with get_conn(db_path) as conn:
        row = conn.execute(sql, params).fetchone()
    return dict(row) if row else None


def fetchall(sql: str, params: tuple = (), db_path: str = DB_PATH) -> list[dict]:
    with get_conn(db_path) as conn:
        rows = conn.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


# ══════════════════════════════════════════════════════════════════════════════
#  Session helpers
# ══════════════════════════════════════════════════════════════════════════════

def get_sessions(db_path: str = DB_PATH) -> list[dict]:
    """
    Return all sessions that have at least one run with PAPI_FP_OPS.
    Each entry: {session_id, run_count, earliest, latest}
    """
    rows = fetchall("""
        SELECT
            e.session_id,
            COUNT(DISTINCT e.run_id)          AS run_count,
            MIN(e.event_item_timestamp)        AS earliest,
            MAX(e.event_item_timestamp)        AS latest
        FROM event_item e
        JOIN event_papi_counter p ON e.event_item_id = p.event_item_id
        WHERE e.session_id IS NOT NULL
          AND p.papi_event_name = 'PAPI_FP_OPS'
        GROUP BY e.session_id
        ORDER BY latest DESC
    """, db_path=db_path)
    return rows


def get_run_id_for_event(session_id: str, papi_event: str, db_path: str = DB_PATH) -> Optional[int]:
    """
    Within a session, find the run_id that has data for the given PAPI event.
    Returns None if not found.
    """
    row = fetchone("""
        SELECT e.run_id
        FROM event_item e
        JOIN event_papi_counter p ON e.event_item_id = p.event_item_id
        WHERE e.session_id = ? AND p.papi_event_name = ?
        LIMIT 1
    """, (session_id, papi_event), db_path=db_path)
    return row["run_id"] if row else None


def get_prefill_token_indices(run_id: int, db_path: str = DB_PATH) -> list[int]:
    rows = fetchall(
        "SELECT DISTINCT event_token_index FROM event_item WHERE event_phase = 'prefill' AND run_id = ? ORDER BY event_token_index",
        (run_id,), db_path=db_path,
    )
    return [r["event_token_index"] for r in rows]


def get_decode_token_indices(run_id: int, db_path: str = DB_PATH) -> list[int]:
    rows = fetchall(
        "SELECT DISTINCT event_token_index FROM event_item WHERE event_phase = 'decode' AND run_id = ? ORDER BY event_token_index",
        (run_id,), db_path=db_path,
    )
    return [r["event_token_index"] for r in rows]


# ══════════════════════════════════════════════════════════════════════════════
#  PAPI / time aggregation
# ══════════════════════════════════════════════════════════════════════════════

def get_papi_sum(
    papi_event: str,
    run_id: int,
    phase: Optional[str] = None,
    token_index: Optional[int] = None,
    db_path: str = DB_PATH,
) -> int:
    conditions = ["p.papi_event_name = ?", "e.run_id = ?"]
    params: list = [papi_event, run_id]
    if phase is not None:
        conditions.append("e.event_phase = ?")
        params.append(phase)
    if token_index is not None:
        conditions.append("e.event_token_index = ?")
        params.append(token_index)
    where = " AND ".join(conditions)
    row = fetchone(
        f"SELECT SUM(p.papi_value) AS total FROM event_item e JOIN event_papi_counter p ON e.event_item_id = p.event_item_id WHERE {where}",
        tuple(params), db_path=db_path,
    )
    return int(row["total"] or 0) if row else 0


def get_time_sum(
    run_id: int,
    phase: Optional[str] = None,
    token_index: Optional[int] = None,
    db_path: str = DB_PATH,
) -> float:
    """Use the FLOPS run_id for time — it's the same hardware so timing is valid."""
    conditions = ["run_id = ?"]
    params: list = [run_id]
    if phase is not None:
        conditions.append("event_phase = ?")
        params.append(phase)
    if token_index is not None:
        conditions.append("event_token_index = ?")
        params.append(token_index)
    where = " AND ".join(conditions)
    row = fetchone(
        f"SELECT SUM(event_time_microseconds) AS total FROM event_item WHERE {where}",
        tuple(params), db_path=db_path,
    )
    return (row["total"] or 0) / 1e6 if row else 0.0


# ══════════════════════════════════════════════════════════════════════════════
#  Hardware detection
# ══════════════════════════════════════════════════════════════════════════════

@lru_cache(maxsize=1)
def get_hardware() -> dict:
    lscpu = subprocess.run(["lscpu"], capture_output=True, text=True).stdout

    cores, threads_per_core, sockets, total_cpus = None, 1, 1, None
    for line in lscpu.splitlines():
        if re.match(r"^CPU\(s\):\s+\d+", line):
            total_cpus = int(line.split(":")[1].strip())
        elif "Thread(s) per core" in line:
            threads_per_core = int(line.split(":")[1].strip())
        elif "Socket(s)" in line:
            sockets = int(line.split(":")[1].strip())
        elif "Core(s) per socket" in line:
            cores = int(line.split(":")[1].strip()) * sockets
    if cores is None and total_cpus:
        cores = total_cpus // threads_per_core
    cores = cores or 1

    base_ghz = boost_ghz = None
    for line in lscpu.splitlines():
        if "Model name" in line:
            m = re.search(r"@\s*([\d.]+)\s*GHz", line, re.I)
            if m:
                base_ghz = float(m.group(1))
        if "CPU max MHz" in line:
            boost_ghz = float(line.split(":")[1].strip()) / 1000

    if boost_ghz is None:
        try:
            with open("/sys/devices/system/cpu/cpu0/cpufreq/cpuinfo_max_freq") as f:
                boost_ghz = int(f.read().strip()) / 1e6
        except OSError:
            pass

    if base_ghz is None:
        base_ghz = (boost_ghz * 0.75) if boost_ghz else 1.0
    if boost_ghz is None:
        boost_ghz = base_ghz * 1.4

    avg_ghz = (base_ghz + boost_ghz) / 2.0

    flags = lscpu.lower()
    if "avx512f" in flags:
        isa, flops_per_cycle = "AVX-512", 32
    elif "avx2" in flags:
        isa, flops_per_cycle = "AVX2", 16
    else:
        isa, flops_per_cycle = "SSE2", 8

    peak_gflops = cores * avg_ghz * flops_per_cycle

    mem_bw_gbs = 30.0
    if os.path.isfile(STREAM_PATH):
        try:
            result = subprocess.run([STREAM_PATH], capture_output=True, text=True, timeout=120)
            for keyword in ("Copy:", "Triad:"):
                m = re.search(rf"{keyword}\s+([\d.]+)", result.stdout)
                if m:
                    mem_bw_gbs = float(m.group(1)) / 1000
                    break
        except Exception as e:
            print(f"[WARN] STREAM failed: {e}. Using 30 GB/s.")
    else:
        print(f"[WARN] STREAM not found at {STREAM_PATH}. Using 30 GB/s.")

    ridge_point = peak_gflops / mem_bw_gbs if mem_bw_gbs > 0 else float("inf")
    cpu_model = next((l.split(":", 1)[1].strip() for l in lscpu.splitlines() if "Model name" in l), "Unknown")

    return {
        "cpu_model"      : cpu_model,
        "cores"          : cores,
        "base_ghz"       : round(base_ghz, 2),
        "boost_ghz"      : round(boost_ghz, 2),
        "avg_ghz"        : round(avg_ghz, 2),
        "isa"            : isa,
        "flops_per_cycle": flops_per_cycle,
        "peak_gflops"    : round(peak_gflops, 2),
        "mem_bw_gbs"     : round(mem_bw_gbs, 2),
        "ridge_point"    : round(ridge_point, 4),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Roofline computation
# ══════════════════════════════════════════════════════════════════════════════

def compute_roofline(
    session_id: str,
    phase: Optional[str] = None,
    token_index: Optional[int] = None,
    label: str = "",
    db_path: str = DB_PATH,
) -> dict:
    """
    Compute roofline for a session.
    Automatically finds which run has PAPI_FP_OPS and which has PAPI_L3_TCM.
    """
    hw = get_hardware()

    flops_run_id = get_run_id_for_event(session_id, PAPI_FLOPS, db_path)
    l3_run_id    = get_run_id_for_event(session_id, PAPI_L3_MISS, db_path)

    if flops_run_id is None:
        print(f"  {YELLOW}[WARN]{RESET} No {PAPI_FLOPS} data in this session. FLOPS = 0.")
    if l3_run_id is None:
        print(f"  {YELLOW}[WARN]{RESET} No {PAPI_L3_MISS} data in this session. L3 misses = 0.")

    flops     = get_papi_sum(PAPI_FLOPS,   flops_run_id, phase, token_index, db_path) if flops_run_id else 0
    l3_misses = get_papi_sum(PAPI_L3_MISS, l3_run_id,    phase, token_index, db_path) if l3_run_id    else 0

    dram_bytes = l3_misses * CACHE_LINE_BYTES

    # Use FLOPS run for timing (or L3 run as fallback)
    time_run_id = flops_run_id or l3_run_id
    time_s = get_time_sum(time_run_id, phase, token_index, db_path) if time_run_id else 0.0

    oi              = flops / dram_bytes        if dram_bytes > 0 else 0.0
    achieved_gflops = (flops / 1e9) / time_s    if time_s     > 0 else 0.0
    bound           = "compute" if oi > hw["ridge_point"] else "memory"

    return {
        "label"          : label,
        "x"              : round(oi, 6),
        "y"              : round(achieved_gflops, 6),
        "total_flops"    : flops,
        "l3_misses"      : l3_misses,
        "dram_bytes"     : dram_bytes,
        "time_seconds"   : round(time_s, 6),
        "oi"             : round(oi, 6),
        "achieved_gflops": round(achieved_gflops, 6),
        "bound"          : bound,
        "flops_run_id"   : flops_run_id,
        "l3_run_id"      : l3_run_id,
        "hardware"       : hw,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Display helpers
# ══════════════════════════════════════════════════════════════════════════════

def print_hardware(hw: dict):
    print()
    print("=" * 50)
    print(f"  CPU      : {hw['cpu_model']}")
    print(f"  Cores    : {hw['cores']}")
    print(f"  Base GHz : {hw['base_ghz']}")
    print(f"  Boost GHz: {hw['boost_ghz']}")
    print(f"  Avg GHz  : {hw['avg_ghz']}  [(base + boost) / 2]")
    print(f"  ISA      : {hw['isa']}  ({hw['flops_per_cycle']} FLOP/cycle)")
    print(f"  Peak     : {hw['peak_gflops']} GFLOPS  [cores × avg_ghz × fpc]")
    print(f"  Mem BW   : {hw['mem_bw_gbs']} GB/s  (STREAM Copy)")
    print(f"  Ridge pt : {hw['ridge_point']} FLOP/byte")
    print("=" * 50)


def print_result(result: dict):
    bound_color = GREEN if result["bound"] == "compute" else YELLOW
    print()
    print(f"{BOLD}--- [RESULT] {result['label']} ---{RESET}")
    print(f"  {'FLOPS':<16}: {result['total_flops']:,}  (run {result['flops_run_id']})")
    print(f"  {'L3 misses':<16}: {result['l3_misses']:,}  (run {result['l3_run_id']})")
    print(f"  {'DRAM bytes':<16}: {result['dram_bytes']:,}")
    print(f"  {'Time (s)':<16}: {result['time_seconds']:.4f}")
    print(f"  {'OI':<16}: {result['oi']:.4f} FLOP/byte")
    print(f"  {'Achieved':<16}: {result['achieved_gflops']:.4f} GFLOPS")
    print(f"  {'Bound':<16}: {bound_color}{result['bound'].upper()}{RESET}")
    print()
    print(f"{BOLD}--- [JSON DATA] ---{RESET}")
    print(json.dumps(result, indent=2))
    print("-------------------")


# ══════════════════════════════════════════════════════════════════════════════
#  Session selector
# ══════════════════════════════════════════════════════════════════════════════

def select_session(db_path: str) -> str:
    sessions = get_sessions(db_path)
    if not sessions:
        print(f"{RED}[ERROR]{RESET} No sessions found in database.")
        print("Run tui.py → 'Run all' with database storage enabled first.")
        sys.exit(1)

    print()
    print("=" * 70)
    print(f"  {'#':<4} {'Session ID':<38} {'Runs':<6} {'Date'}")
    print("=" * 70)
    for i, s in enumerate(sessions):
        date = (s["latest"] or "")[:16]
        print(f"  {i+1:<4} {s['session_id']:<38} {s['run_count']:<6} {date}")
    print("=" * 70)

    print(f"\nSelect session [Enter = 1 (latest)]: ", end="")
    raw = input().strip()
    if raw == "":
        idx = 0
    else:
        try:
            idx = int(raw) - 1
        except ValueError:
            idx = 0
    if idx < 0 or idx >= len(sessions):
        idx = 0

    selected = sessions[idx]["session_id"]
    print(f"  Using session: {selected}")
    return selected


# ══════════════════════════════════════════════════════════════════════════════
#  Menu loop
# ══════════════════════════════════════════════════════════════════════════════

def menu_loop(session_id: str, db_path: str):
    # Find a run_id to use for token index lookups (use flops run)
    flops_run_id = get_run_id_for_event(session_id, PAPI_FLOPS, db_path)
    if flops_run_id is None:
        print(f"{RED}[ERROR]{RESET} No PAPI_FP_OPS data in this session.")
        sys.exit(1)

    while True:
        print()
        print("=" * 50)
        print(f"  {BOLD}ROOFLINE MENU{RESET}  (Session: {session_id[:8]}…)")
        print("=" * 50)
        print("  1. Show hardware")
        print("  2. Entire session (All phases)")
        print("  3. Entire Prefill phase")
        print("  4. Specific Prefill turn (choose from list)")
        print("  5. Specific Decode token (choose from list)")
        print("  0. Exit")
        print("=" * 50)

        choice = input("Make your choice (0-5): ").strip()

        if choice == "0":
            print("Bye!")
            break

        elif choice == "1":
            print_hardware(get_hardware())

        elif choice == "2":
            print_result(compute_roofline(session_id, label="Entire session", db_path=db_path))

        elif choice == "3":
            print_result(compute_roofline(session_id, phase="prefill", label="Entire Prefill phase", db_path=db_path))

        elif choice == "4":
            turns = get_prefill_token_indices(flops_run_id, db_path)
            if not turns:
                print("  No prefill turns found.")
                continue
            print(f"\n  Available Prefill turns: {turns}")
            raw = input("  Enter turn number or press Enter to go back: ").strip()
            if not raw:
                continue
            try:
                ti = int(raw)
            except ValueError:
                print("  Invalid input.")
                continue
            if ti not in turns:
                print(f"  {ti} not in available turns.")
                continue
            print_result(compute_roofline(session_id, phase="prefill", token_index=ti, label=f"Prefill turn {ti}", db_path=db_path))

        elif choice == "5":
            tokens = get_decode_token_indices(flops_run_id, db_path)
            if not tokens:
                print("  No decode tokens found.")
                continue
            print(f"\n  Available Decode tokens: {tokens}")
            raw = input("  Enter token_index (e.g., 1001) or press Enter to go back: ").strip()
            if not raw:
                continue
            try:
                ti = int(raw)
            except ValueError:
                print("  Invalid input.")
                continue
            if ti not in tokens:
                print(f"  {ti} not in available decode tokens.")
                continue
            turn = ti // 1000
            pos  = ti % 1000
            print_result(compute_roofline(session_id, phase="decode", token_index=ti, label=f"Decode turn {turn} pos {pos}", db_path=db_path))

        else:
            print("  Unknown choice, try again.")


# ══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    db_path = sys.argv[1] if len(sys.argv) > 1 else DB_PATH
    if not os.path.isfile(db_path):
        print(f"{RED}[ERROR]{RESET} Database not found: {db_path}")
        sys.exit(1)

    print(f"{CYAN}[INFO]{RESET} Detecting hardware ceiling...")
    get_hardware()

    session_id = select_session(db_path)
    menu_loop(session_id, db_path)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting.")