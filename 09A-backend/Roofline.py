#!/usr/bin/env python3
"""
Roofline.py — Interactive terminal roofline analyser for llama.cpp profiling data.

Data sources:
  JSON (decoder-block-view.json) — entire program, entire prefill, entire decode, specific decode block
  Database (profiling_data.db)   — specific transformer layer (prefill or decode token)

Peak FLOPS : cores × ((base_ghz + boost_ghz) / 2) × flops_per_cycle
Mem BW     : STREAM benchmark (~/stream)
OI         : PAPI_FP_OPS / (PAPI_L3_TCM × 64)
"""

import json
import sys
import os
import re
import sqlite3
import subprocess
import glob
from contextlib import contextmanager
from functools import lru_cache
from typing import Optional

# ── Configuration ─────────────────────────────────────────────────────────────
RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "run_every_view_results",
)
DB_PATH = os.path.join(RESULTS_DIR, "tensor_op_view.db")
STREAM_PATH      = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "stream", "stream_c")
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
#  JSON helpers
# ══════════════════════════════════════════════════════════════════════════════

def find_decoder_block_json() -> Optional[str]:
    path = os.path.join(RESULTS_DIR, "decoder-block-view.json")
    if os.path.isfile(path):
        return path
    matches = glob.glob(os.path.join(RESULTS_DIR, "**", "decoder-block-view.json"), recursive=True)
    return matches[0] if matches else None


def load_json(path: str) -> list:
    with open(path) as f:
        return json.load(f)


def get_blocks(data: list, block_type: str) -> list:
    return [b for b in data if b["block_type"] == block_type]


def get_block(data: list, block_type: str, block_id: int) -> Optional[dict]:
    for b in data:
        if b["block_type"] == block_type and b["block_id"] == block_id:
            return b
    return None


def sum_blocks(blocks: list) -> dict:
    total_flops   = sum(b.get("FLOPs", 0) for b in blocks)
    total_bytes   = sum(b.get("bytes_moved", 0) for b in blocks)
    total_runtime = sum(b.get("runtime_ns", 0) for b in blocks)
    return {
        "total_flops" : total_flops,
        "dram_bytes"  : total_bytes,
        "time_seconds": total_runtime / 1e9,
    }


def roofline_from_aggregated(agg: dict, label: str) -> dict:
    hw              = get_hardware()
    flops           = agg["total_flops"]
    dram_bytes      = agg["dram_bytes"]
    time_s          = agg["time_seconds"]
    oi              = flops / dram_bytes        if dram_bytes > 0 else 0.0
    achieved_gflops = (flops / 1e9) / time_s    if time_s     > 0 else 0.0
    bound           = "compute" if oi > hw["ridge_point"] else "memory"
    return {
        "label"          : label,
        "x"              : round(oi, 6),
        "y"              : round(achieved_gflops, 6),
        "total_flops"    : flops,
        "l3_misses"      : None,
        "dram_bytes"     : dram_bytes,
        "time_seconds"   : round(time_s, 6),
        "oi"             : round(oi, 6),
        "achieved_gflops": round(achieved_gflops, 6),
        "bound"          : bound,
        "source"         : "json",
        "hardware"       : hw,
    }


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


def fetchall(sql: str, params: tuple = (), db_path: str = DB_PATH) -> list:
    with get_conn(db_path) as conn:
        rows = conn.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


def db_available(db_path: str = DB_PATH) -> bool:
    return os.path.isfile(db_path)


def get_layer_numbers(phase: str, token_index: Optional[int] = None, db_path: str = DB_PATH) -> list[int]:
    """Extract distinct layer numbers from tensor names ending with -N."""
    conditions = ["event_phase = ?"]
    params: list = [phase]
    if token_index is not None:
        conditions.append("event_token_index = ?")
        params.append(token_index)
    where = " AND ".join(conditions)

    rows = fetchall(
        f"SELECT DISTINCT event_tensor_name FROM event_item WHERE {where}",
        tuple(params), db_path=db_path,
    )

    layers = set()
    for r in rows:
        m = re.search(r"-(\d+)$", r["event_tensor_name"])
        if m:
            layers.add(int(m.group(1)))
    return sorted(layers)


def get_decode_token_indices(db_path: str = DB_PATH) -> list[int]:
    rows = fetchall(
        "SELECT DISTINCT event_token_index FROM event_item WHERE event_phase = 'decode' AND event_token_index >= 1000 ORDER BY event_token_index",
        db_path=db_path,
    )
    return [r["event_token_index"] for r in rows]



def get_prefill_turn_indices(db_path: str = DB_PATH) -> list[int]:
    rows = fetchall(
        "SELECT DISTINCT event_token_index FROM event_item WHERE event_phase = 'prefill' ORDER BY event_token_index",
        db_path=db_path,
    )
    return [r["event_token_index"] for r in rows]

def get_papi_sum_for_layer(
    papi_event: str,
    phase: str,
    layer: int,
    token_index: Optional[int] = None,
    db_path: str = DB_PATH,
) -> int:
    sql = """
        SELECT SUM(p.papi_value) AS total
        FROM event_item e
        JOIN event_papi_counter p ON e.event_item_id = p.event_item_id
        WHERE p.papi_event_name = ?
          AND e.event_phase = ?
          AND (e.event_tensor_name LIKE ? OR e.event_tensor_name LIKE ?)
    """
    params: list = [papi_event, phase, f"%-{layer}", f"%-{layer} %"]
    if token_index is not None:
        sql += " AND e.event_token_index = ?"
        params.append(token_index)

    row = fetchone(sql, tuple(params), db_path=db_path)
    return int(row["total"] or 0) if row else 0


def get_time_sum_for_layer(
    phase: str,
    layer: int,
    token_index: Optional[int] = None,
    db_path: str = DB_PATH,
) -> float:
    sql = """
        SELECT SUM(event_time_microseconds) AS total
        FROM event_item
        WHERE event_phase = ?
          AND (event_tensor_name LIKE ? OR event_tensor_name LIKE ?)
    """
    params: list = [phase, f"%-{layer}", f"%-{layer} %"]
    if token_index is not None:
        sql += " AND event_token_index = ?"
        params.append(token_index)

    row = fetchone(sql, tuple(params), db_path=db_path)
    return (row["total"] or 0) / 1e6 if row else 0.0


def roofline_from_db_layer(
    phase: str,
    layer: int,
    token_index: Optional[int] = None,
    label: str = "",
    db_path: str = DB_PATH,
) -> dict:
    hw         = get_hardware()
    flops      = get_papi_sum_for_layer(PAPI_FLOPS,   phase, layer, token_index, db_path)
    l3_misses  = get_papi_sum_for_layer(PAPI_L3_MISS, phase, layer, token_index, db_path)
    dram_bytes = l3_misses * CACHE_LINE_BYTES
    time_s     = get_time_sum_for_layer(phase, layer, token_index, db_path)

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
        "source"         : "database",
        "hardware"       : hw,
    }


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
    source_tag  = f"  [{result.get('source', '?')}]"
    print()
    print(f"{BOLD}--- [RESULT] {result['label']}{source_tag} ---{RESET}")
    print(f"  {'FLOPS':<16}: {result['total_flops']:,}")
    if result.get("l3_misses") is not None:
        print(f"  {'L3 misses':<16}: {result['l3_misses']:,}")
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
#  Sub-menus
# ══════════════════════════════════════════════════════════════════════════════

def menu_prefill(data: list, db_path: str):
    while True:
        print()
        print("=" * 44)
        print(f"  {BOLD}PREFILL{RESET}")
        print("=" * 44)
        print("  1. Entire Prefill (all turns)")
        print("  2. Specific prefill turn (from JSON)")
        print("  3. Specific layer for a prefill turn (from DB)")
        print("  0. Back")
        print("=" * 44)
        choice = input("Choice: ").strip()

        if choice == "0":
            break

        elif choice == "1":
            blocks = get_blocks(data, "Prefill")
            if not blocks:
                print("  No Prefill blocks in JSON.")
                continue
            agg    = sum_blocks(blocks)
            result = roofline_from_aggregated(agg, "Entire Prefill")
            print_result(result)

        elif choice == "2":
            blocks = get_blocks(data, "Prefill")
            if not blocks:
                print("  No Prefill blocks in JSON.")
                continue
            ids = [b["block_id"] for b in blocks]
            print(f"\n  Available prefill turns: {ids}")
            raw = input("  Enter block_id or press Enter to go back: ").strip()
            if not raw:
                continue
            try:
                bid = int(raw)
            except ValueError:
                print("  Invalid input.")
                continue
            block = get_block(data, "Prefill", bid)
            if not block:
                print(f"  Prefill turn {bid} not found.")
                continue
            agg = {
                "total_flops" : block["FLOPs"],
                "dram_bytes"  : block["bytes_moved"],
                "time_seconds": block["runtime_ns"] / 1e9,
            }
            result = roofline_from_aggregated(agg, f"Prefill turn {bid}")
            print_result(result)

        elif choice == "3":
            if not db_available(db_path):
                print(f"  {RED}[ERROR]{RESET} Database not found. Run the profiler first.")
                continue
            turns = get_prefill_turn_indices(db_path)
            if not turns:
                print("  No prefill turns found in database.")
                continue
            print(f"\n  Available prefill turns: {turns}")
            raw = input("  Enter turn number or press Enter to go back: ").strip()
            if not raw:
                continue
            try:
                turn = int(raw)
            except ValueError:
                print("  Invalid input.")
                continue
            if turn not in turns:
                print(f"  Turn {turn} not found.")
                continue
            layers = get_layer_numbers("prefill", token_index=turn, db_path=db_path)
            if not layers:
                print("  No layers found for this prefill turn.")
                continue
            print(f"\n  Available layers: {layers[0]} – {layers[-1]}")
            raw = input("  Enter layer number or press Enter to go back: ").strip()
            if not raw:
                continue
            try:
                layer = int(raw)
            except ValueError:
                print("  Invalid input.")
                continue
            if layer not in layers:
                print(f"  Layer {layer} not found.")
                continue
            result = roofline_from_db_layer(
                "prefill", layer,
                token_index=turn,
                label=f"Prefill turn {turn} layer {layer}",
                db_path=db_path,
            )
            print_result(result)

        else:
            print("  Unknown choice.")


def menu_decode(data: list, db_path: str):
    while True:
        print()
        print("=" * 44)
        print(f"  {BOLD}DECODE{RESET}")
        print("=" * 44)
        print("  1. Entire Decode")
        print("  2. Specific decode block (token)")
        print("  3. Specific layer for a decode token")
        print("  0. Back")
        print("=" * 44)
        choice = input("Choice: ").strip()

        if choice == "0":
            break

        elif choice == "1":
            blocks = get_blocks(data, "Decode")
            if not blocks:
                print("  No Decode blocks in JSON.")
                continue
            agg    = sum_blocks(blocks)
            result = roofline_from_aggregated(agg, "Entire Decode")
            print_result(result)

        elif choice == "2":
            blocks = get_blocks(data, "Decode")
            if not blocks:
                print("  No Decode blocks in JSON.")
                continue
            ids = [b["block_id"] for b in blocks]
            print(f"\n  Available decode blocks: {ids}")
            raw = input("  Enter block_id or press Enter to go back: ").strip()
            if not raw:
                continue
            try:
                bid = int(raw)
            except ValueError:
                print("  Invalid input.")
                continue
            block = get_block(data, "Decode", bid)
            if not block:
                print(f"  Block {bid} not found.")
                continue
            agg = {
                "total_flops" : block["FLOPs"],
                "dram_bytes"  : block["bytes_moved"],
                "time_seconds": block["runtime_ns"] / 1e9,
            }
            result = roofline_from_aggregated(agg, f"Decode block {bid}")
            print_result(result)

        elif choice == "3":
            if not db_available(db_path):
                print(f"  {RED}[ERROR]{RESET} Database not found. Run the profiler first.")
                continue
            tokens = get_decode_token_indices(db_path)
            if not tokens:
                print("  No decode tokens found in database.")
                continue
            print(f"\n  Available decode tokens: {tokens}")
            raw = input("  Enter token_index or press Enter to go back: ").strip()
            if not raw:
                continue
            try:
                token_index = int(raw)
            except ValueError:
                print("  Invalid input.")
                continue
            if token_index not in tokens:
                print(f"  Token {token_index} not found.")
                continue
            layers = get_layer_numbers("decode", token_index=token_index, db_path=db_path)
            if not layers:
                print("  No layers found for this token.")
                continue
            print(f"\n  Available layers: {layers[0]} – {layers[-1]}")
            raw = input("  Enter layer number or press Enter to go back: ").strip()
            if not raw:
                continue
            try:
                layer = int(raw)
            except ValueError:
                print("  Invalid input.")
                continue
            if layer not in layers:
                print(f"  Layer {layer} not found.")
                continue
            turn = token_index // 1000
            pos  = token_index % 1000
            result = roofline_from_db_layer(
                "decode", layer,
                token_index=token_index,
                label=f"Decode turn {turn} pos {pos} layer {layer}",
                db_path=db_path,
            )
            print_result(result)

        else:
            print("  Unknown choice.")


# ══════════════════════════════════════════════════════════════════════════════
#  Main menu
# ══════════════════════════════════════════════════════════════════════════════

def menu_loop(data: list, json_path: str, db_path: str):
    while True:
        print()
        print("=" * 44)
        print(f"  {BOLD}ROOFLINE MENU{RESET}")
        print(f"  JSON : {os.path.basename(json_path)}")
        db_status = f"{GREEN}available{RESET}" if db_available(db_path) else f"{RED}not found{RESET}"
        print(f"  DB   : {db_status}")
        print("=" * 44)
        print("  1. Show hardware")
        print("  2. Entire program")
        print("  3. Prefill")
        print("  4. Decode")
        print("  0. Exit")
        print("=" * 44)

        choice = input("Make your choice (0-4): ").strip()

        if choice == "0":
            print("Bye!")
            break
        elif choice == "1":
            print_hardware(get_hardware())
        elif choice == "2":
            if not data:
                print("  No data in JSON.")
                continue
            agg    = sum_blocks(data)
            result = roofline_from_aggregated(agg, "Entire program")
            print_result(result)
        elif choice == "3":
            menu_prefill(data, db_path)
        elif choice == "4":
            menu_decode(data, db_path)
        else:
            print("  Unknown choice, try again.")


# ══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    db_path = DB_PATH

    json_path = find_decoder_block_json()
    if not json_path:
        print(f"{RED}[ERROR]{RESET} decoder-block-view.json not found in {RESULTS_DIR}")
        sys.exit(1)

    print(f"{CYAN}[INFO]{RESET} Detecting hardware ceiling...")
    get_hardware()

    print(f"{CYAN}[INFO]{RESET} Loading {json_path}")
    data = load_json(json_path)

    if not db_available(db_path):
        print(f"{YELLOW}[WARN]{RESET} Database not found — layer analysis unavailable until next run.")

    menu_loop(data, json_path, db_path)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting.")