#!/usr/bin/env python3
"""
Roofline-api.py — FastAPI backend for llama.cpp Roofline Analysis.

Data sources:
  JSON (decoder-block-view.json) — entire program, entire prefill, entire decode, specific blocks
  Database (tensor_op_view.db)   — specific transformer layer (prefill turn or decode token)

Endpoints:
  GET /hardware
  GET /phases                                                → available blocks and tokens
  GET /roofline/all                                          → entire program (JSON)
  GET /roofline/prefill                                      → entire prefill all turns (JSON)
  GET /roofline/prefill/{block_id}                           → specific prefill turn (JSON)
  GET /roofline/prefill/{turn}/layer/{layer}                 → prefill turn + layer (DB)
  GET /roofline/decode                                       → entire decode (JSON)
  GET /roofline/decode/{block_id}                            → specific decode block (JSON)
  GET /roofline/decode/{token_index}/layer/{layer}           → decode token + layer (DB)

Run:
  pip install fastapi uvicorn
  python3 Roofline-api.py
  → http://localhost:8000/docs
"""

import os
import re
import glob
import json
import sqlite3
import subprocess
from contextlib import contextmanager
from functools import lru_cache
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# ── Configuration ─────────────────────────────────────────────────────────────
RESULTS_DIR = os.path.join(
    os.path.expanduser("~"),
    "09A", "profiling-llms-llama-cpp",
    "run_every_view_results",
)
DB_PATH          = os.path.join(RESULTS_DIR, "tensor_op_view.db")
STREAM_PATH      = os.path.expanduser("~/stream")
CACHE_LINE_BYTES = 64
PAPI_FLOPS       = "PAPI_FP_OPS"
PAPI_L3_MISS     = "PAPI_L3_TCM"

app = FastAPI(title="llama.cpp Roofline API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


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
    return {
        "total_flops" : sum(b.get("FLOPs", 0) for b in blocks),
        "dram_bytes"  : sum(b.get("bytes_moved", 0) for b in blocks),
        "time_seconds": sum(b.get("runtime_ns", 0) for b in blocks) / 1e9,
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


def get_prefill_turn_indices(db_path: str = DB_PATH) -> list[int]:
    rows = fetchall(
        "SELECT DISTINCT event_token_index FROM event_item WHERE event_phase = 'prefill' ORDER BY event_token_index",
        db_path=db_path,
    )
    return [r["event_token_index"] for r in rows]


def get_decode_token_indices(db_path: str = DB_PATH) -> list[int]:
    rows = fetchall(
        "SELECT DISTINCT event_token_index FROM event_item WHERE event_phase = 'decode' AND event_token_index >= 1000 ORDER BY event_token_index",
        db_path=db_path,
    )
    return [r["event_token_index"] for r in rows]


def get_papi_sum_for_layer(papi_event: str, phase: str, layer: int, token_index: Optional[int] = None, db_path: str = DB_PATH) -> int:
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


def get_time_sum_for_layer(phase: str, layer: int, token_index: Optional[int] = None, db_path: str = DB_PATH) -> float:
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


def roofline_from_db_layer(phase: str, layer: int, token_index: Optional[int] = None, label: str = "", db_path: str = DB_PATH) -> dict:
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
#  Startup
# ══════════════════════════════════════════════════════════════════════════════

@app.on_event("startup")
async def startup():
    print("\n[STARTUP] Warming up hardware ceiling...")
    hw = get_hardware()
    print(f"[STARTUP] {hw['cpu_model']}")
    print(f"[STARTUP] Peak: {hw['peak_gflops']} GFLOPS | BW: {hw['mem_bw_gbs']} GB/s | Ridge: {hw['ridge_point']} FLOP/byte")
    json_path = find_decoder_block_json()
    print(f"[STARTUP] JSON: {json_path or 'NOT FOUND'}")
    print(f"[STARTUP] DB  : {'found' if db_available() else 'not found'}\n")


# ══════════════════════════════════════════════════════════════════════════════
#  Endpoints
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/hardware")
def hardware():
    """Hardware ceiling: peak FLOPS, memory bandwidth, ridge point."""
    return get_hardware()


@app.get("/phases")
def phases():
    """Available blocks and tokens for UI dropdowns."""
    json_path = find_decoder_block_json()
    if not json_path:
        raise HTTPException(status_code=404, detail="decoder-block-view.json not found.")
    data = load_json(json_path)

    prefill_blocks = [b["block_id"] for b in get_blocks(data, "Prefill")]
    decode_blocks  = [b["block_id"] for b in get_blocks(data, "Decode")]
    prefill_turns  = get_prefill_turn_indices() if db_available() else []
    decode_tokens  = get_decode_token_indices() if db_available() else []

    return {
        "prefill_blocks" : prefill_blocks,
        "decode_blocks"  : decode_blocks,
        "prefill_turns"  : prefill_turns,
        "decode_tokens"  : decode_tokens,
        "db_available"   : db_available(),
    }


@app.get("/roofline/all")
def roofline_all():
    """Entire program (prefill + decode combined) from JSON."""
    json_path = find_decoder_block_json()
    if not json_path:
        raise HTTPException(status_code=404, detail="decoder-block-view.json not found.")
    data = load_json(json_path)
    return roofline_from_aggregated(sum_blocks(data), "Entire program")


@app.get("/roofline/prefill")
def roofline_prefill():
    """Entire prefill phase (all turns) from JSON."""
    json_path = find_decoder_block_json()
    if not json_path:
        raise HTTPException(status_code=404, detail="decoder-block-view.json not found.")
    data   = load_json(json_path)
    blocks = get_blocks(data, "Prefill")
    if not blocks:
        raise HTTPException(status_code=404, detail="No Prefill blocks found.")
    return roofline_from_aggregated(sum_blocks(blocks), "Entire Prefill")


@app.get("/roofline/prefill/{block_id}")
def roofline_prefill_block(block_id: int):
    """Specific prefill turn by block_id from JSON."""
    json_path = find_decoder_block_json()
    if not json_path:
        raise HTTPException(status_code=404, detail="decoder-block-view.json not found.")
    data  = load_json(json_path)
    block = get_block(data, "Prefill", block_id)
    if not block:
        available = [b["block_id"] for b in get_blocks(data, "Prefill")]
        raise HTTPException(status_code=404, detail=f"Prefill block {block_id} not found. Available: {available}")
    agg = {
        "total_flops" : block["FLOPs"],
        "dram_bytes"  : block["bytes_moved"],
        "time_seconds": block["runtime_ns"] / 1e9,
    }
    return roofline_from_aggregated(agg, f"Prefill turn {block_id}")


@app.get("/roofline/prefill/{turn}/layer/{layer}")
def roofline_prefill_layer(turn: int, layer: int):
    """Specific transformer layer for a prefill turn from DB."""
    if not db_available():
        raise HTTPException(status_code=503, detail="Database not available.")
    turns = get_prefill_turn_indices()
    if turn not in turns:
        raise HTTPException(status_code=404, detail=f"Prefill turn {turn} not found. Available: {turns}")
    layers = get_layer_numbers("prefill", token_index=turn)
    if layer not in layers:
        raise HTTPException(status_code=404, detail=f"Layer {layer} not found. Available: {layers}")
    return roofline_from_db_layer("prefill", layer, token_index=turn, label=f"Prefill turn {turn} layer {layer}")


@app.get("/roofline/decode")
def roofline_decode():
    """Entire decode phase from JSON."""
    json_path = find_decoder_block_json()
    if not json_path:
        raise HTTPException(status_code=404, detail="decoder-block-view.json not found.")
    data   = load_json(json_path)
    blocks = get_blocks(data, "Decode")
    if not blocks:
        raise HTTPException(status_code=404, detail="No Decode blocks found.")
    return roofline_from_aggregated(sum_blocks(blocks), "Entire Decode")


@app.get("/roofline/decode/{block_id}")
def roofline_decode_block(block_id: int):
    """Specific decode block by block_id from JSON."""
    json_path = find_decoder_block_json()
    if not json_path:
        raise HTTPException(status_code=404, detail="decoder-block-view.json not found.")
    data  = load_json(json_path)
    block = get_block(data, "Decode", block_id)
    if not block:
        available = [b["block_id"] for b in get_blocks(data, "Decode")]
        raise HTTPException(status_code=404, detail=f"Decode block {block_id} not found. Available: {available}")
    agg = {
        "total_flops" : block["FLOPs"],
        "dram_bytes"  : block["bytes_moved"],
        "time_seconds": block["runtime_ns"] / 1e9,
    }
    return roofline_from_aggregated(agg, f"Decode block {block_id}")


@app.get("/roofline/decode/{token_index}/layer/{layer}")
def roofline_decode_layer(token_index: int, layer: int):
    """Specific transformer layer for a decode token from DB."""
    if not db_available():
        raise HTTPException(status_code=503, detail="Database not available.")
    tokens = get_decode_token_indices()
    if token_index not in tokens:
        raise HTTPException(status_code=404, detail=f"Token {token_index} not found. Available: {tokens}")
    layers = get_layer_numbers("decode", token_index=token_index)
    if layer not in layers:
        raise HTTPException(status_code=404, detail=f"Layer {layer} not found. Available: {layers}")
    turn = token_index // 1000
    pos  = token_index % 1000
    return roofline_from_db_layer(
        "decode", layer,
        token_index=token_index,
        label=f"Decode turn {turn} pos {pos} layer {layer}",
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("Roofline-api:app", host="0.0.0.0", port=8000, reload=False)