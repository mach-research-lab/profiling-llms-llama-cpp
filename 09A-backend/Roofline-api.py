#!/usr/bin/env python3
"""
Roofline-api.py — FastAPI backend for llama.cpp Roofline Analysis.

Endpoints:
  GET /hardware                                              → peak FLOPS, mem BW, ridge point
  GET /sessions                                             → available sessions
  GET /phases?session_id=X                                  → available prefill turns + decode tokens
  GET /roofline/all?session_id=X                            → entire session
  GET /roofline/prefill?session_id=X&token_index=N          → prefill (all or one turn)
  GET /roofline/decode/{token_index}?session_id=X           → one decode step

Run:
  pip install fastapi uvicorn
  python3 Roofline-api.py
  → http://localhost:8000/docs
"""

import os
import re
import sys
import sqlite3
import subprocess
from contextlib import contextmanager
from functools import lru_cache
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

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

app = FastAPI(title="llama.cpp Roofline API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

_cache: dict = {}


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
    return fetchall("""
        SELECT
            e.session_id,
            COUNT(DISTINCT e.run_id)      AS run_count,
            MIN(e.event_item_timestamp)   AS earliest,
            MAX(e.event_item_timestamp)   AS latest
        FROM event_item e
        JOIN event_papi_counter p ON e.event_item_id = p.event_item_id
        WHERE e.session_id IS NOT NULL
          AND p.papi_event_name = 'PAPI_FP_OPS'
        GROUP BY e.session_id
        ORDER BY latest DESC
    """, db_path=db_path)


def get_latest_session_id(db_path: str = DB_PATH) -> str:
    sessions = get_sessions(db_path)
    if not sessions:
        raise RuntimeError("No sessions found in database.")
    return sessions[0]["session_id"]


def get_run_id_for_event(session_id: str, papi_event: str, db_path: str = DB_PATH) -> Optional[int]:
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

def get_papi_sum(papi_event: str, run_id: int, phase: Optional[str] = None, token_index: Optional[int] = None, db_path: str = DB_PATH) -> int:
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


def get_time_sum(run_id: int, phase: Optional[str] = None, token_index: Optional[int] = None, db_path: str = DB_PATH) -> float:
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

def compute_roofline(session_id: str, phase: Optional[str] = None, token_index: Optional[int] = None, label: str = "", db_path: str = DB_PATH) -> dict:
    hw = get_hardware()

    flops_run_id = get_run_id_for_event(session_id, PAPI_FLOPS, db_path)
    l3_run_id    = get_run_id_for_event(session_id, PAPI_L3_MISS, db_path)

    flops     = get_papi_sum(PAPI_FLOPS,   flops_run_id, phase, token_index, db_path) if flops_run_id else 0
    l3_misses = get_papi_sum(PAPI_L3_MISS, l3_run_id,    phase, token_index, db_path) if l3_run_id    else 0

    dram_bytes = l3_misses * CACHE_LINE_BYTES
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
#  Startup
# ══════════════════════════════════════════════════════════════════════════════

def _resolve_session(session_id: Optional[str]) -> str:
    if session_id:
        return session_id
    if "session_id" in _cache:
        return _cache["session_id"]
    return get_latest_session_id()


@app.on_event("startup")
async def startup():
    print("\n[STARTUP] Warming up hardware ceiling...")
    hw = get_hardware()
    print(f"[STARTUP] {hw['cpu_model']}")
    print(f"[STARTUP] Peak: {hw['peak_gflops']} GFLOPS | BW: {hw['mem_bw_gbs']} GB/s | Ridge: {hw['ridge_point']} FLOP/byte")
    try:
        session_id = get_latest_session_id()
        _cache["session_id"] = session_id
        result = compute_roofline(session_id, label="Entire session")
        _cache["all"] = result
        print(f"[STARTUP] Session: {session_id[:8]}…  OI={result['oi']:.4f}  Achieved={result['achieved_gflops']:.4f} GFLOPS\n")
    except RuntimeError as e:
        print(f"[STARTUP WARN] {e}\n")


# ══════════════════════════════════════════════════════════════════════════════
#  Endpoints
# ══════════════════════════════════════════════════════════════════════════════

@app.get("/hardware")
def hardware():
    return get_hardware()


@app.get("/sessions")
def sessions():
    return get_sessions()


@app.get("/phases")
def phases(session_id: Optional[str] = Query(None)):
    sid = _resolve_session(session_id)
    flops_run_id = get_run_id_for_event(sid, PAPI_FLOPS)
    if flops_run_id is None:
        raise HTTPException(status_code=404, detail=f"No PAPI_FP_OPS data for session {sid}")

    phase_list = [{"value": "all", "label": "All phases"}]
    for ti in get_prefill_token_indices(flops_run_id):
        phase_list.append({"value": f"prefill_{ti}", "label": f"Prefill (turn {ti})"})
    for ti in get_decode_token_indices(flops_run_id):
        turn, pos = ti // 1000, ti % 1000
        phase_list.append({"value": f"decode_{ti}", "label": f"Decode turn {turn} pos {pos}  (token {ti})"})

    return {"session_id": sid, "phases": phase_list}


@app.get("/roofline/all")
def roofline_all(session_id: Optional[str] = Query(None)):
    sid = _resolve_session(session_id)
    if session_id is None and "all" in _cache:
        return _cache["all"]
    return compute_roofline(sid, label="Entire session")


@app.get("/roofline/prefill")
def roofline_prefill(
    session_id:  Optional[str] = Query(None),
    token_index: Optional[int] = Query(None),
):
    sid   = _resolve_session(session_id)
    label = f"Prefill turn {token_index}" if token_index else "Entire Prefill phase"
    return compute_roofline(sid, phase="prefill", token_index=token_index, label=label)


@app.get("/roofline/decode/{token_index}")
def roofline_decode(token_index: int, session_id: Optional[str] = Query(None)):
    sid          = _resolve_session(session_id)
    flops_run_id = get_run_id_for_event(sid, PAPI_FLOPS)
    if flops_run_id is None:
        raise HTTPException(status_code=404, detail=f"No PAPI_FP_OPS data for session {sid}")

    available = get_decode_token_indices(flops_run_id)
    if token_index not in available:
        raise HTTPException(status_code=404, detail=f"token_index {token_index} not found for session {sid}.")

    turn, pos = token_index // 1000, token_index % 1000
    return compute_roofline(sid, phase="decode", token_index=token_index, label=f"Decode turn {turn} pos {pos}")


# ══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("Roofline-api:app", host="0.0.0.0", port=8000, reload=False)