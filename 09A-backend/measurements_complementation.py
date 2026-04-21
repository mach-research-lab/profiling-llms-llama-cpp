import json
from typing import Optional

from db_queries import (
    get_run_ids,
    get_time_breakdown_by_operation,
    get_arithmetic_intensity_per_operation,
    get_papi_totals_by_operation,
    get_distinct_papi_events,
    DB_PATH,
)


"""
/////////// PHASE COMPLEMENT //////////////////
"""

def enrich_phase_json(
    json_path:  str,
    db_path:    str = DB_PATH,
    run_id:     Optional[int] = None,
) -> dict:
    with open(json_path) as f:
        data = json.load(f)

    # Use latest run_id if not specified
    if run_id is None:
        ids = get_run_ids(db_path)
        if not ids:
            print("Warning: no runs found in database, skipping DB metrics.")
            for phase in data.values():
                _add_derived_metrics(phase)
            return data
        run_id = ids[-1]

    print(f"Enriching with run_id={run_id}")

    for phase_name, phase in data.items():
        _add_derived_metrics(phase)
        _add_db_metrics(phase, phase_name, run_id, db_path)

    return data


def _add_derived_metrics(phase: dict):
    runtime_ns = phase.get("runtime_ns", 0)
    runtime_s  = runtime_ns / 1e9

    phase["runtime_ms"] = runtime_ns / 1e6

    # IPC
    tot_ins = phase.get("PAPI_TOT_INS", 0)
    tot_cyc = phase.get("PAPI_TOT_CYC", 0)
    phase["IPC"] = (tot_ins / tot_cyc) if tot_cyc > 0 else None

    # FLOPs
    fp_ops = phase.get("PAPI_FP_OPS", 0)
    phase["FLOPs"]  = fp_ops
    phase["FLOP_s"] = (fp_ops / runtime_s) if runtime_s > 0 else None

    # LLC
    llc_accesses     = phase.get("PAPI_L3_TCA", 0)
    llc_misses       = phase.get("PAPI_L3_TCM", 0)
    phase["LLC_hits"]      = llc_accesses - llc_misses
    phase["LLC_misses"]    = llc_misses
    phase["LLC_miss_rate"] = (llc_misses / llc_accesses) if llc_accesses > 0 else None

    # Bytes moved + arithmetic intensity
    bytes_moved              = llc_misses * 64  # 64-byte cache lines
    phase["bytes_moved"]          = bytes_moved
    phase["arithmetic_intensity"] = (fp_ops / bytes_moved) if bytes_moved > 0 else None

    # Energy
    energy = phase.get("energy", {})
    phase["avg_power_pkg_w"] = (
        energy.get("energy-pkg", 0) / 1e6 / runtime_s
    ) if runtime_s > 0 else None

    # Average core utilization
    all_threads = [
        util
        for socket in phase.get("core_utilization", {}).values()
        for core   in socket.values()
        for util   in core.values()
    ]
    phase["avg_core_utilization"] = (
        sum(all_threads) / len(all_threads)
    ) if all_threads else None


def _add_db_metrics(phase: dict, phase_name: str, run_id: int, db_path: str):

    # --- Operation type share (time + count) ---
    op_rows = get_time_breakdown_by_operation(run_id=run_id, phase=phase_name, db_path=db_path)
    if op_rows:
        total_time  = sum(r["total_time_us"] for r in op_rows)
        total_count = sum(r["count"]         for r in op_rows)
        phase["op_type_share"] = {
            r["event_operation_type"]: {
                "count":           r["count"],
                "count_share_pct": round(r["count"]        / total_count * 100, 2) if total_count else 0,
                "total_time_us":   r["total_time_us"],
                "time_share_pct":  round(r["total_time_us"] / total_time  * 100, 2) if total_time  else 0,
                "avg_time_us":     r["avg_time_us"],
            }
            for r in op_rows
        }

    # --- Arithmetic intensity per operation from DB ---
    ai_rows = get_arithmetic_intensity_per_operation(run_id=run_id, phase=phase_name, db_path=db_path)
    if ai_rows and "op_type_share" in phase:
        for r in ai_rows:
            op = r["event_operation_type"]
            if op in phase["op_type_share"]:
                phase["op_type_share"][op]["intensity_ratio"]  = r["intensity_ratio"]
                phase["op_type_share"][op]["total_bytes"]      = r["total_bytes"]
                phase["op_type_share"][op]["total_elements"]   = r["total_elements"]

    # --- PAPI totals per operation for every recorded event ---
    """available_papi = get_distinct_papi_events(db_path=db_path)
    if available_papi and "op_type_share" in phase:
        for event_name in available_papi:
            papi_rows = get_papi_totals_by_operation(
                papi_event=event_name, run_id=run_id, phase=phase_name, db_path=db_path
            )
            for r in papi_rows:
                op = r["event_operation_type"]
                if op in phase["op_type_share"]:
                    phase["op_type_share"][op][event_name] = r["total_papi_value"] """


def complement_phase_json(
    json_path:  str,
    db_path:    str = DB_PATH,
    run_id:     Optional[int] = None,
    out_path:   Optional[str] = None,
) -> dict:
    result   = enrich_phase_json(json_path, db_path, run_id)
    out_path = out_path or json_path.replace(".json", "_enriched.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved enriched JSON to {out_path}")
    return result


"""
/////////// DECODER BLOCK COMPLEMENT //////////////////
"""


PAPI_KEYS = [
    "PAPI_L1_DCM", "PAPI_L1_ICM", "PAPI_L1_TCM",
    "PAPI_L2_DCM", "PAPI_L2_ICM", "PAPI_L2_TCM",
    "PAPI_L2_DCA", "PAPI_L2_TCA", "PAPI_L2_DCR", "PAPI_L2_TCR", "PAPI_L2_LDM",
    "PAPI_L3_DCM", "PAPI_L3_TCA", "PAPI_L3_TCM", "PAPI_L3_DCA", "PAPI_L3_DCR", "PAPI_L3_LDM",
    "PAPI_FP_OPS", "PAPI_TOT_CYC", "PAPI_TOT_INS",
]


def _derive_block_metrics(block: dict, total_runtime_ns: int) -> dict:
    runtime_ns = block.get("runtime_ns", 0)
    runtime_s  = runtime_ns / 1e9

    # --- Runtime ---
    block["runtime_ms"]           = runtime_ns / 1e6
    block["runtime_share_pct"]    = round(runtime_ns / total_runtime_ns * 100, 3) if total_runtime_ns > 0 else None

    # --- FLOPs ---
    fp_ops = block.get("PAPI_FP_OPS", 0)
    block["FLOPs"]  = fp_ops
    block["FLOP_s"] = (fp_ops / runtime_s) if runtime_s > 0 and fp_ops else None

    # --- Cache behavior ---
    l1_misses = block.get("PAPI_L1_TCM", 0)
    l2_misses = block.get("PAPI_L2_TCM", 0)
    l3_access = block.get("PAPI_L3_TCA", 0)
    l3_misses = block.get("PAPI_L3_TCM", 0)
    l3_hits   = l3_access - l3_misses

    block["cache_behavior"] = {
        "L1_misses":      l1_misses,
        "L2_misses":      l2_misses,
        "L3_accesses":    l3_access,
        "L3_hits":        l3_hits,
        "L3_misses":      l3_misses,
        "L3_miss_rate":   round(l3_misses / l3_access, 4) if l3_access > 0 else None,
        "L3_hit_rate":    round(l3_hits   / l3_access, 4) if l3_access > 0 else None,
    }

    # --- Bytes moved (LLC misses × 64-byte cache line) ---
    bytes_moved          = l3_misses * 64
    block["bytes_moved"] = bytes_moved

    # --- Arithmetic intensity ---
    block["arithmetic_intensity"] = (fp_ops / bytes_moved) if bytes_moved > 0 and fp_ops else None

    # --- IPC ---
    tot_ins = block.get("PAPI_TOT_INS", 0)
    tot_cyc = block.get("PAPI_TOT_CYC", 0)
    block["IPC"] = round(tot_ins / tot_cyc, 4) if tot_cyc > 0 else None

    return block

def _get_db_subcomponent_metrics(
    run_id:        int,        # kept for the timing/structural queries
    block_type:    str,
    block_id:      int,
    db_path:       str,
    is_last_block: bool = False,
) -> dict:
    from db_queries import _fetchall

    phase = block_type.lower()
    n     = block_id

    attn_patterns = [
        f"norm-{n}", f"attn_norm-{n}",
        f"Qcur-{n}%", f"Kcur-{n}%", f"Vcur-{n}%",
        f"cache_k_l{n}%", f"cache_v_l{n}%",
        f" (copy)", f"__fattn__-{n}",
        f"kqv_out-{n}%", f"attn_out-{n}", f"ffn_inp-{n}",
    ]
    ffn_patterns = [
        f"ffn_norm-{n}", f"ffn_gate-{n}", f"ffn_up-{n}",
        f"ffn_swiglu-{n}", f"ffn_out-{n}", f"l_out-{n}",
    ]
    if is_last_block:
        ffn_patterns += ["node_490", "node_491", "norm", "result_norm", "result_output"]

    def fetch_for_patterns(patterns: list[str]) -> list[dict]:
        like_clauses = " OR ".join("ei.event_tensor_name LIKE ?" for _ in patterns)
        # Use the specific run_id for timing/structural data
        return _fetchall(f"""
            SELECT ei.event_operation_type,
                   COUNT(*)                        AS count,
                   SUM(ei.event_time_microseconds) AS total_time_us,
                   SUM(ei.event_size_bytes)        AS total_bytes,
                   SUM(ei.event_n_elements)        AS total_elements
            FROM event_item ei
            WHERE ei.run_id      = ?
              AND ei.event_phase = ?
              AND ({like_clauses})
            GROUP BY ei.event_operation_type
            ORDER BY total_time_us DESC
        """, (run_id, phase, *patterns), db_path=db_path)

    def papi_for_patterns_all_runs(patterns: list[str]) -> dict:
        """
        Aggregate PAPI values across ALL runs for these tensor patterns.
        Each run measured a different subset of events, so we take
        the SUM per event_name across all runs that have it.
        """
        like_clauses = " OR ".join("ei.event_tensor_name LIKE ?" for _ in patterns)
        rows = _fetchall(f"""
            SELECT epc.papi_event_name,
                   SUM(epc.papi_value) AS total
            FROM event_item ei
            JOIN event_papi_counter epc ON ei.event_item_id = epc.event_item_id
            WHERE ei.event_phase = ?
              AND ({like_clauses})
            GROUP BY epc.papi_event_name
        """, (phase, *patterns), db_path=db_path)
        # Note: no run_id filter — aggregates across all runs
        return {r["papi_event_name"]: r["total"] for r in rows if r["total"] is not None}

    attn_rows = fetch_for_patterns(attn_patterns)
    ffn_rows  = fetch_for_patterns(ffn_patterns)
    attn_papi = papi_for_patterns_all_runs(attn_patterns)
    ffn_papi  = papi_for_patterns_all_runs(ffn_patterns)

    def summarise(rows, papi) -> dict | None:
        if not rows:
            return None
        total_time_us = sum(r["total_time_us"] for r in rows)
        total_bytes   = sum(r["total_bytes"]   for r in rows)
        fp_ops        = papi.get("PAPI_FP_OPS", 0)
        l3_misses     = papi.get("PAPI_L3_TCM", 0)
        l3_access     = papi.get("PAPI_L3_TCA", 0)
        bytes_moved   = l3_misses * 64
        tot_ins       = papi.get("PAPI_TOT_INS", 0)
        tot_cyc       = papi.get("PAPI_TOT_CYC", 0)

        return {
            "runtime_us":           total_time_us,
            "runtime_ms":           round(total_time_us / 1e3, 3),
            "FLOPs":                fp_ops if "PAPI_FP_OPS" in papi else None,
            "bytes_moved":          bytes_moved or None,
            "arithmetic_intensity": (fp_ops / bytes_moved) if bytes_moved > 0 and fp_ops else None,
            "cache_behavior": {
                "L3_accesses":  l3_access,
                "L3_misses":    l3_misses,
                "L3_miss_rate": round(l3_misses / l3_access, 4) if l3_access > 0 else None,
            },
            "IPC":  round(tot_ins / tot_cyc, 4) if tot_cyc > 0 else None,
            #"op_breakdown": {
            #    r["event_operation_type"]: {
            #        "count":          r["count"],
            #        "total_time_us":  r["total_time_us"],
            #        "total_bytes":    r["total_bytes"],
            #        "total_elements": r["total_elements"],
            #    } for r in rows
            #},
            "papi": papi if papi else None,
        }

    return {
        "attention": summarise(attn_rows, attn_papi),
        "MLP":       summarise(ffn_rows,  ffn_papi),
    }

def complement_decoder_block_json(
    json_path:  str,
    db_path:    str = DB_PATH,
    run_id:     Optional[int] = None,
    out_path:   Optional[str] = None,
) -> list[dict]:

    with open(json_path) as f:
        blocks = json.load(f)

    # Resolve run_id
    if run_id is None:
        from db_queries import get_run_ids
        ids = get_run_ids(db_path)
        run_id = ids[-1] if ids else None

    # Total runtime across all blocks for share calculation
    total_runtime_ns = sum(b.get("runtime_ns", 0) for b in blocks)
    last_block_id = max(b["block_id"] for b in blocks)
    for block in blocks:
        _derive_block_metrics(block, total_runtime_ns)

        if run_id is not None:
            sub = _get_db_subcomponent_metrics(
                run_id     = run_id,
                block_type = block["block_type"],
                block_id   = block["block_id"],
                db_path    = db_path,
                is_last_block= block["block_id"] == last_block_id,
            )
            block["subcomponents"] = sub

    out = out_path or json_path.replace(".json", "_enriched.json")
    with open(out, "w") as f:
        json.dump(blocks, f, indent=2)

    print(f"Saved enriched decoder block JSON to {out}")
    return blocks