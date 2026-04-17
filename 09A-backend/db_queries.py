"""
db_queries.py — SQLite query helpers for profiling_data.db

Schema:
  event_item(
    event_item_id, run_id, event_item_timestamp,
    event_phase TEXT,          -- 'prefill' | 'decode'
    event_token_index INTEGER,
    event_tensor_name TEXT,
    event_operation_type TEXT,
    event_time_microseconds INTEGER,
    event_size_bytes INTEGER,
    event_n_elements INTEGER
  )

  event_papi_counter(
    event_item_id, papi_event_name TEXT, papi_value INTEGER
  )
"""

import sqlite3
import os
from contextlib import contextmanager
from typing import Optional

DB_PATH = os.path.join(
    os.path.expanduser("~"),
    "profiling-llms-llama-cpp",
    "profiling_data.db",
)


# ── Connection helper ──────────────────────────────────────────────────────────

@contextmanager
def _conn(db_path: str = DB_PATH):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def _fetchall(sql: str, params: tuple = (), db_path: str = DB_PATH) -> list[dict]:
    with _conn(db_path) as conn:
        cur = conn.execute(sql, params)
        return [dict(row) for row in cur.fetchall()]


def _fetchone(sql: str, params: tuple = (), db_path: str = DB_PATH) -> Optional[dict]:
    with _conn(db_path) as conn:
        cur = conn.execute(sql, params)
        row = cur.fetchone()
        return dict(row) if row else None


# ── Run-level queries ──────────────────────────────────────────────────────────

def get_run_ids(db_path: str = DB_PATH) -> list[int]:
    """Return all distinct run_ids present in the database."""
    rows = _fetchall("SELECT DISTINCT run_id FROM event_item ORDER BY run_id", db_path=db_path)
    return [r["run_id"] for r in rows]


def get_run_summary(db_path: str = DB_PATH) -> list[dict]:
    """
    Per-run summary: phase breakdown, token range, total time (µs), event count.
    Returns list of dicts keyed by run_id + event_phase.
    """
    return _fetchall("""
        SELECT
            run_id,
            event_phase,
            COUNT(*)                          AS event_count,
            MIN(event_token_index)            AS token_min,
            MAX(event_token_index)            AS token_max,
            SUM(event_time_microseconds)      AS total_time_us,
            AVG(event_time_microseconds)      AS avg_time_us
        FROM event_item
        GROUP BY run_id, event_phase
        ORDER BY run_id, event_phase
    """, db_path=db_path)


# ── Phase-level queries ────────────────────────────────────────────────────────

def get_events_by_phase(
    phase: str,
    run_id: Optional[int] = None,
    db_path: str = DB_PATH,
) -> list[dict]:
    """
    Return all event_item rows for the given phase ('prefill' or 'decode').
    Optionally filter to a single run_id.
    """
    if run_id is not None:
        return _fetchall(
            "SELECT * FROM event_item WHERE event_phase = ? AND run_id = ? ORDER BY event_item_id",
            (phase, run_id), db_path=db_path,
        )
    return _fetchall(
        "SELECT * FROM event_item WHERE event_phase = ? ORDER BY event_item_id",
        (phase,), db_path=db_path,
    )


def get_prefill_events(run_id: Optional[int] = None, db_path: str = DB_PATH) -> list[dict]:
    """All prefill events, optionally filtered by run_id."""
    return get_events_by_phase("prefill", run_id=run_id, db_path=db_path)


def get_decode_events(run_id: Optional[int] = None, db_path: str = DB_PATH) -> list[dict]:
    """All decode events, optionally filtered by run_id."""
    return get_events_by_phase("decode", run_id=run_id, db_path=db_path)


# ── Operation-type queries ─────────────────────────────────────────────────────

def get_distinct_operation_types(db_path: str = DB_PATH) -> list[str]:
    """Return all distinct operation types recorded."""
    rows = _fetchall(
        "SELECT DISTINCT event_operation_type FROM event_item ORDER BY event_operation_type",
        db_path=db_path,
    )
    return [r["event_operation_type"] for r in rows]


def get_events_by_operation(
    operation_type: str,
    run_id: Optional[int] = None,
    phase: Optional[str] = None,
    db_path: str = DB_PATH,
) -> list[dict]:
    """Return all events for a given operation type, with optional run_id / phase filters."""
    sql = "SELECT * FROM event_item WHERE event_operation_type = ?"
    params: list = [operation_type]
    if run_id is not None:
        sql += " AND run_id = ?"
        params.append(run_id)
    if phase is not None:
        sql += " AND event_phase = ?"
        params.append(phase)
    sql += " ORDER BY event_item_id"
    return _fetchall(sql, tuple(params), db_path=db_path)


def get_time_breakdown_by_operation(
    run_id: Optional[int] = None,
    phase: Optional[str] = None,
    db_path: str = DB_PATH,
) -> list[dict]:
    """
    Total / average time (µs) and event count grouped by operation type.
    Results are sorted by total_time_us descending.
    """
    sql = "SELECT event_operation_type, COUNT(*) AS count, SUM(event_time_microseconds) AS total_time_us, AVG(event_time_microseconds) AS avg_time_us FROM event_item WHERE 1=1"
    params: list = []
    if run_id is not None:
        sql += " AND run_id = ?"
        params.append(run_id)
    if phase is not None:
        sql += " AND event_phase = ?"
        params.append(phase)
    sql += " GROUP BY event_operation_type ORDER BY total_time_us DESC"
    return _fetchall(sql, tuple(params), db_path=db_path)


# ── Tensor / layer queries ─────────────────────────────────────────────────────

def get_distinct_tensor_names(db_path: str = DB_PATH) -> list[str]:
    """Return all distinct tensor names in the database."""
    rows = _fetchall(
        "SELECT DISTINCT event_tensor_name FROM event_item ORDER BY event_tensor_name",
        db_path=db_path,
    )
    return [r["event_tensor_name"] for r in rows]


def get_events_by_tensor(
    tensor_name: str,
    run_id: Optional[int] = None,
    phase: Optional[str] = None,
    db_path: str = DB_PATH,
) -> list[dict]:
    """All events for a specific tensor name."""
    sql = "SELECT * FROM event_item WHERE event_tensor_name = ?"
    params: list = [tensor_name]
    if run_id is not None:
        sql += " AND run_id = ?"
        params.append(run_id)
    if phase is not None:
        sql += " AND event_phase = ?"
        params.append(phase)
    sql += " ORDER BY event_item_id"
    return _fetchall(sql, tuple(params), db_path=db_path)


def get_time_breakdown_by_tensor(
    run_id: Optional[int] = None,
    phase: Optional[str] = None,
    db_path: str = DB_PATH,
) -> list[dict]:
    """Total / avg time and count grouped by tensor name, sorted by total time desc."""
    sql = "SELECT event_tensor_name, COUNT(*) AS count, SUM(event_time_microseconds) AS total_time_us, AVG(event_time_microseconds) AS avg_time_us FROM event_item WHERE 1=1"
    params: list = []
    if run_id is not None:
        sql += " AND run_id = ?"
        params.append(run_id)
    if phase is not None:
        sql += " AND event_phase = ?"
        params.append(phase)
    sql += " GROUP BY event_tensor_name ORDER BY total_time_us DESC"
    return _fetchall(sql, tuple(params), db_path=db_path)


# ── Token-timeline queries ─────────────────────────────────────────────────────

def get_time_per_token(
    run_id: int,
    phase: Optional[str] = None,
    db_path: str = DB_PATH,
) -> list[dict]:
    """
    Total time (µs) spent per token index for a run.
    Useful for plotting latency over the decode sequence.
    """
    sql = "SELECT event_token_index, SUM(event_time_microseconds) AS total_time_us FROM event_item WHERE run_id = ?"
    params: list = [run_id]
    if phase is not None:
        sql += " AND event_phase = ?"
        params.append(phase)
    sql += " GROUP BY event_token_index ORDER BY event_token_index"
    return _fetchall(sql, tuple(params), db_path=db_path)


def get_operation_time_per_token(
    run_id: int,
    operation_type: str,
    phase: Optional[str] = None,
    db_path: str = DB_PATH,
) -> list[dict]:
    """Total time (µs) for a specific operation type per token index."""
    sql = """
        SELECT event_token_index,
               SUM(event_time_microseconds) AS total_time_us
        FROM event_item
        WHERE run_id = ? AND event_operation_type = ?
    """
    params: list = [run_id, operation_type]
    if phase is not None:
        sql += " AND event_phase = ?"
        params.append(phase)
    sql += " GROUP BY event_token_index ORDER BY event_token_index"
    return _fetchall(sql, tuple(params), db_path=db_path)


# ── Phase comparison ───────────────────────────────────────────────────────────

def compare_phases(run_id: int, db_path: str = DB_PATH) -> dict:
    """
    Side-by-side totals for prefill vs decode for a single run.
    Returns {'prefill': {...}, 'decode': {...}}.
    """
    rows = _fetchall("""
        SELECT event_phase,
               COUNT(*)                     AS event_count,
               SUM(event_time_microseconds) AS total_time_us,
               AVG(event_time_microseconds) AS avg_time_us,
               MIN(event_time_microseconds) AS min_time_us,
               MAX(event_time_microseconds) AS max_time_us
        FROM event_item
        WHERE run_id = ?
        GROUP BY event_phase
    """, (run_id,), db_path=db_path)
    return {r["event_phase"]: r for r in rows}


def compare_phases_by_operation(run_id: int, db_path: str = DB_PATH) -> list[dict]:
    """
    Time breakdown per (phase, operation_type) for a run.
    Useful for comparing how much each op contributes in prefill vs decode.
    """
    return _fetchall("""
        SELECT event_phase,
               event_operation_type,
               COUNT(*)                     AS count,
               SUM(event_time_microseconds) AS total_time_us,
               AVG(event_time_microseconds) AS avg_time_us
        FROM event_item
        WHERE run_id = ?
        GROUP BY event_phase, event_operation_type
        ORDER BY event_phase, total_time_us DESC
    """, (run_id,), db_path=db_path)


# ── Cross-run comparison ───────────────────────────────────────────────────────

def compare_runs_total_time(
    phase: Optional[str] = None,
    db_path: str = DB_PATH,
) -> list[dict]:
    """
    Total time (µs) per run_id (optionally for one phase).
    Useful for comparing runs against each other.
    """
    sql = "SELECT run_id, event_phase, SUM(event_time_microseconds) AS total_time_us, COUNT(*) AS event_count FROM event_item WHERE 1=1"
    params: list = []
    if phase is not None:
        sql += " AND event_phase = ?"
        params.append(phase)
    sql += " GROUP BY run_id, event_phase ORDER BY run_id, event_phase"
    return _fetchall(sql, tuple(params), db_path=db_path)


def compare_runs_by_operation(
    operation_type: str,
    phase: Optional[str] = None,
    db_path: str = DB_PATH,
) -> list[dict]:
    """Total time per run for a specific operation type."""
    sql = "SELECT run_id, event_phase, SUM(event_time_microseconds) AS total_time_us, COUNT(*) AS count FROM event_item WHERE event_operation_type = ?"
    params: list = [operation_type]
    if phase is not None:
        sql += " AND event_phase = ?"
        params.append(phase)
    sql += " GROUP BY run_id, event_phase ORDER BY run_id, event_phase"
    return _fetchall(sql, tuple(params), db_path=db_path)


# ── PAPI counter queries ───────────────────────────────────────────────────────

def get_distinct_papi_events(db_path: str = DB_PATH) -> list[str]:
    """Return all distinct PAPI event names recorded."""
    rows = _fetchall(
        "SELECT DISTINCT papi_event_name FROM event_papi_counter ORDER BY papi_event_name",
        db_path=db_path,
    )
    return [r["papi_event_name"] for r in rows]


def get_papi_counters_for_event(event_item_id: int, db_path: str = DB_PATH) -> list[dict]:
    """Return all PAPI counter rows for a specific event_item_id."""
    return _fetchall(
        "SELECT * FROM event_papi_counter WHERE event_item_id = ? ORDER BY papi_event_name",
        (event_item_id,), db_path=db_path,
    )


def get_events_with_papi(
    run_id: Optional[int] = None,
    phase: Optional[str] = None,
    papi_event: Optional[str] = None,
    db_path: str = DB_PATH,
) -> list[dict]:
    """
    Join event_item with event_papi_counter.
    Returns rows: event_item fields + papi_event_name + papi_value.
    Optionally filter by run_id, phase, and/or a specific PAPI event name.
    """
    sql = """
        SELECT e.*, p.papi_event_name, p.papi_value
        FROM event_item e
        JOIN event_papi_counter p ON e.event_item_id = p.event_item_id
        WHERE 1=1
    """
    params: list = []
    if run_id is not None:
        sql += " AND e.run_id = ?"
        params.append(run_id)
    if phase is not None:
        sql += " AND e.event_phase = ?"
        params.append(phase)
    if papi_event is not None:
        sql += " AND p.papi_event_name = ?"
        params.append(papi_event)
    sql += " ORDER BY e.event_item_id, p.papi_event_name"
    return _fetchall(sql, tuple(params), db_path=db_path)


def get_papi_totals_by_operation(
    papi_event: str,
    run_id: Optional[int] = None,
    phase: Optional[str] = None,
    db_path: str = DB_PATH,
) -> list[dict]:
    """
    Sum of a PAPI counter grouped by operation type.
    Useful for finding which ops cause the most cache misses, etc.
    """
    sql = """
        SELECT e.event_operation_type,
               SUM(p.papi_value) AS total_papi_value,
               AVG(p.papi_value) AS avg_papi_value,
               COUNT(*)          AS count
        FROM event_item e
        JOIN event_papi_counter p ON e.event_item_id = p.event_item_id
        WHERE p.papi_event_name = ?
    """
    params: list = [papi_event]
    if run_id is not None:
        sql += " AND e.run_id = ?"
        params.append(run_id)
    if phase is not None:
        sql += " AND e.event_phase = ?"
        params.append(phase)
    sql += " GROUP BY e.event_operation_type ORDER BY total_papi_value DESC"
    return _fetchall(sql, tuple(params), db_path=db_path)


def get_papi_totals_by_tensor(
    papi_event: str,
    run_id: Optional[int] = None,
    phase: Optional[str] = None,
    db_path: str = DB_PATH,
) -> list[dict]:
    """Sum of a PAPI counter grouped by tensor name."""
    sql = """
        SELECT e.event_tensor_name,
               SUM(p.papi_value) AS total_papi_value,
               AVG(p.papi_value) AS avg_papi_value,
               COUNT(*)          AS count
        FROM event_item e
        JOIN event_papi_counter p ON e.event_item_id = p.event_item_id
        WHERE p.papi_event_name = ?
    """
    params: list = [papi_event]
    if run_id is not None:
        sql += " AND e.run_id = ?"
        params.append(run_id)
    if phase is not None:
        sql += " AND e.event_phase = ?"
        params.append(phase)
    sql += " GROUP BY e.event_tensor_name ORDER BY total_papi_value DESC"
    return _fetchall(sql, tuple(params), db_path=db_path)


def get_cache_miss_summary(
    run_id: Optional[int] = None,
    phase: Optional[str] = None,
    db_path: str = DB_PATH,
) -> list[dict]:
    """
    Total L1 / L2 data-cache misses per operation type in one query.
    Counter names: PAPI_L1_DCM, PAPI_L2_DCM.
    """
    sql = """
        SELECT e.event_operation_type,
               SUM(CASE WHEN p.papi_event_name = 'PAPI_L1_DCM' THEN p.papi_value ELSE 0 END) AS l1_dcm_total,
               SUM(CASE WHEN p.papi_event_name = 'PAPI_L2_DCM' THEN p.papi_value ELSE 0 END) AS l2_dcm_total,
               COUNT(DISTINCT e.event_item_id) AS event_count
        FROM event_item e
        JOIN event_papi_counter p ON e.event_item_id = p.event_item_id
        WHERE 1=1
    """
    params: list = []
    if run_id is not None:
        sql += " AND e.run_id = ?"
        params.append(run_id)
    if phase is not None:
        sql += " AND e.event_phase = ?"
        params.append(phase)
    sql += " GROUP BY e.event_operation_type ORDER BY l1_dcm_total DESC"
    return _fetchall(sql, tuple(params), db_path=db_path)


def get_papi_per_token(
    papi_event: str,
    run_id: int,
    phase: Optional[str] = None,
    db_path: str = DB_PATH,
) -> list[dict]:
    """Total of a PAPI counter per token index for a run — mirrors get_time_per_token."""
    sql = """
        SELECT e.event_token_index,
               SUM(p.papi_value) AS total_papi_value
        FROM event_item e
        JOIN event_papi_counter p ON e.event_item_id = p.event_item_id
        WHERE p.papi_event_name = ? AND e.run_id = ?
    """
    params: list = [papi_event, run_id]
    if phase is not None:
        sql += " AND e.event_phase = ?"
        params.append(phase)
    sql += " GROUP BY e.event_token_index ORDER BY e.event_token_index"
    return _fetchall(sql, tuple(params), db_path=db_path)


# ── Arithmetic intensity helpers ───────────────────────────────────────────────

def get_arithmetic_intensity_per_operation(
    run_id: Optional[int] = None,
    phase: Optional[str] = None,
    db_path: str = DB_PATH,
) -> list[dict]:
    """
    Estimate arithmetic intensity per operation type as:
        n_elements / size_bytes   (proxy for FLOP/Byte without true FLOP count)

    Also returns total time so you can compute throughput yourself.
    """
    sql = """
        SELECT event_operation_type,
               COUNT(*)                                              AS count,
               SUM(event_n_elements)                                AS total_elements,
               SUM(event_size_bytes)                                AS total_bytes,
               CAST(SUM(event_n_elements) AS REAL) /
                   NULLIF(SUM(event_size_bytes), 0)                 AS intensity_ratio,
               SUM(event_time_microseconds)                         AS total_time_us
        FROM event_item
        WHERE 1=1
    """
    params: list = []
    if run_id is not None:
        sql += " AND run_id = ?"
        params.append(run_id)
    if phase is not None:
        sql += " AND event_phase = ?"
        params.append(phase)
    sql += " GROUP BY event_operation_type ORDER BY intensity_ratio DESC"
    return _fetchall(sql, tuple(params), db_path=db_path)


# ── Raw access helpers ─────────────────────────────────────────────────────────

def get_all_events(
    run_id: Optional[int] = None,
    phase: Optional[str] = None,
    limit: Optional[int] = None,
    db_path: str = DB_PATH,
) -> list[dict]:
    """Return raw event_item rows with optional filters and row limit."""
    sql = "SELECT * FROM event_item WHERE 1=1"
    params: list = []
    if run_id is not None:
        sql += " AND run_id = ?"
        params.append(run_id)
    if phase is not None:
        sql += " AND event_phase = ?"
        params.append(phase)
    sql += " ORDER BY event_item_id"
    if limit is not None:
        sql += f" LIMIT {int(limit)}"
    return _fetchall(sql, tuple(params), db_path=db_path)


def query(sql: str, params: tuple = (), db_path: str = DB_PATH) -> list[dict]:
    """Escape hatch: run any SELECT and get back a list of dicts."""
    return _fetchall(sql, params, db_path=db_path)
