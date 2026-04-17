#!/usr/bin/env python3
"""
Export profiling_data.db to CSV.

Usage:
  python3 db_to_csv.py                        # exports all runs, all tables
  python3 db_to_csv.py --run 1                # only run_id = 1
  python3 db_to_csv.py --out results.csv      # custom output filename
  python3 db_to_csv.py --split                # one CSV file per run
  python3 db_to_csv.py --table events         # events only (no PAPI join)
  python3 db_to_csv.py --table papi           # raw papi counters only
"""

import sqlite3
import csv
import sys
import os
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LLAMA_ROOT = os.path.dirname(SCRIPT_DIR)
DEFAULT_DB  = os.path.join(LLAMA_ROOT, "profiling_data.db")
DEFAULT_OUT = "profiling_export.csv"


def get_runs(conn):
    cur = conn.cursor()
    cur.execute("SELECT DISTINCT run_id FROM event_item ORDER BY run_id")
    return [r[0] for r in cur.fetchall()]


def export_joined(conn, out_path, run_id=None):
    """
    Export event_item LEFT JOINed with event_papi_counter (pivoted per event).
    Each row = one tensor op, with all its PAPI counter values as columns.
    """
    cur = conn.cursor()

    # Get all distinct PAPI event names present in the data
    if run_id is not None:
        cur.execute("""
            SELECT DISTINCT epc.papi_event_name
            FROM event_papi_counter epc
            JOIN event_item ei ON epc.event_item_id = ei.event_item_id
            WHERE ei.run_id = ?
            ORDER BY epc.papi_event_name
        """, (run_id,))
    else:
        cur.execute("""
            SELECT DISTINCT papi_event_name
            FROM event_papi_counter
            ORDER BY papi_event_name
        """)
    papi_cols = [r[0] for r in cur.fetchall()]

    # Build pivot query
    pivot_parts = ", ".join(
        f"MAX(CASE WHEN epc.papi_event_name = '{col}' THEN epc.papi_value END) AS \"{col.lower()}\""
        for col in papi_cols
    )
    pivot_clause = (", " + pivot_parts) if pivot_parts else ""

    where = f"WHERE ei.run_id = {run_id}" if run_id is not None else ""

    query = f"""
        SELECT
            ei.run_id,
            ei.event_item_id,
            ei.event_item_timestamp,
            ei.event_phase,
            ei.event_token_index,
            ei.event_tensor_name,
            ei.event_operation_type,
            ei.event_time_microseconds,
            ei.event_size_bytes,
            ei.event_n_elements
            {pivot_clause}
        FROM event_item ei
        LEFT JOIN event_papi_counter epc ON ei.event_item_id = epc.event_item_id
        {where}
        GROUP BY ei.event_item_id
        ORDER BY ei.run_id, ei.event_item_id
    """

    cur.execute(query)
    rows = cur.fetchall()
    cols = [d[0] for d in cur.description]

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(cols)
        writer.writerows(rows)

    print(f"  Wrote {len(rows):,} rows  →  {out_path}")
    return len(rows)


def export_raw_events(conn, out_path, run_id=None):
    cur = conn.cursor()
    where = f"WHERE run_id = {run_id}" if run_id is not None else ""
    cur.execute(f"SELECT * FROM event_item {where} ORDER BY run_id, event_item_id")
    rows = cur.fetchall()
    cols = [d[0] for d in cur.description]
    with open(out_path, "w", newline="") as f:
        csv.writer(f).writerow(cols)
        csv.writer(f).writerows(rows)
    print(f"  Wrote {len(rows):,} rows  →  {out_path}")


def export_raw_papi(conn, out_path, run_id=None):
    cur = conn.cursor()
    if run_id is not None:
        cur.execute("""
            SELECT epc.*
            FROM event_papi_counter epc
            JOIN event_item ei ON epc.event_item_id = ei.event_item_id
            WHERE ei.run_id = ?
            ORDER BY epc.event_item_id
        """, (run_id,))
    else:
        cur.execute("SELECT * FROM event_papi_counter ORDER BY event_item_id")
    rows = cur.fetchall()
    cols = [d[0] for d in cur.description]
    with open(out_path, "w", newline="") as f:
        csv.writer(f).writerow(cols)
        csv.writer(f).writerows(rows)
    print(f"  Wrote {len(rows):,} rows  →  {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Export profiling_data.db to CSV")
    parser.add_argument("--db",     default=DEFAULT_DB,  help="Path to SQLite database")
    parser.add_argument("--out",    default=DEFAULT_OUT, help="Output CSV filename")
    parser.add_argument("--run",    type=int, default=None, help="Export only this run_id")
    parser.add_argument("--split",  action="store_true", help="One CSV per run_id")
    parser.add_argument("--table",  choices=["joined", "events", "papi"], default="joined",
                        help="joined=events+papi pivoted (default), events=raw events, papi=raw papi counters")
    args = parser.parse_args()

    if not os.path.exists(args.db):
        print(f"Database not found: {args.db}")
        sys.exit(1)

    conn = sqlite3.connect(args.db)
    runs = get_runs(conn)

    if not runs:
        print("No data found in database.")
        conn.close()
        sys.exit(0)

    print(f"Database: {args.db}")
    print(f"Runs found: {runs}")

    export_fn = {
        "joined": export_joined,
        "events": export_raw_events,
        "papi":   export_raw_papi,
    }[args.table]

    if args.split:
        for run in runs:
            base, ext = os.path.splitext(args.out)
            out = f"{base}_run{run}{ext or '.csv'}"
            print(f"\nExporting run {run}...")
            export_fn(conn, out, run_id=run)
    else:
        run = args.run
        if run is not None and run not in runs:
            print(f"run_id {run} not found. Available: {runs}")
            conn.close()
            sys.exit(1)
        print(f"\nExporting {'run ' + str(run) if run else 'all runs'}...")
        export_fn(conn, args.out, run_id=run)

    conn.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
