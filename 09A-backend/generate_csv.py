"""
generate_csv.py
Generates a summary CSV inside each model result folder, and a combined
CSV in the root results folder.

Called automatically at the end of run_every_view(), or manually:
  python3 generate_csv.py ~/profiling-llms-llama-cpp/run_every_view_results

Output per model folder:
  <model_folder>/summary.csv

Output in root:
  <results_root>/all_models_summary.csv

Columns:
  Model,
  Top Runtime (ns), Top L3_TCM,
  Phase Runtime (ns), Phase L3_TCM,
  Decoder Runtime (ns), Decoder L3_TCM,
  Tensor Runtime (ns), Tensor L3_TCM,
  Generated Tokens, Throughput (tok/s), Peak RSS (MB), Model Size (MB), Energy Pkg (J)
"""

import csv
import json
import os
import re
import sqlite3
import sys


# ── Readers ───────────────────────────────────────────────────────────────────

def read_top_view(folder):
    path = os.path.join(folder, "top-view.json")
    with open(path) as f:
        d = json.load(f)
    return {
        "top_runtime_ns":   d.get("runtime_ns", 0),
        "top_L3_TCM":       d.get("PAPI_L3_TCM", 0),
        "generated_tokens": d.get("generated_tokens", 0),
        "throughput":       round(d.get("token_throughput", 0), 2),
        "peak_rss_mb":      round(d.get("peak_rss_mb", 0), 2),
        "model_size_mb":    round(d.get("model_size_mb", 0), 2),
        "energy_pkg_j":     round(d.get("energy-pkg", 0) / 1e6, 2) if d.get("energy-pkg") else 0,
    }


def read_phase_view(folder):
    path = os.path.join(folder, "phase-view.json")
    with open(path) as f:
        d = json.load(f)
    total_runtime = 0
    total_l3      = 0
    for phase in d.values():
        if isinstance(phase, dict):
            total_runtime += phase.get("runtime_ns", 0)
            total_l3      += phase.get("PAPI_L3_TCM", 0)
    return {
        "phase_runtime_ns": total_runtime,
        "phase_L3_TCM":     total_l3,
    }


def read_decoder_block_view(folder):
    path = os.path.join(folder, "decoder-block-view.json")
    with open(path) as f:
        blocks = json.load(f)
    return {
        "decoder_runtime_ns": sum(b.get("runtime_ns", 0)  for b in blocks),
        "decoder_L3_TCM":     sum(b.get("PAPI_L3_TCM", 0) for b in blocks),
    }


def read_tensor_op_view(folder):
    """
    Reads from the SQLite DB produced by llama-papi (tensor_op_view.db).
    Falls back to tensor_op.txt if DB is not present.
    """
    db_path  = os.path.join(folder, "tensor_op_view.db")
    txt_path = os.path.join(folder, "tensor_op.txt")

    if os.path.exists(db_path):
        return _read_tensor_db(db_path)
    elif os.path.exists(txt_path):
        return _read_tensor_txt(txt_path)
    else:
        raise FileNotFoundError(f"No tensor op file found in {folder}")


def _read_tensor_db(db_path):
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Total runtime from event_item (sum of all event times)
    row = conn.execute("""
        SELECT
            COALESCE(SUM(event_time_microseconds), 0) AS total_time_us,
            COALESCE(SUM(p.papi_value), 0)            AS total_l3
        FROM event_item e
        LEFT JOIN event_papi_counter p
               ON e.event_item_id = p.event_item_id
              AND p.papi_event_name = 'PAPI_L3_TCM'
    """).fetchone()
    conn.close()

    return {
        "tensor_runtime_ns": int(row["total_time_us"] * 1000),  # µs -> ns
        "tensor_L3_TCM":     int(row["total_l3"]),
    }


def _read_tensor_txt(txt_path):
    with open(txt_path) as f:
        content = f.read()

    l3_match = re.search(r"Cache misses:\s*(\d+)", content)
    l3 = int(l3_match.group(1)) if l3_match else 0

    total_time_us = 0
    metrics_match = re.search(r"\[.*\]", content, re.DOTALL)
    if metrics_match:
        try:
            metrics = json.loads(metrics_match.group(0))
            total_time_us = sum(m.get("total_time_us", 0) for m in metrics)
        except Exception:
            pass

    return {
        "tensor_runtime_ns": int(total_time_us * 1000),
        "tensor_L3_TCM":     l3,
    }


# ── Per-folder summary ────────────────────────────────────────────────────────

HEADERS = [
    "Model",
    "Top Runtime (ns)",    "Top L3_TCM",
    "Phase Runtime (ns)",  "Phase L3_TCM",
    "Decoder Runtime (ns)","Decoder L3_TCM",
    "Tensor Runtime (ns)", "Tensor L3_TCM",
    "Generated Tokens",    "Throughput (tok/s)",
    "Peak RSS (MB)",       "Model Size (MB)",
    "Energy Pkg (J)",
]

KEYS = [
    "model",
    "top_runtime_ns",     "top_L3_TCM",
    "phase_runtime_ns",   "phase_L3_TCM",
    "decoder_runtime_ns", "decoder_L3_TCM",
    "tensor_runtime_ns",  "tensor_L3_TCM",
    "generated_tokens",   "throughput",
    "peak_rss_mb",        "model_size_mb",
    "energy_pkg_j",
]


def summarize_folder(folder):
    row = {"model": os.path.basename(folder)}

    for reader, label in [
        (read_top_view,           "top"),
        (read_phase_view,         "phase"),
        (read_decoder_block_view, "decoder"),
        (read_tensor_op_view,     "tensor"),
    ]:
        try:
            row.update(reader(folder))
        except FileNotFoundError:
            print(f"  [{label}] file not found in {folder}, skipping.")
        except Exception as e:
            print(f"  [{label}] error reading {folder}: {e}")

    return row


def write_model_csv(folder, row):
    """Write a single-row CSV inside the model folder."""
    out_path = os.path.join(folder, "summary.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=KEYS, extrasaction="ignore")
        writer.writerow(dict(zip(KEYS, HEADERS)))  # header row with friendly names
        writer.writerow(row)
    print(f"  ✓ Written: {out_path}")


def write_combined_csv(root, rows):
    """Write a combined CSV with all models in the results root."""
    out_path = os.path.join(root, "all_models_summary.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=KEYS, extrasaction="ignore")
        writer.writerow(dict(zip(KEYS, HEADERS)))
        for row in rows:
            writer.writerow(row)
    print(f"\n✓ Combined summary: {out_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

def generate_csvs(results_root):
    """
    Main function — call this from run_handler or CLI.
    Scans results_root for model subfolders and generates CSVs.
    """
    results_root = os.path.expanduser(results_root)

    # Find model subfolders
    folders = sorted([
        os.path.join(results_root, d)
        for d in os.listdir(results_root)
        if os.path.isdir(os.path.join(results_root, d))
        and any(f in os.listdir(os.path.join(results_root, d))
                for f in ["top-view.json", "phase-view.json",
                           "decoder-block-view.json", "tensor_op_view.db"])
    ])

    if not folders:
        # Maybe root itself is a single model folder
        folders = [results_root]

    all_rows = []
    for folder in folders:
        print(f"\nProcessing: {os.path.basename(folder)}")
        row = summarize_folder(folder)
        write_model_csv(folder, row)
        all_rows.append(row)

    if len(all_rows) > 1:
        write_combined_csv(results_root, all_rows)

    return all_rows


if __name__ == "__main__":
    root = sys.argv[1] if len(sys.argv) > 1 else "."
    generate_csvs(root)