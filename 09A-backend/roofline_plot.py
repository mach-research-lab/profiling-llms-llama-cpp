#!/usr/bin/env python3
"""
roofline_plot.py — Generates Roofline plots from Roofline.py data.

Same menu structure as Roofline.py.
Saves plots to ./roofline_results/ (cleared on each run).

Usage:
    python3 roofline_plot.py
"""

import os
import sys
import re
import json
import glob
import shutil
import sqlite3
import subprocess
from contextlib import contextmanager
from functools import lru_cache
from typing import Optional

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# ── Import Roofline.py logic ───────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from Roofline import (
    get_hardware,
    find_decoder_block_json,
    load_json,
    get_blocks,
    get_block,
    sum_blocks,
    roofline_from_aggregated,
    roofline_from_db_layer,
    get_layer_numbers,
    get_prefill_turn_indices,
    get_decode_token_indices,
    DB_PATH,
    db_available,
)

# ── Output directory ───────────────────────────────────────────────────────────
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "../roofline_results")

# ── ANSI colours ──────────────────────────────────────────────────────────────
BOLD   = "\033[1m"
CYAN   = "\033[96m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
RESET  = "\033[0m"

# ── Plot style ─────────────────────────────────────────────────────────────────
COLORS = {
    "roof"      : "#2C3E50",
    "memory"    : "#E74C3C",
    "compute"   : "#2ECC71",
    "ridge"     : "#95A5A6",
    "point"     : "#3498DB",
    "point_edge": "#1A252F",
    "bg"        : "#FAFAFA",
    "grid"      : "#ECF0F1",
    "text"      : "#2C3E50",
    "annotation": "#7F8C8D",
}


# ══════════════════════════════════════════════════════════════════════════════
#  Plot function
# ══════════════════════════════════════════════════════════════════════════════

def plot_roofline(result: dict, filename: str):
    """Generate and save a Roofline plot for a single result dict."""
    hw          = result["hardware"]
    peak        = hw["peak_gflops"]
    mem_bw      = hw["mem_bw_gbs"]
    ridge       = hw["ridge_point"]
    oi          = result["oi"]
    achieved    = result["achieved_gflops"]
    label       = result["label"]
    bound       = result["bound"]

    # ── Figure setup ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor(COLORS["bg"])
    ax.set_facecolor(COLORS["bg"])

    # ── OI range for x-axis ───────────────────────────────────────────────────
    x_min = 1e-2
    x_max = max(ridge * 4, oi * 2, 10)
    x     = np.logspace(np.log10(x_min), np.log10(x_max), 500)

    # ── Roofline ceiling ──────────────────────────────────────────────────────
    y_roof = np.minimum(peak, mem_bw * x)

    # Memory slope region (below ridge)
    x_mem = x[x <= ridge]
    y_mem = mem_bw * x_mem
    ax.fill_between(x_mem, 0, y_mem, alpha=0.06, color=COLORS["memory"], zorder=0)
    ax.plot(x_mem, y_mem, color=COLORS["memory"], lw=2.5, label=f"Memory roof  ({mem_bw:.1f} GB/s)", zorder=3)

    # Compute plateau region (above ridge)
    x_cmp = x[x >= ridge]
    y_cmp = np.full_like(x_cmp, peak)
    ax.fill_between(x_cmp, 0, y_cmp, alpha=0.06, color=COLORS["compute"], zorder=0)
    ax.plot(x_cmp, y_cmp, color=COLORS["compute"], lw=2.5, label=f"Compute roof  ({peak:.0f} GFLOPS)", zorder=3)

    # Roofline line (combined)
    ax.plot(x, y_roof, color=COLORS["roof"], lw=1.5, ls="--", alpha=0.4, zorder=2)

    # ── Ridge point ───────────────────────────────────────────────────────────
    ax.axvline(x=ridge, color=COLORS["ridge"], lw=1.2, ls=":", zorder=2)
    ax.text(ridge * 1.05, peak * 0.08,
            f"Ridge\n{ridge:.1f} FLOP/B",
            color=COLORS["ridge"], fontsize=8.5, va="bottom", style="italic")

    # ── Application point ─────────────────────────────────────────────────────
    point_color = COLORS["compute"] if bound == "compute" else COLORS["memory"]
    ax.scatter([oi], [achieved],
               s=160, color=point_color, edgecolors=COLORS["point_edge"],
               linewidths=1.5, zorder=6, label=f"{label}  (OI={oi:.2f}, {achieved:.1f} GFLOPS)")

    # Vertical dashed drop line
    ax.plot([oi, oi], [0, achieved],
            color=point_color, lw=1, ls="--", alpha=0.5, zorder=4)

    # Horizontal dashed line to y-axis
    ax.plot([x_min, oi], [achieved, achieved],
            color=point_color, lw=1, ls="--", alpha=0.5, zorder=4)

    # Annotation box
    gap   = achieved / peak
    va    = "bottom" if gap < 0.8 else "top"
    y_ann = achieved * 1.12 if gap < 0.8 else achieved * 0.88
    bound_label = "COMPUTE-BOUND" if bound == "compute" else "MEMORY-BOUND"
    ax.annotate(
        f"  {bound_label}\n  OI = {oi:.2f} FLOP/B\n  {achieved:.1f} GFLOPS",
        xy=(oi, achieved),
        xytext=(oi * 1.3, y_ann),
        fontsize=8.5,
        color=COLORS["text"],
        va=va,
        bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=point_color, lw=1.2, alpha=0.9),
        arrowprops=dict(arrowstyle="->", color=point_color, lw=1.2),
        zorder=7,
    )

    # ── Axes ──────────────────────────────────────────────────────────────────
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(1e-2, peak * 3)

    ax.set_xlabel("Operational Intensity  (FLOP / Byte)", fontsize=11, color=COLORS["text"], labelpad=8)
    ax.set_ylabel("Achieved Performance  (GFLOPS)", fontsize=11, color=COLORS["text"], labelpad=8)
    ax.set_title(f"Roofline Model — {label}", fontsize=13, fontweight="bold", color=COLORS["text"], pad=14)

    ax.grid(True, which="both", color=COLORS["grid"], lw=0.8, zorder=0)
    ax.tick_params(colors=COLORS["text"], labelsize=9)
    for spine in ax.spines.values():
        spine.set_edgecolor(COLORS["grid"])

    # ── Hardware legend ───────────────────────────────────────────────────────
    hw_text = (f"CPU: {hw['cpu_model']}\n"
               f"Cores: {hw['cores']}  |  ISA: {hw['isa']}  |  "
               f"Avg freq: {hw['avg_ghz']} GHz\n"
               f"Peak: {peak:.0f} GFLOPS  |  BW: {mem_bw:.1f} GB/s  |  Ridge: {ridge:.1f} FLOP/B")
    ax.text(0.01, 0.99, hw_text,
            transform=ax.transAxes, fontsize=7.5,
            va="top", ha="left", color=COLORS["annotation"],
            bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.7, ec=COLORS["grid"]))

    ax.legend(loc="lower right", fontsize=9, framealpha=0.9,
              edgecolor=COLORS["grid"], facecolor="white")

    plt.tight_layout()

    # ── Save ──────────────────────────────────────────────────────────────────
    out_path = os.path.join(OUTPUT_DIR, filename)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=COLORS["bg"])
    plt.close(fig)
    print(f"  {GREEN}✓{RESET} Saved: {out_path}")
    return out_path


def safe_filename(label: str) -> str:
    """Convert a label to a safe filename."""
    name = label.lower().strip()
    name = re.sub(r"[^\w\s-]", "", name)
    name = re.sub(r"[\s]+", "_", name)
    return name + ".png"


# ══════════════════════════════════════════════════════════════════════════════
#  Sub-menus  (mirrors Roofline.py exactly)
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
            plot_roofline(result, safe_filename("Entire Prefill"))

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
            label  = f"Prefill Turn {bid}"
            result = roofline_from_aggregated(agg, label)
            plot_roofline(result, safe_filename(label))

        elif choice == "3":
            if not db_available(db_path):
                print(f"  {RED}[ERROR]{RESET} Database not found.")
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
            label  = f"Prefill Turn {turn} Layer {layer}"
            result = roofline_from_db_layer(
                "prefill", layer, token_index=turn, label=label, db_path=db_path
            )
            plot_roofline(result, safe_filename(label))

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
            plot_roofline(result, safe_filename("Entire Decode"))

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
            label  = f"Decode Block {bid}"
            result = roofline_from_aggregated(agg, label)
            plot_roofline(result, safe_filename(label))

        elif choice == "3":
            if not db_available(db_path):
                print(f"  {RED}[ERROR]{RESET} Database not found.")
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
            turn  = token_index // 1000
            pos   = token_index % 1000
            label = f"Decode Turn {turn} Pos {pos} Layer {layer}"
            result = roofline_from_db_layer(
                "decode", layer, token_index=token_index, label=label, db_path=db_path
            )
            plot_roofline(result, safe_filename(label))

        else:
            print("  Unknown choice.")


# ══════════════════════════════════════════════════════════════════════════════
#  Main menu
# ══════════════════════════════════════════════════════════════════════════════

def menu_loop(data: list, json_path: str, db_path: str):
    while True:
        print()
        print("=" * 44)
        print(f"  {BOLD}ROOFLINE PLOT MENU{RESET}")
        print(f"  JSON : {os.path.basename(json_path)}")
        db_status = f"{GREEN}available{RESET}" if db_available(db_path) else f"{RED}not found{RESET}"
        print(f"  DB   : {db_status}")
        print(f"  OUT  : {OUTPUT_DIR}")
        print("=" * 44)
        print("  1. Entire program")
        print("  2. Prefill")
        print("  3. Decode")
        print("  0. Exit")
        print("=" * 44)

        choice = input("Make your choice (0-3): ").strip()

        if choice == "0":
            print("Bye!")
            break
        elif choice == "1":
            agg    = sum_blocks(data)
            result = roofline_from_aggregated(agg, "Entire Program")
            plot_roofline(result, safe_filename("Entire Program"))
        elif choice == "2":
            menu_prefill(data, db_path)
        elif choice == "3":
            menu_decode(data, db_path)
        else:
            print("  Unknown choice, try again.")


# ══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════════════

def main():
    # Clear and recreate output directory
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)
    print(f"{CYAN}[INFO]{RESET} Output directory cleared: {OUTPUT_DIR}")

    json_path = find_decoder_block_json()
    if not json_path:
        print(f"{RED}[ERROR]{RESET} decoder-block-view.json not found.")
        sys.exit(1)

    print(f"{CYAN}[INFO]{RESET} Detecting hardware ceiling...")
    get_hardware()

    print(f"{CYAN}[INFO]{RESET} Loading {json_path}")
    data = load_json(json_path)

    if not db_available(DB_PATH):
        print(f"{YELLOW}[WARN]{RESET} Database not found — layer analysis unavailable.")

    menu_loop(data, json_path, DB_PATH)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting.")