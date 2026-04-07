#!/usr/bin/env python3
"""
Roofline analysis for llama.cpp
- Problem 1: Hardware ceiling from lscpu + stream benchmark
- Problem 2: Operational Intensity from PAPI CSV files
"""

import subprocess
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import re

# ─────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────
STREAM_PATH = os.path.expanduser("~/stream")
CSV_DIR = os.path.join(os.path.dirname(__file__), "..", "run_all_results")
CACHE_LINE_BYTES = 64


# ─────────────────────────────────────────
# PROBLEM 1: HARDWARE CEILING
# ─────────────────────────────────────────

def get_cpu_info():
    """Extract CPU specs from lscpu."""
    result = subprocess.run(["lscpu"], capture_output=True, text=True)
    output = result.stdout

    # Physical cores (not hyperthreads)
    cores = None
    sockets = 1
    threads_per_core = 1
    total_cpus = None

    for line in output.splitlines():
        if re.match(r"^CPU\(s\):\s+\d+", line):
            total_cpus = int(line.split(":")[1].strip())
        elif "Thread(s) per core" in line:
            threads_per_core = int(line.split(":")[1].strip())
        elif "Socket(s)" in line:
            sockets = int(line.split(":")[1].strip())
        elif "Core(s) per socket" in line:
            cores_per_socket = int(line.split(":")[1].strip())
            cores = cores_per_socket * sockets

    if cores is None and total_cpus and threads_per_core:
        cores = total_cpus // threads_per_core

    # Frequencies
    min_mhz, max_mhz = None, None
    for line in output.splitlines():
        if "CPU min MHz" in line:
            min_mhz = float(line.split(":")[1].strip())
        elif "CPU max MHz" in line:
            max_mhz = float(line.split(":")[1].strip())

    # Fallback: parse GHz from model name
    if min_mhz is None or max_mhz is None:
        for line in output.splitlines():
            if "Model name" in line:
                match = re.search(r"(\d+\.\d+)\s*GHz", line)
                if match:
                    base_ghz = float(match.group(1))
                    min_mhz = base_ghz * 1000
                    max_mhz = base_ghz * 1000

    # AVX support → FLOPS per cycle
    avx_result = subprocess.run(["lscpu"], capture_output=True, text=True)
    flags = avx_result.stdout
    if "avx512" in flags:
        flops_per_cycle = 32  # AVX-512 FP32 with FMA
        isa = "AVX-512"
    elif "avx2" in flags:
        flops_per_cycle = 16  # AVX2 FP32 with FMA
        isa = "AVX2"
    else:
        flops_per_cycle = 8
        isa = "SSE"

    # Extract base frequency from model name (e.g. "@ 3.00GHz")
    base_ghz = None
    for line in output.splitlines():
        if "Model name" in line:
            match = re.search(r"@\s*(\d+\.\d+)\s*GHz", line)
            if match:
                base_ghz = float(match.group(1))
    if base_ghz is None:
        base_ghz = (min_mhz + max_mhz) / 2 / 1000  # fallback

    # Average of base (from model name) and max boost frequency
    avg_ghz = (base_ghz + max_mhz / 1000) / 2 if base_ghz else max_mhz / 1000

    return {
        "cores": cores,
        "min_ghz": min_mhz / 1000,
        "max_ghz": max_mhz / 1000,
        "base_ghz": base_ghz,
        "avg_ghz": avg_ghz,  # mean of base + boost
        "flops_per_cycle": flops_per_cycle,
        "isa": isa,
    }


def compute_peak_flops(cpu):
    """Peak FLOPS in GFLOPS using average frequency."""
    # ×2 for FMA (multiply-add counts as 2 FLOPS)
    peak = cpu["cores"] * cpu["avg_ghz"] * cpu["flops_per_cycle"] * 2
    print(f"\n── Problem 1: Hardware Ceiling ──────────────────")
    print(f"  CPU cores (physical): {cpu['cores']}")
    print(f"  Frequency min/max:    {cpu['min_ghz']:.2f} / {cpu['max_ghz']:.2f} GHz")
    print(f"  Frequency (bas+boost)/2: {cpu['avg_ghz']:.2f} GHz  (bas={cpu['base_ghz']:.2f}, boost={cpu['max_ghz']:.2f})")
    print(f"  ISA:                  {cpu['isa']} → {cpu['flops_per_cycle']} FP32/cycle")
    print(f"  Peak FLOPS:           {peak:.1f} GFLOPS  (with FMA ×2)")
    return peak  # GFLOPS


def get_memory_bandwidth():
    """Run stream benchmark and extract Triad bandwidth in GB/s."""
    if not os.path.exists(STREAM_PATH):
        raise FileNotFoundError(f"stream not found at {STREAM_PATH}")

    result = subprocess.run([STREAM_PATH], capture_output=True, text=True)
    for line in result.stdout.splitlines():
        if line.startswith("Triad"):
            bw = float(line.split()[1]) / 1000  # MB/s → GB/s
            print(f"  Memory bandwidth:     {bw:.1f} GB/s  (stream Triad)")
            return bw

    raise RuntimeError("Could not parse Triad value from stream output")


# ─────────────────────────────────────────
# PROBLEM 2: OPERATIONAL INTENSITY
# ─────────────────────────────────────────

def load_csv_group(csv_dir, group_num):
    """Load a specific events_group CSV."""
    pattern = os.path.join(csv_dir, f"events_group_{group_num}.csv")
    files = glob.glob(pattern)
    if not files:
        raise FileNotFoundError(f"No file matching {pattern}")
    return pd.read_csv(files[0])


def compute_operational_intensity(csv_dir):
    """
    FLOPs  : papi_fp_ops  from group_4
    Bytes  : papi_l3_tcm × 64  from group_2
    OI     : FLOPs / Bytes
    """
    print(f"\n── Problem 2: Operational Intensity ─────────────")

    # FLOPs
    g4 = load_csv_group(csv_dir, 4)
    total_flops = g4["papi_fp_ops"].sum()
    print(f"  Total FLOPs (papi_fp_ops):   {total_flops:,.0f}")

    # Memory bytes via L3 misses → DRAM traffic
    g2 = load_csv_group(csv_dir, 2)
    total_l3_misses = g2["papi_l3_tcm"].sum()
    total_bytes = total_l3_misses * CACHE_LINE_BYTES
    print(f"  Total L3 misses:             {total_l3_misses:,.0f}")
    print(f"  Total DRAM bytes (×64):      {total_bytes:,.0f}")

    oi = total_flops / total_bytes if total_bytes > 0 else 0
    print(f"  Operational Intensity (OI):  {oi:.4f} FLOP/byte")

    # Achieved performance
    g4_time = g4["time_ns"].sum()
    time_sec = g4_time / 1e9
    achieved_gflops = (total_flops / 1e9) / time_sec
    print(f"  Total time:                  {time_sec:.3f} s")
    print(f"  Achieved performance:        {achieved_gflops:.2f} GFLOPS")

    return oi, achieved_gflops


# ─────────────────────────────────────────
# ROOFLINE PLOT
# ─────────────────────────────────────────

def plot_roofline(peak_flops, mem_bw, oi, achieved_gflops, cpu_info):
    """Plot the roofline model with one program point."""

    ridge_point = peak_flops / mem_bw  # FLOP/byte where roofline bends

    # X axis range
    x_min = 1e-3
    x_max = max(ridge_point * 10, oi * 10)
    x = np.logspace(np.log10(x_min), np.log10(x_max), 500)

    # Roofline ceiling
    y_roof = np.minimum(mem_bw * x, peak_flops)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_facecolor("#0f1117")
    fig.patch.set_facecolor("#0f1117")

    # Roofline
    ax.loglog(x, y_roof, color="#00d4ff", linewidth=2.5, label="Roofline ceiling")

    # Ceiling lines (dashed)
    ax.axhline(peak_flops, color="#ff6b6b", linestyle="--", linewidth=1.2, alpha=0.7)
    ax.axvline(ridge_point, color="#ffd93d", linestyle=":", linewidth=1.2, alpha=0.6)

    # Program point
    bound = "Memory-bound" if oi < ridge_point else "Compute-bound"
    point_color = "#ff9f43" if oi < ridge_point else "#48dbfb"
    ax.scatter([oi], [achieved_gflops], color=point_color, s=150, zorder=5,
               label=f"llama.cpp ({bound})", edgecolors="white", linewidths=0.8)

    # Vertical line from point to roofline ceiling
    roof_at_oi = min(mem_bw * oi, peak_flops)
    ax.plot([oi, oi], [achieved_gflops, roof_at_oi],
            color=point_color, linestyle=":", alpha=0.5, linewidth=1)

    # Labels
    ax.text(oi * 1.15, achieved_gflops * 1.1,
            f"OI={oi:.3f}\n{achieved_gflops:.2f} GFLOPS",
            color="white", fontsize=9, va="bottom")

    ax.text(x_max * 0.6, peak_flops * 1.05,
            f"Peak: {peak_flops:.0f} GFLOPS ({cpu_info['isa']})",
            color="#ff6b6b", fontsize=9)

    ax.text(ridge_point * 1.05, x_min * mem_bw * 1.5,
            f"Ridge: {ridge_point:.2f} FLOP/B",
            color="#ffd93d", fontsize=9, rotation=90, va="bottom")

    # Memory bandwidth slope label
    ax.text(x_min * 3, mem_bw * x_min * 3 * 1.5,
            f"Mem BW: {mem_bw:.1f} GB/s",
            color="#00d4ff", fontsize=9, rotation=35)

    # Style
    ax.set_xlabel("Operational Intensity (FLOP/byte)", color="white", fontsize=12)
    ax.set_ylabel("Performance (GFLOPS)", color="white", fontsize=12)
    ax.set_title(f"Roofline Model — llama.cpp\n"
                 f"Intel i7-1185G7 · {cpu_info['cores']} cores · "
                 f"{cpu_info['avg_ghz']:.2f} GHz avg · {cpu_info['isa']}",
                 color="white", fontsize=13)

    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_edgecolor("#444")

    ax.grid(True, which="both", color="#333", linestyle="--", linewidth=0.5)
    ax.legend(facecolor="#1a1a2e", edgecolor="#444", labelcolor="white", fontsize=10)

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), "roofline_output.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    print(f"\n  Plot saved → {out_path}")
    return out_path


# ─────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────

def main():
    print("=" * 52)
    print("  Roofline Analysis — llama.cpp")
    print("=" * 52)

    # Problem 1
    cpu = get_cpu_info()
    peak_flops = compute_peak_flops(cpu)
    mem_bw = get_memory_bandwidth()

    # Problem 2
    oi, achieved_gflops = compute_operational_intensity(CSV_DIR)

    # Plot
    print("\n── Plotting ─────────────────────────────────────")
    plot_roofline(peak_flops, mem_bw, oi, achieved_gflops, cpu)

    print("\n── Summary ──────────────────────────────────────")
    print(f"  Peak FLOPS:     {peak_flops:.1f} GFLOPS")
    print(f"  Memory BW:      {mem_bw:.1f} GB/s")
    print(f"  Ridge point:    {peak_flops/mem_bw:.4f} FLOP/byte")
    print(f"  OI (llama.cpp): {oi:.4f} FLOP/byte")
    print(f"  Achieved:       {achieved_gflops:.2f} GFLOPS")
    bound = "MEMORY-BOUND" if oi < peak_flops / mem_bw else "COMPUTE-BOUND"
    print(f"  Verdict:        {bound}")
    print("=" * 52)


if __name__ == "__main__":
    main()