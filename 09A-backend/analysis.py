"""
roofline.py — three styles: "academic", "loglog", "advisor"
"""

import os, math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Hardware presets ───────────────────────────────────────────────────────────
HARDWARE = {
    "i7-1185G7": {
        "label": "Intel Core i7-1185G7 (Tiger Lake)",
        "compute_roofs": [
            {"name": "Scalar Peak (FP64)",      "val": 24.0},
            {"name": "AVX2 Vector Peak (FP64)", "val": 96.0},
            {"name": "AVX2 FMA Peak (FP64)",    "val": 192.0},
        ],
        "mem_roofs": [
            {"name": "L1 BW",   "val": 800.0},
            {"name": "L2 BW",   "val": 400.0},
            {"name": "L3 BW",   "val": 100.0},
            {"name": "DRAM BW", "val": 51.2},
        ],
        "peak_flops":    192e9,
        "mem_bandwidth":  51.2e9,
    },
    "A100": {
        "label": "NVIDIA A100 80 GB SXM",
        "compute_roofs": [
            {"name": "FP64 Peak",        "val": 9700.0},
            {"name": "FP32 Peak",        "val": 19500.0},
            {"name": "FP16 Tensor Peak", "val": 312000.0},
        ],
        "mem_roofs": [
            {"name": "L2 BW",    "val": 12000.0},
            {"name": "HBM2e BW", "val": 2000.0},
        ],
        "peak_flops":    9.7e12,
        "mem_bandwidth": 2000e9,
    },
}

# ── Core calculation ───────────────────────────────────────────────────────────
def calculate_roofline(peak_flops, mem_bandwidth, num_points=600):
    """
    P(I) = min(peak_flops, mem_bandwidth * I)

    Returns intensity array, performance array, and ridge_point.
    """
    ridge_point = peak_flops / mem_bandwidth
    intensity   = np.logspace(np.log10(ridge_point / 1000),
                              np.log10(ridge_point * 100), num_points)
    performance = np.minimum(peak_flops, mem_bandwidth * intensity)
    return intensity, performance, ridge_point

# ── Style: academic ────────────────────────────────────────────────────────────
def _plot_academic(ax, intensity, performance, peak_flops, mem_bandwidth,
                   ridge_point, pts):
    x_max = ridge_point * 2.5
    y_max = peak_flops  * 1.35
    mask  = intensity <= x_max
    x, p  = intensity[mask], performance[mask]

    ax.fill_between(x, p, y_max, alpha=0.07, color="gray", zorder=0)
    ax.text(ridge_point * 0.42, peak_flops * 1.18,
            "Unattainable\n(Greater than Peak)",
            fontsize=8.5, color="dimgray", ha="center", style="italic")

    ax.plot([0, x_max], [0, mem_bandwidth * x_max],
            color="mediumpurple", lw=1.5, ls="--", alpha=0.5, zorder=1)
    lx = ridge_point * 0.35
    ax.text(lx, mem_bandwidth * lx * 0.52,
            f"Bandwidth\n{mem_bandwidth/1e9:.0f} GB/s",
            fontsize=9, color="mediumpurple", ha="center",
            rotation=55, rotation_mode="anchor")

    ax.plot(x, p, color="deeppink", lw=2.8, zorder=3)
    ax.text(ridge_point * 1.5, peak_flops * 1.04,
            f"Peak  {peak_flops/1e9:.0f} GFLOP/s",
            fontsize=9, color="deeppink", va="bottom")
    ax.axvline(x=ridge_point, color="limegreen", ls="--", lw=1.4, alpha=0.85, zorder=2)

    ay = peak_flops * -0.13
    for a, b in [((0.02, ay), (ridge_point*0.98, ay)),
                 ((ridge_point*1.02, ay), (x_max*0.98, ay))]:
        ax.annotate("", xy=a, xytext=b,
                    arrowprops=dict(arrowstyle="<->", color="limegreen", lw=1.5),
                    annotation_clip=False)
    for txt, xp in [("Bandwidth-bound", ridge_point*0.5),
                    ("Compute-bound",   ridge_point*1.75)]:
        ax.text(xp, ay*1.5, txt, fontsize=9, color="limegreen",
                ha="center").set_clip_on(False)
    ax.text(ridge_point, ay*2.8,
            f"Transition @ AI ≈ {ridge_point:.2f}  (Machine Balance)",
            fontsize=8, color="limegreen", ha="center",
            style="italic").set_clip_on(False)

    palette = ["#1f77b4","#ff7f0e","#2ca02c","#d62728",
               "#9467bd","#8c564b","#e377c2","#17becf"]
    if pts:
        for i, (oi, perf, label) in enumerate(pts):
            c = palette[i % len(palette)]
            ax.plot(oi, perf, "o", ms=10, color=c, zorder=5,
                    mec="black", mew=0.7)
            attainable = min(peak_flops, mem_bandwidth * oi)
            if perf < attainable * 0.93:
                ax.annotate("", xy=(oi, attainable), xytext=(oi, perf),
                            arrowprops=dict(arrowstyle="->", color=c,
                                            lw=1.4, ls="dashed"))
            ax.text(oi, perf - peak_flops*0.06, label,
                    fontsize=8, ha="center", color=c)

    ax.set_xlim(0, x_max); ax.set_ylim(0, y_max)
    def fmt(v, _):
        if v >= 1e12: return f"{v/1e12:.3g}T"
        if v >= 1e9:  return f"{v/1e9:.3g}G"
        if v >= 1e6:  return f"{v/1e6:.3g}M"
        return f"{v:.3g}"
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(fmt))
    ax.set_xlabel("Arithmetic Intensity (FLOP/Byte)", fontsize=12)
    ax.set_ylabel("Attainable Performance (FLOP/s)", fontsize=12)
    ax.grid(True, ls="--", alpha=0.25)
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout(rect=[0, 0.1, 1, 1])

# ── Style: loglog ──────────────────────────────────────────────────────────────
def _plot_loglog(ax, intensity, performance, peak_flops, mem_bandwidth,
                 ridge_point, pts):
    ax.loglog(intensity, mem_bandwidth * intensity,
              color="gray", lw=1.4, ls="--", label="Memory BW limit")
    ax.axhline(y=peak_flops, color="red", lw=1.4, ls="--", label="Compute peak")
    ax.loglog(intensity, performance, color="black", lw=2, label="Roofline")

    ax.plot(ridge_point, peak_flops, "x", color="green", ms=12, mew=2.5, zorder=5,
            label=f"Ridge ({ridge_point:.2f} FLOP/B)")
    ax.annotate(f"({ridge_point:.2f}, {peak_flops/1e9:.1f} GFLOP/s)",
                xy=(ridge_point, peak_flops),
                xytext=(ridge_point*1.5, peak_flops*1.6),
                color="green", fontsize=9,
                arrowprops=dict(arrowstyle="->", color="green", lw=1))

    palette = ["#e04040","#9370DB","#8B5E3C","#FF80C0",
               "#1f77b4","#ff7f0e","#2ca02c","#17becf"]
    if pts:
        for i, (oi, perf, label) in enumerate(pts):
            c = palette[i % len(palette)]
            ax.loglog(oi, perf, "o", ms=10, color=c, zorder=6,
                      mec="black", mew=0.7, label=label)

    ax.set_xlabel("Operational Intensity (FLOP/Byte)", fontsize=12)
    ax.set_ylabel("Performance (FLOP/s)", fontsize=12)
    def fmt(v, _):
        if v >= 1e12: return f"{v/1e12:.3g}T"
        if v >= 1e9:  return f"{v/1e9:.3g}G"
        if v >= 1e6:  return f"{v/1e6:.3g}M"
        return f"{v:.3g}"
    ax.yaxis.set_major_formatter(ticker.FuncFormatter(fmt))
    ax.grid(True, which="both", ls="--", alpha=0.3)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.85)

    if pts:
        ymin = ax.get_ylim()[0]
        for i, (oi, perf, _) in enumerate(pts):
            c = palette[i % len(palette)]
            ax.plot([oi, oi], [ymin, perf],
                    color=c, lw=1, ls="--", alpha=0.55, zorder=4)
    plt.tight_layout()

# ── Style: advisor ─────────────────────────────────────────────────────────────
def _plot_advisor(ax, hw, pts, axis_limits=None):
    """
    Intel Advisor-style: multiple memory slopes + multiple compute ceilings.
    pts should be in (AI [FLOP/Byte], perf [GFLOP/s], label).
    """
    compute_roofs = hw["compute_roofs"]  # GFLOP/s
    mem_roofs     = hw["mem_roofs"]      # GB/s

    max_compute = max(r["val"] for r in compute_roofs)
    max_bw      = max(r["val"] for r in mem_roofs)
    min_bw      = min(r["val"] for r in mem_roofs)

    if axis_limits:
        xmin, xmax, ymin, ymax = axis_limits
    else:
        xmin = 0.01
        xmax = max_compute / min_bw * 20
        ymin = min(r["val"] for r in compute_roofs) * 0.05
        ymax = max_compute * 3.0

    xlogsize = math.log10(xmax / xmin)
    ylogsize = math.log10(ymax / ymin)
    m = xlogsize / ylogsize  # log-space ratio for label rotation angle

    # Memory slopes
    slope_colors = plt.cm.cool(np.linspace(0.15, 0.85, len(mem_roofs)))
    for idx, slope in enumerate(mem_roofs):
        bw = slope["val"]
        y_line = np.array([ymin, max_compute])
        x_line = y_line / bw
        ax.loglog(x_line, y_line, lw=1.2, ls="-.",
                  color=slope_colors[idx], zorder=10)
        xpos = xmin * (10 ** (xlogsize * 0.04))
        ypos = xpos * bw
        if ypos < ymin:
            ypos = ymin * (10 ** (ylogsize * 0.03))
            xpos = ypos / bw
        angle = math.degrees(math.atan(m)) * 0.9
        ax.annotate(
            f"{slope['name']}: {bw:.0f} GB/s", (xpos, ypos),
            rotation=angle, rotation_mode="anchor",
            fontsize=9, ha="left", va="bottom", color=slope_colors[idx],
        )

    # Compute ceilings
    roof_colors = plt.cm.autumn(np.linspace(0.1, 0.7, len(compute_roofs)))
    for idx, roof in enumerate(compute_roofs):
        perf  = roof["val"]
        ridge = perf / max_bw
        ax.loglog([ridge, xmax*10], [perf, perf],
                  lw=1.4, ls="-.", color=roof_colors[idx], zorder=10)
        ax.text(xmax / (10 ** (xlogsize * 0.01)),
                perf * (10 ** (ylogsize * 0.012)),
                f"{roof['name']}: {perf:.0f} GFLOP/s",
                ha="right", fontsize=9, color=roof_colors[idx])

    # Kernel points
    palette = ["#1f77b4","#ff7f0e","#2ca02c","#d62728",
               "#9467bd","#8c564b","#e377c2","#17becf"]
    if pts:
        for i, (oi, perf, label) in enumerate(pts):
            c = palette[i % len(palette)]
            ax.scatter(oi, perf, color=c, s=90, zorder=100,
                       edgecolors="black", linewidths=0.6, label=label)
            ax.plot([oi, oi], [ymin, perf],
                    color=c, lw=0.8, ls="--", alpha=0.5, zorder=4)
        ax.legend(loc="lower right", fontsize=9, framealpha=0.85)

    ax.set_xlim(xmin, xmax); ax.set_ylim(ymin, ymax)
    ax.grid(color="#dddddd", which="both", zorder=-1)
    ax.set_xlabel("Arithmetic Intensity [FLOP/Byte]", fontsize=13)
    ax.set_ylabel("Performance [GFLOP/s]", fontsize=13)
    plt.tight_layout()

# ── Public API ─────────────────────────────────────────────────────────────────
def plot_roofline(
    peak_flops=None,
    mem_bandwidth=None,
    title="Roofline Model",
    application_points=None,
    style="loglog",
    hw=None,
    axis_limits=None,
    ax=None,
):
    """
    Plot a Roofline Model.

    Args:
        peak_flops (float):        Peak compute (FLOPS). Needed for academic/loglog.
        mem_bandwidth (float):     Memory bandwidth (bytes/sec). Same.
        title (str):               Plot title.
        application_points (list): [(AI, perf, label), ...]
                                   For "advisor": perf in GFLOP/s.
                                   For others:   perf in FLOP/s.
        style (str):               "academic" | "loglog" | "advisor"
        hw (dict):                 A HARDWARE dict entry.
                                   Required for "advisor".
                                   For others, fills peak_flops/mem_bandwidth.
        axis_limits (tuple):       (xmin, xmax, ymin, ymax) for "advisor".
        ax (plt.Axes):             Existing axes; new figure created if None.

    Returns:
        fig, ax
    """
    if style not in ("academic", "loglog", "advisor"):
        raise ValueError("style must be 'academic', 'loglog', or 'advisor'")

    if hw is not None and style != "advisor":
        peak_flops    = peak_flops    or hw["peak_flops"]
        mem_bandwidth = mem_bandwidth or hw["mem_bandwidth"]

    if ax is None:
        fig, ax = plt.subplots(figsize=(11, 6))
    else:
        fig = ax.get_figure()

    ax.set_title(title, fontsize=14, fontweight="bold", pad=14)

    if style == "academic":
        I, P, rp = calculate_roofline(peak_flops, mem_bandwidth)
        _plot_academic(ax, I, P, peak_flops, mem_bandwidth, rp, application_points)
    elif style == "loglog":
        I, P, rp = calculate_roofline(peak_flops, mem_bandwidth)
        _plot_loglog(ax, I, P, peak_flops, mem_bandwidth, rp, application_points)
    elif style == "advisor":
        if hw is None:
            raise ValueError("'advisor' style requires hw=")
        _plot_advisor(ax, hw, application_points, axis_limits)

    return fig, ax

# ── Demo ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    cpu = HARDWARE["i7-1185G7"]
    gpu = HARDWARE["A100"]

    # CPU kernels derived from baseline latency measurements
    cpu_kernels_flops = [
        (0.16e9 / (10.135e-3 * cpu["mem_bandwidth"]), 0.16e9 / 10.135e-3, "preproc_baseline"),
        (2.25e9 / (56.327e-3 * cpu["mem_bandwidth"]), 2.25e9 / 56.327e-3, "conv_baseline"),
        (0.16e9 / (3.5e-3   * cpu["mem_bandwidth"]), 0.16e9 / 3.5e-3,    "preproc_optimized"),
        (2.25e9 / (10.0e-3  * cpu["mem_bandwidth"]), 2.25e9 / 10.0e-3,   "conv_optimized"),
    ]
    cpu_kernels_gflops = [(oi, p/1e9, l) for oi, p, l in cpu_kernels_flops]

    gpu_kernels_flops  = [
        (0.5,  200e9,  "Embedding lookup"),
        (4.0,  1.5e12, "Attention (small)"),
        (80.0, 8.0e12, "GEMM (large)"),
    ]
    gpu_kernels_gflops = [(oi, p/1e9, l) for oi, p, l in gpu_kernels_flops]

    plots = [
        (cpu, cpu_kernels_flops,  "academic", "cpu_academic"),
        (cpu, cpu_kernels_flops,  "loglog",   "cpu_loglog"),
        (cpu, cpu_kernels_gflops, "advisor",  "cpu_advisor"),
        (gpu, gpu_kernels_flops,  "academic", "gpu_academic"),
        (gpu, gpu_kernels_flops,  "loglog",   "gpu_loglog"),
        (gpu, gpu_kernels_gflops, "advisor",  "gpu_advisor"),
    ]

    for hw, kernels, style, suffix in plots:
        fig, _ = plot_roofline(
            title=f"Roofline — {hw['label']}  [{style}]",
            application_points=kernels,
            style=style,
            hw=hw,
        )
        out = os.path.join(OUTPUT_DIR, f"roofline_{suffix}.png")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved: {out}")
        plt.close(fig)

    print("Done.")