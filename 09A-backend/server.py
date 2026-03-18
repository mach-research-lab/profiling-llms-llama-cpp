#!/usr/bin/env python3
"""
FastAPI server for PAPI profiling frontend.
Provides API endpoints to list models and run profiling.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import subprocess
import os
import glob
import json
import re
from typing import List, Optional
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from analysis import plot_roofline, HARDWARE

app = FastAPI()

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LLAMA_ROOT = os.path.dirname(SCRIPT_DIR)
BINARY = os.path.join(LLAMA_ROOT, "build/bin/llama-eval-callback")
MODELS_ROOT = os.path.join(os.path.expanduser("~"), "shared/models")
PLOTS_DIR = os.path.join(SCRIPT_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

AVAILABLE_EVENTS = [
    ("PAPI_TOT_CYC", "Total cycles"),
    ("PAPI_TOT_INS", "Instructions completed"),
    ("PAPI_L1_DCM", "L1 data cache misses"),
    ("PAPI_L1_ICM", "L1 instruction cache misses"),
    ("PAPI_L1_TCM", "L1 cache misses (total)"),
    ("PAPI_L2_DCM", "L2 data cache misses"),
    ("PAPI_L2_ICM", "L2 instruction cache misses"),
    ("PAPI_L2_TCM", "L2 cache misses (total)"),
    ("PAPI_L3_TCM", "L3 cache misses (total)"),
    ("PAPI_L3_LDM", "L3 load misses"),
    ("PAPI_FP_OPS", "Floating point operations"),
    ("PAPI_VEC_INS", "Vector/SIMD instructions"),
]


class ModelInfo(BaseModel):
    path: str
    display_name: str


class ProfileRequest(BaseModel):
    model_path: str
    events: List[str]
    prompt: str
    n_predict: int = 64


class AnalysisRequest(BaseModel):
    profiling_output: str
    hardware: str = "i7-1185G7"
    style: str = "academic"


def find_models():
    """Find all .gguf model files."""
    pattern = os.path.join(MODELS_ROOT, "**", "*.gguf")
    return sorted(glob.glob(pattern, recursive=True))


@app.get("/")
async def root():
    return {"message": "PAPI Profiler API"}


@app.get("/models", response_model=List[ModelInfo])
async def get_models():
    """Get list of available models."""
    models = find_models()
    if not models:
        return []

    return [
        ModelInfo(
            path=path,
            display_name=os.path.relpath(path, MODELS_ROOT)
        )
        for path in models
    ]


@app.get("/events")
async def get_events():
    """Get list of available PAPI events."""
    return [
        {"name": name, "description": desc}
        for name, desc in AVAILABLE_EVENTS
    ]


@app.post("/profile")
async def run_profile(request: ProfileRequest):
    """Run PAPI profiling with selected parameters."""

    # Check if binary exists
    if not os.path.isfile(BINARY):
        raise HTTPException(
            status_code=404,
            detail=f"Binary not found at {BINARY}. Please build llama-eval-callback first."
        )

    # Validate events (max 4)
    if len(request.events) > 4:
        raise HTTPException(status_code=400, detail="Maximum 4 events allowed")

    # Build command
    events_arg = ",".join(request.events)
    cmd = [
        BINARY,
        "--papi-events", events_arg,
        "-m", request.model_path,
        "-p", request.prompt,
        "-n", str(request.n_predict),
        "--log-disable",
    ]

    try:
        # Run the profiling
        result = subprocess.run(
            cmd,
            cwd=LLAMA_ROOT,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )

        return {
            "success": result.returncode == 0,
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode
        }
    except subprocess.TimeoutExpired:
        raise HTTPException(status_code=408, detail="Profiling timed out after 5 minutes")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


def parse_papi_output(output: str):
    """
    Parse PAPI profiling output to extract metrics.
    Returns dict with parsed metrics or None if parsing fails.

    TODO: This function is ready for future integration with live profiling.
    Currently not used as we're showing hardcoded examples.
    Will be integrated when connecting to actual PAPI output.
    """
    lines = output.strip().split('\n')
    metrics = {}

    for line in lines:
        # Look for lines like "PAPI_FP_OPS: 1234567"
        match = re.match(r'(PAPI_\w+):\s*(\d+)', line)
        if match:
            event_name = match.group(1)
            value = int(match.group(2))
            metrics[event_name] = value

        # Look for timing information
        if "Time:" in line or "time:" in line.lower():
            # Try to extract time in seconds
            time_match = re.search(r'([\d.]+)\s*(s|ms|seconds?|milliseconds?)', line.lower())
            if time_match:
                time_val = float(time_match.group(1))
                unit = time_match.group(2)
                if 'ms' in unit or 'milli' in unit:
                    time_val /= 1000.0
                metrics['time_seconds'] = time_val

    return metrics if metrics else None


def calculate_roofline_metrics(metrics: dict, mem_bandwidth: float):
    """
    Calculate arithmetic intensity and performance from PAPI metrics.

    Args:
        metrics: Dict with PAPI counters
        mem_bandwidth: Hardware memory bandwidth in bytes/sec

    Returns:
        (arithmetic_intensity, performance) or None

    TODO: This function is ready for future integration with live profiling.
    Currently not used as we're showing hardcoded examples.
    Will be integrated when connecting to actual PAPI output.
    """
    flops = metrics.get('PAPI_FP_OPS', 0)
    time_sec = metrics.get('time_seconds', 0)

    if time_sec <= 0:
        return None

    # Calculate performance (FLOP/s)
    performance = flops / time_sec

    # Estimate memory traffic from cache misses
    # This is a simplified model - in practice you'd want more sophisticated analysis
    l1_misses = metrics.get('PAPI_L1_TCM', 0)
    l2_misses = metrics.get('PAPI_L2_TCM', 0)
    l3_misses = metrics.get('PAPI_L3_TCM', 0)

    # Assume 64 bytes per cache line
    cache_line_size = 64

    # Estimate bytes transferred (prioritize L3 misses as they go to DRAM)
    if l3_misses > 0:
        bytes_transferred = l3_misses * cache_line_size
    elif l2_misses > 0:
        bytes_transferred = l2_misses * cache_line_size
    elif l1_misses > 0:
        bytes_transferred = l1_misses * cache_line_size
    else:
        # Fallback: assume memory-bound based on time
        bytes_transferred = mem_bandwidth * time_sec * 0.5

    # Calculate arithmetic intensity (FLOP/Byte)
    arithmetic_intensity = flops / bytes_transferred if bytes_transferred > 0 else 0

    return arithmetic_intensity, performance


@app.get("/hardware")
async def get_hardware():
    """Get list of available hardware profiles."""
    return [
        {"name": key, "label": val["label"]}
        for key, val in HARDWARE.items()
    ]


@app.post("/analyze")
async def analyze_profiling(request: AnalysisRequest):
    """
    Generate hardcoded roofline plot examples.

    TODO: Future integration with live PAPI profiling data
    - Parse PAPI output from request.profiling_output using parse_papi_output()
    - Calculate arithmetic intensity and performance using calculate_roofline_metrics()
    - Plot actual kernel measurements instead of hardcoded examples
    - See parse_papi_output() and calculate_roofline_metrics() functions below for reference
    """
    # Validate hardware
    if request.hardware not in HARDWARE:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown hardware '{request.hardware}'. Available: {list(HARDWARE.keys())}"
        )

    # Validate style
    if request.style not in ("academic", "loglog", "advisor"):
        raise HTTPException(
            status_code=400,
            detail="style must be 'academic', 'loglog', or 'advisor'"
        )

    # Get hardware config
    hw = HARDWARE[request.hardware]

    # Generate hardcoded example plots (same as analysis.py main function)
    try:
        # Use hardcoded kernel examples based on hardware type
        if request.hardware == "i7-1185G7":
            # CPU kernels derived from baseline latency measurements
            cpu_kernels_flops = [
                (0.16e9 / (10.135e-3 * hw["mem_bandwidth"]), 0.16e9 / 10.135e-3, "preproc_baseline"),
                (2.25e9 / (56.327e-3 * hw["mem_bandwidth"]), 2.25e9 / 56.327e-3, "conv_baseline"),
                (0.16e9 / (3.5e-3   * hw["mem_bandwidth"]), 0.16e9 / 3.5e-3,    "preproc_optimized"),
                (2.25e9 / (10.0e-3  * hw["mem_bandwidth"]), 2.25e9 / 10.0e-3,   "conv_optimized"),
            ]
            cpu_kernels_gflops = [(oi, p/1e9, l) for oi, p, l in cpu_kernels_flops]

            if request.style == "advisor":
                app_points = cpu_kernels_gflops
            else:
                app_points = cpu_kernels_flops

        elif request.hardware == "A100":
            # GPU kernels
            gpu_kernels_flops  = [
                (0.5,  200e9,  "Embedding lookup"),
                (4.0,  1.5e12, "Attention (small)"),
                (80.0, 8.0e12, "GEMM (large)"),
            ]
            gpu_kernels_gflops = [(oi, p/1e9, l) for oi, p, l in gpu_kernels_flops]

            if request.style == "advisor":
                app_points = gpu_kernels_gflops
            else:
                app_points = gpu_kernels_flops
        else:
            app_points = []

        # Generate plot
        fig, ax = plot_roofline(
            title=f"Roofline — {hw['label']}  [{request.style}]",
            application_points=app_points,
            style=request.style,
            hw=hw,
        )

        # Save plot
        import time
        timestamp = int(time.time())
        plot_filename = f"roofline_{request.hardware}_{request.style}_{timestamp}.png"
        plot_path = os.path.join(PLOTS_DIR, plot_filename)

        fig.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)

        return {
            "success": True,
            "plot_url": f"/plots/{plot_filename}",
            "metrics": {
                "note": "Using hardcoded example kernels. TODO: Integrate with live PAPI profiling data.",
                "hardware": request.hardware,
                "style": request.style,
                "kernels": [{"ai": k[0], "perf_gflops": k[1] if request.style == "advisor" else k[1]/1e9, "label": k[2]} for k in app_points]
            }
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating plot: {str(e)}")


@app.get("/plots/{filename}")
async def get_plot(filename: str):
    """Serve generated plot images."""
    plot_path = os.path.join(PLOTS_DIR, filename)

    if not os.path.isfile(plot_path):
        raise HTTPException(status_code=404, detail="Plot not found")

    return FileResponse(plot_path, media_type="image/png")


@app.get("/measurements.csv")
async def get_measurements():
    """Serve the measurements CSV file."""
    csv_path = os.path.join(LLAMA_ROOT, "measurements.csv")

    if not os.path.isfile(csv_path):
        raise HTTPException(status_code=404, detail="measurements.csv not found. Run profiling first.")

    return FileResponse(csv_path, media_type="text/csv")

@app.get("/cpu_info")
async def get_cpu_info():
    """Get CPU architecture and model name using lscpu."""
    try:
        result = subprocess.run(["lscpu"], capture_output=True, text=True, timeout=5)
        if result.returncode != 0:
            raise Exception("lscpu command failed")

        cpu_info = {}
        for line in result.stdout.splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                cpu_info[key.strip()] = value.strip()

        return {
            "architecture": cpu_info.get("Architecture"),
            "model_name": cpu_info.get("Model name"),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error getting CPU info: {str(e)}")

@app.get("/roofline_data")
async def get_roofline_data():
    """Serve hardcoded roofline data for frontend visualization."""
    # This is a placeholder endpoint. In the future, this will serve actual data derived from PAPI profiling.
    # For now, it returns the same hardcoded kernel examples used in the analysis endpoint.

    data = {
        "i7-1185G7": [
            {"arithmetic_intensity": 0.16e9 / (10.135e-3 * HARDWARE["i7-1185G7"]["mem_bandwidth"]), "performance_gflops": 0.16e9 / 10.135e-3 / 1e9, "label": "preproc_baseline"},
            {"arithmetic_intensity": 2.25e9 / (56.327e-3 * HARDWARE["i7-1185G7"]["mem_bandwidth"]), "performance_gflops": 2.25e9 / 56.327e-3 / 1e9, "label": "conv_baseline"},
            {"arithmetic_intensity": 0.16e9 / (3.5e-3   * HARDWARE["i7-1185G7"]["mem_bandwidth"]), "performance_gflops": 0.16e9 / 3.5e-3 / 1e9,    "label": "preproc_optimized"},
            {"arithmetic_intensity": 2.25e9 / (10.0e-3  * HARDWARE["i7-1185G7"]["mem_bandwidth"]), "performance_gflops": 2.25e9 / 10.0e-3 / 1e9,   "label": "conv_optimized"},
        ],
        "A100": [
            {"arithmetic_intensity": 0.5,  "performance_gflops": 200,   "label": "Embedding lookup"},
            {"arithmetic_intensity": 4.0,  "performance_gflops": 1500,  "label": "Attention (small)"},
            {"arithmetic_intensity": 80.0, "performance_gflops": 8000,  "label": "GEMM (large)"},
        ]
    }

    return data


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
