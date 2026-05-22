#!/usr/bin/env python3
"""
FastAPI server for PAPI profiling frontend.
Provides API endpoints to list models and run profiling.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
import subprocess, threading, uuid, asyncio
import os
import glob
import json
import re
from typing import List, Optional
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from analysis import plot_roofline, HARDWARE
from event_retriever import get_available_events

app = FastAPI()

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
LLAMA_ROOT = os.path.dirname(SCRIPT_DIR)
BINARY = os.path.join(LLAMA_ROOT, "build/bin/llama-papi")
MODELS_ROOT = os.path.join(LLAMA_ROOT, "models")
PLOTS_DIR = os.path.join(SCRIPT_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

AVAILABLE_EVENTS = get_available_events()

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


## --- Session store ---
active_sessions: dict[str, dict] = {}
# Each entry: { "proc": Popen|None, "queue": asyncio.Queue, "loop": loop, "cfg": Config, "papi_events_per_run": int|None }


class RunStartRequest(BaseModel):
    model_path: str
    prompt: str
    n_predict: int = 64
    k_cache_type: str = "f16"
    v_cache_type: str = "f16"
    papi_events_per_run: int | None = None
    custom_events: Optional[List[str]] = None


class SendPromptRequest(BaseModel):
    prompt: str


@app.post("/run/start")
async def start_run(request: RunStartRequest):
    """Start ONLY the phase-view binary in conversation mode for interactive chat."""
    session_id = str(uuid.uuid4())
    loop = asyncio.get_event_loop()
    queue: asyncio.Queue = asyncio.Queue()
    from run_handler import Config, Run_type, LLAMA_ROOT as LR
    from event_retriever import get_valid_runs_from_list

    # Clean up any existing active sessions/processes first to free up PAPI/GPU
    for old_sid, session in list(active_sessions.items()):
        proc = session.get("proc")
        if proc and proc.poll() is None:
            print(f"[chat] Terminating orphaned process from session {old_sid}")
            try:
                proc.terminate()
                proc.wait(timeout=1.5)
            except Exception as e:
                print(f"[chat] Failed to terminate orphaned process: {e}")
                try:
                    proc.kill()
                except Exception:
                    pass
            try:
                old_loop = session.get("loop")
                old_queue = session.get("queue")
                if old_loop and old_queue:
                    asyncio.run_coroutine_threadsafe(old_queue.put({"chat_done": True, "done": True}), old_loop)
            except Exception:
                pass
            active_sessions.pop(old_sid, None)

    model_path = request.model_path
    if not os.path.isabs(model_path):
        model_path = os.path.join(MODELS_ROOT, model_path)
    if not os.path.isfile(model_path):
        raise HTTPException(status_code=404, detail=f"Model not found at {model_path}.")

    cfg = Config(
        model_path=model_path,
        custom_events=request.custom_events,
        prompt=request.prompt,
        n_predict=request.n_predict,
        k_cache_type=request.k_cache_type,
        v_cache_type=request.v_cache_type,
        binary_path=os.path.join(LR, "build/bin/llama-measurement-phase-view"),
    )

    RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "run_every_view_results")
    if os.path.exists(RESULTS_DIR):
        import shutil
        shutil.rmtree(RESULTS_DIR)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    result_path = os.path.join(RESULTS_DIR, Run_type.PHASE_VIEW.path)

    active_sessions[session_id] = {
        "proc": None, "queue": queue, "loop": loop,
        "cfg": cfg, "papi_events_per_run": request.papi_events_per_run,
    }

    def chat_thread():
        """Run the phase-view binary in conversation+collect-prompts mode."""
        try:
            events_list = request.custom_events if request.custom_events is not None else Run_type.PHASE_VIEW.events
            event_groups = get_valid_runs_from_list(events_list, request.papi_events_per_run)
        except Exception as e:
            import traceback; traceback.print_exc()
            asyncio.run_coroutine_threadsafe(queue.put({"error": str(e), "done": True}), loop)
            return

        # Only run the FIRST event group interactively (with --collect-prompts)
        group = event_groups[0]
        events_arg = ",".join(group)
        cmd = [
            cfg.binary_path,
            "--papi-events", events_arg,
            "--result-path", result_path,
            "--papi-events-unrestricted",
            "--conversation",
            "-m", cfg.model_path,
            "-p", cfg.prompt,
            "-n", str(cfg.n_predict),
            "--cache-type-k", cfg.k_cache_type,
            "--cache-type-v", cfg.v_cache_type,
            "--temp", "0",
            "--log-disable",
            "--collect-prompts",
        ]

        print(f"\n[chat] Running: {' '.join(cmd)}\n")
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            stdin=subprocess.PIPE, text=True, bufsize=1,
            cwd=LLAMA_ROOT,
        )
        active_sessions[session_id]["proc"] = proc

        for line in iter(proc.stdout.readline, ""):
            asyncio.run_coroutine_threadsafe(
                queue.put({"line": line.rstrip()}), loop
            )
        proc.wait()

        # Chat phase done
        asyncio.run_coroutine_threadsafe(queue.put({"chat_done": True}), loop)

    threading.Thread(target=chat_thread, daemon=True).start()
    return {"sessionId": session_id}


@app.get("/run/stream/{session_id}")
async def stream_run(session_id: str):
    """SSE endpoint — streams stdout lines for a session."""
    session = active_sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    queue = session["queue"]

    async def event_generator():
        while True:
            msg = await queue.get()
            yield f"data: {json.dumps(msg)}\n\n"
            if msg.get("done") or msg.get("chat_done"):
                break

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/run/cancel")
async def cancel_runs():
    """Cancel all active sessions and terminate their running processes."""
    count = 0
    for sid, session in list(active_sessions.items()):
        proc = session.get("proc")
        if proc and proc.poll() is None:
            print(f"[chat] Cancelling active process for session {sid}")
            try:
                proc.terminate()
                proc.wait(timeout=1.5)
            except Exception as e:
                try:
                    proc.kill()
                except Exception:
                    pass
            count += 1
        try:
            loop = session.get("loop")
            queue = session.get("queue")
            if loop and queue:
                asyncio.run_coroutine_threadsafe(queue.put({"chat_done": True, "done": True}), loop)
        except Exception:
            pass
    active_sessions.clear()
    return {"terminated_count": count}


@app.post("/run/prompt/{session_id}")
async def send_prompt(session_id: str, body: SendPromptRequest):
    """Send a follow-up prompt to the running chat process via stdin."""
    session = active_sessions.get(session_id)
    if not session or not session.get("proc"):
        raise HTTPException(status_code=404, detail="No active process for this session")
    proc = session["proc"]
    try:
        proc.stdin.write(body.prompt + "\n")
        proc.stdin.flush()
    except BrokenPipeError:
        raise HTTPException(status_code=410, detail="Process has already exited")
    return {"ok": True}


@app.post("/run/profile/{session_id}")
async def run_profile(session_id: str):
    """End the chat (send 'quit') and run remaining profiling views.
    Returns a new SSE stream with profiling progress.
    """
    session = active_sessions.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Send quit to end the conversation
    proc = session.get("proc")
    if proc and proc.poll() is None:
        try:
            proc.stdin.write("quit\n")
            proc.stdin.flush()
        except BrokenPipeError:
            pass

    cfg = session["cfg"]
    papi_events_per_run = session["papi_events_per_run"]
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    active_sessions[session_id]["queue"] = queue
    active_sessions[session_id]["loop"] = loop

    def profile_thread():
        """Run remaining views (top, decoder-block, tensor-op) + roofline."""
        # Wait for the interactive chat process to finish and write collected_prompts.json
        proc = session.get("proc")
        if proc and proc.poll() is None:
            try:
                proc.wait()
            except Exception:
                pass

        import run_handler
        from run_handler import Run_type
        import event_retriever

        # Save original builtins print and assign custom print to route python logs to SSE stream
        original_print = print

        def custom_print(*args, **kwargs):
            sep = kwargs.get("sep", " ")
            line = sep.join(str(arg) for arg in args)
            original_print(*args, **kwargs)
            for subline in line.split('\n'):
                trimmed = subline.strip()
                if trimmed:
                    asyncio.run_coroutine_threadsafe(queue.put({"line": trimmed}), loop)

        run_handler.print = custom_print
        event_retriever.print = custom_print

        RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "run_every_view_results")
        original_run = subprocess.run
        original_popen = subprocess.Popen

        def patched_popen(cmd, **kwargs):
            kwargs["stdout"] = subprocess.PIPE
            kwargs["stderr"] = subprocess.STDOUT
            kwargs["stdin"]  = subprocess.PIPE
            kwargs["text"]   = True
            kwargs["bufsize"] = 1
            p = original_popen(cmd, **kwargs)
            active_sessions.get(session_id, {})["proc"] = p
            for line in iter(p.stdout.readline, ""):
                asyncio.run_coroutine_threadsafe(queue.put({"line": line.rstrip()}), loop)
            p.wait()
            return p

        def patched_run(cmd, **kwargs):
            if kwargs.get("capture_output"):
                return original_run(cmd, **kwargs)
            p = patched_popen(cmd, cwd=kwargs.get("cwd", LLAMA_ROOT))
            p.wait()
            return p

        run_handler.subprocess.run = patched_run

        try:
            # Also run remaining phase-view event groups (2nd, 3rd, etc.) with --user-prompts
            from event_retriever import get_valid_runs_from_list
            from run_handler import get_user_prompts
            result_path_phase = os.path.join(RESULTS_DIR, Run_type.PHASE_VIEW.path)
            events_list = cfg.custom_events if cfg.custom_events is not None else Run_type.PHASE_VIEW.events
            event_groups = get_valid_runs_from_list(events_list, papi_events_per_run)
            for i, group in enumerate(event_groups[1:], start=2):
                asyncio.run_coroutine_threadsafe(
                    queue.put({"line": f"===== Phase View group {i}/{len(event_groups)} ====="}), loop)
                events_arg = ",".join(group)
                cmd = [
                    cfg.binary_path,
                    "--papi-events", events_arg,
                    "--result-path", result_path_phase,
                    "--papi-events-unrestricted", "--conversation",
                    "-m", cfg.model_path, "-p", cfg.prompt,
                    "-n", str(cfg.n_predict),
                    "--cache-type-k", cfg.k_cache_type, "--cache-type-v", cfg.v_cache_type,
                    "--temp", "0", "--log-disable",
                    "--user-prompts", get_user_prompts(result_path_phase), "--disable-prints",
                ]
                patched_run(cmd, cwd=LLAMA_ROOT)

            asyncio.run_coroutine_threadsafe(
                queue.put({"line": "===== Starting Top View ====="}), loop)
            cfg.binary_path = os.path.join(LLAMA_ROOT, "build/bin/llama-measurement-top-view")
            run_handler.run_view(cfg, False, Run_type.TOP_VIEW, False)

            asyncio.run_coroutine_threadsafe(
                queue.put({"line": "===== Starting Decoder Block View ====="}), loop)
            cfg.binary_path = os.path.join(LLAMA_ROOT, "build/bin/llama-measurement-decoder-block-view")
            run_handler.run_view(cfg, False, Run_type.DECODER_BLOCK_VIEW, False)

            asyncio.run_coroutine_threadsafe(
                queue.put({"line": "===== Starting Tensor-Op View ====="}), loop)
            cfg.binary_path = os.path.join(LLAMA_ROOT, "build/bin/llama-papi")
            run_handler.run_view(cfg, False, Run_type.TENSOR_OP_VIEW, False, papi_events_per_run)

            try:
                run_handler.complement_phase_json(
                    os.path.join(RESULTS_DIR, Run_type.PHASE_VIEW.path),
                    os.path.join(RESULTS_DIR, Run_type.TENSOR_OP_VIEW.path),
                    None, os.path.join(RESULTS_DIR, Run_type.PHASE_VIEW.path))
            except Exception as pe:
                print(f"Warning: Failed to complement phase-view.json: {pe}")

            try:
                run_handler.complement_decoder_block_json(
                    os.path.join(RESULTS_DIR, Run_type.DECODER_BLOCK_VIEW.path),
                    os.path.join(RESULTS_DIR, Run_type.TENSOR_OP_VIEW.path),
                    None, os.path.join(RESULTS_DIR, Run_type.DECODER_BLOCK_VIEW.path))
            except Exception as de:
                print(f"Warning: Failed to complement decoder-block-view.json: {de}")

            try:
                from Roofline import sum_blocks, roofline_from_aggregated, load_json, find_decoder_block_json
                json_path = find_decoder_block_json()
                if json_path:
                    data = load_json(json_path)
                    agg = sum_blocks(data)
                    roofline_data = roofline_from_aggregated(agg, "Entire program")
                    roofline_path = os.path.join(os.path.dirname(json_path), "roofline.json")
                    with open(roofline_path, "w") as f:
                        json.dump(roofline_data, f, indent=2)
            except Exception as re:
                print(f"Warning: Failed to generate roofline.json: {re}")

            asyncio.run_coroutine_threadsafe(queue.put({"done": True}), loop)
        except Exception as e:
            import traceback; traceback.print_exc()
            asyncio.run_coroutine_threadsafe(queue.put({"error": str(e), "done": True}), loop)
        finally:
            active_sessions.pop(session_id, None)

    threading.Thread(target=profile_thread, daemon=True).start()
    return {"ok": True}


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


RESULTS_DIR = os.path.join(LLAMA_ROOT, "run_every_view_results")

@app.get("/top-view.json")
async def get_top_view():
    path = os.path.join(RESULTS_DIR, "top-view.json")
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="top-view.json not found. Run profiling first.")
    return FileResponse(path, media_type="application/json")

@app.get("/phase-view.json")
async def get_phase_view():
    path = os.path.join(RESULTS_DIR, "phase-view.json")
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="phase-view.json not found. Run profiling first.")
    return FileResponse(path, media_type="application/json")

@app.get("/decoder-block-view.json")
async def get_decoder_block_view():
    path = os.path.join(RESULTS_DIR, "decoder-block-view.json")
    if not os.path.isfile(path):
        raise HTTPException(status_code=404, detail="decoder-block-view.json not found. Run profiling first.")
    return FileResponse(path, media_type="application/json")

@app.get("/roofline.json")
async def get_roofline_json():
    path = os.path.join(RESULTS_DIR, "roofline.json")
    if not os.path.isfile(path):
        from Roofline import sum_blocks, roofline_from_aggregated, load_json, find_decoder_block_json
        json_path = find_decoder_block_json()
        if json_path:
            try:
                data = load_json(json_path)
                agg = sum_blocks(data)
                roofline_data = roofline_from_aggregated(agg, "Entire program")
                return roofline_data
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Error generating roofline: {str(e)}")
        raise HTTPException(status_code=404, detail="roofline.json not found. Run profiling first.")
    return FileResponse(path, media_type="application/json")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
