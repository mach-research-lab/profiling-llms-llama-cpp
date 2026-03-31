#!/usr/bin/env python3
"""
energy_measure.py — Interactive energy measurement tool.

Wraps the llama-energy C++ binary. Measures CPU energy per transformer
layer using Linux perf_event_open (no sudo required).

Usage:
    python3 09A-backend/energy_measure.py

Requirements:
    1. Build llama-energy:
           cd ~/09A/profiling-llms-llama-cpp
           cmake -B build
           cmake --build build --target llama-energy -j$(nproc)

    2. perf_event_paranoid must be <= 1:
           cat /proc/sys/kernel/perf_event_paranoid
       If > 1, fix permanently:
           echo kernel.perf_event_paranoid=-1 | sudo tee /etc/sysctl.d/99-perf.conf
           sudo sysctl -p /etc/sysctl.d/99-perf.conf

Output:
    energy.csv — per-layer energy for cpu_package, cpu_cores, full_system domains
"""

import subprocess
import sys
import os
import glob

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
LLAMA_ROOT  = os.path.dirname(SCRIPT_DIR)
BINARY      = os.path.join(LLAMA_ROOT, "build/bin/llama-energy")
MODELS_ROOT = os.path.join(os.path.expanduser("~"), "shared/models")


def check_perf_paranoid():
    """Check that perf_event_paranoid allows energy measurements."""
    try:
        with open("/proc/sys/kernel/perf_event_paranoid") as f:
            val = int(f.read().strip())
        if val <= 1:
            print(f"  ✓ perf_event_paranoid = {val} (OK)")
        else:
            print(f"  ✗ perf_event_paranoid = {val} (must be <= 1)")
            print("\n  Fix permanently:")
            print("    echo kernel.perf_event_paranoid=-1 | sudo tee /etc/sysctl.d/99-perf.conf")
            print("    sudo sysctl -p /etc/sysctl.d/99-perf.conf")
            if input("\n  Continue anyway? (y/n): ").strip().lower() != "y":
                sys.exit(1)
    except FileNotFoundError:
        print("  ✗ Could not read perf_event_paranoid")


def check_binary():
    if not os.path.isfile(BINARY):
        print(f"\n  ✗ Binary not found: {BINARY}")
        print("  Build it first:")
        print("    cd ~/09A/profiling-llms-llama-cpp")
        print("    cmake -B build")
        print("    cmake --build build --target llama-energy -j$(nproc)")
        sys.exit(1)
    print(f"  ✓ Binary found: {BINARY}")


def find_models():
    pattern = os.path.join(MODELS_ROOT, "**", "*.gguf")
    return sorted(glob.glob(pattern, recursive=True))


def select_model():
    models = find_models()
    if not models:
        print(f"No .gguf models found in {MODELS_ROOT}")
        sys.exit(1)

    print("\n╔══════════════════════════════════════════════════════════════════════╗")
    print("║                        Available models                             ║")
    print("╠══════════════════════════════════════════════════════════════════════╣")
    for i, path in enumerate(models):
        rel = os.path.relpath(path, MODELS_ROOT)
        display = rel if len(rel) <= 68 else "..." + rel[-65:]
        print(f"║  {i+1:2d}. {display:<68}║")
    print("╚══════════════════════════════════════════════════════════════════════╝")

    print("\nSelect a model (enter number):")
    while True:
        raw = input("> ").strip()
        if not raw.isdigit():
            print("Invalid input. Enter a number.")
            continue
        idx = int(raw) - 1
        if idx < 0 or idx >= len(models):
            print(f"Choose between 1 and {len(models)}.")
            continue
        selected = models[idx]
        print(f"\n  ✓ {os.path.relpath(selected, MODELS_ROOT)}")
        if input("Confirm? (y/n): ").strip().lower() == "y":
            return selected


def run_binary(model_path, prompt, n_predict):
    cmd = [
        BINARY,
        "--result-path", os.path.join(LLAMA_ROOT, "energy.csv"),
        "-m", model_path,
        "-p", prompt,
        "-n", str(n_predict),
        "--log-disable",
    ]

    print(f"\nRunning: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, cwd=LLAMA_ROOT)

    if result.returncode != 0:
        print(f"\n  ✗ llama-energy failed (returncode {result.returncode})")
        sys.exit(1)

    csv_path = os.path.join(LLAMA_ROOT, "energy.csv")
    if os.path.isfile(csv_path):
        print(f"\n  ✓ Results saved to: {csv_path}")
    else:
        print("\n  ✗ energy.csv not found after run.")


def main():
    print("═" * 60)
    print("   llama.cpp energy measurement (perf per layer)")
    print("═" * 60)
    print()

    check_binary()
    check_perf_paranoid()
    model_path = select_model()

    print("\nEnter your prompt:")
    prompt = input("> ").strip()
    if not prompt:
        prompt = "What is the capital of Sweden?"

    print("How many tokens to generate? (default: 64)")
    raw       = input("> ").strip()
    n_predict = int(raw) if raw.isdigit() else 64

    run_binary(model_path, prompt, n_predict)


if __name__ == "__main__":
    main()