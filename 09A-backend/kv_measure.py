#!/usr/bin/env python3
"""
kv_measure.py — Interactive KV-cache measurement tool.

Wraps the kv-measure C++ binary. Asks the user to select a model
and enter a prompt, then runs the measurement and saves results to
kv_sizes.csv in the llama.cpp root directory.

The KV cache type (f16, q8_0, etc.) is detected automatically from
the model filename — no manual configuration needed.

Usage:
    python3 09A-backend/kv_measure.py

Requirements:
    Build kv-measure first:
        cd ~/09A/profiling-llms-llama-cpp
        cmake -B build
        cmake --build build --target kv-measure -j$(nproc)

Output:
    kv_sizes.csv — KV cache size per token (K bytes, V bytes, total)
"""

import subprocess
import sys
import os
import glob

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
LLAMA_ROOT  = os.path.dirname(SCRIPT_DIR)
BINARY      = os.path.join(LLAMA_ROOT, "build/bin/kv-measure")
MODELS_ROOT = os.path.join(os.path.expanduser("~"), "shared/models")


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


def detect_cache_type(model_path: str) -> str:
    """
    Detect KV cache type from the model filename.
    Maps common quantization suffixes to llama.cpp cache type names.
    Falls back to f16 if nothing matches.
    """
    name = os.path.basename(model_path).lower()

    # Map filename patterns to cache type
    patterns = [
        ("q8_0",  "q8_0"),
        ("q6_k",  "q8_0"),   # Q6_K weights → use q8_0 cache (best match)
        ("q4_k",  "q8_0"),   # Q4_K weights → use q8_0 cache
        ("q4_0",  "q4_0"),
        ("q5_0",  "q8_0"),
        ("q5_k",  "q8_0"),
        ("f32",   "f32"),
        ("fp32",  "f32"),
        ("f16",   "f16"),
        ("fp16",  "f16"),
    ]

    for pattern, cache_type in patterns:
        if pattern in name:
            return cache_type

    return "f16"  # default


def check_binary():
    if not os.path.isfile(BINARY):
        print(f"\nBinary not found: {BINARY}")
        print("Build it first:")
        print("  cd ~/09A/profiling-llms-llama-cpp")
        print("  cmake --build build --target kv-measure -j$(nproc)")
        sys.exit(1)
    print(f"\n  ✓ Binary found: {BINARY}")


def run_binary(model_path, prompt, n_predict, cache_type):
    cmd = [
        BINARY,
        "--result-path", os.path.join(LLAMA_ROOT, "kv_sizes.csv"),
        "-m", model_path,
        "-p", prompt,
        "-n", str(n_predict),
        "--cache-type-k", cache_type,
        "--cache-type-v", cache_type,
        "--log-disable",
    ]

    print(f"\nRunning: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, cwd=LLAMA_ROOT)

    if result.returncode != 0:
        print(f"\n  ✗ kv-measure failed (returncode {result.returncode})")
        sys.exit(1)

    csv_path = os.path.join(LLAMA_ROOT, "kv_sizes.csv")
    if os.path.isfile(csv_path):
        print(f"\n  ✓ Results saved to: {csv_path}")
    else:
        print("\n  ✗ kv_sizes.csv not found after run.")


def main():
    print("═" * 60)
    print("   KV-cache measurement")
    print("═" * 60)

    check_binary()
    model_path = select_model()

    cache_type = detect_cache_type(model_path)
    print(f"\n  ✓ KV cache type: {cache_type} (detected from filename)")

    print("\nEnter your prompt:")
    prompt = input("> ").strip()
    if not prompt:
        prompt = "What is the capital of Sweden?"

    print("How many tokens to generate? (default: 64)")
    raw       = input("> ").strip()
    n_predict = int(raw) if raw.isdigit() else 64

    run_binary(model_path, prompt, n_predict, cache_type)


if __name__ == "__main__":
    main()