#!/usr/bin/env python3
"""
Interactive tool for selecting PAPI events and running llama-eval-callback.
Run from the 09A-backend directory inside the llama.cpp repo.

The C++ binary is compiled once. This script selects events at runtime
and passes them via --papi-events.
"""

import subprocess
import sys
import os
import glob
from event_retriever import get_available_events

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
LLAMA_ROOT  = os.path.dirname(SCRIPT_DIR)
BINARY      = os.path.join(LLAMA_ROOT, "build/bin/llama-probe")
MODELS_ROOT = os.path.join(os.path.expanduser("~"), "shared/models")

AVAILABLE_EVENTS = get_available_events()


def find_models():
    pattern = os.path.join(MODELS_ROOT, "**", "*.gguf")
    return sorted(glob.glob(pattern, recursive=True))


def select_model():
    models = find_models()
    if not models:
        print(f"No .gguf models found in {MODELS_ROOT}")
        sys.exit(1)

    print("\n╔══════════════════════════════════════════════════════════════════════╗")
    print("║                        Available models                              ║")
    print("╠══════════════════════════════════════════════════════════════════════╣")
    for i, path in enumerate(models):
        rel = os.path.relpath(path, MODELS_ROOT)
        display = rel if len(rel) <= 68 else "..." + rel[-65:]
        print(f"║  {i+1:2d}. {display:<64}║")
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
        confirm = input("Confirm? (y/n): ").strip().lower()
        if confirm == "y":
            return selected


def print_events():
    print("\n╔═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╗")
    print("║                                                  Available PAPI events                                                  ║")
    print("╠═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╣")
    for i, (name, desc) in enumerate(AVAILABLE_EVENTS):
        print(f"║  {i+1:2d}. {name:<12} — {desc:<100}║")
    print("╚═════════════════════════════════════════════════════════════════════════════════════════════════════════════════════════╝")


def select_events():
    print_events()
    print("\nSelect up to 4 events (enter numbers separated by commas, e.g. 1,2,8,24):")
    while True:
        raw = input("> ").strip()
        parts = [p.strip() for p in raw.split(",")]
        if len(parts) > 4:
            print("Maximum 4 events! Try again.")
            continue
        try:
            indices = [int(p) - 1 for p in parts]
        except ValueError:
            print("Invalid format. Use numbers separated by commas.")
            continue
        if any(i < 0 or i >= len(AVAILABLE_EVENTS) for i in indices):
            print(f"Invalid numbers. Choose between 1 and {len(AVAILABLE_EVENTS)}.")
            continue
        selected = [AVAILABLE_EVENTS[i] for i in indices]
        print("\nYou selected:")
        for name, desc in selected:
            print(f"  ✓ {name} — {desc}")
        confirm = input("\nConfirm? (y/n): ").strip().lower()
        if confirm == "y":
            return selected


def check_binary():
    if not os.path.isfile(BINARY):
        print(f"\nBinary not found at {BINARY}")
        print("Building llama-eval-callback...")
        result = subprocess.run(
            ["cmake", "--build", "build", "--target", "llama-eval-callback",
             f"-j{os.cpu_count()}"],
            cwd=LLAMA_ROOT
        )
        if result.returncode != 0:
            print("Build failed! Make sure eval-callback.cpp is in place and cmake is configured.")
            sys.exit(1)
        print("Build successful!")
    else:
        print(f"\n  ✓ Binary found: {BINARY}")


def run_binary(model_path, events, prompt, n_predict):
    event_names = [e[0] for e in events]
    events_arg = ",".join(event_names)

    cmd = [
        BINARY,
        "--papi-events", events_arg,
        "-m", model_path,
        "-p", prompt,
        "-n", str(n_predict),
        "--log-disable",
    ]

    print(f"\nRunning: {' '.join(cmd)}\n")
    subprocess.run(cmd, cwd=LLAMA_ROOT)


def main():
    print("═" * 66)
    print("   llama.cpp PAPI Profiler")
    print("═" * 66)

    check_binary()
    model_path = select_model()
    events = select_events()

    print("\nEnter your prompt:")
    prompt = input("> ").strip()
    if not prompt:
        prompt = "What is the capital of Sweden?"

    print("How many tokens to generate? (default: 64)")
    raw = input("> ").strip()
    n_predict = int(raw) if raw.isdigit() else 64

    run_binary(model_path, events, prompt, n_predict)


if __name__ == "__main__":
    main()