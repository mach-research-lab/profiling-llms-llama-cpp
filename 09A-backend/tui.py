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
import shutil

# Homemade module
from event_retriever import get_available_events, get_valid_runs

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
LLAMA_ROOT  = os.path.dirname(SCRIPT_DIR)
MODELS_ROOT = os.path.join(os.path.expanduser("~"), "shared/models")

AVAILABLE_EVENTS = get_available_events()

#Used for selecting storage type (database, file, or both)
def select_storage_type():
    storage_types = {
        "1": ("file", "Save to CSV file only"),
        "2": ("database", "Save to SQLite database only"),
        "3": ("both", "Save to both CSV file and database"),
    }

    print("\nSelect storage type:")
    for key, (_, desc) in storage_types.items():
        print(f"  {key}. {desc}")

    while True:
        raw = input("> ").strip()
        if raw not in storage_types:
            print("Invalid choice. Enter 1, 2, or 3.")
            continue

        key, desc = storage_types[raw]
        print(f"\n  ✓ {desc}")
        confirm = input("Confirm? (y/n): ").strip().lower()
        if confirm == "y":
            return key

#Used for selecting run type (PAPI events, energy, KV cache or everything)
def select_run_type():
    run_types = {
        "1": ("single", "PAPI events measurement (single batch)"),
        "2": ("energy", "Energy measurement"),
        "3": ("kv",     "KV cache footprint"),
        "4": ("all",    "Run all (Multi-batch with event groups)"),
        "5": ("conversation", "Conversation mode with PAPI events"),
        "6": ("TOP-VIEW", "Run TOP-VIEW measurements with PAPI events (experimental)"),
        "7": ("PHASE-VIEW", "Run PHASE-VIEW measurements with PAPI events (experimental)"),
        "8": ("DECODER-BLOCK-VIEW", "Run DECODER-BLOCK-VIEW measurements with PAPI events (experimental)")
    }

    print("\nSelect run type:")
    for key, (_, desc) in run_types.items():
        print(f"  {key}. {desc}")

    while True:
        raw = input("> ").strip()
        if raw not in run_types:
            print("Invalid choice. Enter 1, 2, 3, 4, 5, 6, 7, or 8.")
            continue

        key, desc = run_types[raw]
        print(f"\n  ✓ {desc}")
        confirm = input("Confirm? (y/n): ").strip().lower()
        if confirm == "y":
            return key


#--------- PAPI events measurement ---------

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

#--------- KV cache dtype selection  ---------
def select_kv_cache_type():
    cache_types = {
        "1": ("f16",  "FP16 (default, recommended)"),
        "2": ("f32",  "FP32 (higher precision, more memory)"),
        "3": ("q8_0", "Q8_0 (quantized, less memory)"),
        "4": ("q4_0", "Q4_0 (quantized, least memory)"),
    }

    print("\nUse default KV cache types? (F16 for both K and V)")
    print("  1. Yes, use defaults")
    print("  2. No, specify manually")
    while True:
        raw = input("> ").strip()
        if raw == "" or raw == "1":
            print("\n  ✓ K-cache: f16, V-cache: f16 (defaults)")
            return "f16", "f16"
        elif raw == "2":
            break
        else:
            print("Invalid choice. Enter 1 or 2.")

    selected = {}
    for cache in ("K", "V"):
        print(f"\nSelect {cache}-cache data type:")
        for key, (_, desc) in cache_types.items():
            print(f"  {key}. {desc}")

        while True:
            raw = input("> ").strip()
            if raw == "":
                raw = "1"  # default to f16
            if raw not in cache_types:
                print("Invalid choice. Enter 1, 2, 3, or 4.")
                continue
            cache_type, desc = cache_types[raw]
            print(f"\n  ✓ {desc}")
            confirm = input("Confirm? (y/n): ").strip().lower()
            if confirm == "y":
                selected[cache] = cache_type
                break

    print(f"\n  ✓ K-cache: {selected['K']}, V-cache: {selected['V']}")
    return selected["K"], selected["V"]


#--------- Common functions ---------
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

def check_binary(binary_path):
    if not os.path.isfile(binary_path):
        return False
    else:
        print(f"\n  ✓ Binary found: {binary_path}")
        return True

#Used for running llama-papi
def run_single_papi(model_path, events, prompt, n_predict, binary_path, storage_type="file", db_path="profiling_data.db"):
    event_names = [e[0] for e in events]
    events_arg = ",".join(event_names)

    cmd = [
        binary_path,
        "--papi-events", events_arg,
        "--result-path", "measurements.csv",
        "-m", model_path,
        "-p", prompt,
        "-n", str(n_predict),
        "--temp", "0",  # fixed temp for consistent measurements
        "--log-disable",
    ]

    # Add database flags based on storage type
    if storage_type in ["database", "both"]:
        cmd.extend(["--use-db", "--db-path", db_path])

    print(f"\nRunning: {' '.join(cmd)}\n")
    subprocess.run(cmd, cwd=LLAMA_ROOT)

#Used for running llama-papi in conversation mode
def run_conversation_papi(model_path, events, prompt, n_predict, binary_path, storage_type="file", db_path="profiling_data.db"):
    event_names = [e[0] for e in events]
    events_arg = ",".join(event_names)

    cmd = [
        binary_path,
        "--papi-events", events_arg,
        "--result-path", "conversation_measurements.csv",
        "--conversation",  # Enable conversation mode
        "-m", model_path,
        "-p", prompt,
        "-n", str(n_predict),
        "--temp", "0",  # fixed temp for consistent measurements
        "--log-disable",
    ]

    # Add database flags based on storage type
    if storage_type in ["database", "both"]:
        cmd.extend(["--use-db", "--db-path", db_path])

    print(f"\nStarting conversation mode...")
    print(f"Running: {' '.join(cmd)}\n")
    print("You can type 'quit' or 'exit' to end the conversation.\n")

    # Run interactively so user can input multiple turns
    subprocess.run(cmd, cwd=LLAMA_ROOT)

#Used for running llama-papi in TOP-VIEW mode (experimental)
def run_top_view_papi(model_path, events, prompt, n_predict, k_cache_type, v_cache_type, binary_path):
    event_names = [e[0] for e in events]
    events_arg = ",".join(event_names)

    cmd = [
        binary_path,
        "--papi-events", events_arg,
        "--result-path", "top_view_measurements.json",
        "--conversation",  # Enable conversation mode for TOP-VIEW
        "-m", model_path,
        "-p", prompt,
        "-n", str(n_predict),
        "--cache-type-k", k_cache_type,
        "--cache-type-v", v_cache_type,
        "--temp", "0",  # fixed temp for consistent measurements
        "--log-disable",
    ]

    print(f"\nStarting TOP-VIEW measurement with PAPI events...")
    print(f"Running: {' '.join(cmd)}\n")
    print("You can type 'quit' or 'exit' to end the conversation.\n")

    # Run interactively so user can input multiple turns
    subprocess.run(cmd, cwd=LLAMA_ROOT)

def run_phase_view_papi(model_path, events, prompt, n_predict, binary_path):
    event_names = [e[0] for e in events]
    events_arg = ",".join(event_names)

    cmd = [
        binary_path,
        "--papi-events", events_arg,
        "--result-path", "phase_view_measurements.json",
        "--conversation",  # Enable conversation mode for PHASE-VIEW
        "-m", model_path,
        "-p", prompt,
        "-n", str(n_predict),
        "--temp", "0",  # fixed temp for consistent measurements
        "--log-disable",
    ]

    print(f"\nStarting PHASE-VIEW measurement with PAPI events...")
    print(f"Running: {' '.join(cmd)}\n")
    print("You can type 'quit' or 'exit' to end the conversation.\n")

    # Run interactively so user can input multiple turns
    subprocess.run(cmd, cwd=LLAMA_ROOT)

def run_decoder_block_view_papi(model_path, events, prompt, n_predict, binary_path):
    event_names = [e[0] for e in events]
    events_arg = ",".join(event_names)
    result_path = os.path.abspath("../decoder_block_view_measurements.json")

    cmd = [
        binary_path,
        "--papi-events", events_arg,
        "--result-path", result_path,
        "--conversation",  # Enable conversation mode for DECODER-BLOCK-VIEW
        "-m", model_path,
        "-p", prompt,
        "-n", str(n_predict),
        "--temp", "0",  # fixed temp for consistent measurements
        "--log-disable",
    ]

    print(f"\nStarting DECODER-BLOCK-VIEW measurement with PAPI events...")
    print(f"Running: {' '.join(cmd)}\n")
    print("You can type 'quit' or 'exit' to end the conversation.\n")

    # Run interactively so user can input multiple turns
    subprocess.run(cmd, cwd=LLAMA_ROOT)



#Used for running kv-measure
def run_kv_measurement(model_path, prompt, n_predict, k_cache_type, v_cache_type, binary_path):
    cmd = [
        binary_path,
        "--result-path", "kv_sizes.csv",
        "-m", model_path,
        "-p", prompt,
        "-n", str(n_predict),
        "--cache-type-k", k_cache_type,
        "--cache-type-v", v_cache_type,
        "--temp", "0",  # fixed temp for consistent measurements
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

#Used for running llama-energy
def run_energy(model_path, prompt, n_predict, binary_path):
    cmd = [
        binary_path,
        "--result-path", "energy.csv",
        "-m", model_path,
        "-p", prompt,
        "-n", str(n_predict),
        "--temp", "0",  # fixed temp for consistent measurements
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



# --------- RUN ALL (Multi-batch with event groups) ---------
def run_all(model_path, prompt, n_predict, k_cache_type, v_cache_type, storage_type="file", db_path="profiling_data.db"):
    # Remove existing directory and recreate it fresh
    output_dir = os.path.join(LLAMA_ROOT, "run_all_results")
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir)

    # ---- Running PAPI events in groups -----

    binary_path = os.path.join(LLAMA_ROOT, "build/bin/llama-papi")
    #Get event groups for multibatch runs
    event_groups = get_valid_runs()
    for group in event_groups:
        print(f"\nEvent group: {group}")
    print(f"\nRunning {len(event_groups)} event groups for multi-batch measurement...")
    for i, group in enumerate(event_groups):
        events_arg = ",".join(group)
        csv_path = os.path.join(output_dir, f"events_group_{i+1}.csv")

        cmd = [
            binary_path,
            "--papi-events", events_arg,
            "--result-path", csv_path,
            "--papi-events-unrestricted",  # allow all events for multibatch runs
            "-m", model_path,
            "-p", prompt,
            "-n", str(n_predict),
            "--temp", "0",  # fixed temp for consistent measurements
            "--log-disable",
        ]

        # Add database flags based on storage type
        if storage_type in ["database", "both"]:
            cmd.extend(["--use-db", "--db-path", db_path])

        print(f"\nRunning group {i+1}/{len(event_groups)}: {' '.join(cmd)}\n")
        result = subprocess.run(cmd, cwd=LLAMA_ROOT)

        if result.returncode != 0:
            print(f"\n  ✗ Group {i+1} failed (returncode {result.returncode})")
            sys.exit(1)

        if os.path.isfile(csv_path):
            print(f"\n  ✓ Group {i+1} results saved to: {csv_path}")
        else:
            print(f"\n  ✗ {csv_path} not found after run.")
    
    # ---- Running energy -----

    binary_path = os.path.join(LLAMA_ROOT, "build/bin/llama-energy")
    csv_path = os.path.join(output_dir, f"energy.csv")

    cmd = [
        binary_path,
        "--result-path", csv_path,
        "-m", model_path,
        "-p", prompt,
        "-n", str(n_predict),
        "--temp", "0",  # fixed temp for consistent measurements
        "--log-disable",
    ]

    print(f"\nRunning: {' '.join(cmd)}\n")
    result_energy = subprocess.run(cmd, cwd=LLAMA_ROOT)

    if result_energy.returncode != 0:
        print(f"\n  ✗ llama-energy failed (returncode {result_energy.returncode})")
        sys.exit(1)
    
    
    if os.path.isfile(csv_path):
        print(f"\n  ✓ Results saved to: {csv_path}")
    else:
        print(f"\n  ✗ energy.csv not found after run.")

    # ---- Running KV cache footprint -----

    binary_path = os.path.join(LLAMA_ROOT, "build/bin/kv-measure")
    csv_path = os.path.join(output_dir, f"kv_measurement.csv")
    cmd = [
        binary_path,
        "--result-path", csv_path,
        "-m", model_path,
        "-p", prompt,
        "-n", str(n_predict),
        "--cache-type-k", k_cache_type,
        "--cache-type-v", v_cache_type,
        "--temp", "0",  # fixed temp for consistent measurements
        "--log-disable",
    ]

    print(f"\nRunning: {' '.join(cmd)}\n")
    result_kv = subprocess.run(cmd, cwd=LLAMA_ROOT)

    if result_kv.returncode != 0:
        print(f"\n  ✗ kv-measure failed (returncode {result_kv.returncode})")
        sys.exit(1)
    
    
    if os.path.isfile(csv_path):
        print(f"\n  ✓ Results saved to: {csv_path}")
    else:
        print(f"\n  ✗ kv_measurement.csv not found after run.")


def main():
    print("═" * 66)
    print("   llama.cpp Profiler")
    print("═" * 66)

    binary_path = None

    run_type = select_run_type()
    print(run_type)

    # Select storage type for PAPI-based measurements
    storage_type = "file"  # Default for energy and KV measurements
    db_path = os.path.join(LLAMA_ROOT, "profiling_data.db")

    if run_type in ["single", "conversation", "all"]:
        storage_type = select_storage_type()

    #Add path to the correct binary based on the run type
    if run_type == "energy":
        binary_path = os.path.join(LLAMA_ROOT, "build/bin/llama-energy")
    elif run_type == "kv":
        binary_path = os.path.join(LLAMA_ROOT, "build/bin/kv-measure")
    elif run_type == "single":
        binary_path  = os.path.join(LLAMA_ROOT, "build/bin/llama-papi")
    elif run_type == "all":
        binary_path = os.path.join(LLAMA_ROOT, "build/bin/llama-papi")
    elif run_type == "conversation":
        binary_path = os.path.join(LLAMA_ROOT, "build/bin/llama-papi")
    elif run_type == "TOP-VIEW":
        binary_path = os.path.join(LLAMA_ROOT, "build/bin/llama-measurement-top-view")
    elif run_type == "PHASE-VIEW":
        binary_path = os.path.join(LLAMA_ROOT, "build/bin/llama-measurement-phase-view")
    elif run_type == "DECODER-BLOCK-VIEW":
        binary_path = os.path.join(LLAMA_ROOT, "build/bin/llama-measurement-decoder-block-view")

    # Check if the selected binary exists
    if(not check_binary(binary_path)):
        print(f"Binary not found: {binary_path}")
        print("Please compile binaries in /09A-backend/llama-measuring first.")
        sys.exit(1)

    model_path = select_model()

    #If single batch with PAPI events or conversation mode, allow event selection. Otherwise skip to prompt input.
    events = []
    if run_type == "single" or run_type == "conversation" or run_type == "TOP-VIEW" or run_type == "PHASE-VIEW" or run_type == "DECODER-BLOCK-VIEW":
        events = select_events()

    #If KV cache measurement, detect cache type from model name. Otherwise skip to prompt input.
    k_cache_type = None
    v_cache_type = None
    if run_type == "kv" or run_type == "all" or run_type == "TOP-VIEW":
        k_cache_type, v_cache_type = select_kv_cache_type()
        print(f"\nSelected KV cache type: {k_cache_type}, {v_cache_type}")

    # ---- COMMON PROMPT AND N_PREDICT INPUT ----
    print("\nEnter your initial prompt:")
    prompt = input("> ").strip()
    if not prompt:
        prompt = "What is the capital of Sweden?"

    print("How many tokens to generate per response? (default: 64)")
    raw = input("> ").strip()
    n_predict = int(raw) if raw.isdigit() else 64

    if run_type == "single":
        run_single_papi(model_path, events, prompt, n_predict, binary_path, storage_type, db_path)
    elif run_type == "kv":
        run_kv_measurement(model_path, prompt, n_predict, k_cache_type, v_cache_type, binary_path)
    elif run_type == "energy":
        run_energy(model_path, prompt, n_predict, binary_path)
    elif run_type == "all":
        run_all(model_path, prompt, n_predict, k_cache_type, v_cache_type, storage_type, db_path)
    elif run_type == "conversation":
        run_conversation_papi(model_path, events, prompt, n_predict, binary_path, storage_type, db_path)
    elif run_type == "TOP-VIEW":
        run_top_view_papi(model_path, events, prompt, n_predict, k_cache_type, v_cache_type, binary_path)
    elif run_type == "PHASE-VIEW":
        run_phase_view_papi(model_path, events, prompt, n_predict, binary_path)
    elif run_type == "DECODER-BLOCK-VIEW":
        run_decoder_block_view_papi(model_path, events, prompt, n_predict, binary_path)

   


if __name__ == "__main__":
    main()