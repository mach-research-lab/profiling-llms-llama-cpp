
from dataclasses import dataclass
import json
import re
import subprocess
import sys
import os
import glob
import shutil
from enum import Enum
# Homemade module
from event_retriever import  get_valid_runs_from_list
from measurements_complementation import complement_phase_json, complement_decoder_block_json

SCRIPT_DIR  = os.path.dirname(os.path.abspath(__file__))
LLAMA_ROOT  = os.path.dirname(SCRIPT_DIR)
MODELS_ROOT = os.path.join(LLAMA_ROOT, "models")
OUTPUT_PATH = os.path.join(LLAMA_ROOT, "run_every_view_results")

#Data class holding arguments
@dataclass
class Config:
    model_path: str
    custom_events: list
    prompt: str
    n_predict: int
    k_cache_type: str
    v_cache_type: str
    binary_path: str



#PAPI events for each runtype
TOP_VIEW_PAPI_EVENTS           = ["PAPI_L3_TCM", "PAPI_L2_TCM", "PAPI_L1_TCM"]
PHASE_VIEW_PAPI_EVENTS         = ["PAPI_FP_OPS", "PAPI_L3_TCM", "PAPI_L3_TCA", "PAPI_TOT_INS", "PAPI_TOT_CYC"]
DECODER_BLOCK_VIEW_PAPI_EVENTS = [
    # L1 Cache
    "PAPI_L1_DCM",  # L1 data cache misses
    "PAPI_L1_ICM",  # L1 instruction cache misses
    "PAPI_L1_TCM",  # L1 total cache misses

    # L2 Cache
    "PAPI_L2_DCM",  # L2 data cache misses
    "PAPI_L2_ICM",  # L2 instruction cache misses
    "PAPI_L2_TCM",  # L2 total cache misses
    "PAPI_L2_DCA",  # L2 data cache accesses
    "PAPI_L2_TCA",  # L2 total cache accesses
    "PAPI_L2_DCR",  # L2 data cache reads
    "PAPI_L2_TCR",  # L2 total cache reads
    "PAPI_L2_LDM",  # L2 load misses

    # L3 Cache
    "PAPI_L3_DCM",  # L3 data cache misses
    "PAPI_L3_ICM",  # L3 instruction cache misses
    "PAPI_L3_TCM",  # L3 total cache misses  ← already in your list
    "PAPI_L3_DCA",  # L3 data cache accesses
    "PAPI_L3_TCA",  # L3 total cache accesses ← already in your list
    "PAPI_L3_DCR",  # L3 data cache reads
    "PAPI_L3_TCR",  # L3 total cache reads
    "PAPI_L3_LDM",  # L3 load misses

    # TLB (impacts effective cache behavior)
    "PAPI_TLB_DM",  # Data TLB misses
    "PAPI_TLB_IM",  # Instruction TLB misses
    "PAPI_TLB_TL",  # Total TLB misses

    # FLOPS
    "PAPI_FP_OPS",

    #IPC
    "PAPI_TOT_INS",
    "PAPI_TOT_CYC"
]
TENSOR_OP_VIEW_PAPI_EVENTS     = [
    # L1 Cache
    "PAPI_L1_DCM",  # L1 data cache misses
    "PAPI_L1_ICM",  # L1 instruction cache misses
    "PAPI_L1_TCM",  # L1 total cache misses

    # L2 Cache
    "PAPI_L2_DCM",  # L2 data cache misses
    "PAPI_L2_ICM",  # L2 instruction cache misses
    "PAPI_L2_TCM",  # L2 total cache misses
    "PAPI_L2_DCA",  # L2 data cache accesses
    "PAPI_L2_TCA",  # L2 total cache accesses
    "PAPI_L2_DCR",  # L2 data cache reads
    "PAPI_L2_TCR",  # L2 total cache reads
    "PAPI_L2_LDM",  # L2 load misses

    # L3 Cache
    "PAPI_L3_DCM",  # L3 data cache misses
    "PAPI_L3_ICM",  # L3 instruction cache misses
    "PAPI_L3_TCM",  # L3 total cache misses  ← already in your list
    "PAPI_L3_DCA",  # L3 data cache accesses
    "PAPI_L3_TCA",  # L3 total cache accesses ← already in your list
    "PAPI_L3_DCR",  # L3 data cache reads
    "PAPI_L3_TCR",  # L3 total cache reads
    "PAPI_L3_LDM",  # L3 load misses

    # TLB (impacts effective cache behavior)
    "PAPI_TLB_DM",  # Data TLB misses
    "PAPI_TLB_IM",  # Instruction TLB misses
    "PAPI_TLB_TL",  # Total TLB misses

       # FLOPS
    "PAPI_FP_OPS",

    #IPC
    "PAPI_TOT_INS",
    "PAPI_TOT_CYC"
]


#Enum class used for specifying run type
class Run_type(Enum):
    TOP_VIEW           = ("top-view.json", TOP_VIEW_PAPI_EVENTS)
    PHASE_VIEW         = ("phase-view.json", PHASE_VIEW_PAPI_EVENTS)
    DECODER_BLOCK_VIEW = ("decoder-block-view.json", DECODER_BLOCK_VIEW_PAPI_EVENTS)
    TENSOR_OP_VIEW     = ("tensor_op_view.db", TENSOR_OP_VIEW_PAPI_EVENTS) #Regular llama-papi

    def __init__(self, path, events):
        self.path = path
        self.events = events


#Helper to read previous prompts from generated JSON
def get_user_prompts(result_path: str, default_prompt: str = "hello") -> str:
    prompts_path = os.path.join(os.path.dirname(result_path), "collected_prompts.json")
    if not os.path.exists(prompts_path):
        return json.dumps([default_prompt, "quit"])
    try:
        with open(prompts_path) as f:
            return json.dumps(json.load(f)["prompts"])
    except Exception:
        return json.dumps([default_prompt, "quit"])


#Function used to run specified view, is multibatch
def run_view(config: Config, clean_folder: bool, type: Run_type, gather_prompts, event_per_run: int | None = None):

    #Output storage
    output_dir = OUTPUT_PATH
    #If clean_folder is true, delete old run measurements and create new folder
    if clean_folder:
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        os.makedirs(output_dir)

    result_path = os.path.join(output_dir, type.path)

    #If no custom PAPI events are given, run ENUM default events
    if config.custom_events is None:
        event_groups = get_valid_runs_from_list(type.events, event_per_run)
    else:
        event_groups = get_valid_runs_from_list(config.custom_events, event_per_run)
        print(f"Custom events added: {config.custom_events}")

    #Run multiple times to collect all events
    for i, group in enumerate(event_groups):

        events_arg = ",".join(group)

        cmd = [
                config.binary_path,
                "--papi-events", events_arg,
                "--result-path", result_path,
                "--papi-events-unrestricted",  # allow all events for multibatch runs
                "--conversation",              # Mulitple prompts
                "-m", config.model_path,
                "-p", config.prompt,
                "-n", str(config.n_predict),
                "--cache-type-k", config.k_cache_type,
                "--cache-type-v", config.v_cache_type,
                "--temp", "0",  # fixed temp for consistent measurements
                "--log-disable",
        ]

        if gather_prompts:
            gather_prompts = False #Change to false so we don't have prints or collection for next iteration
            cmd.extend(["--collect-prompts"])
            
        else: 
            cmd.extend(["--user-prompts", get_user_prompts(result_path, config.prompt),"--disable-prints"])
            
        # Add database flags based on storage type
        if type == Run_type.TENSOR_OP_VIEW:
            cmd.extend(["--no-csv", "--use-db", "--db-path", result_path])

        print(f"\nRunning group {i+1}/{len(event_groups)}: {' '.join(cmd)}\n")
        subprocess.run(cmd, cwd=LLAMA_ROOT)
        

#Runs every view with conversation mode, is multibatch
def run_every_view(config: Config, event_per_tensor: int | None):

    #PHASE VIEW START
    print("\n" + "="*5 + " Starting phase view " + "="*5 + "\n")
    config.binary_path = os.path.join(LLAMA_ROOT, "build/bin/llama-measurement-phase-view")
    run_view(config, True, Run_type.PHASE_VIEW, True)

    #TOP VIEW START
    print("\n" + "="*5 + " Starting top view " + "="*5 + "\n")
    config.binary_path = os.path.join(LLAMA_ROOT, "build/bin/llama-measurement-top-view")
    run_view(config, False, Run_type.TOP_VIEW, False)

    #DECODER-BLOCKS VIEW START
    print("\n" + "="*5 + " Starting decoder block view " + "="*5 + "\n")
    config.binary_path = os.path.join(LLAMA_ROOT, "build/bin/llama-measurement-decoder-block-view")
    run_view(config, False, Run_type.DECODER_BLOCK_VIEW, False)

    #TENSOR-OP VIEW START
    print("\n" + "="*5 + " Starting decoder block view " + "="*5 + "\n")
    config.binary_path = os.path.join(LLAMA_ROOT, "build/bin/llama-papi")
    run_view(config, False, Run_type.TENSOR_OP_VIEW, False, event_per_tensor)

    #Complement JSON:s with additional fields
    try:
        complement_phase_json(os.path.join(OUTPUT_PATH, Run_type.PHASE_VIEW.path),
                              os.path.join(OUTPUT_PATH, Run_type.TENSOR_OP_VIEW.path),
                              None, os.path.join(OUTPUT_PATH, Run_type.PHASE_VIEW.path))
    except Exception as pe:
        print(f"Warning: Failed to complement phase-view.json: {pe}")

    try:
        complement_decoder_block_json(os.path.join(OUTPUT_PATH, Run_type.DECODER_BLOCK_VIEW.path),
                                  os.path.join(OUTPUT_PATH, Run_type.TENSOR_OP_VIEW.path),
                                  None,
                                  os.path.join(OUTPUT_PATH, Run_type.DECODER_BLOCK_VIEW.path))
    except Exception as de:
        print(f"Warning: Failed to complement decoder-block-view.json: {de}")
    

"""
#used for test run
cfg_test = Config(
    model_path=os.path.join(MODELS_ROOT, "Llama3.2/Llama-3.2-1B-Instruct-Q4_K_M.gguf"),
    custom_events= None,
    prompt="hello",
    n_predict=64,
    k_cache_type="f16",
    v_cache_type="f16",
    binary_path="",
)

run_every_view(cfg_test)

"""