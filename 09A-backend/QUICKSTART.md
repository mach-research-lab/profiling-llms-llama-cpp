# Quick Start Guide - Hierarchical LLM Profiling

Get started with hierarchical LLM profiling in 3 simple steps.

## Prerequisites

- Built llama.cpp with `llama-probe` target
- Python 3.12+ with venv
- PAPI library installed (`libpapi-dev` on Ubuntu/Debian)

## Step 1: Build the Profiler (One-time)

```bash
cd /home/spiderman/profiling-llms-llama-cpp

# Configure and build
mkdir -p build && cd build
cmake .. -DLLAMA_BUILD_TESTS=ON
cmake --build . --target llama-probe -j$(nproc)

# Verify
./bin/llama-probe --help
```

## Step 2: Install Python Environment (One-time)

```bash
cd /home/spiderman/profiling-llms-llama-cpp/09A-backend

# Create virtual environment
python3 -m venv .venv

# Activate it
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Step 3: Run Profiling & Generate Visualizations

### Option A: Interactive Mode (Recommended for first-time)

```bash
cd /home/spiderman/profiling-llms-llama-cpp/09A-backend
source .venv/bin/activate

# Run interactive profiler
python papi.py
```

This will guide you through:
1. Selecting a model from `~/shared/models`
2. Choosing PAPI events (e.g., cycles, instructions, cache misses)
3. Entering a prompt
4. Running the profiler

Output: `measurements.csv` in the parent directory

### Option B: Direct Command

```bash
cd /home/spiderman/profiling-llms-llama-cpp

# Run profiling directly
./build/bin/llama-probe \
  --papi-events PAPI_TOT_CYC,PAPI_TOT_INS,PAPI_L3_TCM,PAPI_L1_DCM \
  -m ~/shared/models/your-model.gguf \
  -p "What is the capital of France?" \
  -n 32 \
  --log-disable
```

### Generate Visualizations

```bash
cd 09A-backend
source .venv/bin/activate

# Generate all 5 levels
python visualization.py ../measurements.csv ./plots

# Check output
ls -lh plots/
```

You should see:
- `level1_top_view.png` - Overall summary
- `level2_phase_view.png` - Prefill vs Decode
- `level3_decoder_blocks.png` - Per-block analysis
- `level4_attention_mlp.png` - Attention vs MLP
- `level5_layer_view.png` - Individual operations

### View Results

```bash
# Open plots (Linux with GUI)
xdg-open plots/level1_top_view.png

# Or copy to shared directory for viewing elsewhere
cp plots/*.png /srv/shared/
```

## Step 4 (Optional): Use the Web API

```bash
cd 09A-backend
source .venv/bin/activate

# Start server
python server.py
# Server runs on http://localhost:8000
```

### Available Endpoints

Open in browser or use curl:

```bash
# Get all visualization levels info
curl http://localhost:8000/visualization-levels | jq

# Generate full hierarchical analysis
curl http://localhost:8000/hierarchical-analysis | jq

# Get specific level metrics
curl http://localhost:8000/metrics/phase | jq
curl http://localhost:8000/metrics/block | jq

# Generate comprehensive report
curl -X POST http://localhost:8000/generate-report \
  -H "Content-Type: application/json" \
  -d '{"include_roofline": true, "hardware": "i7-1185G7"}' | jq

# View plots in browser
# Open: http://localhost:8000/plots/level1_top_view.png
```

## Common PAPI Event Combinations

### Basic (Always Works)
```
PAPI_TOT_CYC,PAPI_TOT_INS
```
Gives you IPC (Instructions Per Cycle)

### Cache Analysis
```
PAPI_TOT_CYC,PAPI_TOT_INS,PAPI_L3_TCM,PAPI_L1_DCM
```
Adds L3 and L1 cache misses

### Compute Focus
```
PAPI_TOT_CYC,PAPI_TOT_INS,PAPI_FP_OPS,PAPI_VEC_INS
```
Tracks floating point ops and SIMD instructions

### Memory Focus
```
PAPI_TOT_CYC,PAPI_L1_TCM,PAPI_L2_TCM,PAPI_L3_TCM
```
Full cache hierarchy

### Check Available Events
```bash
papi_avail | grep -i "cache\|fp\|vec"
```

## Understanding the Visualizations

### Level 1: Top View
**What it shows:** Overall model performance
**Key metrics:** Total runtime, throughput, model size
**Color coding:** Blue (compute), Red (memory), Purple (mixed)

### Level 2: Phase View
**What it shows:** Prefill vs Decode comparison
**Key question:** Which phase is slower and why?
**Look for:** Time bars, IPC comparison, operation type distribution

### Level 3: Decoder Block View
**What it shows:** Per-block metrics across all layers
**Key question:** Which blocks dominate runtime?
**Look for:** Runtime bars, share of total runtime pie chart

### Level 4: Attention vs MLP
**What it shows:** Component breakdown within blocks
**Key question:** Is attention or MLP the bottleneck?
**Look for:** Grouped bars comparing attention (green) vs MLP (orange)

### Level 5: Layer View
**What it shows:** Individual operations (top 50 by runtime)
**Key question:** Which specific ops are slow?
**Look for:** Heatmap showing runtime/bytes/IPC/cache for each op

## Interpreting Results

### Compute-Bound Indicators
✅ High IPC (> 1.5)
✅ Low cache miss rate
✅ High arithmetic intensity
➡️ **Action:** Optimize algorithms, use better SIMD, reduce FLOPs

### Memory-Bound Indicators
❌ Low IPC (< 1.0)
❌ High L3 cache misses
❌ Low arithmetic intensity
➡️ **Action:** Improve data locality, cache blocking, reduce memory traffic

## Troubleshooting

### "PAPI: unknown event"
Some events not available on your CPU. Try basic set:
```
--papi-events PAPI_TOT_CYC,PAPI_TOT_INS
```

### "Permission denied" for PAPI
```bash
sudo sysctl -w kernel.perf_event_paranoid=-1
# Or run with sudo (not recommended)
```

### No decoder blocks detected
Model uses different tensor naming. Check `measurements.csv` for patterns like:
- `blk.0.*` (standard)
- `layers.0.*` (alternative)
- `transformer.h.0.*` (GPT-style)

Update regex in `data_processor.py::_extract_block_number()` if needed.

### Empty visualizations
Check `measurements.csv` has compute operations (not just metadata rows).
Should have rows with phase=prefill or phase=decode and non-n/a op_type.

## Next Steps

1. **Compare models:** Run profiling on different model sizes
2. **Compare quantization:** Profile Q4 vs Q8 vs F16
3. **Optimize:** Use insights to guide optimization efforts
4. **Automate:** Create scripts to profile multiple configs
5. **Extend:** Add custom metrics or new visualization levels

## Example Output Interpretation

```
Level 1 (Top View):
  Total Runtime: 5.23s
  Throughput: 12.3 tokens/s
  Total Cache Misses: 45M
  ➡️ Overall, ~12 tokens/s throughput

Level 2 (Phase View):
  Prefill: 2.1s, IPC=1.8
  Decode: 3.1s, IPC=0.9
  ➡️ Decode is slower and more memory-bound (lower IPC)

Level 3 (Block View):
  Block 0: 0.4s (7.6%)
  Block 11: 0.5s (9.5%)
  ➡️ Later blocks slightly slower (larger KV-cache)

Level 4 (Attention vs MLP):
  Attention: 2.8s total
  MLP: 2.4s total
  ➡️ Attention slightly dominates

Level 5 (Layer View):
  Top op: MUL_MAT (0.12s, IPC=1.5)
  ➡️ Matrix multiplications are the hotspot, reasonably compute-bound
```

## Help & Support

- Full documentation: `README_HIERARCHICAL.md`
- Metrics reference: See "Metrics Reference" section in README
- API docs: http://localhost:8000/docs (when server running)

## License

MIT (same as llama.cpp)
