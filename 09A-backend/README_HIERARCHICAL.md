# Hierarchical LLM Profiling System

A comprehensive profiling and visualization system for analyzing LLM performance across 5 hierarchical levels, from top-level summary down to individual operations.

## Overview

This system provides deep insights into LLM inference performance by:

1. **Capturing detailed metrics** using PAPI hardware counters and timing instrumentation
2. **Organizing data hierarchically** from model-level down to operation-level
3. **Visualizing performance** with clear separation of compute vs memory metrics
4. **Answering key questions** about bottlenecks, phase differences, and component costs

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      llama-probe.cpp                        │
│  C++ profiler with PAPI instrumentation                    │
│  Outputs: measurements.csv                                 │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                   data_processor.py                         │
│  Parses CSV and builds 5-level hierarchy                   │
└─────────────────────┬───────────────────────────────────────┘
                      │
        ┌─────────────┼─────────────┐
        │             │             │
        ▼             ▼             ▼
┌──────────────┐  ┌──────────┐  ┌────────────────┐
│ metrics.py   │  │ visual-  │  │  server.py     │
│ FLOPs, AI    │  │ ization  │  │  FastAPI REST  │
│ IPC, cache   │  │ .py      │  │  API           │
└──────────────┘  └────┬─────┘  └────────────────┘
                       │
                       ▼
                  ┌─────────────────┐
                  │  5 Level Plots  │
                  │  PNG outputs    │
                  └─────────────────┘
```

## 5-Level Hierarchy

### Level 1: Top View
**Whole LLM as one box**

Answers: *How much time and energy does the full run take?*

Metrics:
- Total runtime (seconds)
- Number of input/output tokens
- Throughput (tokens/sec)
- Total energy/power (if available)
- Peak RSS memory
- Total model size
- Total KV-cache size
- Average CPU utilization
- Total cache misses

**Visualization**: Summary card + gauges + bars

---

### Level 2: Phase View
**Separate prefill and decode**

Answers: *How do prefill and decode differ?*

Metrics for each phase:
- Time spent
- FLOPs (estimated)
- Bytes moved
- Arithmetic intensity
- LLC misses and hits
- IPC (Instructions Per Cycle)
- Energy
- Operation type share (pie chart showing % of each operation)
- Core utilization

**Visualization**: Side-by-side comparison bars + operation distribution

---

### Level 3: Decoder Block View
**Show metrics for each decoder block**

Answers: *Which blocks are most expensive?*

Metrics for each block:
- Runtime
- FLOPs
- Bytes moved
- KV-cache footprint
- Arithmetic intensity
- Cache behavior (L1/L2/L3 misses)
- Share of total runtime (%)

**Visualization**: Bar charts per block + pie chart for runtime share

---

### Level 4: Attention vs MLP View
**Break each decoder block into attention and MLP**

Answers: *Is attention or MLP the main bottleneck?*

Metrics for attention and MLP separately:
- Runtime
- FLOPs
- Bytes moved
- Arithmetic intensity
- Cache behavior
- Energy/CPU efficiency

**Visualization**: Grouped bar charts comparing attention vs MLP

---

### Level 5: Layer View
**Finest-grained view for individual operations**

Answers: *Which specific operations are slow? Memory vs compute bound per operation?*

Metrics for each operation:
- Runtime
- FLOPs
- Bytes moved
- Arithmetic intensity
- Cache behavior (detailed L1/L2/L3)
- IPC/utilization
- Operation type (MUL_MAT, ADD, ROPE, etc.)

**Visualization**: Heatmap + detailed table for top operations

---

## Compute vs Memory Distinction

Every level uses **color coding** to distinguish metrics:

- 🔵 **Blue** = Compute-related (FLOPs, IPC, CPU utilization)
- 🔴 **Red** = Memory-related (Bytes moved, cache misses, bandwidth)
- 🟣 **Purple** = Mixed/Other (Arithmetic intensity, total runtime)

This helps identify whether bottlenecks are:
- **Compute-bound**: High arithmetic intensity, low IPC, underutilized compute
- **Memory-bound**: Low arithmetic intensity, high cache misses, memory BW saturated

---

## Files

### Core Modules

| File | Purpose |
|------|---------|
| `llama-probe.cpp` | C++ profiler with PAPI instrumentation |
| `data_processor.py` | Parse CSV and build hierarchical data structure |
| `metrics.py` | Calculate FLOPs, arithmetic intensity, IPC, etc. |
| `visualization.py` | Generate 5-level visualizations |
| `server.py` | FastAPI REST API endpoints |
| `analysis.py` | Roofline model visualization (existing) |

### Supporting Files

| File | Purpose |
|------|---------|
| `papi.py` | Interactive PAPI event selector |
| `requirements.txt` | Python dependencies |
| `CMakeLists.txt` | Build configuration for llama-probe |

---

## Installation

### 1. Build the C++ Profiler

```bash
cd /path/to/profiling-llms-llama-cpp
mkdir -p build && cd build
cmake .. -DLLAMA_BUILD_TESTS=ON
cmake --build . --target llama-probe -j
```

This creates `build/bin/llama-probe`

### 2. Install Python Dependencies

```bash
cd 09A-backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Dependencies:
- pandas (CSV parsing)
- numpy (numerical operations)
- matplotlib (plotting)
- seaborn (enhanced visualizations)
- fastapi + uvicorn (REST API)
- pydantic (data validation)

---

## Usage

### Step 1: Run Profiling

Using the interactive tool:

```bash
cd 09A-backend
source .venv/bin/activate
python papi.py
```

This will:
1. List available models
2. Let you select PAPI events (max 4)
3. Enter a prompt
4. Run profiling and save `measurements.csv`

**Or** run directly:

```bash
../build/bin/llama-probe \
  --papi-events PAPI_TOT_CYC,PAPI_TOT_INS,PAPI_L3_TCM,PAPI_L1_DCM \
  -m /path/to/model.gguf \
  -p "What is the capital of France?" \
  -n 64 \
  --log-disable
```

### Step 2: Generate Visualizations

**Option A: Command-line**

```bash
python visualization.py ../measurements.csv ./plots
```

This generates:
- `level1_top_view.png`
- `level2_phase_view.png`
- `level3_decoder_blocks.png`
- `level4_attention_mlp.png`
- `level5_layer_view.png`

**Option B: Via API**

Start the server:

```bash
python server.py
# Server runs on http://localhost:8000
```

Then access:

```bash
# Generate all visualizations
curl http://localhost:8000/hierarchical-analysis

# Get specific level metrics
curl http://localhost:8000/metrics/phase

# Generate comprehensive report
curl -X POST http://localhost:8000/generate-report \
  -H "Content-Type: application/json" \
  -d '{"include_roofline": true, "hardware": "i7-1185G7"}'
```

### Step 3: View Results

Open the generated PNG files in `plots/` or access via the API:

```bash
# View plots in browser
open http://localhost:8000/plots/level1_top_view.png
open http://localhost:8000/plots/level2_phase_view.png
# etc.
```

---

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/hierarchical-analysis` | GET | Generate all 5 levels + return metrics |
| `/metrics/{level}` | GET | Get metrics for specific level (top/phase/block/attention/layer) |
| `/generate-report` | POST | Generate comprehensive report with all visualizations |
| `/visualization-levels` | GET | Get information about available levels |
| `/plots/{filename}` | GET | Serve generated plot images |
| `/measurements.csv` | GET | Download raw measurements CSV |

### Example Responses

**GET /hierarchical-analysis**

```json
{
  "success": true,
  "plots": {
    "top_view": "/plots/level1_top_view.png",
    "phase_view": "/plots/level2_phase_view.png",
    "block_view": "/plots/level3_decoder_blocks.png",
    "attention_mlp_view": "/plots/level4_attention_mlp.png",
    "layer_view": "/plots/level5_layer_view.png"
  },
  "summary": {
    "total_operations": 1234,
    "num_decoder_blocks": 12,
    "phases": ["prefill", "decode"],
    "total_runtime_sec": 5.432,
    "throughput_tokens_per_sec": 15.6
  },
  "levels": {
    "top_view": { ... },
    "phase_view": { ... },
    "block_view": { ... },
    "attention_mlp_view": { ... },
    "layer_view_sample": [ ... ]
  }
}
```

**GET /metrics/phase**

```json
{
  "success": true,
  "level": "phase",
  "data": {
    "prefill": {
      "time_sec": 2.5,
      "bytes_moved": 1048576000,
      "ipc": 1.25,
      "llc_misses": 5000000,
      "operation_type_share": {
        "MUL_MAT": 45.2,
        "ADD": 15.3,
        "ROPE": 10.1
      }
    },
    "decode": {
      "time_sec": 2.9,
      "bytes_moved": 524288000,
      "ipc": 0.95,
      "llc_misses": 3000000,
      "operation_type_share": { ... }
    }
  }
}
```

---

## Answering Key Questions

The visualization system is designed to answer specific questions:

### 1. How much time and energy does the full run take?

**Level 1 (Top View)** shows:
- `total_runtime_sec`: Total end-to-end time
- `total_energy_joules`: Total energy (if available via power monitoring)
- `throughput_tokens_per_sec`: Overall efficiency

### 2. How do prefill and decode differ?

**Level 2 (Phase View)** compares:
- **Prefill**: Processes all input tokens at once (parallel, high throughput)
  - Higher bytes moved (loading full KV-cache)
  - More compute-bound (batch matmuls)
  - Different operation mix
- **Decode**: Generates one token at a time (sequential, lower throughput)
  - Lower bytes moved per token
  - More memory-bound (reading KV-cache)
  - Focus on autoregressive operations

Look at the side-by-side bars and operation type distributions.

### 3. Which blocks are most expensive?

**Level 3 (Decoder Block View)** shows:
- Runtime bar chart per block
- "Share of total runtime" pie chart
- Top 3 most expensive blocks in the summary table

Typically:
- Early blocks may be faster (smaller KV-cache)
- Later blocks slower (larger KV-cache to attend to)
- But variations depend on model architecture

### 4. Is attention or MLP the main bottleneck?

**Level 4 (Attention vs MLP View)** compares:
- **Attention**: O(n²) with sequence length
  - QKV projections (matmuls)
  - Attention scores (softmax, multiply)
  - Output projection
- **MLP**: O(n) with sequence length
  - Up-projection (d → 4d typically)
  - Activation (GELU, SiLU)
  - Down-projection (4d → d)

Look at:
- Stacked runtime chart (shows total per block)
- Grouped bar charts (compare attention vs MLP directly)
- IPC comparison (lower IPC = more memory-bound)

Generally:
- **Prefill**: Attention dominates (large batch matmuls)
- **Decode**: MLP can dominate (attention is incremental)

### 5. Is the bottleneck computation or memory?

Check multiple indicators:

**Compute-bound indicators:**
- High IPC (> 1.5)
- Low cache miss rate
- High arithmetic intensity (> 10 FLOPs/Byte)
- Roofline: operating near compute ceiling

**Memory-bound indicators:**
- Low IPC (< 1.0)
- High L3 cache miss rate
- Low arithmetic intensity (< 5 FLOPs/Byte)
- High bytes moved relative to FLOPs
- Roofline: operating near memory bandwidth line

Use **Level 5 (Layer View)** to identify specific bottleneck operations:
- MUL_MAT on large matrices: often compute-bound
- ROPE, element-wise ops: often memory-bound
- Small matmuls: memory-bound

---

## Metrics Reference

### Compute Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **FLOPs** | Estimated from op type × elements | Total floating point operations |
| **GFLOP/s** | FLOPs / time_sec / 1e9 | Throughput in billions of FLOP/s |
| **IPC** | Instructions / Cycles | Instructions per cycle (higher = better) |
| **CPU Efficiency** | Actual GFLOP/s / Attainable GFLOP/s | How close to roofline limit |

### Memory Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Bytes Moved** | Tensor sizes + cache_misses × 64 | Data transferred |
| **Memory BW** | Bytes / time_sec / 1e9 | GB/s bandwidth utilization |
| **Cache Misses** | From PAPI counters | L1/L2/L3 miss counts |
| **Cache Hit Rate** | 1 - (misses / accesses) | % of accesses hitting cache |

### Mixed Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Arithmetic Intensity** | FLOPs / Bytes | FLOPs per byte (higher = more compute-bound) |
| **Bottleneck Type** | Compare AI to ridge point | "compute" or "memory" |
| **Ridge Point** | Peak FLOPs / Peak BW | AI threshold for compute vs memory bound |

---

## Customization

### Adding New Metrics

1. **Capture in C++** (`llama-probe.cpp`):
   - Add new PAPI events
   - Add custom timing/instrumentation
   - Write to CSV

2. **Parse in Python** (`data_processor.py`):
   - Update `_build_*_view()` functions
   - Extract from CSV columns

3. **Calculate** (`metrics.py`):
   - Add new calculation functions
   - Update `calculate_all_metrics()`

4. **Visualize** (`visualization.py`):
   - Add to appropriate level's plot function
   - Update color coding (compute=blue, memory=red)

### Adding New Visualization Levels

1. Add data structure in `data_processor.py`
2. Implement `plot_*_view()` in `visualization.py`
3. Add to `generate_all_visualizations()`
4. Create API endpoint in `server.py`

---

## Troubleshooting

### PAPI Events Not Available

Some PAPI events require specific CPU features or permissions:

```bash
# Check available events
papi_avail

# On Linux, may need perf_event permissions
sudo sysctl -w kernel.perf_event_paranoid=-1
```

Common issues:
- `PAPI_L3_TCM` not on all CPUs → use `PAPI_L2_TCM`
- Event conflicts → reduce to 3 events instead of 4
- Permission denied → run with `--privileged` (Docker) or adjust `/proc/sys/kernel/perf_event_paranoid`

### Missing Decoder Blocks

If `block_view` is empty:
- Check tensor names in CSV (need `blk.X` pattern)
- Some models use different naming (e.g., `layers.X`)
- Update `_extract_block_number()` regex in `data_processor.py`

### Visualization Errors

```bash
# Test each level independently
python visualization.py measurements.csv plots/
```

Common issues:
- Missing dependencies → `pip install -r requirements.txt`
- Matplotlib backend issues → check `matplotlib.use('Agg')` is set
- Empty data → check CSV has compute operations (not just metadata)

---

## Performance Tips

### For Faster Profiling

- Reduce `-n` (number of tokens to generate)
- Use smaller model
- Disable PAPI events you don't need
- Run on dedicated system (reduce noise)

### For Faster Analysis

- Process only top N operations in layer view
- Cache parsed data between visualization runs
- Generate only needed levels (not all 5)

---

## Examples

### Example 1: Quick Profile

```bash
# Profile with minimal events
../build/bin/llama-probe \
  --papi-events PAPI_TOT_CYC,PAPI_TOT_INS \
  -m /models/tinyllama-1.1b.gguf \
  -p "Hello" \
  -n 10

# Generate visualizations
python visualization.py ../measurements.csv ./plots

# View top-level summary
python -c "
from data_processor import parse_measurements
data = parse_measurements('../measurements.csv')
print(data.top_view)
"
```

### Example 2: Full Analysis via API

```bash
# Start server
python server.py &

# Run profiling
curl -X POST http://localhost:8000/profile \
  -H "Content-Type: application/json" \
  -d '{
    "model_path": "/models/llama-2-7b.gguf",
    "events": ["PAPI_TOT_CYC", "PAPI_TOT_INS", "PAPI_L3_TCM", "PAPI_FP_OPS"],
    "prompt": "Explain how transformers work.",
    "n_predict": 100
  }'

# Generate analysis
curl http://localhost:8000/hierarchical-analysis | jq

# Download report
curl -X POST http://localhost:8000/generate-report \
  -H "Content-Type: application/json" \
  -d '{"include_roofline": true}' \
  > report.json
```

### Example 3: Compare Prefill vs Decode

```python
from data_processor import parse_measurements

data = parse_measurements('measurements.csv')

prefill = data.phase_view['prefill']
decode = data.phase_view['decode']

print("Prefill:")
print(f"  Time: {prefill['time_sec']:.4f}s")
print(f"  IPC: {prefill['ipc']:.3f}")
print(f"  LLC Misses: {prefill['llc_misses'] / 1e6:.2f}M")

print("\nDecode:")
print(f"  Time: {decode['time_sec']:.4f}s")
print(f"  IPC: {decode['ipc']:.3f}")
print(f"  LLC Misses: {decode['llc_misses'] / 1e6:.2f}M")

print(f"\nDecode is {decode['time_sec'] / prefill['time_sec']:.2f}x slower")
```

---

## Contributing

To extend this system:

1. **Add new hardware configs** in `analysis.py` (HARDWARE dict)
2. **Improve FLOP estimation** in `metrics.py` (operation-specific formulas)
3. **Add energy monitoring** (integrate RAPL or external power meters)
4. **Create frontend** (React app consuming the REST API)
5. **Add PDF report generation** (use matplotlib.backends.backend_pdf)

---

## References

- [Roofline Model](https://en.wikipedia.org/wiki/Roofline_model)
- [PAPI Performance API](http://icl.utk.edu/papi/)
- [llama.cpp](https://github.com/ggml-org/llama.cpp)
- [Transformer Architecture](https://arxiv.org/abs/1706.03762)

---

## License

Same as llama.cpp (MIT)
