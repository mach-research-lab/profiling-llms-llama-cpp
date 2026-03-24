#!/usr/bin/env python3
"""
Data processor for hierarchical LLM profiling analysis.
Parses measurements.csv and aggregates metrics across 5 levels:
1. Top view (whole model)
2. Phase view (prefill vs decode)
3. Decoder block view
4. Attention vs MLP view
5. Layer view (individual operations)
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import re


class ProfilingData:
    """Structured container for hierarchical profiling data."""

    def __init__(self):
        self.raw_df: Optional[pd.DataFrame] = None
        self.model_info: Dict = {}
        self.top_view: Dict = {}
        self.phase_view: Dict = {}
        self.block_view: Dict = {}
        self.attention_mlp_view: Dict = {}
        self.layer_view: Dict = {}


def parse_measurements(csv_path: str) -> ProfilingData:
    """
    Parse measurements.csv and create hierarchical data structure.

    Args:
        csv_path: Path to measurements.csv file

    Returns:
        ProfilingData object with all 5 levels populated
    """
    data = ProfilingData()

    # Read CSV, skipping comment lines starting with #
    try:
        df = pd.read_csv(csv_path, comment='#')
    except pd.errors.EmptyDataError:
        print(f"\n❌ ERROR: {csv_path} is empty or contains only comments.")
        print("Did you run profiling first?")
        print("\nTo run profiling:")
        print("  cd /home/spiderman/profiling-llms-llama-cpp/09A-backend")
        print("  python papi.py")
        raise

    if len(df) == 0:
        print(f"\n⚠️  WARNING: {csv_path} has no data rows (only header/comments).")
        print("The profiling may have failed. Check llama-probe output.")

    data.raw_df = df

    # Detect available PAPI columns dynamically
    papi_columns = [col for col in df.columns if col.startswith('papi_')]

    # Build hierarchical views
    data.top_view = _build_top_view(df, papi_columns)
    data.phase_view = _build_phase_view(df, papi_columns)
    data.block_view = _build_block_view(df, papi_columns)
    data.attention_mlp_view = _build_attention_mlp_view(df, papi_columns)
    data.layer_view = _build_layer_view(df, papi_columns)

    return data


def _extract_block_number(tensor_name) -> Optional[int]:
    """
    Extract decoder block number from tensor name.
    Examples:
      - 'blk.0.attn_q.weight' -> 0, 'blk.15.ffn_down.weight' -> 15  (llama.cpp old format)
      - 'attn_out-0' -> 0, 'ffn_gate-15' -> 15  (llama.cpp new format)
      - 'Qcur-0' -> 0, 'Kcur-5' -> 5
    Handles non-string values (NaN, None) gracefully.
    """
    # Handle non-string values (pandas may have NaN)
    if not isinstance(tensor_name, str):
        return None

    # Try old format first: blk.N
    match = re.search(r'blk\.(\d+)', tensor_name)
    if match:
        return int(match.group(1))

    # Try new format: word-N (where word contains letters)
    # This matches patterns like: attn_out-0, ffn_gate-15, Qcur-3, cache_k_l0 (skip cache for now)
    match = re.search(r'[a-zA-Z]+[a-zA-Z_]*-(\d+)', tensor_name)
    if match:
        return int(match.group(1))

    return None


def _get_cache_misses(df: pd.DataFrame, papi_columns: List[str], level: str) -> int:
    """
    Get cache misses for a specific cache level, trying multiple PAPI counter variants.

    Args:
        df: DataFrame with PAPI columns
        papi_columns: List of available PAPI column names
        level: Cache level ('l1', 'l2', or 'l3')

    Returns:
        Total cache misses for that level
    """
    # Try different PAPI counter variants in order of preference
    variants = {
        'l1': ['papi_l1_tcm', 'papi_l1_dcm', 'papi_l1_icm'],  # total, data, instruction
        'l2': ['papi_l2_tcm', 'papi_l2_dcm', 'papi_l2_icm'],
        'l3': ['papi_l3_tcm', 'papi_l3_dcm', 'papi_l3_icm', 'papi_l3_ldm'],
    }

    for counter in variants.get(level, []):
        if counter in papi_columns:
            return int(df[counter].sum())

    return 0


def _classify_operation(tensor_name, op_type: str) -> str:
    """
    Classify operation as attention, MLP, or other.

    Args:
        tensor_name: Name of the tensor (e.g., 'blk.0.attn_q')
        op_type: Operation type (e.g., 'MUL_MAT', 'ADD')

    Returns:
        'attention', 'mlp', or 'other'
    """
    # Handle non-string values
    if not isinstance(tensor_name, str):
        return 'other'

    name_lower = tensor_name.lower()

    # Attention patterns
    if any(pattern in name_lower for pattern in ['attn', 'attention', '_q', '_k', '_v', 'qkv']):
        return 'attention'

    # MLP patterns (FFN - Feed Forward Network)
    if any(pattern in name_lower for pattern in ['ffn', 'mlp', 'fc', 'feed_forward']):
        return 'mlp'

    # Other (embeddings, normalization, etc.)
    return 'other'


def _build_top_view(df: pd.DataFrame, papi_columns: List[str]) -> Dict:
    """
    Build top-level view: whole model as one box.

    Returns metrics:
    - total_runtime_ns
    - num_input_tokens
    - num_output_tokens
    - throughput_tokens_per_sec
    - total_energy (if available)
    - peak_rss_memory (placeholder - would need external tracking)
    - total_model_size_bytes
    - total_kv_cache_size_bytes (estimated)
    - avg_cpu_utilization (placeholder)
    - total_cache_misses
    """
    # Filter out metadata rows (tokenization, sampling)
    compute_df = df[~df['op_type'].isin(['n/a', 'tokenization', 'sampling'])]

    # Total runtime (sum of all operation times)
    total_runtime_ns = compute_df['time_ns'].sum()

    # Token counts (from phases)
    prefill_df = df[df['phase'] == 'prefill']
    decode_df = df[df['phase'] == 'decode']

    # Estimate input tokens from prefill (rough approximation)
    num_input_tokens = len(prefill_df['token_index'].unique()) if len(prefill_df) > 0 else 0
    num_output_tokens = len(decode_df['token_index'].unique()) if len(decode_df) > 0 else 0

    # Throughput
    total_tokens = num_input_tokens + num_output_tokens
    throughput = total_tokens / (total_runtime_ns / 1e9) if total_runtime_ns > 0 else 0

    # Model size (sum of unique tensor sizes)
    total_model_size = compute_df.groupby('tensor_name')['size_bytes'].first().sum()

    # Cache misses (sum across all PAPI cache miss counters)
    cache_miss_columns = [col for col in papi_columns if 'tcm' in col or 'dcm' in col or 'icm' in col]
    total_cache_misses = compute_df[cache_miss_columns].sum().sum() if cache_miss_columns else 0

    return {
        'total_runtime_ns': int(total_runtime_ns),
        'total_runtime_sec': total_runtime_ns / 1e9,
        'num_input_tokens': int(num_input_tokens),
        'num_output_tokens': int(num_output_tokens),
        'total_tokens': int(total_tokens),
        'throughput_tokens_per_sec': float(throughput),
        'total_energy_joules': None,  # Placeholder - requires energy monitoring
        'peak_rss_memory_bytes': None,  # Placeholder - requires external tracking
        'total_model_size_bytes': int(total_model_size),
        'total_kv_cache_size_bytes': None,  # Placeholder - needs model-specific calculation
        'avg_cpu_utilization': None,  # Placeholder - requires system monitoring
        'total_cache_misses': int(total_cache_misses),
    }


def _build_phase_view(df: pd.DataFrame, papi_columns: List[str]) -> Dict:
    """
    Build phase-level view: separate prefill and decode.

    For each phase:
    - time_ns
    - flops (estimated)
    - bytes_moved
    - arithmetic_intensity
    - llc_misses, llc_hits
    - ipc (instructions per cycle)
    - energy
    - operation_type_share
    - core_utilization
    """
    phases = {}

    for phase_name in df['phase'].unique():
        if phase_name in ['n/a']:
            continue

        phase_df = df[df['phase'] == phase_name]
        compute_df = phase_df[~phase_df['op_type'].isin(['n/a'])]

        # Time
        time_ns = compute_df['time_ns'].sum()

        # Operation type distribution
        op_counts = compute_df['op_type'].value_counts()
        op_share = (op_counts / len(compute_df) * 100).to_dict()

        # PAPI metrics
        total_cycles = compute_df['papi_tot_cyc'].sum() if 'papi_tot_cyc' in papi_columns else 0
        total_ins = compute_df['papi_tot_ins'].sum() if 'papi_tot_ins' in papi_columns else 0
        ipc = total_ins / total_cycles if total_cycles > 0 else 0

        # L3 cache (LLC)
        llc_misses = compute_df['papi_l3_tcm'].sum() if 'papi_l3_tcm' in papi_columns else 0

        # Bytes moved (estimated from cache misses * cache line size + tensor movements)
        cache_line_size = 64
        bytes_from_cache_misses = llc_misses * cache_line_size
        bytes_from_tensors = compute_df['size_bytes'].sum()
        bytes_moved = bytes_from_cache_misses + bytes_from_tensors

        phases[phase_name] = {
            'time_ns': int(time_ns),
            'time_sec': time_ns / 1e9,
            'flops': None,  # Will be calculated by metrics.py
            'bytes_moved': int(bytes_moved),
            'arithmetic_intensity': None,  # Will be calculated by metrics.py
            'llc_misses': int(llc_misses),
            'llc_hits': None,  # Needs LLC access counter
            'ipc': float(ipc),
            'energy_joules': None,  # Placeholder
            'operation_type_share': op_share,
            'core_utilization': None,  # Placeholder
            'num_operations': len(compute_df),
        }

    return phases


def _build_block_view(df: pd.DataFrame, papi_columns: List[str]) -> Dict:
    """
    Build decoder block view: metrics for each decoder block.

    For each block:
    - runtime_ns
    - flops
    - bytes_moved
    - kv_cache_footprint
    - arithmetic_intensity
    - cache_behavior
    - share_of_total_runtime
    """
    # Add block number column
    df['block_num'] = df['tensor_name'].apply(_extract_block_number)

    # Get compute operations with valid block numbers
    compute_df = df[~df['op_type'].isin(['n/a']) & df['block_num'].notna()]

    if len(compute_df) == 0:
        return {}

    blocks = {}
    total_runtime = compute_df['time_ns'].sum()

    for block_num in sorted(compute_df['block_num'].unique()):
        block_df = compute_df[compute_df['block_num'] == block_num]

        # Runtime
        runtime_ns = block_df['time_ns'].sum()

        # Cache behavior
        l1_misses = _get_cache_misses(block_df, papi_columns, 'l1')
        l2_misses = _get_cache_misses(block_df, papi_columns, 'l2')
        l3_misses = _get_cache_misses(block_df, papi_columns, 'l3')

        # Bytes moved
        bytes_moved = block_df['size_bytes'].sum() + (l3_misses * 64)

        blocks[int(block_num)] = {
            'runtime_ns': int(runtime_ns),
            'runtime_sec': runtime_ns / 1e9,
            'flops': None,  # Will be calculated by metrics.py
            'bytes_moved': int(bytes_moved),
            'kv_cache_footprint_bytes': None,  # Needs model-specific calculation
            'arithmetic_intensity': None,  # Will be calculated by metrics.py
            'cache_behavior': {
                'l1_misses': int(l1_misses),
                'l2_misses': int(l2_misses),
                'l3_misses': int(l3_misses),
            },
            'share_of_total_runtime': float(runtime_ns / total_runtime * 100) if total_runtime > 0 else 0,
            'num_operations': len(block_df),
        }

    return blocks


def _build_attention_mlp_view(df: pd.DataFrame, papi_columns: List[str]) -> Dict:
    """
    Build attention vs MLP view: separate attention and MLP within each block.

    For each block:
        For attention and MLP:
            - runtime_ns
            - flops
            - bytes_moved
            - arithmetic_intensity
            - cache_behavior
            - energy/cpu_efficiency
    """
    # Add block number and operation classification
    df['block_num'] = df['tensor_name'].apply(_extract_block_number)
    df['op_class'] = df.apply(lambda row: _classify_operation(row['tensor_name'], row['op_type']), axis=1)

    compute_df = df[~df['op_type'].isin(['n/a']) & df['block_num'].notna()]

    if len(compute_df) == 0:
        return {}

    blocks = {}

    for block_num in sorted(compute_df['block_num'].unique()):
        block_df = compute_df[compute_df['block_num'] == block_num]

        blocks[int(block_num)] = {}

        for op_class in ['attention', 'mlp', 'other']:
            class_df = block_df[block_df['op_class'] == op_class]

            if len(class_df) == 0:
                continue

            # Runtime
            runtime_ns = class_df['time_ns'].sum()

            # Cache behavior
            l1_misses = _get_cache_misses(class_df, papi_columns, 'l1')
            l2_misses = _get_cache_misses(class_df, papi_columns, 'l2')
            l3_misses = _get_cache_misses(class_df, papi_columns, 'l3')

            # Bytes moved
            bytes_moved = class_df['size_bytes'].sum() + (l3_misses * 64)

            # IPC
            total_cycles = class_df['papi_tot_cyc'].sum() if 'papi_tot_cyc' in papi_columns else 0
            total_ins = class_df['papi_tot_ins'].sum() if 'papi_tot_ins' in papi_columns else 0
            ipc = total_ins / total_cycles if total_cycles > 0 else 0

            blocks[int(block_num)][op_class] = {
                'runtime_ns': int(runtime_ns),
                'runtime_sec': runtime_ns / 1e9,
                'flops': None,  # Will be calculated by metrics.py
                'bytes_moved': int(bytes_moved),
                'arithmetic_intensity': None,  # Will be calculated by metrics.py
                'cache_behavior': {
                    'l1_misses': int(l1_misses),
                    'l2_misses': int(l2_misses),
                    'l3_misses': int(l3_misses),
                },
                'ipc': float(ipc),
                'cpu_efficiency': None,  # Placeholder
                'num_operations': len(class_df),
            }

    return blocks


def _build_layer_view(df: pd.DataFrame, papi_columns: List[str]) -> List[Dict]:
    """
    Build layer view: individual operations with full detail.

    Returns list of operations with:
    - phase, block_num, tensor_name, op_type
    - runtime_ns
    - flops
    - bytes_moved
    - arithmetic_intensity
    - cache_behavior (L1/L2/L3 misses)
    - ipc
    - utilization
    """
    # Add derived columns
    df['block_num'] = df['tensor_name'].apply(_extract_block_number)
    df['op_class'] = df.apply(lambda row: _classify_operation(row['tensor_name'], row['op_type']), axis=1)

    # Filter compute operations
    compute_df = df[~df['op_type'].isin(['n/a'])].copy()

    layers = []

    for idx, row in compute_df.iterrows():
        # Cache behavior
        cache_behavior = {}
        for col in papi_columns:
            if 'tcm' in col or 'dcm' in col or 'icm' in col:
                cache_behavior[col] = int(row[col]) if pd.notna(row[col]) else 0

        # IPC
        cycles = row['papi_tot_cyc'] if 'papi_tot_cyc' in papi_columns and pd.notna(row['papi_tot_cyc']) else 0
        ins = row['papi_tot_ins'] if 'papi_tot_ins' in papi_columns and pd.notna(row['papi_tot_ins']) else 0
        ipc = ins / cycles if cycles > 0 else 0

        layer_info = {
            'phase': row['phase'],
            'token_index': int(row['token_index']) if pd.notna(row['token_index']) else None,
            'block_num': int(row['block_num']) if pd.notna(row['block_num']) else None,
            'op_class': row['op_class'],
            'tensor_name': row['tensor_name'],
            'op_type': row['op_type'],
            'runtime_ns': int(row['time_ns']),
            'runtime_sec': row['time_ns'] / 1e9,
            'size_bytes': int(row['size_bytes']),
            'n_elements': int(row['n_elements']),
            'flops': None,  # Will be calculated by metrics.py
            'bytes_moved': int(row['size_bytes']),  # Simplified
            'arithmetic_intensity': None,  # Will be calculated by metrics.py
            'cache_behavior': cache_behavior,
            'ipc': float(ipc),
            'utilization': None,  # Placeholder
        }

        layers.append(layer_info)

    return layers


def get_summary_stats(data: ProfilingData) -> Dict:
    """
    Get summary statistics across all hierarchy levels.

    Returns:
        Dict with key statistics for quick overview
    """
    if data.raw_df is None:
        return {}

    return {
        'total_operations': len(data.raw_df[~data.raw_df['op_type'].isin(['n/a'])]),
        'num_decoder_blocks': len(data.block_view),
        'phases': list(data.phase_view.keys()),
        'total_runtime_sec': data.top_view.get('total_runtime_sec', 0),
        'throughput_tokens_per_sec': data.top_view.get('throughput_tokens_per_sec', 0),
    }


if __name__ == '__main__':
    import sys

    if len(sys.argv) < 2:
        print("Usage: python data_processor.py <measurements.csv>")
        sys.exit(1)

    csv_path = sys.argv[1]

    if not Path(csv_path).exists():
        print(f"Error: {csv_path} not found")
        sys.exit(1)

    print(f"Parsing {csv_path}...")
    data = parse_measurements(csv_path)

    print("\n=== Summary Statistics ===")
    stats = get_summary_stats(data)
    for key, value in stats.items():
        print(f"{key}: {value}")

    print("\n=== Top View ===")
    for key, value in data.top_view.items():
        print(f"{key}: {value}")

    print("\n=== Phase View ===")
    for phase, metrics in data.phase_view.items():
        print(f"\n{phase}:")
        for key, value in metrics.items():
            print(f"  {key}: {value}")

    print("\n=== Decoder Blocks ===")
    print(f"Number of blocks: {len(data.block_view)}")

    print("\n=== Attention vs MLP ===")
    for block_num, components in data.attention_mlp_view.items():
        print(f"\nBlock {block_num}:")
        for component, metrics in components.items():
            print(f"  {component}: {metrics['runtime_sec']:.6f}s")
