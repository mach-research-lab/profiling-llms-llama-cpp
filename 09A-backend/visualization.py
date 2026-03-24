#!/usr/bin/env python3
"""
Hierarchical visualization module for LLM profiling.

Implements 5 levels of visualization:
1. Top view - Whole model summary
2. Phase view - Prefill vs Decode comparison
3. Decoder block view - Per-block metrics
4. Attention vs MLP view - Component breakdown
5. Layer view - Individual operations

Each level distinguishes compute vs memory metrics using color coding.
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import numpy as np
import seaborn as sns
from typing import Dict, List, Optional, Tuple
import pandas as pd

# Color scheme
COLOR_COMPUTE = '#3498db'  # Blue for compute metrics
COLOR_MEMORY = '#e74c3c'   # Red for memory metrics
COLOR_MIXED = '#9b59b6'    # Purple for mixed/other metrics
COLOR_ATTENTION = '#2ecc71'  # Green for attention
COLOR_MLP = '#f39c12'      # Orange for MLP
COLOR_OTHER = '#95a5a6'    # Gray for other operations

sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['font.size'] = 9


# ══════════════════════════════════════════════════════════════════════════════
# Level 1: Top View
# ══════════════════════════════════════════════════════════════════════════════

def plot_top_view(top_data: Dict, output_path: str) -> str:
    """
    Plot top-level view: whole LLM as one box.

    Shows:
    - Total runtime
    - Number of input/output tokens
    - Throughput (tokens/s)
    - Total energy/power
    - Peak RSS memory
    - Total model size
    - Total KV-cache size
    - Average CPU utilization
    - Total cache misses

    Args:
        top_data: Dict from data_processor.ProfilingData.top_view
        output_path: Path to save the plot

    Returns:
        Path to saved plot
    """
    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

    # Main summary box (top-left, spans 2 columns)
    ax_summary = fig.add_subplot(gs[0, :2])
    ax_summary.axis('off')

    summary_text = f"""
    ╔══════════════════════════════════════════════════════════════════╗
    ║                 LLM PROFILING - TOP VIEW                         ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║  Total Runtime:           {top_data['total_runtime_sec']:.4f} seconds
    ║  Input Tokens:            {top_data['num_input_tokens']}
    ║  Output Tokens:           {top_data['num_output_tokens']}
    ║  Total Tokens:            {top_data['total_tokens']}
    ║  Throughput:              {top_data['throughput_tokens_per_sec']:.2f} tokens/sec
    ║
    ║  Model Size:              {_format_bytes(top_data['total_model_size_bytes'])}
    ║  Total Cache Misses:      {top_data['total_cache_misses']:,}
    ╚══════════════════════════════════════════════════════════════════╝
    """

    ax_summary.text(0.1, 0.5, summary_text, fontfamily='monospace',
                    fontsize=10, verticalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    # Throughput gauge (top-right)
    ax_throughput = fig.add_subplot(gs[0, 2])
    throughput = top_data['throughput_tokens_per_sec']
    _plot_gauge(ax_throughput, throughput, 0, 100, 'Throughput\n(tokens/s)',
                COLOR_COMPUTE)

    # Memory usage (bottom-left)
    ax_memory = fig.add_subplot(gs[1, 0])
    model_size_gb = top_data['total_model_size_bytes'] / 1e9
    ax_memory.bar(['Model Size'], [model_size_gb], color=COLOR_MEMORY, width=0.5)
    ax_memory.set_ylabel('Size (GB)', color=COLOR_MEMORY)
    ax_memory.set_title('Memory Footprint')
    ax_memory.tick_params(axis='y', labelcolor=COLOR_MEMORY)

    # Cache misses (bottom-middle)
    ax_cache = fig.add_subplot(gs[1, 1])
    cache_misses_millions = top_data['total_cache_misses'] / 1e6
    ax_cache.bar(['Cache Misses'], [cache_misses_millions], color=COLOR_MEMORY, width=0.5)
    ax_cache.set_ylabel('Millions', color=COLOR_MEMORY)
    ax_cache.set_title('Total Cache Misses')
    ax_cache.tick_params(axis='y', labelcolor=COLOR_MEMORY)

    # Runtime breakdown placeholder (bottom-right)
    ax_runtime = fig.add_subplot(gs[1, 2])
    runtime_sec = top_data['total_runtime_sec']
    ax_runtime.bar(['Runtime'], [runtime_sec], color=COLOR_MIXED, width=0.5)
    ax_runtime.set_ylabel('Seconds')
    ax_runtime.set_title('Total Runtime')

    plt.suptitle('LLM Profiling - Top View', fontsize=14, fontweight='bold')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

    return output_path


# ══════════════════════════════════════════════════════════════════════════════
# Level 2: Phase View
# ══════════════════════════════════════════════════════════════════════════════

def plot_phase_view(phase_data: Dict, output_path: str) -> str:
    """
    Plot phase-level view: prefill vs decode comparison.

    For each phase shows:
    - Time spent
    - FLOPs
    - Bytes moved
    - Arithmetic intensity
    - LLC misses and hits
    - IPC
    - Energy
    - Operation type share
    - Core utilization

    Args:
        phase_data: Dict from data_processor.ProfilingData.phase_view
        output_path: Path to save the plot

    Returns:
        Path to saved plot
    """
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.35)

    phases = list(phase_data.keys())
    if len(phases) == 0:
        phases = ['prefill', 'decode']  # Fallback

    # Time comparison (compute metric)
    ax_time = fig.add_subplot(gs[0, 0])
    times = [phase_data.get(p, {}).get('time_sec', 0) for p in phases]
    ax_time.bar(phases, times, color=COLOR_COMPUTE, edgecolor='black', linewidth=1.5)
    ax_time.set_ylabel('Time (seconds)', fontweight='bold')
    ax_time.set_title('Phase Time', fontweight='bold', color=COLOR_COMPUTE)

    # Bytes moved comparison (memory metric)
    ax_bytes = fig.add_subplot(gs[0, 1])
    bytes_vals = [phase_data.get(p, {}).get('bytes_moved', 0) / 1e6 for p in phases]
    ax_bytes.bar(phases, bytes_vals, color=COLOR_MEMORY, edgecolor='black', linewidth=1.5)
    ax_bytes.set_ylabel('MB', fontweight='bold')
    ax_bytes.set_title('Bytes Moved', fontweight='bold', color=COLOR_MEMORY)

    # IPC comparison (compute metric)
    ax_ipc = fig.add_subplot(gs[0, 2])
    ipcs = [phase_data.get(p, {}).get('ipc', 0) for p in phases]
    ax_ipc.bar(phases, ipcs, color=COLOR_COMPUTE, edgecolor='black', linewidth=1.5)
    ax_ipc.set_ylabel('IPC', fontweight='bold')
    ax_ipc.set_title('Instructions Per Cycle', fontweight='bold', color=COLOR_COMPUTE)
    ax_ipc.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)

    # LLC misses (memory metric)
    ax_llc = fig.add_subplot(gs[0, 3])
    llc_misses = [phase_data.get(p, {}).get('llc_misses', 0) / 1e6 for p in phases]
    ax_llc.bar(phases, llc_misses, color=COLOR_MEMORY, edgecolor='black', linewidth=1.5)
    ax_llc.set_ylabel('Millions', fontweight='bold')
    ax_llc.set_title('LLC Misses', fontweight='bold', color=COLOR_MEMORY)

    # Operation type share - Prefill (if available)
    if 'prefill' in phase_data and 'operation_type_share' in phase_data['prefill']:
        ax_ops_prefill = fig.add_subplot(gs[1, :2])
        op_share = phase_data['prefill']['operation_type_share']
        if op_share:
            ops = list(op_share.keys())[:10]  # Top 10
            vals = [op_share[op] for op in ops]
            colors = plt.cm.Set3(np.linspace(0, 1, len(ops)))
            ax_ops_prefill.barh(ops, vals, color=colors, edgecolor='black', linewidth=0.5)
            ax_ops_prefill.set_xlabel('Percentage (%)', fontweight='bold')
            ax_ops_prefill.set_title('Prefill - Operation Type Distribution',
                                     fontweight='bold', color=COLOR_MIXED)
            ax_ops_prefill.invert_yaxis()
        else:
            ax_ops_prefill.text(0.5, 0.5, 'No operation data', ha='center', va='center')
            ax_ops_prefill.axis('off')

    # Operation type share - Decode (if available)
    if 'decode' in phase_data and 'operation_type_share' in phase_data['decode']:
        ax_ops_decode = fig.add_subplot(gs[1, 2:])
        op_share = phase_data['decode']['operation_type_share']
        if op_share:
            ops = list(op_share.keys())[:10]  # Top 10
            vals = [op_share[op] for op in ops]
            colors = plt.cm.Set3(np.linspace(0, 1, len(ops)))
            ax_ops_decode.barh(ops, vals, color=colors, edgecolor='black', linewidth=0.5)
            ax_ops_decode.set_xlabel('Percentage (%)', fontweight='bold')
            ax_ops_decode.set_title('Decode - Operation Type Distribution',
                                    fontweight='bold', color=COLOR_MIXED)
            ax_ops_decode.invert_yaxis()
        else:
            ax_ops_decode.text(0.5, 0.5, 'No operation data', ha='center', va='center')
            ax_ops_decode.axis('off')

    # Summary table
    ax_table = fig.add_subplot(gs[2, :])
    ax_table.axis('off')

    table_data = []
    for phase in phases:
        pdata = phase_data.get(phase, {})
        row = [
            phase,
            f"{pdata.get('time_sec', 0):.4f}s",
            f"{pdata.get('bytes_moved', 0) / 1e6:.2f} MB",
            f"{pdata.get('ipc', 0):.3f}",
            f"{pdata.get('llc_misses', 0) / 1e6:.2f}M",
            f"{pdata.get('num_operations', 0):,}",
        ]
        table_data.append(row)

    table = ax_table.table(cellText=table_data,
                           colLabels=['Phase', 'Time', 'Bytes Moved', 'IPC', 'LLC Misses', 'Operations'],
                           cellLoc='center',
                           loc='center',
                           bbox=[0.1, 0.3, 0.8, 0.6])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2)

    # Color header
    for i in range(6):
        cell = table[(0, i)]
        cell.set_facecolor('#cccccc')
        cell.set_text_props(weight='bold')

    plt.suptitle('LLM Profiling - Phase View (Prefill vs Decode)',
                 fontsize=14, fontweight='bold')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

    return output_path


# ══════════════════════════════════════════════════════════════════════════════
# Level 3: Decoder Block View
# ══════════════════════════════════════════════════════════════════════════════

def plot_decoder_blocks(block_data: Dict, output_path: str) -> str:
    """
    Plot decoder block view: metrics for each decoder block.

    For each block shows:
    - Runtime
    - FLOPs (if calculated)
    - Bytes moved
    - KV-cache footprint
    - Arithmetic intensity (if calculated)
    - Cache behavior
    - Share of total runtime

    Args:
        block_data: Dict from data_processor.ProfilingData.block_view
        output_path: Path to save the plot

    Returns:
        Path to saved plot
    """
    if len(block_data) == 0:
        # Create empty plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No decoder block data available',
                ha='center', va='center', fontsize=14)
        ax.axis('off')
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        return output_path

    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.3)

    block_nums = sorted(block_data.keys())
    block_labels = [f'Block {i}' for i in block_nums]

    # Runtime per block (compute metric)
    ax_runtime = fig.add_subplot(gs[0, 0])
    runtimes = [block_data[i]['runtime_sec'] for i in block_nums]
    bars = ax_runtime.bar(block_labels, runtimes, color=COLOR_COMPUTE,
                          edgecolor='black', linewidth=0.5)
    ax_runtime.set_ylabel('Time (seconds)', fontweight='bold')
    ax_runtime.set_title('Runtime per Decoder Block', fontweight='bold', color=COLOR_COMPUTE)
    ax_runtime.tick_params(axis='x', rotation=45)
    _add_value_labels(ax_runtime, bars, fmt='{:.4f}')

    # Bytes moved per block (memory metric)
    ax_bytes = fig.add_subplot(gs[0, 1])
    bytes_vals = [block_data[i]['bytes_moved'] / 1e6 for i in block_nums]
    bars = ax_bytes.bar(block_labels, bytes_vals, color=COLOR_MEMORY,
                        edgecolor='black', linewidth=0.5)
    ax_bytes.set_ylabel('MB', fontweight='bold')
    ax_bytes.set_title('Bytes Moved per Block', fontweight='bold', color=COLOR_MEMORY)
    ax_bytes.tick_params(axis='x', rotation=45)
    _add_value_labels(ax_bytes, bars, fmt='{:.1f}')

    # Share of total runtime (pie chart)
    ax_share = fig.add_subplot(gs[1, 0])
    shares = [block_data[i]['share_of_total_runtime'] for i in block_nums]
    colors = plt.cm.viridis(np.linspace(0, 1, len(block_nums)))
    wedges, texts, autotexts = ax_share.pie(shares, labels=block_labels, autopct='%1.1f%%',
                                             colors=colors, startangle=90)
    ax_share.set_title('Share of Total Runtime', fontweight='bold', color=COLOR_MIXED)

    # Cache misses per block (memory metric)
    ax_cache = fig.add_subplot(gs[1, 1])
    l1_misses = [block_data[i]['cache_behavior']['l1_misses'] / 1e6 for i in block_nums]
    l2_misses = [block_data[i]['cache_behavior']['l2_misses'] / 1e6 for i in block_nums]
    l3_misses = [block_data[i]['cache_behavior']['l3_misses'] / 1e6 for i in block_nums]

    x = np.arange(len(block_labels))
    width = 0.25

    ax_cache.bar(x - width, l1_misses, width, label='L1', color='#3498db')
    ax_cache.bar(x, l2_misses, width, label='L2', color='#e74c3c')
    ax_cache.bar(x + width, l3_misses, width, label='L3', color='#f39c12')

    ax_cache.set_ylabel('Millions', fontweight='bold')
    ax_cache.set_title('Cache Misses per Block', fontweight='bold', color=COLOR_MEMORY)
    ax_cache.set_xticks(x)
    ax_cache.set_xticklabels(block_labels, rotation=45, ha='right')
    ax_cache.legend()

    # Summary table
    ax_table = fig.add_subplot(gs[2, :])
    ax_table.axis('off')

    table_data = []
    for i in block_nums:
        bdata = block_data[i]
        row = [
            f"Block {i}",
            f"{bdata['runtime_sec']:.4f}s",
            f"{bdata['bytes_moved'] / 1e6:.2f} MB",
            f"{bdata['cache_behavior']['l3_misses'] / 1e6:.2f}M",
            f"{bdata['share_of_total_runtime']:.1f}%",
            f"{bdata['num_operations']:,}",
        ]
        table_data.append(row)

    table = ax_table.table(cellText=table_data,
                           colLabels=['Block', 'Runtime', 'Bytes Moved', 'L3 Misses', '% Runtime', 'Ops'],
                           cellLoc='center',
                           loc='center',
                           bbox=[0.05, 0.2, 0.9, 0.7])
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.8)

    # Color header
    for i in range(6):
        cell = table[(0, i)]
        cell.set_facecolor('#cccccc')
        cell.set_text_props(weight='bold')

    plt.suptitle('LLM Profiling - Decoder Block View', fontsize=14, fontweight='bold')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

    return output_path


# ══════════════════════════════════════════════════════════════════════════════
# Level 4: Attention vs MLP View
# ══════════════════════════════════════════════════════════════════════════════

def plot_attention_mlp_view(attn_mlp_data: Dict, output_path: str) -> str:
    """
    Plot attention vs MLP view: component breakdown within each decoder block.

    For each component (attention/MLP) in each block shows:
    - Runtime
    - FLOPs (if calculated)
    - Bytes moved
    - Arithmetic intensity (if calculated)
    - Cache behavior
    - Energy/CPU efficiency (if available)

    Args:
        attn_mlp_data: Dict from data_processor.ProfilingData.attention_mlp_view
        output_path: Path to save the plot

    Returns:
        Path to saved plot
    """
    if len(attn_mlp_data) == 0:
        # Create empty plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No attention/MLP data available',
                ha='center', va='center', fontsize=14)
        ax.axis('off')
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        return output_path

    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)

    block_nums = sorted(attn_mlp_data.keys())

    # Prepare data
    attention_runtimes = []
    mlp_runtimes = []
    attention_bytes = []
    mlp_bytes = []
    attention_ipc = []
    mlp_ipc = []

    for block_num in block_nums:
        components = attn_mlp_data[block_num]

        attention_runtimes.append(components.get('attention', {}).get('runtime_sec', 0))
        mlp_runtimes.append(components.get('mlp', {}).get('runtime_sec', 0))

        attention_bytes.append(components.get('attention', {}).get('bytes_moved', 0) / 1e6)
        mlp_bytes.append(components.get('mlp', {}).get('bytes_moved', 0) / 1e6)

        attention_ipc.append(components.get('attention', {}).get('ipc', 0))
        mlp_ipc.append(components.get('mlp', {}).get('ipc', 0))

    x = np.arange(len(block_nums))
    width = 0.35

    # Runtime comparison (compute metric)
    ax_runtime = fig.add_subplot(gs[0, 0])
    ax_runtime.bar(x - width/2, attention_runtimes, width, label='Attention',
                   color=COLOR_ATTENTION, edgecolor='black', linewidth=0.5)
    ax_runtime.bar(x + width/2, mlp_runtimes, width, label='MLP',
                   color=COLOR_MLP, edgecolor='black', linewidth=0.5)
    ax_runtime.set_ylabel('Time (seconds)', fontweight='bold')
    ax_runtime.set_title('Runtime: Attention vs MLP', fontweight='bold', color=COLOR_COMPUTE)
    ax_runtime.set_xticks(x)
    ax_runtime.set_xticklabels([f'Block {i}' for i in block_nums], rotation=45, ha='right')
    ax_runtime.legend()

    # Bytes moved comparison (memory metric)
    ax_bytes = fig.add_subplot(gs[0, 1])
    ax_bytes.bar(x - width/2, attention_bytes, width, label='Attention',
                 color=COLOR_ATTENTION, edgecolor='black', linewidth=0.5)
    ax_bytes.bar(x + width/2, mlp_bytes, width, label='MLP',
                 color=COLOR_MLP, edgecolor='black', linewidth=0.5)
    ax_bytes.set_ylabel('MB', fontweight='bold')
    ax_bytes.set_title('Bytes Moved: Attention vs MLP', fontweight='bold', color=COLOR_MEMORY)
    ax_bytes.set_xticks(x)
    ax_bytes.set_xticklabels([f'Block {i}' for i in block_nums], rotation=45, ha='right')
    ax_bytes.legend()

    # IPC comparison (compute metric)
    ax_ipc = fig.add_subplot(gs[1, 0])
    ax_ipc.bar(x - width/2, attention_ipc, width, label='Attention',
               color=COLOR_ATTENTION, edgecolor='black', linewidth=0.5)
    ax_ipc.bar(x + width/2, mlp_ipc, width, label='MLP',
               color=COLOR_MLP, edgecolor='black', linewidth=0.5)
    ax_ipc.set_ylabel('IPC', fontweight='bold')
    ax_ipc.set_title('IPC: Attention vs MLP', fontweight='bold', color=COLOR_COMPUTE)
    ax_ipc.set_xticks(x)
    ax_ipc.set_xticklabels([f'Block {i}' for i in block_nums], rotation=45, ha='right')
    ax_ipc.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5)
    ax_ipc.legend()

    # Stacked runtime for total picture
    ax_stacked = fig.add_subplot(gs[1, 1])
    ax_stacked.bar(x, attention_runtimes, width*2, label='Attention',
                   color=COLOR_ATTENTION, edgecolor='black', linewidth=0.5)
    ax_stacked.bar(x, mlp_runtimes, width*2, bottom=attention_runtimes,
                   label='MLP', color=COLOR_MLP, edgecolor='black', linewidth=0.5)
    ax_stacked.set_ylabel('Time (seconds)', fontweight='bold')
    ax_stacked.set_title('Total Runtime Breakdown', fontweight='bold', color=COLOR_MIXED)
    ax_stacked.set_xticks(x)
    ax_stacked.set_xticklabels([f'Block {i}' for i in block_nums], rotation=45, ha='right')
    ax_stacked.legend()

    plt.suptitle('LLM Profiling - Attention vs MLP View', fontsize=14, fontweight='bold')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

    return output_path


# ══════════════════════════════════════════════════════════════════════════════
# Level 5: Layer View
# ══════════════════════════════════════════════════════════════════════════════

def plot_layer_view(layer_data: List[Dict], output_path: str, top_n: int = 50) -> str:
    """
    Plot layer view: individual operations with full detail.

    Shows heatmap/table of top N operations by:
    - Runtime
    - FLOPs (if calculated)
    - Bytes moved
    - Arithmetic intensity (if calculated)
    - Cache behavior
    - IPC/utilization

    Args:
        layer_data: List from data_processor.ProfilingData.layer_view
        output_path: Path to save the plot
        top_n: Number of top operations to display

    Returns:
        Path to saved plot
    """
    if len(layer_data) == 0:
        # Create empty plot
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, 'No layer data available',
                ha='center', va='center', fontsize=14)
        ax.axis('off')
        plt.savefig(output_path, bbox_inches='tight')
        plt.close()
        return output_path

    # Sort by runtime and take top N
    sorted_layers = sorted(layer_data, key=lambda x: x['runtime_ns'], reverse=True)[:top_n]

    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 1, figure=fig, height_ratios=[2, 1], hspace=0.3)

    # Top operations heatmap
    ax_heatmap = fig.add_subplot(gs[0])

    # Prepare data for heatmap
    labels = []
    runtimes = []
    bytes_moved = []
    ipcs = []
    l3_misses = []

    for layer in sorted_layers[:30]:  # Top 30 for heatmap
        label = f"{layer['op_type'][:15]}\n{layer['tensor_name'][:20]}"
        labels.append(label)
        runtimes.append(layer['runtime_ns'] / 1e6)  # Convert to ms
        bytes_moved.append(layer['bytes_moved'] / 1e3)  # Convert to KB
        ipcs.append(layer['ipc'])
        l3_miss_count = layer['cache_behavior'].get('papi_l3_tcm', 0)
        l3_misses.append(l3_miss_count / 1e3)  # Convert to thousands

    # Normalize for heatmap
    def normalize(vals):
        vmin, vmax = min(vals), max(vals)
        if vmax == vmin:
            return [0.5] * len(vals)
        return [(v - vmin) / (vmax - vmin) for v in vals]

    data = np.array([
        normalize(runtimes),
        normalize(bytes_moved),
        normalize(ipcs),
        normalize(l3_misses),
    ]).T

    im = ax_heatmap.imshow(data, cmap='RdYlGn_r', aspect='auto')
    ax_heatmap.set_xticks(np.arange(4))
    ax_heatmap.set_xticklabels(['Runtime (ms)', 'Bytes (KB)', 'IPC', 'L3 Misses (K)'], rotation=15)
    ax_heatmap.set_yticks(np.arange(len(labels)))
    ax_heatmap.set_yticklabels(labels, fontsize=7)
    ax_heatmap.set_title('Top 30 Operations Heatmap (Normalized)', fontweight='bold')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax_heatmap)
    cbar.set_label('Normalized Value', rotation=270, labelpad=15)

    # Detailed table for top 10
    ax_table = fig.add_subplot(gs[1])
    ax_table.axis('off')

    table_data = []
    for layer in sorted_layers[:10]:
        row = [
            layer['op_type'][:12],
            layer['tensor_name'][:25],
            f"{layer['runtime_ns'] / 1e6:.3f}ms",
            f"{layer['bytes_moved'] / 1e3:.1f}KB",
            f"{layer['ipc']:.3f}",
            f"{layer['cache_behavior'].get('papi_l3_tcm', 0) / 1e3:.1f}K",
        ]
        table_data.append(row)

    table = ax_table.table(cellText=table_data,
                           colLabels=['Op Type', 'Tensor Name', 'Runtime', 'Bytes', 'IPC', 'L3 Misses'],
                           cellLoc='left',
                           loc='center',
                           bbox=[0, 0, 1, 1])
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 2)

    # Color header
    for i in range(6):
        cell = table[(0, i)]
        cell.set_facecolor('#cccccc')
        cell.set_text_props(weight='bold')

    # Color rows alternately
    for i in range(1, len(table_data) + 1):
        for j in range(6):
            cell = table[(i, j)]
            if i % 2 == 0:
                cell.set_facecolor('#f0f0f0')

    plt.suptitle('LLM Profiling - Layer View (Individual Operations)',
                 fontsize=14, fontweight='bold')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()

    return output_path


# ══════════════════════════════════════════════════════════════════════════════
# Utilities
# ══════════════════════════════════════════════════════════════════════════════

def _format_bytes(bytes_val: Optional[int]) -> str:
    """Format bytes in human-readable form."""
    if bytes_val is None:
        return 'N/A'
    if bytes_val >= 1e9:
        return f"{bytes_val / 1e9:.2f} GB"
    elif bytes_val >= 1e6:
        return f"{bytes_val / 1e6:.2f} MB"
    elif bytes_val >= 1e3:
        return f"{bytes_val / 1e3:.2f} KB"
    else:
        return f"{bytes_val} B"


def _plot_gauge(ax, value, vmin, vmax, label, color):
    """Plot a simple gauge/indicator."""
    normalized = (value - vmin) / (vmax - vmin) if vmax > vmin else 0
    normalized = max(0, min(1, normalized))

    ax.barh([0], [normalized], height=0.5, color=color, edgecolor='black', linewidth=2)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, 0.5)
    ax.set_xticks([0, 0.5, 1])
    ax.set_xticklabels([f'{vmin:.0f}', f'{(vmin+vmax)/2:.0f}', f'{vmax:.0f}'])
    ax.set_yticks([])
    ax.set_xlabel(label, fontweight='bold')
    ax.text(0.5, 0, f'{value:.2f}', ha='center', va='center',
            fontsize=12, fontweight='bold')


def _add_value_labels(ax, bars, fmt='{:.2f}'):
    """Add value labels on top of bars."""
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                fmt.format(height),
                ha='center', va='bottom', fontsize=7)


# ══════════════════════════════════════════════════════════════════════════════
# Main Entry Point
# ══════════════════════════════════════════════════════════════════════════════

def generate_all_visualizations(data, output_dir: str) -> Dict[str, str]:
    """
    Generate all 5 levels of visualization.

    Args:
        data: ProfilingData object
        output_dir: Directory to save plots

    Returns:
        Dict mapping level name to output file path
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    plots = {}

    # Level 1: Top view
    plots['top_view'] = plot_top_view(
        data.top_view,
        os.path.join(output_dir, 'level1_top_view.png')
    )

    # Level 2: Phase view
    plots['phase_view'] = plot_phase_view(
        data.phase_view,
        os.path.join(output_dir, 'level2_phase_view.png')
    )

    # Level 3: Decoder blocks
    plots['block_view'] = plot_decoder_blocks(
        data.block_view,
        os.path.join(output_dir, 'level3_decoder_blocks.png')
    )

    # Level 4: Attention vs MLP
    plots['attention_mlp_view'] = plot_attention_mlp_view(
        data.attention_mlp_view,
        os.path.join(output_dir, 'level4_attention_mlp.png')
    )

    # Level 5: Layer view
    plots['layer_view'] = plot_layer_view(
        data.layer_view,
        os.path.join(output_dir, 'level5_layer_view.png')
    )

    return plots


if __name__ == '__main__':
    import sys
    from pathlib import Path
    from data_processor import parse_measurements

    if len(sys.argv) < 2:
        print("Usage: python visualization.py <measurements.csv> [output_dir]")
        sys.exit(1)

    csv_path = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else './plots'

    if not Path(csv_path).exists():
        print(f"Error: {csv_path} not found")
        sys.exit(1)

    print(f"Parsing {csv_path}...")
    data = parse_measurements(csv_path)

    print(f"Generating visualizations in {output_dir}...")
    plots = generate_all_visualizations(data, output_dir)

    print("\n=== Generated Plots ===")
    for level, path in plots.items():
        print(f"{level}: {path}")

    print("\nDone!")
