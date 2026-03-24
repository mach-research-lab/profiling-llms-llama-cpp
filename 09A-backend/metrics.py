#!/usr/bin/env python3
"""
Metrics calculation module for LLM profiling.

Calculates:
- FLOPs (Floating Point Operations) from operation types and tensor dimensions
- Arithmetic Intensity (FLOPs per Byte)
- IPC (Instructions Per Cycle)
- Cache hit rates
- Memory bandwidth utilization
- Energy estimates
"""

import numpy as np
from typing import Dict, Optional, Tuple
import pandas as pd


# ══════════════════════════════════════════════════════════════════════════════
# FLOP Estimation
# ══════════════════════════════════════════════════════════════════════════════

def estimate_flops_from_operation(op_type: str, n_elements: int, tensor_name: str = "") -> float:
    """
    Estimate FLOPs for a single operation based on type and size.

    Args:
        op_type: Operation type (e.g., 'MUL_MAT', 'ADD', 'ROPE')
        n_elements: Number of elements in the tensor
        tensor_name: Tensor name for additional context (optional)

    Returns:
        Estimated FLOPs for this operation
    """
    op_type = op_type.upper()

    # Matrix multiplication: 2 * m * n * k FLOPs
    # We approximate: for MUL_MAT with n elements, assume ~2n FLOPs
    if 'MUL_MAT' in op_type or 'MATMUL' in op_type or 'GEMM' in op_type:
        # Rough estimate: 2 FLOPs per element (multiply + accumulate)
        # For more accurate estimation, would need matrix dimensions
        return 2.0 * n_elements

    # Element-wise operations: 1 FLOP per element
    if any(op in op_type for op in ['ADD', 'SUB', 'MUL', 'DIV', 'SCALE']):
        return float(n_elements)

    # Activation functions: ~1 FLOP per element
    if any(op in op_type for op in ['RELU', 'GELU', 'SILU', 'SWIGLU', 'SIGMOID', 'TANH']):
        return float(n_elements)

    # Normalization: ~2-3 FLOPs per element (mean, variance, normalize)
    if any(op in op_type for op in ['NORM', 'RMS_NORM', 'LAYER_NORM', 'GROUP_NORM']):
        return 3.0 * n_elements

    # Softmax: ~5 FLOPs per element (exp, sum, divide)
    if 'SOFTMAX' in op_type:
        return 5.0 * n_elements

    # Attention-specific operations
    if 'ROPE' in op_type:  # Rotary Position Embedding
        return 6.0 * n_elements  # sin, cos, multiply operations

    if 'FLASH_ATTN' in op_type or 'ATTN' in op_type:
        # Flash attention: O(n^2 * d) where n is sequence length, d is dimension
        # Rough estimate: 4 FLOPs per element
        return 4.0 * n_elements

    # Reduction operations: ~1 FLOP per element
    if any(op in op_type for op in ['SUM', 'MEAN', 'MAX', 'MIN']):
        return float(n_elements)

    # Convolution: ~2 FLOPs per element
    if 'CONV' in op_type:
        return 2.0 * n_elements

    # Copy/reshape operations: 0 FLOPs (memory-bound)
    if any(op in op_type for op in ['CPY', 'RESHAPE', 'VIEW', 'PERMUTE', 'TRANSPOSE']):
        return 0.0

    # Default: assume 1 FLOP per element for unknown operations
    return float(n_elements)


def calculate_total_flops(operations: pd.DataFrame) -> float:
    """
    Calculate total FLOPs across multiple operations.

    Args:
        operations: DataFrame with columns 'op_type', 'n_elements', 'tensor_name'

    Returns:
        Total FLOPs
    """
    total_flops = 0.0

    for _, row in operations.iterrows():
        flops = estimate_flops_from_operation(
            row['op_type'],
            row['n_elements'],
            row.get('tensor_name', '')
        )
        total_flops += flops

    return total_flops


# ══════════════════════════════════════════════════════════════════════════════
# Arithmetic Intensity
# ══════════════════════════════════════════════════════════════════════════════

def calculate_arithmetic_intensity(flops: float, bytes_moved: int) -> float:
    """
    Calculate arithmetic intensity (FLOPs per Byte).

    Args:
        flops: Total floating point operations
        bytes_moved: Total bytes transferred

    Returns:
        Arithmetic intensity (FLOPs/Byte)
    """
    if bytes_moved == 0:
        return 0.0
    return flops / bytes_moved


def estimate_bytes_moved(
    tensor_sizes: list,
    l1_misses: int = 0,
    l2_misses: int = 0,
    l3_misses: int = 0,
    cache_line_size: int = 64
) -> int:
    """
    Estimate total bytes moved considering cache hierarchy.

    Args:
        tensor_sizes: List of tensor sizes in bytes
        l1_misses: L1 cache misses
        l2_misses: L2 cache misses
        l3_misses: L3 cache misses (goes to DRAM)
        cache_line_size: Cache line size in bytes (typically 64)

    Returns:
        Estimated bytes moved
    """
    # Bytes from tensor data movement
    tensor_bytes = sum(tensor_sizes)

    # Bytes from cache misses
    # L3 misses go to DRAM (most expensive)
    bytes_from_l3_misses = l3_misses * cache_line_size

    # Add some fraction of L2 misses (they hit L3)
    bytes_from_l2_misses = l2_misses * cache_line_size * 0.5

    # L1 misses hit L2 (less significant for DRAM traffic)
    bytes_from_l1_misses = l1_misses * cache_line_size * 0.1

    total_bytes = (
        tensor_bytes +
        bytes_from_l3_misses +
        bytes_from_l2_misses +
        bytes_from_l1_misses
    )

    return int(total_bytes)


# ══════════════════════════════════════════════════════════════════════════════
# Performance Metrics
# ══════════════════════════════════════════════════════════════════════════════

def calculate_ipc(instructions: int, cycles: int) -> float:
    """
    Calculate Instructions Per Cycle.

    Args:
        instructions: Total instructions executed
        cycles: Total CPU cycles

    Returns:
        IPC value
    """
    if cycles == 0:
        return 0.0
    return instructions / cycles


def calculate_gflops(flops: float, time_seconds: float) -> float:
    """
    Calculate GFLOP/s (billions of FLOPs per second).

    Args:
        flops: Total floating point operations
        time_seconds: Time in seconds

    Returns:
        GFLOP/s
    """
    if time_seconds == 0:
        return 0.0
    return flops / time_seconds / 1e9


def calculate_memory_bandwidth(bytes_moved: int, time_seconds: float) -> float:
    """
    Calculate memory bandwidth in GB/s.

    Args:
        bytes_moved: Total bytes transferred
        time_seconds: Time in seconds

    Returns:
        Memory bandwidth in GB/s
    """
    if time_seconds == 0:
        return 0.0
    return bytes_moved / time_seconds / 1e9


# ══════════════════════════════════════════════════════════════════════════════
# Cache Metrics
# ══════════════════════════════════════════════════════════════════════════════

def calculate_cache_hit_rate(accesses: int, misses: int) -> float:
    """
    Calculate cache hit rate.

    Args:
        accesses: Total cache accesses
        misses: Cache misses

    Returns:
        Hit rate (0.0 to 1.0)
    """
    if accesses == 0:
        return 0.0
    hits = accesses - misses
    return max(0.0, min(1.0, hits / accesses))


def calculate_cache_miss_rate(accesses: int, misses: int) -> float:
    """
    Calculate cache miss rate.

    Args:
        accesses: Total cache accesses
        misses: Cache misses

    Returns:
        Miss rate (0.0 to 1.0)
    """
    return 1.0 - calculate_cache_hit_rate(accesses, misses)


# ══════════════════════════════════════════════════════════════════════════════
# Roofline Model Metrics
# ══════════════════════════════════════════════════════════════════════════════

def calculate_roofline_performance(
    arithmetic_intensity: float,
    peak_flops: float,
    memory_bandwidth: float
) -> float:
    """
    Calculate attainable performance using the roofline model.

    Performance = min(peak_flops, memory_bandwidth * arithmetic_intensity)

    Args:
        arithmetic_intensity: FLOPs per Byte
        peak_flops: Peak compute performance (FLOP/s)
        memory_bandwidth: Memory bandwidth (Bytes/s)

    Returns:
        Attainable performance (FLOP/s)
    """
    compute_bound = peak_flops
    memory_bound = memory_bandwidth * arithmetic_intensity

    return min(compute_bound, memory_bound)


def classify_bottleneck(
    arithmetic_intensity: float,
    peak_flops: float,
    memory_bandwidth: float
) -> Tuple[str, float]:
    """
    Classify whether operation is compute-bound or memory-bound.

    Args:
        arithmetic_intensity: FLOPs per Byte
        peak_flops: Peak compute performance (FLOP/s)
        memory_bandwidth: Memory bandwidth (Bytes/s)

    Returns:
        Tuple of (bottleneck_type, ridge_point)
        bottleneck_type: 'compute' or 'memory'
        ridge_point: Arithmetic intensity at the ridge point
    """
    ridge_point = peak_flops / memory_bandwidth

    if arithmetic_intensity < ridge_point:
        bottleneck = 'memory'
    else:
        bottleneck = 'compute'

    return bottleneck, ridge_point


# ══════════════════════════════════════════════════════════════════════════════
# Energy Metrics
# ══════════════════════════════════════════════════════════════════════════════

def estimate_energy_from_power(power_watts: float, time_seconds: float) -> float:
    """
    Estimate energy consumption.

    Energy = Power * Time

    Args:
        power_watts: Power consumption in watts
        time_seconds: Time in seconds

    Returns:
        Energy in joules
    """
    return power_watts * time_seconds


def estimate_power_from_utilization(
    cpu_utilization: float,
    tdp_watts: float = 28.0  # Default for i7-1185G7
) -> float:
    """
    Estimate power consumption from CPU utilization.

    Very rough estimate: power scales with utilization.

    Args:
        cpu_utilization: CPU utilization (0.0 to 1.0)
        tdp_watts: CPU TDP (Thermal Design Power)

    Returns:
        Estimated power in watts
    """
    # Rough model: idle power + (TDP - idle) * utilization
    idle_power = tdp_watts * 0.2  # ~20% at idle
    active_power = tdp_watts - idle_power

    return idle_power + (active_power * cpu_utilization)


# ══════════════════════════════════════════════════════════════════════════════
# Composite Metrics
# ══════════════════════════════════════════════════════════════════════════════

def calculate_all_metrics(
    operations_df: pd.DataFrame,
    time_seconds: float,
    papi_counters: Optional[Dict] = None,
    hardware_config: Optional[Dict] = None
) -> Dict:
    """
    Calculate comprehensive metrics for a set of operations.

    Args:
        operations_df: DataFrame with operation details
        time_seconds: Total time in seconds
        papi_counters: Dict with PAPI counter totals (optional)
        hardware_config: Hardware configuration (peak_flops, mem_bandwidth, etc.)

    Returns:
        Dict with all calculated metrics
    """
    if papi_counters is None:
        papi_counters = {}

    if hardware_config is None:
        # Default to i7-1185G7
        hardware_config = {
            'peak_flops': 192e9,  # 192 GFLOPS (AVX2 FMA)
            'mem_bandwidth': 51.2e9,  # 51.2 GB/s
            'tdp_watts': 28.0,
        }

    # Calculate FLOPs
    total_flops = calculate_total_flops(operations_df)

    # Get cache misses
    l1_misses = papi_counters.get('papi_l1_tcm', 0)
    l2_misses = papi_counters.get('papi_l2_tcm', 0)
    l3_misses = papi_counters.get('papi_l3_tcm', 0)

    # Calculate bytes moved
    tensor_sizes = operations_df['size_bytes'].tolist()
    bytes_moved = estimate_bytes_moved(tensor_sizes, l1_misses, l2_misses, l3_misses)

    # Arithmetic intensity
    arithmetic_intensity = calculate_arithmetic_intensity(total_flops, bytes_moved)

    # Performance metrics
    gflops = calculate_gflops(total_flops, time_seconds)
    memory_bw = calculate_memory_bandwidth(bytes_moved, time_seconds)

    # IPC
    total_cycles = papi_counters.get('papi_tot_cyc', 0)
    total_ins = papi_counters.get('papi_tot_ins', 0)
    ipc = calculate_ipc(total_ins, total_cycles)

    # Roofline analysis
    bottleneck, ridge_point = classify_bottleneck(
        arithmetic_intensity,
        hardware_config['peak_flops'],
        hardware_config['mem_bandwidth']
    )

    attainable_perf = calculate_roofline_performance(
        arithmetic_intensity,
        hardware_config['peak_flops'],
        hardware_config['mem_bandwidth']
    )

    # CPU efficiency (actual vs attainable)
    actual_flops = total_flops / time_seconds
    cpu_efficiency = actual_flops / attainable_perf if attainable_perf > 0 else 0

    return {
        # Compute metrics
        'total_flops': total_flops,
        'gflops': gflops,
        'peak_gflops': hardware_config['peak_flops'] / 1e9,
        'compute_utilization': gflops / (hardware_config['peak_flops'] / 1e9),

        # Memory metrics
        'bytes_moved': bytes_moved,
        'memory_bandwidth_gbs': memory_bw,
        'peak_memory_bandwidth_gbs': hardware_config['mem_bandwidth'] / 1e9,
        'memory_bandwidth_utilization': memory_bw / (hardware_config['mem_bandwidth'] / 1e9),

        # Arithmetic intensity
        'arithmetic_intensity': arithmetic_intensity,
        'ridge_point': ridge_point,
        'bottleneck': bottleneck,

        # Performance
        'ipc': ipc,
        'cpu_efficiency': cpu_efficiency,
        'attainable_gflops': attainable_perf / 1e9,

        # Cache metrics
        'l1_misses': l1_misses,
        'l2_misses': l2_misses,
        'l3_misses': l3_misses,

        # Time
        'time_seconds': time_seconds,
    }


# ══════════════════════════════════════════════════════════════════════════════
# Utilities
# ══════════════════════════════════════════════════════════════════════════════

def format_flops(flops: float) -> str:
    """Format FLOPs in human-readable form."""
    if flops >= 1e12:
        return f"{flops/1e12:.2f} TFLOPs"
    elif flops >= 1e9:
        return f"{flops/1e9:.2f} GFLOPs"
    elif flops >= 1e6:
        return f"{flops/1e6:.2f} MFLOPs"
    elif flops >= 1e3:
        return f"{flops/1e3:.2f} KFLOPs"
    else:
        return f"{flops:.2f} FLOPs"


def format_bytes(bytes_val: int) -> str:
    """Format bytes in human-readable form."""
    if bytes_val >= 1e12:
        return f"{bytes_val/1e12:.2f} TB"
    elif bytes_val >= 1e9:
        return f"{bytes_val/1e9:.2f} GB"
    elif bytes_val >= 1e6:
        return f"{bytes_val/1e6:.2f} MB"
    elif bytes_val >= 1e3:
        return f"{bytes_val/1e3:.2f} KB"
    else:
        return f"{bytes_val} B"


if __name__ == '__main__':
    # Example usage
    print("=== Metrics Calculation Examples ===\n")

    # Example 1: Matrix multiplication
    print("Example 1: Matrix Multiplication (1024x1024)")
    n_elements = 1024 * 1024
    flops = estimate_flops_from_operation('MUL_MAT', n_elements)
    print(f"  Estimated FLOPs: {format_flops(flops)}")

    # Example 2: Arithmetic intensity
    print("\nExample 2: Arithmetic Intensity")
    bytes_moved = 8 * 1024 * 1024  # 8 MB
    ai = calculate_arithmetic_intensity(flops, bytes_moved)
    print(f"  FLOPs: {format_flops(flops)}")
    print(f"  Bytes moved: {format_bytes(bytes_moved)}")
    print(f"  Arithmetic Intensity: {ai:.2f} FLOPs/Byte")

    # Example 3: Roofline classification
    print("\nExample 3: Bottleneck Classification")
    peak_flops = 192e9
    mem_bw = 51.2e9
    bottleneck, ridge = classify_bottleneck(ai, peak_flops, mem_bw)
    print(f"  Ridge point: {ridge:.2f} FLOPs/Byte")
    print(f"  Operation AI: {ai:.2f} FLOPs/Byte")
    print(f"  Bottleneck: {bottleneck}-bound")
