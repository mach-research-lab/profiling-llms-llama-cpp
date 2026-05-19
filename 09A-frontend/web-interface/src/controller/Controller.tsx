import { Model, PapiEvent } from '../types';
import { AppState } from './AppContext';

type SetFn = <K extends keyof AppState>(key: K, value: AppState[K]) => void;

function avg(arr: number[]): number {
    return arr.reduce((a, b) => a + b, 0) / arr.length;
}

export function fmt(value: number, precision: number): string {
    return value.toFixed(precision);
}

export function fmtSI(value: number, unit: string, precision: number): string {
    const sep = unit ? ' ' : '';
    const abs = Math.abs(value);
    if (abs >= 1e15) return `${(value / 1e15).toFixed(precision)} P${unit}`;
    if (abs >= 1e12) return `${(value / 1e12).toFixed(precision)} T${unit}`;
    if (abs >= 1e9)  return `${(value / 1e9).toFixed(precision)} G${unit}`;
    if (abs >= 1e6)  return `${(value / 1e6).toFixed(precision)} M${unit}`;
    if (abs >= 1e3)  return `${(value / 1e3).toFixed(precision)} K${unit}`;
    if (abs >= 1)    return `${value.toFixed(precision)}${sep}${unit}`;
    if (abs >= 1e-3) return `${(value * 1e3).toFixed(precision)} m${unit}`;
    if (abs >= 1e-6) return `${(value * 1e6).toFixed(precision)} μ${unit}`;
    if (abs >= 1e-9) return `${(value * 1e9).toFixed(precision)} n${unit}`;
    return `${value.toFixed(precision)}${sep}${unit}`;
}

// Mock — replace with fetch('/api/models') when integrating.
async function queryModels(): Promise<Model[]> {
    return [
        { id: 'gpt-kinetic-4-v2.1', name: 'August-Kinetic-4-v2.1', type: 'Transformer', params: '175B', status: 'Active',  latency: '42ms',  energy: 'Low'       },
        { id: 'gpt-kinetic-4-v2.0', name: 'GPT-Kinetic-4-v2.0', type: 'Transformer', params: '175B', status: 'Standby', latency: '48ms',  energy: 'Medium'    },
        { id: 'kinetic-light-v1',   name: 'Kinetic-Light-v1',   type: 'MoE',         params: '32B',  status: 'Offline', latency: '12ms',  energy: 'Ultra-Low' },
        { id: 'synapse-heavy-x',    name: 'Synapse-Heavy-X',    type: 'Dense',        params: '1.2T', status: 'Offline', latency: '180ms', energy: 'High'      },
    ];
}

// Mock — replace with fetch('/api/papi-events') when integrating.
async function queryPapiEvents(): Promise<PapiEvent[]> {
    return [
        { id: 'pre_token',       label: 'Tokenization something' },
        { id: 'l3_cache_misses', label: 'L3 Cache Misses'        },
    ];
}

export async function fetchAndSetModels(set: SetFn): Promise<void> {
    const models = await queryModels();
    set('models', models);
}

export async function fetchAndSetPapiEvents(set: SetFn): Promise<void> {
    const events = await queryPapiEvents();
    set('availableHooks', events);
}

// Fetches all inference result JSON files in parallel and updates app state.
export async function fetchAndSetResults(set: SetFn): Promise<void> {
    const [topView, phaseView, decoderBlocks, roofline] = await Promise.all([
        fetch('/top-view.json').then(r => r.json()),
        fetch('/phase-view.json').then(r => r.json()),
        fetch('/decoder-block-view.json').then(r => r.json()),
        fetch('/roofline.json').then(r => r.json()),
    ]);

    // --- top-view ---
    set('tokensPerSecond',  topView.token_throughput);
    set('totalRuntimeS',   topView.runtime_s);
    set('modelSizeBytes',  topView.model_size_mb * 1_048_576);
    set('kvCacheBytes',    topView.kv_size_used_bytes);
    set('kvCapacityBytes', topView.kv_size_capacity_bytes);
    set('kvUsedBytes',     topView.kv_size_used_bytes);
    set('kvTokensCapacity', topView.kv_tokens_capacity);
    set('kvTokensUsed',    topView.kv_tokens_used);
    set('kvUtilPercent',   topView.kv_tokens_used / topView.kv_tokens_capacity * 100);
    set('cpuUtilPercent',  topView.avg_cpu_usage);
    set('totalEnergy',     topView['energy-pkg'] / 1_000_000);
    set('energyPsysJ',     topView['energy-psys'] / 1_000_000);
    set('energyPkgJ',      topView['energy-pkg']  / 1_000_000);
    set('energyCoresJ',    topView['energy-cores'] / 1_000_000);
    set('memoryUsedBytes', topView.peak_rss_mb * 1_048_576);
    set('outputTokens',    topView.generated_tokens);
    set('inputTokens',     topView.total_tokens - topView.generated_tokens);
    set('papiL1Misses',    topView.PAPI_L1_TCM);
    set('papiL2Misses',    topView.PAPI_L2_TCM);
    set('papiL3Misses',    topView.PAPI_L3_TCM);

    // --- phase-view ---
    const { prefill, decode } = phaseView;
    const totalPhaseMs = prefill.runtime_ms + decode.runtime_ms;

    set('prefillTimeS',         prefill.runtime_ms / 1000);
    set('prefillTimePercent',   prefill.runtime_ms / totalPhaseMs * 100);
    set('prefillFLOPs',         prefill.FLOPs);
    set('prefillIntensity',     prefill.arithmetic_intensity);
    set('prefillBytesMoved',    prefill.bytes_moved);
    set('prefillIPC',           prefill.IPC);
    set('prefillEnergyJ',       prefill.energy['energy-pkg'] / 1_000_000);
    set('prefillHitRate',       (1 - prefill.LLC_miss_rate) * 100);
    set('prefillMatmulPct',     prefill.op_type_share.MUL_MAT.time_share_pct);

    set('decodeTimeS',          decode.runtime_ms / 1000);
    set('decodeTimePercent',    decode.runtime_ms / totalPhaseMs * 100);
    set('decodeFLOPs',          decode.FLOPs);
    set('decodeIntensity',      decode.arithmetic_intensity);
    set('decodeBytesMoved',     decode.bytes_moved);
    set('decodeIPC',            decode.IPC);
    set('decodeEnergyJ',        decode.energy['energy-pkg'] / 1_000_000);
    set('decodeHitRate',        (1 - decode.LLC_miss_rate) * 100);
    set('decodeMatmulPct',      decode.op_type_share.MUL_MAT.time_share_pct);
    set('cacheMissPercent',     decode.LLC_miss_rate * 100);
    set('powerWatts',           decode.avg_power_pkg_w);

    // --- decoder-block-view (averaged over all decode blocks) ---
    const decodeBlocks = decoderBlocks.filter((b: any) => b.block_type === 'Decode');
    set('decoderBlockList', decoderBlocks); // all blocks — prefill + decode

    const blockLatencyMs = avg(decodeBlocks.map((b: any) => b.runtime_ms));
    const attnRuntimeMs  = avg(decodeBlocks.map((b: any) => b.subcomponents.attention.runtime_us / 1000));
    const mlpRuntimeMs   = avg(decodeBlocks.map((b: any) => b.subcomponents.MLP.runtime_us / 1000));

    set('blockLatencyS',          blockLatencyMs / 1000);
    set('attentionRuntimeS',      attnRuntimeMs / 1000);
    set('attentionRuntimePct',    attnRuntimeMs  / blockLatencyMs * 100);
    set('attentionFLOPs',         avg(decodeBlocks.map((b: any) => b.subcomponents.attention.FLOPs)));
    set('attentionIntensity',     avg(decodeBlocks.map((b: any) => b.subcomponents.attention.arithmetic_intensity)));
    set('attentionBytesMoved',    avg(decodeBlocks.map((b: any) => b.subcomponents.attention.bytes_moved)));
    set('attentionHitRate',       avg(decodeBlocks.map((b: any) => (1 - b.subcomponents.attention.cache_behavior.L3_miss_rate) * 100)));
    set('attentionIPC',           avg(decodeBlocks.map((b: any) => b.subcomponents.attention.IPC)));
    set('attentionFLOPsPerS',     avg(decodeBlocks.map((b: any) => b.subcomponents.attention.FLOPs / (b.subcomponents.attention.runtime_us / 1e6))));
    set('attentionL1Misses',      avg(decodeBlocks.map((b: any) => b.subcomponents.attention.papi.PAPI_L1_TCM)));
    set('attentionL2Misses',      avg(decodeBlocks.map((b: any) => b.subcomponents.attention.papi.PAPI_L2_TCM)));
    set('attentionL3Misses',      avg(decodeBlocks.map((b: any) => b.subcomponents.attention.cache_behavior.L3_misses)));
    set('attentionL3Accesses',    avg(decodeBlocks.map((b: any) => b.subcomponents.attention.cache_behavior.L3_accesses)));

    set('mlpRuntimeS',      mlpRuntimeMs / 1000);
    set('mlpRuntimePct',    mlpRuntimeMs / blockLatencyMs * 100);
    set('mlpFLOPs',         avg(decodeBlocks.map((b: any) => b.subcomponents.MLP.FLOPs)));
    set('mlpIntensity',     avg(decodeBlocks.map((b: any) => b.subcomponents.MLP.arithmetic_intensity)));
    set('mlpBytesMoved',    avg(decodeBlocks.map((b: any) => b.subcomponents.MLP.bytes_moved)));
    set('mlpHitRate',       avg(decodeBlocks.map((b: any) => (1 - b.subcomponents.MLP.cache_behavior.L3_miss_rate) * 100)));
    set('mlpIPC',           avg(decodeBlocks.map((b: any) => b.subcomponents.MLP.IPC)));
    set('mlpFLOPsPerS',     avg(decodeBlocks.map((b: any) => b.subcomponents.MLP.FLOPs / (b.subcomponents.MLP.runtime_us / 1e6))));
    set('mlpL1Misses',      avg(decodeBlocks.map((b: any) => b.subcomponents.MLP.papi.PAPI_L1_TCM)));
    set('mlpL2Misses',      avg(decodeBlocks.map((b: any) => b.subcomponents.MLP.papi.PAPI_L2_TCM)));
    set('mlpL3Misses',      avg(decodeBlocks.map((b: any) => b.subcomponents.MLP.cache_behavior.L3_misses)));
    set('mlpL3Accesses',    avg(decodeBlocks.map((b: any) => b.subcomponents.MLP.cache_behavior.L3_accesses)));

    // --- roofline ---
    set('arithmeticIntensity', roofline.oi);
    set('achievedFLOPS',       roofline.achieved_gflops * 1e9);
    set('peakFLOPS',           roofline.hardware.peak_gflops * 1e9);
    set('memBwBs',             roofline.hardware.mem_bw_gbs * 1e9);
    set('ridgePoint',          roofline.hardware.ridge_point);
    set('totalFLOPs',          roofline.total_flops);
    set('dramBytes',           roofline.dram_bytes);
    set('hwCpuModel',          roofline.hardware.cpu_model);
    set('hwCores',             roofline.hardware.cores);
    set('hwBaseGHz',           roofline.hardware.base_ghz);
    set('hwBoostGHz',          roofline.hardware.boost_ghz);
    set('hwAvgGHz',            roofline.hardware.avg_ghz);
    set('hwISA',               roofline.hardware.isa);
    set('hwFlopsPerCycle',     roofline.hardware.flops_per_cycle);
}
