import { Model, PapiEvent } from '../types';
import { AppState, OpTypeShareItem, CoreThreadUtilization } from './AppContext';

type SetFn = <K extends keyof AppState>(key: K, value: AppState[K]) => void;

function parseOpTypeShare(phase: any): OpTypeShareItem[] {
    const opTypeShare = phase?.op_type_share ?? {};
    return Object.entries(opTypeShare).map(([label, details]) => {
        const info = details as any;
        return {
            label,
            timeSharePct: info?.time_share_pct ?? 0,
            countSharePct: info?.count_share_pct ?? 0,
            totalTimeUs: info?.total_time_us ?? 0,
        };
    }).sort((a, b) => b.timeSharePct - a.timeSharePct);
}

function parseCoreUtilization(phase: any): CoreThreadUtilization[] {
    const coreUtil = phase?.core_utilization ?? {};
    const cores: CoreThreadUtilization[] = [];

    for (const [socket, socketData] of Object.entries(coreUtil)) {
        if (typeof socketData !== 'object' || socketData === null) continue;
        for (const [core, threadData] of Object.entries(socketData as Record<string, any>)) {
            if (typeof threadData !== 'object' || threadData === null) continue;
            let activeThread = '';
            let activeUtil = 0;
            for (const [thread, rawValue] of Object.entries(threadData as Record<string, any>)) {
                const util = typeof rawValue === 'number' ? rawValue : Number(rawValue) || 0;
                cores.push({ socket, core, thread, utilizationPct: util });
                if (util > activeUtil) {
                    activeUtil = util;
                    activeThread = thread;
                }
            }
        }
    }

    return cores.sort((a, b) => {
        if (a.socket !== b.socket) return a.socket.localeCompare(b.socket);
        if (a.core !== b.core) return a.core.localeCompare(b.core);
        return a.thread.localeCompare(b.thread);
    });
}

function avg(arr: number[]): number {
    return arr.length > 0 ? arr.reduce((a, b) => a + b, 0) / arr.length : 0;
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

// Replace with fetch('/api/models') when integrating.
async function queryModels(): Promise<Model[]> {
    try {
        const response = await fetch('/api/models');
        if (!response.ok) throw new Error('Failed to fetch models');
        const data = await response.json() as { path: string; display_name: string }[];
        
        return data.map(m => {
            const nameParts = m.display_name.split('/');
            const filename = nameParts[nameParts.length - 1];
            
            // Extract parameter count (e.g. 1.5B or 1B)
            const paramMatch = filename.match(/(\d+(?:\.\d+)?)[Bb]/);
            const params = paramMatch ? `${paramMatch[1]}B` : '1.5B';
            
            // Format name nicely
            const cleanName = filename.replace('.gguf', '').replace(/-/g, ' ');
            
            return {
                id: m.path,
                name: cleanName,
                type: 'Transformer',
                params: params,
                status: 'Active',
                latency: 'N/A',
                energy: 'Low'
            };
        });
    } catch (e) {
        console.error('Error fetching models:', e);
        return [];
    }
}

// Replace with fetch('/api/papi-events') when integrating.
async function queryPapiEvents(): Promise<PapiEvent[]> {
    try {
        const response = await fetch('/api/events');
        if (!response.ok) throw new Error('Failed to fetch PAPI events');
        const data = await response.json() as { name: string; description: string }[];
        return data.map(d => ({
            id: d.name,
            label: d.description || d.name
        }));
    } catch (e) {
        console.error('Error fetching PAPI events:', e);
        return [];
    }
}

export async function fetchAndSetModels(set: SetFn): Promise<void> {
    const models = await queryModels();
    set('models', models);
    if (models.length > 0) {
        set('selectedModelId', models[0].id);
        set('modelName', models[0].name);
    }
}

export async function fetchAndSetPapiEvents(set: SetFn): Promise<void> {
    const events = await queryPapiEvents();
    set('availableHooks', events);
}

// Fetches all inference result JSON files in parallel and updates app state.
export async function fetchAndSetResults(set: SetFn): Promise<void> {
    const [topView, phaseView, decoderBlocks, rooflineALL, rooflinePrefill, rooflineDecode] = await Promise.all([
        fetch('/api/top-view.json').then(r => r.json()).catch(() => ({})),
        fetch('/api/phase-view.json').then(r => r.json()).catch(() => ({})),
        fetch('/api/decoder-block-view.json').then(r => r.json()).catch(() => ([])),
        fetch('/api/roofline/all').then(r => r.json()).catch(() => ({})),
        fetch('/api/roofline/prefill').then(r => r.ok ? r.json() : {}).catch(() => ({})),
        fetch('/api/roofline/decode').then(r => r.ok ? r.json() : {}).catch(() => ({})),
    ]);

    // --- top-view ---
    set('tokensPerSecond',  topView?.token_throughput ?? 0);
    set('totalRuntimeS',   topView?.runtime_s ?? 0);
    set('modelSizeBytes',  (topView?.model_size_mb ?? 0) * 1_048_576);
    set('kvCacheBytes',    topView?.kv_size_used_bytes ?? 0);
    set('kvCapacityBytes', topView?.kv_size_capacity_bytes ?? 1e-9);
    set('kvUsedBytes',     topView?.kv_size_used_bytes ?? 0);
    set('kvTokensCapacity', topView?.kv_tokens_capacity ?? 1e-9);
    set('kvTokensUsed',    topView?.kv_tokens_used ?? 0);
    set('kvUtilPercent',   (topView?.kv_tokens_used ?? 0) / (topView?.kv_tokens_capacity ?? 1e-9) * 100);
    set('cpuUtilPercent',  topView?.avg_cpu_usage ?? 0);
    set('totalEnergy',     (topView?.['energy-pkg'] ?? 0) / 1_000_000);
    set('energyPsysJ',     (topView?.['energy-psys'] ?? 0) / 1_000_000);
    set('energyPkgJ',      (topView?.['energy-pkg'] ?? 0)  / 1_000_000);
    set('energyCoresJ',    (topView?.['energy-cores'] ?? 0) / 1_000_000);
    set('memoryUsedBytes', (topView?.peak_rss_mb ?? 0) * 1_048_576);
    set('outputTokens',    topView?.generated_tokens ?? 0);
    set('totalTokens',     (topView?.total_tokens ?? 0));
    set('papiL1Misses',    topView?.PAPI_L1_TCM ?? 0);
    set('papiL2Misses',    topView?.PAPI_L2_TCM ?? 0);
    set('papiL3Misses',    topView?.PAPI_L3_TCM ?? 0);

    // --- phase-view ---
    const prefill = phaseView?.prefill ?? {};
    const decode = phaseView?.decode ?? {};
    const totalPhaseMs = (prefill?.runtime_ms ?? 0) + (decode?.runtime_ms ?? 0);

    set('prefillTimeS',         (prefill?.runtime_ms ?? 0) / 1000);
    set('prefillTimePercent',   totalPhaseMs > 0 ? (prefill?.runtime_ms ?? 0) / totalPhaseMs * 100 : 0);
    set('prefillFLOPs',         prefill?.FLOPs ?? 0);
    set('prefillIntensity',     prefill?.arithmetic_intensity ?? 0);
    set('prefillBytesMoved',    prefill?.bytes_moved ?? 0);
    set('prefillIPC',           prefill?.IPC ?? 0);
    set('prefillEnergyJ',       (prefill?.energy?.['energy-pkg'] ?? 0) / 1_000_000);
    set('prefillHitRate',       (1 - (prefill?.LLC_miss_rate ?? 0)) * 100);
    set('prefillMatmulPct',     prefill?.op_type_share?.MUL_MAT?.time_share_pct ?? 0);
    set('prefillOpTypeShare',   parseOpTypeShare(prefill));
    set('prefillCoreUtilPercent', prefill?.avg_core_utilization ?? prefill?.core_utilization ?? 0);
    set('prefillCoreThreads',   parseCoreUtilization(prefill));
    set('prefillEnergyCoresJ',   (prefill?.energy?.['energy-cores'] ?? 0) / 1_000_000);
    set('prefillEnergyPkgJ',     (prefill?.energy?.['energy-pkg'] ?? 0) / 1_000_000);
    set('prefillEnergyPsysJ',    (prefill?.energy?.['energy-psys'] ?? 0) / 1_000_000);
    set('prefillAvgPowerPkgW',   prefill?.avg_power_pkg_w ?? 0);

    set('decodeTimeS',          (decode?.runtime_ms ?? 0) / 1000);
    set('decodeTimePercent',    totalPhaseMs > 0 ? (decode?.runtime_ms ?? 0) / totalPhaseMs * 100 : 0);
    set('decodeFLOPs',          decode?.FLOPs ?? 0);
    set('decodeIntensity',      decode?.arithmetic_intensity ?? 0);
    set('decodeBytesMoved',     decode?.bytes_moved ?? 0);
    set('decodeIPC',            decode?.IPC ?? 0);
    set('decodeEnergyJ',        (decode?.energy?.['energy-pkg'] ?? 0) / 1_000_000);
    set('decodeHitRate',        (1 - (decode?.LLC_miss_rate ?? 0)) * 100);
    set('decodeMatmulPct',      decode?.op_type_share?.MUL_MAT?.time_share_pct ?? 0);
    set('decodeOpTypeShare',    parseOpTypeShare(decode));
    set('decodeCoreUtilPercent', decode?.avg_core_utilization ?? decode?.core_utilization ?? 0);
    set('decodeCoreThreads',   parseCoreUtilization(decode));
    set('decodeEnergyCoresJ',   (decode?.energy?.['energy-cores'] ?? 0) / 1_000_000);
    set('decodeEnergyPkgJ',     (decode?.energy?.['energy-pkg'] ?? 0) / 1_000_000);
    set('decodeEnergyPsysJ',    (decode?.energy?.['energy-psys'] ?? 0) / 1_000_000);
    set('decodeAvgPowerPkgW',   decode?.avg_power_pkg_w ?? 0);
    set('cacheMissPercent',     (decode?.LLC_miss_rate ?? 0) * 100);
    set('powerWatts',           decode?.avg_power_pkg_w ?? 0);

    // --- decoder-block-view (averaged over all decode blocks) ---
    const decodeBlocks = Array.isArray(decoderBlocks) ? decoderBlocks.filter((b: any) => b?.block_type === 'Decode') : [];
    set('decoderBlockList', Array.isArray(decoderBlocks) ? decoderBlocks : []); // all blocks — prefill + decode

    const blockLatencyMs = avg(decodeBlocks.map((b: any) => b?.runtime_ms ?? 0));
    const attnRuntimeMs  = avg(decodeBlocks.map((b: any) => (b?.subcomponents?.attention?.runtime_us ?? 0) / 1000));
    const mlpRuntimeMs   = avg(decodeBlocks.map((b: any) => (b?.subcomponents?.MLP?.runtime_us ?? 0) / 1000));

    set('blockLatencyS',          blockLatencyMs / 1000);
    set('attentionRuntimeS',      attnRuntimeMs / 1000);
    set('attentionRuntimePct',    blockLatencyMs > 0 ? attnRuntimeMs  / blockLatencyMs * 100 : 0);
    set('attentionFLOPs',         avg(decodeBlocks.map((b: any) => b?.subcomponents?.attention?.FLOPs ?? 0)));
    set('attentionIntensity',     avg(decodeBlocks.map((b: any) => b?.subcomponents?.attention?.arithmetic_intensity ?? 0)));
    set('attentionBytesMoved',    avg(decodeBlocks.map((b: any) => b?.subcomponents?.attention?.bytes_moved ?? 0)));
    set('attentionHitRate',       avg(decodeBlocks.map((b: any) => (1 - (b?.subcomponents?.attention?.cache_behavior?.L3_miss_rate ?? 0)) * 100)));
    set('attentionIPC',           avg(decodeBlocks.map((b: any) => b?.subcomponents?.attention?.IPC ?? 0)));
    set('attentionFLOPsPerS',     avg(decodeBlocks.map((b: any) => {
        const flops = b?.subcomponents?.attention?.FLOPs ?? 0;
        const us = b?.subcomponents?.attention?.runtime_us ?? 1;
        return flops / (us / 1e6);
    })));
    set('attentionL1Misses',      avg(decodeBlocks.map((b: any) => b?.subcomponents?.attention?.papi?.PAPI_L1_TCM ?? 0)));
    set('attentionL2Misses',      avg(decodeBlocks.map((b: any) => b?.subcomponents?.attention?.papi?.PAPI_L2_TCM ?? 0)));
    set('attentionL3Misses',      avg(decodeBlocks.map((b: any) => b?.subcomponents?.attention?.cache_behavior?.L3_misses ?? 0)));
    set('attentionL3Accesses',    avg(decodeBlocks.map((b: any) => b?.subcomponents?.attention?.cache_behavior?.L3_accesses ?? 0)));

    set('mlpRuntimeS',      mlpRuntimeMs / 1000);
    set('mlpRuntimePct',    blockLatencyMs > 0 ? mlpRuntimeMs / blockLatencyMs * 100 : 0);
    set('mlpFLOPs',         avg(decodeBlocks.map((b: any) => b?.subcomponents?.MLP?.FLOPs ?? 0)));
    set('mlpIntensity',     avg(decodeBlocks.map((b: any) => b?.subcomponents?.MLP?.arithmetic_intensity ?? 0)));
    set('mlpBytesMoved',    avg(decodeBlocks.map((b: any) => b?.subcomponents?.MLP?.bytes_moved ?? 0)));
    set('mlpHitRate',       avg(decodeBlocks.map((b: any) => (1 - (b?.subcomponents?.MLP?.cache_behavior?.L3_miss_rate ?? 0)) * 100)));
    set('mlpIPC',           avg(decodeBlocks.map((b: any) => b?.subcomponents?.MLP?.IPC ?? 0)));
    set('mlpFLOPsPerS',     avg(decodeBlocks.map((b: any) => {
        const flops = b?.subcomponents?.MLP?.FLOPs ?? 0;
        const us = b?.subcomponents?.MLP?.runtime_us ?? 1;
        return flops / (us / 1e6);
    })));
    set('mlpL1Misses',      avg(decodeBlocks.map((b: any) => b?.subcomponents?.MLP?.papi?.PAPI_L1_TCM ?? 0)));
    set('mlpL2Misses',      avg(decodeBlocks.map((b: any) => b?.subcomponents?.MLP?.papi?.PAPI_L2_TCM ?? 0)));
    set('mlpL3Misses',      avg(decodeBlocks.map((b: any) => b?.subcomponents?.MLP?.cache_behavior?.L3_misses ?? 0)));
    set('mlpL3Accesses',    avg(decodeBlocks.map((b: any) => b?.subcomponents?.MLP?.cache_behavior?.L3_accesses ?? 0)));

    // --- roofline ---
    set('arithmeticIntensity', rooflineALL?.oi ?? 0);
    set('achievedFLOPS',       (rooflineALL?.achieved_gflops ?? 0) * 1e9);
    set('peakFLOPS',           (rooflineALL?.hardware?.peak_gflops ?? 0) * 1e9);
    set('memBwBs',             (rooflineALL?.hardware?.mem_bw_gbs ?? 0) * 1e9);
    set('ridgePoint',          rooflineALL?.hardware?.ridge_point ?? 0);
    set('totalFLOPs',          rooflineALL?.total_flops ?? 0);
    set('dramBytes',           rooflineALL?.dram_bytes ?? 0);
    set('hwCpuModel',          rooflineALL?.hardware?.cpu_model ?? 'N/A');
    set('hwCores',             rooflineALL?.hardware?.cores ?? 0);
    set('hwBaseGHz',           rooflineALL?.hardware?.base_ghz ?? 0);
    set('hwBoostGHz',          rooflineALL?.hardware?.boost_ghz ?? 0);
    set('hwAvgGHz',            rooflineALL?.hardware?.avg_ghz ?? 0);
    set('hwISA',               rooflineALL?.hardware?.isa ?? 'N/A');
    set('hwFlopsPerCycle',     rooflineALL?.hardware?.flops_per_cycle ?? 0);

    // Prefill roofline (used by PhaseView prefill section)
    set('prefillRooflineOI',            rooflinePrefill?.oi ?? 0);
    set('prefillRooflineAchievedGFLOPS', rooflinePrefill?.achieved_gflops ?? 0);

    // Decode roofline (used by PhaseView decode section)  
    set('decodeRooflineOI',             rooflineDecode?.oi ?? 0);
    set('decodeRooflineAchievedGFLOPS',  rooflineDecode?.achieved_gflops ?? 0);
}
