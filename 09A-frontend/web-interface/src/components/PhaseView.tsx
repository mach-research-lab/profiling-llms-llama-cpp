import React from 'react';
import { 
  Zap, 
  RefreshCw, 
  Clock, 
  Activity,
  AlertCircle
} from 'lucide-react';
import { motion } from 'motion/react';
import { useAppState } from '@/src/controller/AppContext.tsx';
import { fmt, fmtSI } from '@/src/controller/Controller.tsx';
import Heatmap from '@/src/components/Heatmap.tsx';

// TODO: Tokenization, Sampling eventuellt

// TODO: Heatmap component that takes in a massive matrix
  // TODO: (Tiles)

export default function PhaseView() {
  const { state } = useAppState();
  const {
    latencyS, tokensPerSecond,
    prefillTimeS, prefillTimePercent, prefillFLOPs, prefillFlopsTrendPct,
    prefillIntensity, prefillBytesMoved, prefillIPC, prefillEnergyJ,
    prefillHitRate, prefillMatmulPct,
    decodeTimeS, decodeTimePercent, decodeFLOPs, decodeFlopsTrendPct,
    decodeIntensity, decodeBytesMoved, decodeIPC, decodeEnergyJ,
    decodeHitRate, decodeMatmulPct,
    decimalPrecision, decoderBlockList,
  } = state;

  // ── Block-driven heatmap ───────────────────────────────────────────────────
  const decodeBlocks = decoderBlockList.filter((b: any) => b.block_type === 'Decode');
  const blockStages  = decodeBlocks.map((b: any) => `B${b.block_id}`);

  function norm(arr: number[]): number[] {
    const max = Math.max(...arr, 1e-9);
    return arr.map(v => v / max);
  }

  // Raw values per row (used for cell labels in raw display mode)
  const rawMatMul     = decodeBlocks.map((b: any) => (b.subcomponents?.attention?.FLOPs ?? 0) + (b.subcomponents?.MLP?.FLOPs ?? 0));
  const rawSoftmax    = decodeBlocks.map((b: any) => b.subcomponents?.attention?.FLOPs ?? 0);
  const rawKvCache    = decodeBlocks.map((b: any) => b.subcomponents?.attention?.bytes_moved ?? 0);
  const rawActivation = decodeBlocks.map((b: any) => b.subcomponents?.MLP?.FLOPs ?? 0);
  const rawResidual   = decodeBlocks.map((b: any) => b.runtime_ms ?? 0);

  // Fake quantitative values for ops without real profiling data yet
  const fakeRope      = decodeBlocks.map((_: any, i: number) => 1.2e6 + i * 8e3);
  const fakeLayerNorm = decodeBlocks.map((_: any, i: number) => 0.9e6 + i * 5e3);

  // Normalise each column across all blocks individually to preserve relative hotspot colors
  const normMatMul     = norm(rawMatMul);
  const normRope        = norm(fakeRope);
  const normSoftmax     = norm(rawSoftmax);
  const normKvCache     = norm(rawKvCache);
  const normActivation  = norm(rawActivation);
  const normLayerNorm   = norm(fakeLayerNorm);
  const normResidual    = norm(rawResidual);

  const OP_COLUMNS = ['MAT_MUL', 'ROPE', 'SOFTMAX', 'KV_CACHE', 'ACTIVATION', 'LAYER_NORM', 'RESIDUAL'];

  const blockOpRowsInverted = decodeBlocks.length > 0 ? decodeBlocks.map((b: any, i: number) => {
    const label = `B${i}`;
    const values = [
      normMatMul[i],
      normRope[i],
      normSoftmax[i],
      normKvCache[i],
      normActivation[i],
      normLayerNorm[i],
      normResidual[i]
    ];
    const rawValues = [
      rawMatMul[i],
      fakeRope[i],
      rawSoftmax[i],
      rawKvCache[i],
      rawActivation[i],
      fakeLayerNorm[i],
      rawResidual[i]
    ];
    return { label, values, rawValues };
  }) : [];
  const f = (n: number) => fmt(n, decimalPrecision);
  const si = (n: number, unit: string) => fmtSI(n, unit, decimalPrecision);
  const trend = (pct: number) => `${pct >= 0 ? '↑' : '↓'} ${Math.abs(pct)}%`;

  return (
    <div className="p-8 space-y-8 animate-in fade-in duration-500">
      {/* Header Status */}
      <div className="flex justify-between items-end mb-8">
        <div>
          <h2 className="font-headline text-3xl font-light text-primary mb-1">Phase Comparative Analytics</h2>
          <p className="text-on-surface-variant font-mono text-xs uppercase tracking-widest">
            Inference Session: <span className="text-secondary">#0988-X-OMEGA</span>
          </p>
        </div>
        <div className="flex gap-4">
          <div className="bg-surface-container p-3 rounded-lg border-l-2 border-tertiary">
            <div className="text-[10px] text-on-surface-variant mb-1 uppercase tracking-tighter font-bold">Total Latency</div>
            <div className="text-xl font-headline font-bold text-tertiary">{si(latencyS, 's')}</div>
          </div>
          <div className="bg-surface-container p-3 rounded-lg border-l-2 border-secondary">
            <div className="text-[10px] text-on-surface-variant mb-1 uppercase tracking-tighter font-bold">Throughput</div>
            <div className="text-xl font-headline font-bold text-secondary">{f(tokensPerSecond)} tok/s</div>
          </div>
        </div>
      </div>

      {/* Side-by-Side Comparison Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* PREFILL PHASE */}
        <PhaseSection
          title="PHASE 01: PREFILL"
          icon={<Zap className="w-5 h-5 text-primary fill-current" />}
          badge="COMPUTE BOUND"
          badgeColor="text-primary bg-primary/10 border-primary/30"
          time={si(prefillTimeS, 's')}
          timePercent={prefillTimePercent}
          f={f}
          flops={si(prefillFLOPs, 'FLOPs')}
          flopsTrend={trend(prefillFlopsTrendPct)}
          intensity={`${f(prefillIntensity)} FLOPs/Byte`}
          intensityPoint={{ x: 280, y: 80 }}
          bytesMoved={si(prefillBytesMoved, 'B')}
          ipc={f(prefillIPC)}
          energy={si(prefillEnergyJ, 'J')}
          hitRate={prefillHitRate}
          matmul={prefillMatmulPct}
          primaryColor="bg-primary"
        />

        {/* DECODE PHASE */}
        <PhaseSection 
          title="PHASE 02: DECODE" 
          icon={<RefreshCw className="w-5 h-5 text-secondary" />}
          badge="MEMORY BOUND"
          badgeColor="text-secondary bg-secondary/10 border-secondary/30"
          time={si(decodeTimeS, 's')}
          timePercent={decodeTimePercent}
          f={f}
          flops={si(decodeFLOPs, 'FLOPs')}
          flopsTrend={trend(decodeFlopsTrendPct)}
          intensity={`${f(decodeIntensity)} FLOPs/Byte`}
          intensityPoint={{ x: 20, y: 144 }}
          bytesMoved={si(decodeBytesMoved, 'B')}
          ipc={f(decodeIPC)}
          energy={si(decodeEnergyJ, 'J')}
          hitRate={decodeHitRate}
          matmul={decodeMatmulPct}
          primaryColor="bg-secondary"
          isWarning
        />
      </div>

      {/* Operation Heatmap — inverted block × op grid layout */}
      <Heatmap
        title="Operation Intensity per Decode Block"
        description="Decode block (Y) × Operation (X) — raw FLOPs / bytes / ms"
        stages={OP_COLUMNS}
        cellSize={45}
        displayMode="raw"
        formatValue={v => fmtSI(v, '', decimalPrecision)}
        tabs={blockOpRowsInverted.length > 0
          ? [{ label: 'Decode', rows: blockOpRowsInverted }]
          : [{ label: 'Decode', rows: [{ label: 'B0', values: [] }] }]
        }
      />

      {/* Cache Miss Rate Heatmap */}
      <Heatmap
        title="L3 Cache Miss Rate Heatmap"
        description="Cache level (Y) × Transformer layer stage (X)"
        stages={OP_STAGES}
        cellSize={40}
        defaultCollapsed
        tabs={[
          {
            label: 'Prefill',
            rows: CACHE_ROWS.map(r => ({ label: r.label, values: r.prefill })),
          },
          {
            label: 'Decode',
            rows: CACHE_ROWS.map(r => ({ label: r.label, values: r.decode })),
          },
        ]}
      />

      {/* Memory Bandwidth Heatmap */}
      <Heatmap
        title="IPC Heatmap"
        description="Data source (Y) × Transformer layer stage (X)"
        stages={OP_STAGES}
        cellSize={40}
        defaultCollapsed
        tabs={[
          {
            label: 'Prefill',
            rows: MEMBW_ROWS.map(r => ({ label: r.label, values: r.prefill })),
          },
          {
            label: 'Decode',
            rows: MEMBW_ROWS.map(r => ({ label: r.label, values: r.decode })),
          },
        ]}
      />

    </div>
  );
}

// ─── Heatmap data for the operation × layer-stage view ────────────────────────

const OP_STAGES = [
  'Q_PROJ', 'K_PROJ', 'V_PROJ', 'ROPE',
  'QK_MATMUL', 'SOFTMAX', 'AV_MATMUL', 'O_PROJ',
  'FFN_GATE', 'FFN_ACT', 'FFN_DOWN', 'LAYER_NORM',
];

const OP_ROWS: { label: string; prefill: number[]; decode: number[] }[] = [
  { label: 'MAT_MUL',    prefill: [0.95, 0.90, 0.90, 0.00, 0.82, 0.00, 0.85, 0.92, 0.88, 0.00, 0.93, 0.00], decode: [0.55, 0.52, 0.52, 0.00, 0.30, 0.00, 0.32, 0.54, 0.50, 0.00, 0.55, 0.00] },
  { label: 'ROPE',       prefill: [0.00, 0.00, 0.00, 0.95, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00], decode: [0.00, 0.00, 0.00, 0.90, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00] },
  { label: 'SOFTMAX',    prefill: [0.00, 0.00, 0.00, 0.00, 0.00, 0.88, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00], decode: [0.00, 0.00, 0.00, 0.00, 0.00, 0.72, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00] },
  { label: 'KV_CACHE',   prefill: [0.00, 0.55, 0.55, 0.00, 0.40, 0.00, 0.50, 0.00, 0.00, 0.00, 0.00, 0.00], decode: [0.00, 0.85, 0.85, 0.00, 0.78, 0.00, 0.80, 0.00, 0.00, 0.00, 0.00, 0.00] },
  { label: 'ACTIVATION', prefill: [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.82, 0.00, 0.00], decode: [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.78, 0.00, 0.00] },
  { label: 'LAYER_NORM', prefill: [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.92], decode: [0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.00, 0.88] },
  { label: 'RESIDUAL',   prefill: [0.18, 0.18, 0.18, 0.12, 0.12, 0.12, 0.18, 0.22, 0.16, 0.12, 0.22, 0.28], decode: [0.22, 0.22, 0.22, 0.16, 0.16, 0.16, 0.22, 0.28, 0.20, 0.16, 0.28, 0.34] },
];

// ─── Cache miss rate × layer stage ────────────────────────────────────────────
//     Values represent miss rate 0..1 per cache level per stage

const CACHE_ROWS: { label: string; prefill: number[]; decode: number[] }[] = [
  // L1 misses — high during weight loads (projections, FFN), low elsewhere
  { label: 'L1_MISS', prefill: [0.72, 0.68, 0.68, 0.12, 0.55, 0.10, 0.58, 0.70, 0.65, 0.08, 0.74, 0.15], decode: [0.45, 0.42, 0.42, 0.10, 0.62, 0.12, 0.65, 0.44, 0.40, 0.06, 0.46, 0.12] },
  // L2 misses — spill from L1, especially bad in FFN and KV stages
  { label: 'L2_MISS', prefill: [0.48, 0.52, 0.52, 0.08, 0.44, 0.07, 0.46, 0.50, 0.60, 0.05, 0.62, 0.10], decode: [0.60, 0.65, 0.65, 0.10, 0.72, 0.09, 0.74, 0.61, 0.55, 0.06, 0.58, 0.11] },
  // L3 misses — worst in decode (KV cache thrashing)
  { label: 'L3_MISS', prefill: [0.28, 0.35, 0.35, 0.05, 0.30, 0.04, 0.32, 0.30, 0.38, 0.03, 0.40, 0.06], decode: [0.55, 0.82, 0.82, 0.08, 0.78, 0.06, 0.80, 0.54, 0.48, 0.04, 0.50, 0.08] },
];

// ─── Memory bandwidth pressure × layer stage ──────────────────────────────────
//     Values represent relative BW demand 0..1 per data source per stage

const MEMBW_ROWS: { label: string; prefill: number[]; decode: number[] }[] = [
  // Weights — loaded once per token in prefill, every token in decode
  { label: 'WEIGHTS',  prefill: [0.90, 0.85, 0.85, 0.00, 0.78, 0.00, 0.82, 0.88, 0.85, 0.00, 0.90, 0.00], decode: [0.95, 0.92, 0.92, 0.00, 0.88, 0.00, 0.90, 0.94, 0.92, 0.00, 0.96, 0.00] },
  // Activations — intermediate tensors between ops
  { label: 'ACTIV',    prefill: [0.55, 0.50, 0.50, 0.40, 0.60, 0.45, 0.62, 0.58, 0.52, 0.48, 0.55, 0.35], decode: [0.35, 0.32, 0.32, 0.28, 0.40, 0.30, 0.42, 0.38, 0.34, 0.30, 0.36, 0.22] },
  // KV cache — minimal in prefill, dominant in decode
  { label: 'KV_CACHE', prefill: [0.00, 0.30, 0.30, 0.00, 0.35, 0.00, 0.38, 0.00, 0.00, 0.00, 0.00, 0.00], decode: [0.00, 0.88, 0.88, 0.00, 0.92, 0.00, 0.90, 0.00, 0.00, 0.00, 0.00, 0.00] },
  // DRAM — last resort, driven by cache misses
  { label: 'DRAM',     prefill: [0.30, 0.38, 0.38, 0.05, 0.28, 0.04, 0.30, 0.32, 0.40, 0.03, 0.42, 0.06], decode: [0.55, 0.75, 0.75, 0.08, 0.80, 0.06, 0.82, 0.56, 0.50, 0.04, 0.52, 0.08] },
];


function PhaseSection({
  title, icon, badge, badgeColor, time, timePercent, flops, flopsTrend,
  intensity, intensityPoint, bytesMoved, ipc, energy, hitRate, matmul, primaryColor, isWarning, f
}: any) {
  return (
    <section className="space-y-6">
      <div className="flex items-center gap-2 border-b border-outline-variant/20 pb-2">
        {icon}
        <h3 className="font-headline text-xl font-bold tracking-tight">{title}</h3>
        <span className={`ml-auto text-[10px] px-2 py-0.5 rounded-full border font-bold ${badgeColor}`}>{badge}</span>
      </div>
      
      <div className="grid grid-cols-2 gap-4">
        <div className="bg-surface-container p-4 rounded-lg relative overflow-hidden group">
          <div className="absolute top-0 right-0 p-2 opacity-10 group-hover:opacity-20 transition-opacity">
            <Clock className="w-12 h-12" />
          </div>
          <div className="text-[10px] text-on-surface-variant font-mono uppercase font-bold">Time Spent</div>
          <div className="text-3xl font-headline font-bold text-white mt-1">{time}</div>
          <div className="w-full bg-surface-container-highest h-1.5 mt-3 rounded-full overflow-hidden">
            <motion.div 
              initial={{ width: 0 }}
              animate={{ width: `${timePercent}%` }}
              transition={{ duration: 1 }}
              className={`h-full ${primaryColor}`} 
            />
          </div>
          <div className={`text-[10px] mt-1 font-bold ${primaryColor.replace('bg-', 'text-')}`}>{f(timePercent)}% of total inference</div>
        </div>
        
        <div className="bg-surface-container p-4 rounded-lg relative overflow-hidden group">
          <div className="absolute top-0 right-0 p-2 opacity-10 group-hover:opacity-20 transition-opacity">
            <Activity className="w-12 h-12" />
          </div>
          <div className="text-[10px] text-on-surface-variant font-mono uppercase font-bold">FLOPs</div>
          <div className="text-3xl font-headline font-bold text-white mt-1">{flops}</div>
          <div className="flex items-center gap-2 mt-2">
            <span className={`${flopsTrend.includes('↑') ? 'text-secondary' : 'text-error'} text-[10px] font-bold`}>{flopsTrend}</span>
            <span className="text-[10px] text-on-surface-variant">vs. baseline</span>
          </div>
        </div>
      </div>

      <div className="bg-surface-container p-6 rounded-lg border border-outline-variant/10">
        <div className="flex justify-between items-center mb-6">
          <h4 className="text-xs font-bold uppercase tracking-widest text-on-surface-variant flex items-center gap-2">
            <span className={`w-2 h-2 rounded-full animate-pulse ${primaryColor}`}></span>
            Arithmetic Intensity
          </h4>
          <div className="text-[10px] font-mono text-outline">{intensity}</div>
        </div>
        <div className="h-40 w-full roofline-grid border-b border-l border-outline/30 relative">
          <svg className="absolute inset-0 w-full h-full">
            <path d="M 0 160 L 100 80 L 400 80" fill="none" stroke="#3e4850" strokeDasharray="4 4" strokeWidth="2" />
            <motion.circle 
              initial={{ opacity: 0, scale: 0 }}
              animate={{ opacity: 1, scale: 1 }}
              cx={intensityPoint.x} 
              cy={intensityPoint.y} 
              fill={primaryColor === 'bg-primary' ? '#89ceff' : '#4edea3'} 
              r="6" 
              className="drop-shadow-[0_0_8px_currentColor]"
            />
          </svg>
          <div className="absolute bottom-2 right-2 text-[8px] text-outline font-mono">Memory Bound → Compute Bound</div>
        </div>
      </div>

      <div className="grid grid-cols-3 gap-4">
        <div className="bg-surface-container-low p-4 border border-outline-variant/10">
          <div className="text-[10px] text-on-surface-variant mb-2 font-bold uppercase">Bytes Moved</div>
          <div className="text-lg font-headline font-bold">{bytesMoved}</div>
        </div>
        <div className="bg-surface-container-low p-4 border border-outline-variant/10">
          <div className="text-[10px] text-on-surface-variant mb-2 font-bold uppercase">IPC</div>
          <div className="text-lg font-headline font-bold">{ipc}</div>
        </div>
        <div className="bg-surface-container-low p-4 border border-outline-variant/10">
          <div className="text-[10px] text-on-surface-variant mb-2 font-bold uppercase">Energy</div>
          <div className="text-lg font-headline font-bold text-tertiary">{energy}</div>
        </div>
      </div>

      <div className="grid grid-cols-2 gap-4">
        <div className="bg-surface-container p-4">
          <h4 className="text-[10px] font-bold text-on-surface-variant uppercase mb-4">LLC Efficiency</h4>
          <div className="flex items-end gap-2 mb-2">
            <div className="text-2xl font-headline font-bold">{f(hitRate)}%</div>
            <div className={`text-[10px] mb-1 font-bold ${hitRate > 50 ? 'text-secondary' : 'text-error'}`}>HIT RATE</div>
          </div>
          <div className="flex gap-1 h-2">
            <div className="h-full bg-secondary" style={{ width: `${hitRate}%` }}></div>
            <div className="h-full bg-error" style={{ width: `${100 - hitRate}%` }}></div>
          </div>
        </div>
        <div className="bg-surface-container p-4 flex flex-col justify-center items-center relative">
          <svg className="w-20 h-20 -rotate-90">
            <circle cx="40" cy="40" fill="transparent" r="32" stroke="#2d3449" strokeWidth="8" />
            <motion.circle 
              initial={{ strokeDashoffset: 201 }}
              animate={{ strokeDashoffset: 201 - (201 * matmul / 100) }}
              transition={{ duration: 1.5 }}
              cx="40" cy="40" fill="transparent" r="32" 
              stroke={primaryColor === 'bg-primary' ? '#4edea3' : '#89ceff'} 
              strokeDasharray="201" 
              strokeWidth="8" 
            />
          </svg>
          <div className="absolute inset-0 flex flex-col items-center justify-center pt-2">
            <span className="text-xs font-bold">{matmul}%</span>
            <span className="text-[8px] text-on-surface-variant uppercase font-bold">Matmul</span>
          </div>
          <div className="mt-2 text-[10px] text-on-surface-variant font-bold uppercase">Op Type Share</div>
        </div>
      </div>
    </section>
  );
}
