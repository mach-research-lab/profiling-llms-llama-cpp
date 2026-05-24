import React, { useState } from 'react';
import {
  Layers,
  ChevronDown,
  ChevronRight,
  CornerDownRight,
  Activity,
  Clock,
  Zap
} from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';
import { useAppState } from '@/src/controller/AppContext.tsx';
import { fmt, fmtSI } from '@/src/controller/Controller.tsx';

export default function LayerView() {
  const { state } = useAppState();
  const {
    latencyS, memoryUsedBytes, memoryTotalBytes,
    computeEfficiencyPercent, decimalPrecision,
    decoderBlockList,
  } = state;

  const f  = (n: number) => fmt(n, decimalPrecision);
  const si = (n: number, unit: string) => fmtSI(n, unit, decimalPrecision);

  const [expandedId, setExpandedId]   = useState<number | null>(null);
  const [selectedId, setSelectedId]   = useState<number | null>(null);

  const prefillBlocks = decoderBlockList.filter((b: any) => b.block_type === 'Prefill');
  const decodeBlocks  = decoderBlockList.filter((b: any) => b.block_type === 'Decode');
  const allBlocks     = [...prefillBlocks, ...decodeBlocks];

  const selected = allBlocks.find((b: any) => b.block_id === selectedId) ?? allBlocks[0] ?? null;

  function toggle(id: number) {
    setExpandedId(prev => (prev === id ? null : id));
    setSelectedId(id);
  }

  function BlockRow({ block }: { block: any }) {
    const isExpanded = expandedId === block.block_id;
    const isSelected = selectedId === block.block_id;
    const isPrefill  = block.block_type === 'Prefill';
    const accentColor = isPrefill ? 'border-primary/50' : 'border-secondary/50';
    const ChevronIcon = isExpanded ? ChevronDown : ChevronRight;

    return (
      <>
        <div
          onClick={() => toggle(block.block_id)}
          className={`p-3 border-l-4 ${accentColor} rounded flex items-center justify-between cursor-pointer transition-colors ${
            isSelected ? 'bg-surface-container-highest' : 'bg-surface-container-high hover:bg-surface-container-highest'
          }`}
        >
          <div className="flex items-center gap-3">
            <ChevronIcon className="w-4 h-4 text-outline" />
            <span className={`text-xs font-bold uppercase ${isSelected ? (isPrefill ? 'text-primary' : 'text-secondary') : 'text-on-surface'}`}>
              {block.block_type.toUpperCase()}_BLOCK_{String(block.block_id).padStart(2, '0')}
            </span>
          </div>
          <div className="flex items-center gap-8">
            <span className={`text-xs font-bold ${isPrefill ? 'text-secondary' : 'text-secondary/80'}`}>
              {si(block.FLOPs, 'FLOPs')}
            </span>
            <span className={`text-xs font-bold ${isPrefill ? 'text-primary' : 'text-primary/80'}`}>
              {si(block.runtime_ms / 1000, 's')}
            </span>
          </div>
        </div>

        <AnimatePresence initial={false}>
          {isExpanded && (
            <motion.div
              key="sub"
              initial={{ height: 0, opacity: 0 }}
              animate={{ height: 'auto', opacity: 1 }}
              exit={{ height: 0, opacity: 0 }}
              transition={{ duration: 0.2 }}
              className="overflow-hidden ml-6 space-y-1"
            >
              {/* Attention row */}
              <div className="p-2 bg-surface-container-low border-l-2 border-primary/40 flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <CornerDownRight className="w-4 h-4 text-outline" />
                  <span className="text-[11px] text-on-surface-variant font-medium">ATTENTION</span>
                </div>
                <div className="flex items-center gap-8">
                  <span className="text-[11px] text-secondary/70">{si(block.subcomponents.attention.FLOPs, 'FLOPs')}</span>
                  <span className="text-[11px] text-primary/70">{si(block.subcomponents.attention.runtime_us / 1e6, 's')}</span>
                </div>
              </div>
              {/* MLP row */}
              <div className="p-2 bg-surface-container-low border-l-2 border-tertiary/40 flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <CornerDownRight className="w-4 h-4 text-outline" />
                  <span className="text-[11px] text-on-surface-variant font-medium">MLP (FFN)</span>
                </div>
                <div className="flex items-center gap-8">
                  <span className="text-[11px] text-secondary/70">{si(block.subcomponents.MLP.FLOPs, 'FLOPs')}</span>
                  <span className="text-[11px] text-primary/70">{si(block.subcomponents.MLP.runtime_us / 1e6, 's')}</span>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </>
    );
  }

  return (
    <div className="p-8 space-y-8 animate-in fade-in duration-500">
      <div className="flex justify-between items-end mb-8">
        <div>
          <h2 className="font-headline text-3xl font-light text-primary mb-1">Layer Operation Trace</h2>
          <p className="text-on-surface-variant font-mono text-xs uppercase tracking-widest">
            {allBlocks.length > 0
              ? <><span className="text-primary">{prefillBlocks.length}</span> Prefill · <span className="text-secondary">{decodeBlocks.length}</span> Decode blocks</>
              : 'No data — run inference first'
            }
          </p>
        </div>
        <div className="flex gap-4">
          <div className="bg-surface-container p-3 rounded-lg border-l-2 border-primary">
            <div className="text-[10px] text-on-surface-variant mb-1 uppercase tracking-tighter font-bold">Inference Latency</div>
            <div className="text-xl font-headline font-bold text-primary">{si(latencyS, 's')}</div>
          </div>
          <div className="bg-surface-container p-3 rounded-lg border-l-2 border-secondary">
            <div className="text-[10px] text-on-surface-variant mb-1 uppercase tracking-tighter font-bold">Compute Efficiency</div>
            <div className="text-xl font-headline font-bold text-secondary">{f(computeEfficiencyPercent)}%</div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-12 gap-8">
        {/* Left: Trace */}
        <div className="col-span-12 lg:col-span-7">
          <section className="space-y-4">
            <div className="flex items-center gap-2 border-b border-outline-variant/20 pb-2">
              <Layers className="w-5 h-5 text-primary" />
              <h3 className="font-headline text-xl font-bold tracking-tight uppercase">Trace Execution Map</h3>
            </div>

            <div className="bg-surface-container rounded-lg overflow-hidden border border-outline-variant/10">
              <div className="p-4 space-y-2 font-mono max-h-[600px] overflow-y-auto">
                {allBlocks.length === 0 ? (
                  <p className="text-outline text-xs text-center py-8">Run inference to populate trace data.</p>
                ) : (
                  allBlocks.map((block: any) => (
                    <BlockRow key={block.block_id} block={block} />
                  ))
                )}
              </div>
            </div>
          </section>
        </div>

        {/* Right: Properties panel for selected block */}
        <div className="col-span-12 lg:col-span-5 space-y-6">
          {selected ? (
            <section className="space-y-4">
              <div className="flex items-center gap-2 border-b border-outline-variant/20 pb-2">
                <Activity className="w-5 h-5 text-secondary" />
                <h3 className="font-headline text-xl font-bold tracking-tight uppercase">Block Analysis</h3>
                <span className="ml-auto text-[10px] bg-secondary/10 text-secondary px-2 py-0.5 rounded-full border border-secondary/30 uppercase font-bold">
                  {selected.block_type} #{selected.block_id}
                </span>
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div className="bg-surface-container p-4 rounded-lg relative overflow-hidden group">
                  <div className="absolute top-0 right-0 p-2 opacity-10 group-hover:opacity-20 transition-opacity">
                    <Clock className="w-12 h-12" />
                  </div>
                  <div className="text-[10px] text-on-surface-variant font-mono uppercase font-bold">Runtime</div>
                  <div className="text-3xl font-headline font-bold text-white mt-1">{si(selected.runtime_ms / 1000, 's')}</div>
                  <div className="w-full bg-surface-container-highest h-1.5 mt-3 rounded-full overflow-hidden">
                    <div className="bg-primary h-full" style={{ width: `${Math.min(selected.runtime_share_pct, 100)}%` }} />
                  </div>
                  <div className="text-[10px] text-primary mt-1 font-bold">{f(selected.runtime_share_pct)}% of total runtime</div>
                </div>

                <div className="bg-surface-container p-4 rounded-lg relative overflow-hidden group">
                  <div className="absolute top-0 right-0 p-2 opacity-10 group-hover:opacity-20 transition-opacity">
                    <Activity className="w-12 h-12" />
                  </div>
                  <div className="text-[10px] text-on-surface-variant font-mono uppercase font-bold">FLOPs</div>
                  <div className="text-3xl font-headline font-bold text-white mt-1">{si(selected.FLOPs, 'FLOPs')}</div>
                  <div className="flex items-center gap-2 mt-2">
                    <span className="text-[10px] text-on-surface-variant">IPC: <span className="text-white font-bold">{f(selected.IPC)}</span></span>
                  </div>
                </div>
              </div>

              {/* Arithmetic Intensity mini-roofline */}
              <div className="bg-surface-container p-6 rounded-lg border border-outline-variant/10">
                <div className="flex justify-between items-center mb-6">
                  <h4 className="text-xs font-bold uppercase tracking-widest text-on-surface-variant flex items-center gap-2">
                    <span className="w-2 h-2 rounded-full bg-primary animate-pulse"></span>
                    Arithmetic Intensity
                  </h4>
                  <div className="text-[10px] font-mono text-outline">{f(selected.arithmetic_intensity)} FLOPs/Byte</div>
                </div>
                <div className="h-40 w-full roofline-grid border-b border-l border-outline/30 relative">
                  <svg className="absolute inset-0 w-full h-full">
                    <path d="M 0 160 L 100 80 L 400 80" fill="none" stroke="#3e4850" strokeDasharray="4 4" strokeWidth="2" />
                    {/* compute dot from roofline JSON-driven hardware values */}
                    {
                      (() => {
                        const peakGF = (state.peakFLOPS || 0) / 1e9;
                        const ridge = state.ridgePoint || 1;
                        const achieved = selected.FLOPs && selected.runtime_ms ? ((selected.FLOPs / 1e9) / (selected.runtime_ms / 1000)) : 0.0;
                        const svgSpec = { x0: 0, y0: 64, xRidge: 100, yRidge: 80, xMax: 400, yMax: 160 };
                        const { dotX, dotY } = computeRooflineSVG(selected.arithmetic_intensity, achieved, ridge, peakGF, svgSpec);
                        const fill = selected.block_type === 'Prefill' ? '#89ceff' : '#a8d8a8';
                        return (
                          <motion.circle
                            key={selected.block_id}
                            initial={{ opacity: 0, scale: 0 }}
                            animate={{ opacity: 1, scale: 1 }}
                            cx={dotX}
                            cy={dotY}
                            fill={fill}
                            r="6"
                            className="drop-shadow-[0_0_8px_currentColor]"
                          />
                        );
                      })()
                    }
                  </svg>
                  <div className="absolute bottom-2 right-2 text-[8px] text-outline font-mono">Memory Bound → Compute Bound</div>
                </div>
              </div>

              <div className="grid grid-cols-3 gap-4">
                <div className="bg-surface-container-low p-4 border border-outline-variant/10">
                  <div className="text-[10px] text-on-surface-variant mb-2 font-bold uppercase">Bytes Moved</div>
                  <div className="text-lg font-headline font-bold">{si(selected.bytes_moved, 'B')}</div>
                </div>
                <div className="bg-surface-container-low p-4 border border-outline-variant/10">
                  <div className="text-[10px] text-on-surface-variant mb-2 font-bold uppercase">IPC</div>
                  <div className="text-lg font-headline font-bold">{f(selected.IPC)}</div>
                </div>
                <div className="bg-surface-container-low p-4 border border-outline-variant/10">
                  <div className="text-[10px] text-on-surface-variant mb-2 font-bold uppercase">KV Cache</div>
                  <div className="text-lg font-headline font-bold text-tertiary">{si(selected.kv_cache_footprint_bytes, 'B')}</div>
                </div>
              </div>

              {/* Cache behaviour */}
              <div className="grid grid-cols-2 gap-4">
                <div className="bg-surface-container p-4">
                  <h4 className="text-[10px] font-bold text-on-surface-variant uppercase mb-4">L3 Cache</h4>
                  <div className="flex items-end gap-2 mb-2">
                    <div className={`text-2xl font-headline font-bold ${selected.cache_behavior.L3_hit_rate > 0.5 ? 'text-secondary' : 'text-error'}`}>
                      {f(selected.cache_behavior.L3_hit_rate * 100)}%
                    </div>
                    <div className="text-[10px] text-secondary mb-1 font-bold">HIT RATE</div>
                  </div>
                  <div className="flex gap-1 h-2">
                    <div className="h-full bg-secondary" style={{ width: `${selected.cache_behavior.L3_hit_rate * 100}%` }} />
                    <div className="h-full bg-error" style={{ width: `${selected.cache_behavior.L3_miss_rate * 100}%` }} />
                  </div>
                </div>

                {/* Attention vs MLP split */}
                <div className="bg-surface-container p-4">
                  <h4 className="text-[10px] font-bold text-on-surface-variant uppercase mb-3">Attn / MLP Split</h4>
                  {(() => {
                    const attnMs = selected.subcomponents.attention.runtime_us / 1000;
                    const mlpMs  = selected.subcomponents.MLP.runtime_us / 1000;
                    const total  = attnMs + mlpMs || 1;
                    const attnPct = attnMs / total * 100;
                    const mlpPct  = mlpMs  / total * 100;
                    return (
                      <>
                        <div className="flex gap-1 h-4 rounded overflow-hidden mb-2">
                          <div className="bg-primary h-full transition-all" style={{ width: `${attnPct}%` }} />
                          <div className="bg-tertiary h-full transition-all" style={{ width: `${mlpPct}%` }} />
                        </div>
                        <div className="flex justify-between text-[10px] font-mono">
                          <span className="text-primary font-bold">Attn {f(attnPct)}%</span>
                          <span className="text-tertiary font-bold">MLP {f(mlpPct)}%</span>
                        </div>
                      </>
                    );
                  })()}
                </div>
              </div>
            </section>
          ) : (
            <div className="flex items-center justify-center h-64 text-outline text-xs uppercase tracking-widest">
              Select a block to inspect
            </div>
          )}
        </div>
      </div>

      {/* Footer */}
      <footer className="fixed bottom-0 left-64 right-0 bg-surface-container-lowest border-t border-outline-variant/10 px-6 h-8 flex items-center justify-between text-[9px] uppercase tracking-widest font-bold text-outline z-40">
        <div className="flex items-center gap-4">
          <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-secondary"></span> System Online</span>
          <span>{allBlocks.length} blocks loaded</span>
        </div>
        <div className="flex items-center gap-4">
          <span>Mem: {si(memoryUsedBytes, 'B')} / {si(memoryTotalBytes, 'B')}</span>
        </div>
      </footer>
    </div>
  );
}
