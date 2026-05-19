import React, { useState } from 'react';
import {
  AlertTriangle,
  Zap,
  MemoryStick,
  ArrowRightLeft,
  Brain,
  Cpu,
  ArrowRight,
} from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';
import { useAppState } from '@/src/controller/AppContext.tsx';
import { fmt, fmtSI } from '@/src/controller/Controller.tsx';
import { View } from '@/src/types';

export default function DecoderBlockView({ onViewChange }: { onViewChange: (v: View) => void }) {
  const { state, set } = useAppState();
  const {
    totalEnergy, memoryUsedBytes, interBlockLatencyS,
    parallelismFactor, ioWaitState, decoderBlockList, decimalPrecision,
  } = state;
  const f  = (n: number) => fmt(n, decimalPrecision);
  const si = (n: number, unit: string) => fmtSI(n, unit, decimalPrecision);

  const [selectedIdx, setSelectedIdx] = useState<number | null>(null);

  function navigateToBlock(block: any, blockId: number | string) {
    const attn = block.subcomponents.attention;
    const mlp  = block.subcomponents.MLP;
    const totalUs = attn.runtime_us + mlp.runtime_us;
    set('blockLatencyS',       block.runtime_ms / 1000);
    set('attentionRuntimeS',   attn.runtime_us / 1_000_000);
    set('attentionRuntimePct', totalUs > 0 ? attn.runtime_us / totalUs * 100 : 0);
    set('attentionFLOPs',      attn.FLOPs);
    set('attentionIntensity',  attn.arithmetic_intensity);
    set('attentionBytesMoved', attn.bytes_moved);
    set('attentionHitRate',    (1 - attn.cache_behavior.L3_miss_rate) * 100);
    set('attentionIPC',        attn.IPC);
    set('attentionFLOPsPerS',  attn.FLOPs / (attn.runtime_us / 1e6));
    set('attentionL1Misses',   attn.papi.PAPI_L1_TCM);
    set('attentionL2Misses',   attn.papi.PAPI_L2_TCM);
    set('attentionL3Misses',   attn.cache_behavior.L3_misses);
    set('attentionL3Accesses', attn.cache_behavior.L3_accesses);
    set('mlpRuntimeS',         mlp.runtime_us / 1_000_000);
    set('mlpRuntimePct',       totalUs > 0 ? mlp.runtime_us / totalUs * 100 : 0);
    set('mlpFLOPs',            mlp.FLOPs);
    set('mlpIntensity',        mlp.arithmetic_intensity);
    set('mlpBytesMoved',       mlp.bytes_moved);
    set('mlpHitRate',          (1 - mlp.cache_behavior.L3_miss_rate) * 100);
    set('mlpIPC',              mlp.IPC);
    set('mlpFLOPsPerS',        mlp.FLOPs / (mlp.runtime_us / 1e6));
    set('mlpL1Misses',         mlp.papi.PAPI_L1_TCM);
    set('mlpL2Misses',         mlp.papi.PAPI_L2_TCM);
    set('mlpL3Misses',         mlp.cache_behavior.L3_misses);
    set('mlpL3Accesses',       mlp.cache_behavior.L3_accesses);
    set('selectedBlockLabel',  `Block #${blockId} (${block.block_type})`);
    onViewChange('attention');
  }

  const totalRuntimeMs = decoderBlockList.reduce((sum: number, b: any) => sum + b.runtime_ms, 0);

  const prefillBlocks = decoderBlockList
    .map((b: any, i: number) => ({ block: b, idx: i }))
    .filter(({ block }) => block.block_type === 'Prefill');
  const decodeBlocks = decoderBlockList
    .map((b: any, i: number) => ({ block: b, idx: i }))
    .filter(({ block }) => block.block_type === 'Decode');

  function blockMetrics(b: any) {
    const attn = b.subcomponents.attention;
    const mlp  = b.subcomponents.MLP;
    const totalFLOPs = attn.FLOPs + mlp.FLOPs;
    const totalBytes = attn.bytes_moved + mlp.bytes_moved;
    const hitRate    = (1 - (attn.cache_behavior.L3_miss_rate + mlp.cache_behavior.L3_miss_rate) / 2) * 100;
    const intensity  = totalBytes > 0 ? totalFLOPs / totalBytes : 0;
    const sharePct   = totalRuntimeMs > 0 ? b.runtime_ms / totalRuntimeMs * 100 : 0;
    const isWarning  = hitRate < 50;
    return { attn, mlp, totalFLOPs, totalBytes, hitRate, intensity, sharePct, isWarning };
  }


  return (
    <div className="p-8 space-y-8 animate-in fade-in duration-500">
      <div className="flex justify-between items-end mb-8">
        <div>
          <h2 className="font-headline text-3xl font-light text-primary mb-1">Decoder Blocks</h2>
          <p className="text-on-surface-variant font-mono text-xs uppercase tracking-widest">
            {decoderBlockList.length > 0
              ? <><span className="text-primary">{prefillBlocks.length}</span> prefill · <span className="text-secondary">{decodeBlocks.length}</span> decode — click a row to inspect</>
              : 'Run inference to load block data'}
          </p>
        </div>
        <div className="flex gap-4">
          <div className="bg-surface-container p-3 rounded-lg border-l-2 border-tertiary">
            <div className="text-[10px] text-on-surface-variant mb-1 uppercase tracking-tighter font-bold">Total Energy</div>
            <div className="text-xl font-headline font-bold text-tertiary">{si(totalEnergy, 'J')}</div>
          </div>
          <div className="bg-surface-container p-3 rounded-lg border-l-2 border-secondary">
            <div className="text-[10px] text-on-surface-variant mb-1 uppercase tracking-tighter font-bold">Peak RSS</div>
            <div className="text-xl font-headline font-bold text-secondary">{si(memoryUsedBytes, 'B')}</div>
          </div>
        </div>
      </div>

      {/* Block table */}
      <div className="space-y-2">
        {/* Column headers */}
        <div className="grid grid-cols-12 px-4 py-2 text-[10px] font-mono uppercase tracking-widest text-on-surface-variant bg-surface-container/30 rounded-t-lg font-bold">
          <div className="col-span-1">Block</div>
          <div className="col-span-2">Runtime</div>
          <div className="col-span-2">FLOPs</div>
          <div className="col-span-2">Bytes Moved</div>
          <div className="col-span-2">Intensity</div>
          <div className="col-span-2">L3 Hit Rate</div>
          <div className="col-span-1 text-right">Share</div>
        </div>

        {/* Prefill section */}
        {prefillBlocks.length > 0 && (
          <div className="flex items-center gap-3 px-2 pt-2 pb-1">
            <span className="text-[10px] font-mono font-bold uppercase tracking-widest text-primary">Phase 01 — Prefill</span>
            <div className="flex-1 h-px bg-primary/20" />
            <span className="text-[10px] font-mono text-on-surface-variant">{prefillBlocks.length} block{prefillBlocks.length !== 1 ? 's' : ''}</span>
          </div>
        )}
        {prefillBlocks.map(({ block, idx }) => {
          const { totalFLOPs, totalBytes, hitRate, intensity, sharePct, isWarning } = blockMetrics(block);
          const isSelected = selectedIdx === idx;
          const blockId = block.layer_index ?? idx;
          return (
            <React.Fragment key={idx}>
              <motion.div
                whileHover={{ scale: 1.002 }}
                onClick={() => setSelectedIdx(isSelected ? null : idx)}
                className={`p-4 rounded-lg border transition-all cursor-pointer select-none ${
                  isSelected
                    ? 'bg-primary/10 border-primary/50'
                    : isWarning
                    ? 'bg-surface-container border-error/20 hover:border-error/50'
                    : 'bg-surface-container border-outline-variant/10 hover:border-primary/30'
                }`}
              >
                <div className="grid grid-cols-12 items-center">
                  <div className="col-span-1 flex items-center gap-2">
                    <span className={`font-headline font-bold text-sm ${isWarning ? 'text-error' : 'text-primary'}`}>
                      #{blockId}
                    </span>
                    {isWarning && <AlertTriangle className="w-3 h-3 text-error" />}
                  </div>
                  <div className="col-span-2">
                    <span className={`text-lg font-headline font-bold leading-none ${isWarning ? 'text-error' : 'text-white'}`}>
                      {si(block.runtime_ms / 1000, 's')}
                    </span>
                  </div>
                  <div className="col-span-2">
                    <span className="text-base font-headline font-bold text-white leading-none">
                      {si(totalFLOPs, 'FLOPs')}
                    </span>
                  </div>
                  <div className="col-span-2">
                    <span className="text-base font-headline font-bold text-white leading-none">
                      {si(totalBytes, 'B')}
                    </span>
                  </div>
                  <div className="col-span-2">
                    <span className={`text-[10px] px-2 py-0.5 border rounded font-mono font-bold ${
                      isWarning ? 'bg-error/10 text-error border-error/20' : 'bg-primary/10 text-primary border-primary/20'
                    }`}>
                      {f(intensity)} FLOPs/B
                    </span>
                  </div>
                  <div className="col-span-2 pr-4">
                    <div className="flex items-center gap-2">
                      <span className={`text-xs font-bold ${hitRate < 50 ? 'text-error' : 'text-secondary'}`}>{f(hitRate)}%</span>
                      <div className="flex-1 h-1 bg-surface-container-highest rounded-full overflow-hidden">
                        <div className={`h-full ${hitRate < 50 ? 'bg-error' : 'bg-secondary'}`} style={{ width: `${hitRate}%` }} />
                      </div>
                    </div>
                  </div>
                  <div className="col-span-1 text-right">
                    <span className={`text-base font-headline font-bold ${isWarning ? 'text-error' : 'text-secondary'}`}>
                      {f(sharePct)}%
                    </span>
                  </div>
                </div>
              </motion.div>

              <AnimatePresence>
                {isSelected && (
                  <motion.section
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: 'auto' }}
                    exit={{ opacity: 0, height: 0 }}
                    transition={{ duration: 0.2 }}
                    className="overflow-hidden"
                  >
                    <div className="bg-surface-container rounded-lg border border-primary/30 mb-2 overflow-hidden">
                      <div className="flex items-center justify-between px-6 py-3 border-b border-outline-variant/20 bg-primary/5">
                        <div className="flex items-center gap-3">
                          <span className="w-2 h-2 rounded-full bg-primary animate-pulse" />
                          <span className="font-headline font-bold text-primary text-sm uppercase tracking-widest">
                            Block #{blockId} — Subcomponent Breakdown
                          </span>
                        </div>
                        <div className="flex gap-6 text-[10px] font-mono text-on-surface-variant">
                          <span>Runtime: <span className="text-white font-bold">{si(block.runtime_ms / 1000, 's')}</span></span>
                          <span>FLOPs: <span className="text-white font-bold">{si(totalFLOPs, 'FLOPs')}</span></span>
                          <span>Bytes: <span className="text-white font-bold">{si(totalBytes, 'B')}</span></span>
                        </div>
                      </div>
                      <div className="grid grid-cols-2 divide-x divide-outline-variant/20">
                        {[
                          { label: 'Attention', icon: <Brain className="w-4 h-4 text-primary"   />, color: 'text-primary',   data: block.subcomponents.attention },
                          { label: 'MLP (FFN)',  icon: <Cpu   className="w-4 h-4 text-secondary" />, color: 'text-secondary', data: block.subcomponents.MLP       },
                        ].map(({ label, icon, color, data }) => {
                          const subHitRate = (1 - data.cache_behavior.L3_miss_rate) * 100;
                          const runtimeS   = (data.runtime_us ?? 0) / 1_000_000;
                          return (
                            <div key={label} className="p-6 space-y-4">
                              <div className="flex items-center gap-2 mb-4">
                                {icon}
                                <h4 className={`font-headline font-bold text-sm uppercase tracking-widest ${color}`}>{label}</h4>
                              </div>
                              <div className="space-y-3 font-mono text-[11px]">
                                {[
                                  { label: 'Runtime',     value: si(runtimeS, 's') },
                                  { label: 'FLOPs',       value: si(data.FLOPs, 'FLOPs') },
                                  { label: 'Bytes Moved', value: si(data.bytes_moved, 'B') },
                                  { label: 'Intensity',   value: `${f(data.arithmetic_intensity)} FLOPs/B` },
                                ].map(({ label: rowLabel, value }) => (
                                  <div key={rowLabel} className="flex justify-between items-center border-b border-outline-variant/10 pb-2">
                                    <span className="text-on-surface-variant">{rowLabel}</span>
                                    <span className="text-white font-bold">{value}</span>
                                  </div>
                                ))}
                                <div className="pt-1">
                                  <div className="flex justify-between items-center mb-1.5">
                                    <span className="text-on-surface-variant">L3 Hit Rate</span>
                                    <span className={`font-bold ${subHitRate < 50 ? 'text-error' : 'text-secondary'}`}>{f(subHitRate)}%</span>
                                  </div>
                                  <div className="flex gap-0.5 h-1.5 rounded-full overflow-hidden">
                                    <div className="h-full bg-secondary rounded-l-full" style={{ width: `${subHitRate}%` }} />
                                    <div className="h-full bg-error rounded-r-full" style={{ width: `${100 - subHitRate}%` }} />
                                  </div>
                                </div>
                              </div>
                            </div>
                          );
                        })}
                      </div>
                      <div className="border-t border-outline-variant/20 px-6 py-4 flex justify-end">
                        <button
                          onClick={() => navigateToBlock(block, blockId)}
                          className="flex items-center gap-2 text-[11px] font-mono font-bold uppercase tracking-widest text-primary bg-primary/10 hover:bg-primary/20 border border-primary/30 hover:border-primary/60 px-4 py-2 rounded transition-all"
                        >
                          Advanced Layer Details
                          <ArrowRight className="w-3.5 h-3.5" />
                        </button>
                      </div>
                    </div>
                  </motion.section>
                )}
              </AnimatePresence>
            </React.Fragment>
          );
        })}

        {/* Decode section */}
        {decodeBlocks.length > 0 && (
          <div className="flex items-center gap-3 px-2 pt-4 pb-1">
            <span className="text-[10px] font-mono font-bold uppercase tracking-widest text-secondary">Phase 02 — Decode</span>
            <div className="flex-1 h-px bg-secondary/20" />
            <span className="text-[10px] font-mono text-on-surface-variant">{decodeBlocks.length} block{decodeBlocks.length !== 1 ? 's' : ''}</span>
          </div>
        )}
        {decodeBlocks.map(({ block, idx }) => {
          const { totalFLOPs, totalBytes, hitRate, intensity, sharePct, isWarning } = blockMetrics(block);
          const isSelected = selectedIdx === idx;
          const blockId = block.layer_index ?? idx;
          return (
            <React.Fragment key={idx}>
              <motion.div
                whileHover={{ scale: 1.002 }}
                onClick={() => setSelectedIdx(isSelected ? null : idx)}
                className={`p-4 rounded-lg border transition-all cursor-pointer select-none ${
                  isSelected
                    ? 'bg-secondary/10 border-secondary/50'
                    : isWarning
                    ? 'bg-surface-container border-error/20 hover:border-error/50'
                    : 'bg-surface-container border-outline-variant/10 hover:border-secondary/30'
                }`}
              >
                <div className="grid grid-cols-12 items-center">
                  <div className="col-span-1 flex items-center gap-2">
                    <span className={`font-headline font-bold text-sm ${isWarning ? 'text-error' : 'text-secondary'}`}>
                      #{blockId}
                    </span>
                    {isWarning && <AlertTriangle className="w-3 h-3 text-error" />}
                  </div>
                  <div className="col-span-2">
                    <span className={`text-lg font-headline font-bold leading-none ${isWarning ? 'text-error' : 'text-white'}`}>
                      {si(block.runtime_ms / 1000, 's')}
                    </span>
                  </div>
                  <div className="col-span-2">
                    <span className="text-base font-headline font-bold text-white leading-none">{si(totalFLOPs, 'FLOPs')}</span>
                  </div>
                  <div className="col-span-2">
                    <span className="text-base font-headline font-bold text-white leading-none">{si(totalBytes, 'B')}</span>
                  </div>
                  <div className="col-span-2">
                    <span className={`text-[10px] px-2 py-0.5 border rounded font-mono font-bold ${
                      isWarning ? 'bg-error/10 text-error border-error/20' : 'bg-secondary/10 text-secondary border-secondary/20'
                    }`}>
                      {f(intensity)} FLOPs/B
                    </span>
                  </div>
                  <div className="col-span-2 pr-4">
                    <div className="flex items-center gap-2">
                      <span className={`text-xs font-bold ${hitRate < 50 ? 'text-error' : 'text-secondary'}`}>{f(hitRate)}%</span>
                      <div className="flex-1 h-1 bg-surface-container-highest rounded-full overflow-hidden">
                        <div className={`h-full ${hitRate < 50 ? 'bg-error' : 'bg-secondary'}`} style={{ width: `${hitRate}%` }} />
                      </div>
                    </div>
                  </div>
                  <div className="col-span-1 text-right">
                    <span className={`text-base font-headline font-bold ${isWarning ? 'text-error' : 'text-secondary'}`}>
                      {f(sharePct)}%
                    </span>
                  </div>
                </div>
              </motion.div>

              <AnimatePresence>
                {isSelected && (
                  <motion.section
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: 'auto' }}
                    exit={{ opacity: 0, height: 0 }}
                    transition={{ duration: 0.2 }}
                    className="overflow-hidden"
                  >
                    <div className="bg-surface-container rounded-lg border border-secondary/30 mb-2 overflow-hidden">
                      <div className="flex items-center justify-between px-6 py-3 border-b border-outline-variant/20 bg-secondary/5">
                        <div className="flex items-center gap-3">
                          <span className="w-2 h-2 rounded-full bg-secondary animate-pulse" />
                          <span className="font-headline font-bold text-secondary text-sm uppercase tracking-widest">
                            Block #{blockId} — Subcomponent Breakdown
                          </span>
                        </div>
                        <div className="flex gap-6 text-[10px] font-mono text-on-surface-variant">
                          <span>Runtime: <span className="text-white font-bold">{si(block.runtime_ms / 1000, 's')}</span></span>
                          <span>FLOPs: <span className="text-white font-bold">{si(totalFLOPs, 'FLOPs')}</span></span>
                          <span>Bytes: <span className="text-white font-bold">{si(totalBytes, 'B')}</span></span>
                        </div>
                      </div>
                      <div className="grid grid-cols-2 divide-x divide-outline-variant/20">
                        {[
                          { label: 'Attention', icon: <Brain className="w-4 h-4 text-secondary" />, color: 'text-secondary', data: block.subcomponents.attention },
                          { label: 'MLP (FFN)',  icon: <Cpu   className="w-4 h-4 text-tertiary"  />, color: 'text-tertiary',  data: block.subcomponents.MLP  },
                        ].map(({ label, icon, color, data }) => {
                          const subHitRate = (1 - data.cache_behavior.L3_miss_rate) * 100;
                          const runtimeS   = (data.runtime_us ?? 0) / 1_000_000;
                          return (
                            <div key={label} className="p-6 space-y-4">
                              <div className="flex items-center gap-2 mb-4">
                                {icon}
                                <h4 className={`font-headline font-bold text-sm uppercase tracking-widest ${color}`}>{label}</h4>
                              </div>
                              <div className="space-y-3 font-mono text-[11px]">
                                {[
                                  { label: 'Runtime',     value: si(runtimeS, 's') },
                                  { label: 'FLOPs',       value: si(data.FLOPs, 'FLOPs') },
                                  { label: 'Bytes Moved', value: si(data.bytes_moved, 'B') },
                                  { label: 'Intensity',   value: `${f(data.arithmetic_intensity)} FLOPs/B` },
                                ].map(({ label: rowLabel, value }) => (
                                  <div key={rowLabel} className="flex justify-between items-center border-b border-outline-variant/10 pb-2">
                                    <span className="text-on-surface-variant">{rowLabel}</span>
                                    <span className="text-white font-bold">{value}</span>
                                  </div>
                                ))}
                                <div className="pt-1">
                                  <div className="flex justify-between items-center mb-1.5">
                                    <span className="text-on-surface-variant">L3 Hit Rate</span>
                                    <span className={`font-bold ${subHitRate < 50 ? 'text-error' : 'text-secondary'}`}>{f(subHitRate)}%</span>
                                  </div>
                                  <div className="flex gap-0.5 h-1.5 rounded-full overflow-hidden">
                                    <div className="h-full bg-secondary rounded-l-full" style={{ width: `${subHitRate}%` }} />
                                    <div className="h-full bg-error rounded-r-full"   style={{ width: `${100 - subHitRate}%` }} />
                                  </div>
                                </div>
                              </div>
                            </div>
                          );
                        })}
                      </div>
                      <div className="border-t border-outline-variant/20 px-6 py-4 flex justify-end">
                        <button
                          onClick={() => navigateToBlock(block, blockId)}
                          className="flex items-center gap-2 text-[11px] font-mono font-bold uppercase tracking-widest text-secondary bg-secondary/10 hover:bg-secondary/20 border border-secondary/30 hover:border-secondary/60 px-4 py-2 rounded transition-all"
                        >
                          Advanced Layer Details
                          <ArrowRight className="w-3.5 h-3.5" />
                        </button>
                      </div>
                    </div>
                  </motion.section>
                )}
              </AnimatePresence>
            </React.Fragment>
          );
        })}
      </div>

      {/* Summary cards */}
      <div className="grid grid-cols-3 gap-6">
        <MetricCardSmall
          title="Inter-Block Latency"
          value={si(interBlockLatencyS, 's')}
          subtext="Avg transition between blocks"
          icon={<ArrowRightLeft className="w-8 h-8" />}
        />
        <MetricCardSmall
          title="Parallelism Factor"
          value={`${parallelismFactor}x`}
          subtext="Attention Head Pool"
          icon={<Zap className="w-8 h-8" />}
        />
        <MetricCardSmall
          title="IO Wait State"
          value={ioWaitState}
          subtext="Weight Offload Bottleneck"
          icon={<MemoryStick className="w-8 h-8" />}
          color="text-tertiary"
        />
      </div>
    </div>
  );
}

function MetricCardSmall({ title, value, subtext, icon, color = 'text-white' }: any) {
  return (
    <div className="bg-surface-container p-6 rounded-lg relative overflow-hidden group border border-outline-variant/10">
      <div className="absolute top-0 right-0 p-2 opacity-10 group-hover:opacity-20 transition-opacity">
        {icon}
      </div>
      <div className="text-[10px] text-on-surface-variant font-mono uppercase font-bold">{title}</div>
      <div className={`text-3xl font-headline font-bold mt-1 ${color}`}>{value}</div>
      <div className="text-[10px] text-outline mt-2 font-bold">{subtext}</div>
    </div>
  );
}
