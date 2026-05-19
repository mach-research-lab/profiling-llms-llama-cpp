import React, {useEffect} from 'react';
import {Info, Timer, Zap, Activity} from 'lucide-react';
import { motion } from 'motion/react';
import { useAppState } from "@/src/controller/AppContext.tsx";
import { fetchAndSetModels, fetchAndSetResults, fmt, fmtSI } from "@/src/controller/Controller.tsx";


export default function TopView() {
  const { state, set } = useAppState();
  const {
    modelName, totalRuntimeS, tokensPerSecond, memoryUsedBytes, maxTokens, papiEventsPerRun,
    arithmeticIntensity, achievedFLOPS, peakFLOPS, memBwBs, ridgePoint,
    totalFLOPs, dramBytes, hwCpuModel, hwCores, hwBaseGHz, hwBoostGHz, hwAvgGHz, hwISA, hwFlopsPerCycle,
    modelSizeBytes, kvCapacityBytes, kvUsedBytes, kvTokensCapacity, kvTokensUsed, kvUtilPercent,
    inputTokens, outputTokens, cpuUtilPercent,
    papiL1Misses, papiL2Misses, papiL3Misses,
    energyPsysJ, energyPkgJ, energyCoresJ,
    decimalPrecision,
  } = state;

  const f = (n: number) => fmt(n, decimalPrecision);
  const si = (n: number, unit: string) => fmtSI(n, unit, decimalPrecision);

  useEffect(() => {
    fetchAndSetResults(set);
  }, []);

  return (
    <div className="p-8 space-y-8 animate-in fade-in duration-500">
      {/* Header Section */}
      <header className="flex justify-between items-end">
        <div>
          <h1 className="font-headline text-3xl font-light text-primary mb-1">Top View</h1>
          <p className="text-on-surface-variant font-mono text-xs uppercase tracking-widest">
            Model Analytics Overview: <span className="text-secondary">{modelName}</span>
          </p>
        </div>
        <div className="flex gap-4">
          <div className="bg-surface-container p-3 rounded-lg border-l-2 border-primary">
            <div className="text-[10px] text-on-surface-variant mb-1 uppercase tracking-tighter font-bold">Max Tokens</div>
            <div className="text-xl font-headline font-bold text-primary">{maxTokens}</div>
          </div>
          <div className="bg-surface-container p-3 rounded-lg border-l-2 border-secondary">
            <div className="text-[10px] text-on-surface-variant mb-1 uppercase tracking-tighter font-bold">PAPI Events / Run</div>
            <div className="text-xl font-headline font-bold text-secondary">{papiEventsPerRun}</div>
          </div>
        </div>
      </header>

      {/* Metric Cards — full width */}
      <div className="grid grid-cols-3 md:grid-cols-5 gap-6">
        <MetricCard title="Total Runtime" value={si(totalRuntimeS, 's')} icon={<Timer className="w-8 h-8" />} />
        <MetricCard title="Token Throughput" value={f(tokensPerSecond)} unit="tok/s" trend="↑ 4.2% Peak" icon={<Activity className="w-8 h-8" />} />
<div className="bg-surface-container p-6 rounded-lg border-l-2 border-primary">
          <div className="text-[10px] text-on-surface-variant font-mono uppercase font-bold">Input Tokens</div>
          <div className="text-2xl font-headline font-bold text-white mt-2">{inputTokens}</div>
        </div>
        <div className="bg-surface-container p-6 rounded-lg border-l-2 border-secondary">
          <div className="text-[10px] text-on-surface-variant font-mono uppercase font-bold">Output Tokens</div>
          <div className="text-2xl font-headline font-bold text-white mt-2">{outputTokens}</div>
        </div>
      </div>

      {/* Summary boxes — three columns */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">

        {/* Model Summary */}
        <section className="bg-surface-container p-6 rounded-lg border border-outline-variant/10 relative overflow-hidden group">
          <div className="absolute top-0 right-0 p-4 opacity-10 group-hover:opacity-20 transition-opacity">
            <Zap className="w-16 h-16 text-primary" />
          </div>
          <h2 className="font-headline text-xl font-bold text-white mb-4 flex items-center gap-2">
            <Info className="w-5 h-5 text-primary" />
            Model Summary
          </h2>
          <p className="text-sm text-on-surface-variant leading-relaxed mb-6">
            Transformer-based architecture optimized for low-latency active inference. FP16 precision kernels with quantization-aware paths.
          </p>
          <div className="grid grid-cols-2 gap-4">
            <div className="bg-surface-container-low p-4 rounded-lg border border-outline-variant/10">
              <div className="text-[10px] text-on-surface-variant font-mono uppercase font-bold">Model Size</div>
              <div className="text-2xl font-headline font-bold text-white mt-1">{si(modelSizeBytes, 'B')}</div>
            </div>
            <div className="bg-surface-container-low p-4 rounded-lg border border-outline-variant/10">
              <div className="text-[10px] text-on-surface-variant font-mono uppercase font-bold">Peak RSS</div>
              <div className="text-2xl font-headline font-bold text-white mt-1">{si(memoryUsedBytes, 'B')}</div>
            </div>
          </div>
          <div className="grid grid-cols-3 gap-4 mt-4">
            <div className="bg-surface-container-low p-4 rounded-lg border border-outline-variant/10">
              <div className="text-[10px] text-on-surface-variant font-mono uppercase font-bold mb-1">L1 Misses</div>
              <div className="text-lg font-headline font-bold text-white">{si(papiL1Misses, '')}</div>
            </div>
            <div className="bg-surface-container-low p-4 rounded-lg border border-outline-variant/10">
              <div className="text-[10px] text-on-surface-variant font-mono uppercase font-bold mb-1">L2 Misses</div>
              <div className="text-lg font-headline font-bold text-white">{si(papiL2Misses, '')}</div>
            </div>
            <div className="bg-surface-container-low p-4 rounded-lg border border-outline-variant/10">
              <div className="text-[10px] text-on-surface-variant font-mono uppercase font-bold mb-1">L3 Misses</div>
              <div className="text-lg font-headline font-bold text-white">{si(papiL3Misses, '')}</div>
            </div>
          </div>
        </section>

        {/* KV Cache */}
        <section className="bg-surface-container p-6 rounded-lg border border-outline-variant/10 relative overflow-hidden group">
          <div className="absolute top-0 right-0 p-4 opacity-10 group-hover:opacity-20 transition-opacity">
            <Activity className="w-16 h-16 text-secondary" />
          </div>
          <h2 className="font-headline text-xl font-bold text-white mb-4 flex items-center gap-2">
            <Activity className="w-5 h-5 text-secondary" />
            KV Cache
          </h2>
          <div className="grid grid-cols-2 gap-4">
            <div className="bg-surface-container-low p-4 rounded-lg border border-outline-variant/10">
              <div className="text-[10px] text-on-surface-variant font-mono uppercase font-bold">Capacity</div>
              <div className="text-2xl font-headline font-bold text-white mt-1">{si(kvCapacityBytes, 'B')}</div>
            </div>
            <div className="bg-surface-container-low p-4 rounded-lg border border-outline-variant/10">
              <div className="text-[10px] text-on-surface-variant font-mono uppercase font-bold">Used</div>
              <div className="text-2xl font-headline font-bold text-white mt-1">{si(kvUsedBytes, 'B')}</div>
            </div>
          </div>
          <div className="grid grid-cols-2 gap-4 mt-4">
            <div className="bg-surface-container-low p-4 rounded-lg border border-outline-variant/10">
              <div className="text-[10px] text-on-surface-variant font-mono uppercase font-bold">Token Capacity</div>
              <div className="text-2xl font-headline font-bold text-white mt-1">{si(kvTokensCapacity, '')}</div>
            </div>
            <div className="bg-surface-container-low p-4 rounded-lg border border-outline-variant/10">
              <div className="text-[10px] text-on-surface-variant font-mono uppercase font-bold">Tokens Used</div>
              <div className="text-2xl font-headline font-bold text-white mt-1">{kvTokensUsed}</div>
            </div>
          </div>
          <div className="mt-6">
            <div className="flex justify-between items-center text-[10px] text-on-surface-variant uppercase mb-2 font-bold">
              <span>Utilization</span>
              <span className="text-secondary font-mono">{f(kvUtilPercent)}%</span>
            </div>
            <div className="h-1.5 bg-surface-container-lowest rounded-full overflow-hidden">
              <motion.div
                initial={{ width: 0 }}
                animate={{ width: `${kvUtilPercent}%` }}
                transition={{ duration: 1, ease: "easeOut" }}
                className="h-full bg-secondary"
              />
            </div>
          </div>
        </section>

        {/* Energy & CPU */}
        <section className="bg-surface-container p-6 rounded-lg border border-outline-variant/10 relative overflow-hidden group">
          <div className="absolute top-0 right-0 p-4 opacity-10 group-hover:opacity-20 transition-opacity">
            <Zap className="w-16 h-16 text-tertiary" />
          </div>
          <h2 className="font-headline text-xl font-bold text-white mb-4 flex items-center gap-2">
            <Zap className="w-5 h-5 text-tertiary" />
            Energy & CPU
          </h2>
          <div className="grid grid-cols-2 gap-4">
            <div className="bg-surface-container-low p-4 rounded-lg border border-outline-variant/10">
              <div className="text-[10px] text-on-surface-variant font-mono uppercase font-bold">CPU Util</div>
              <div className="text-2xl font-headline font-bold text-white mt-1">{f(cpuUtilPercent)} <span className="text-xs text-on-surface-variant">%</span></div>
            </div>
            <div className="bg-surface-container-low p-4 rounded-lg border border-outline-variant/10">
              <div className="text-[10px] text-on-surface-variant font-mono uppercase font-bold">Total Energy</div>
              <div className="text-2xl font-headline font-bold text-tertiary mt-1">{si(energyPsysJ, 'J')}</div>
            </div>
          </div>
          <div className="grid grid-cols-2 gap-4 mt-4">
            <div className="bg-surface-container-low p-4 rounded-lg border border-outline-variant/10">
              <div className="text-[10px] text-on-surface-variant font-mono uppercase font-bold">CPU Energy</div>
              <div className="text-2xl font-headline font-bold text-white mt-1">{si(energyPkgJ, 'J')}</div>
            </div>
            <div className="bg-surface-container-low p-4 rounded-lg border border-outline-variant/10">
              <div className="text-[10px] text-on-surface-variant font-mono uppercase font-bold">Cores Energy</div>
              <div className="text-2xl font-headline font-bold text-white mt-1">{si(energyCoresJ, 'J')}</div>
            </div>
          </div>
        </section>

      </div>

      {/* Roofline */}
      {(() => {
        const achievedGFLOPS = achievedFLOPS / 1e9;
        const peakGFLOPS     = peakFLOPS / 1e9;
        const logRidge    = Math.log10(Math.max(ridgePoint, 1e-9));
        const logXMin     = -2;
        const logPerfMin  = -2;
        const logPeakGF   = Math.log10(Math.max(peakGFLOPS, 1e-9));
        const logOI       = Math.log10(Math.max(arithmeticIntensity, 1e-9));
        const dotX = logOI <= logRidge
          ? 100 * (logOI - logXMin) / (logRidge - logXMin)
          : 100 + 300 * (logOI - logRidge) / (3 - logRidge);
        const dotY = Math.min(284, Math.max(148,
          288 - 144 * (Math.log10(Math.max(achievedGFLOPS, 1e-9)) - logPerfMin) / (logPeakGF - logPerfMin)
        ));
        const isMemBound = arithmeticIntensity < ridgePoint;
        return (
          <section className="bg-surface-container p-6 rounded-lg border border-outline-variant/10">
            <div className="flex justify-between items-center mb-6">
              <h4 className="text-xs font-bold uppercase tracking-widest text-on-surface-variant flex items-center gap-2">
                <span className="w-2 h-2 rounded-full bg-primary animate-pulse"></span>
                Arithmetic Intensity — Entire Run
              </h4>
              <div className="text-[10px] font-mono text-outline">{f(arithmeticIntensity)} FLOPs/Byte</div>
            </div>
            <div className="flex gap-6">
              <div className="flex-[2] h-72 roofline-grid border-b border-l border-outline/30 relative">
                <svg className="absolute inset-0 w-full h-full">
                  <path d="M 0 288 L 100 144 L 400 144" fill="none" stroke="#3e4850" strokeDasharray="4 4" strokeWidth="2" />
                  <motion.circle
                    initial={{ opacity: 0, scale: 0 }}
                    animate={{ opacity: 1, scale: 1 }}
                    cx={dotX} cy={dotY}
                    fill="#89ceff" r="6"
                    className="drop-shadow-[0_0_8px_#89ceff]"
                  />
                </svg>
                <div className="absolute bottom-2 right-2 text-[8px] text-outline font-mono">Memory Bound → Compute Bound</div>
                <div className={`absolute top-2 left-2 text-xs font-mono font-bold ${isMemBound ? 'text-error' : 'text-secondary'}`}>
                  {isMemBound ? 'MEMORY BOUND' : 'COMPUTE BOUND'}
                </div>
              </div>

              {/* Data panel */}
              <div className="flex-1 space-y-4 font-mono text-[10px]">
                <div>
                  <div className="text-on-surface-variant uppercase font-bold mb-2 tracking-widest">Workload</div>
                  <div className="space-y-2">
                    {[
                      { label: 'Achieved', value: si(achievedFLOPS, 'FLOPS/s') },
                      { label: 'Total FLOPs', value: si(totalFLOPs, 'FLOPs') },
                      { label: 'DRAM Bytes', value: si(dramBytes, 'B') },
                      { label: 'OI', value: `${f(arithmeticIntensity)} FLOPs/B` },
                    ].map(({ label, value }) => (
                      <div key={label} className="flex justify-between items-center border-b border-outline-variant/10 pb-1">
                        <span className="text-on-surface-variant">{label}</span>
                        <span className="text-white font-bold">{value}</span>
                      </div>
                    ))}
                  </div>
                </div>
                <div>
                  <div className="text-on-surface-variant uppercase font-bold mb-2 tracking-widest">Hardware</div>
                  <div className="space-y-2">
                    {[
                      { label: 'CPU', value: hwCpuModel.split('@')[0].trim() },
                      { label: 'Cores', value: String(hwCores) },
                      { label: 'Base / Boost', value: `${hwBaseGHz} / ${hwBoostGHz} GHz` },
                      { label: 'Avg Clock', value: `${hwAvgGHz} GHz` },
                      { label: 'ISA', value: hwISA },
                      { label: 'FLOPs/Cycle', value: String(hwFlopsPerCycle) },
                      { label: 'Peak', value: si(peakFLOPS, 'FLOPS/s') },
                      { label: 'Mem BW', value: si(memBwBs, 'B/s') },
                      { label: 'Ridge Point', value: `${f(ridgePoint)} FLOPs/B` },
                    ].map(({ label, value }) => (
                      <div key={label} className="flex justify-between items-center border-b border-outline-variant/10 pb-1">
                        <span className="text-on-surface-variant">{label}</span>
                        <span className="text-white font-bold">{value}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </div>
            </div>
          </section>
        );
      })()}
    </div>
  );
}

function MetricCard({ title, value, unit, trend, icon, color = "text-white", progress, isWarning }: any) {
  return (
    <div className={`bg-surface-container p-6 rounded-lg relative overflow-hidden group border ${isWarning ? 'border-error/30' : 'border-outline-variant/10'}`}>
      <div className="absolute top-0 right-0 p-2 opacity-10 group-hover:opacity-20 transition-opacity">
        {icon}
      </div>
      <div className="text-[10px] text-on-surface-variant font-mono uppercase font-bold">{title}</div>
      <div className={`text-3xl font-headline font-bold mt-2 ${color}`}>
        {value} {unit && <span className="text-lg text-on-surface-variant">{unit}</span>}
      </div>
      {trend && <div className="mt-3 text-[10px] text-secondary font-bold">{trend}</div>}
      {progress !== undefined && (
        <div className="w-full bg-surface-container-highest h-1 mt-4 rounded-full overflow-hidden">
          <motion.div 
            initial={{ width: 0 }}
            animate={{ width: `${progress}%` }}
            transition={{ duration: 1, ease: "easeOut" }}
            className="bg-primary h-full" 
          />
        </div>
      )}
    </div>
  );
}
