import React from 'react';
import { 
  AlertTriangle, 
  Zap, 
  MemoryStick, 
  Cpu, 
  ArrowRightLeft,
  AlertCircle
} from 'lucide-react';
import { motion } from 'motion/react';
import { useAppState } from '@/src/controller/AppContext.tsx';

export default function DecoderBlockView() {
  const { state } = useAppState();
  const { totalEnergy, memoryUsedGB, interBlockLatencyMs, parallelismFactor, ioWaitState } = state;
  const blocks = [
    { id: 'DB-001', runtime: '12.45ms', flops: '4.82', bytes: '1.2GB', cache: '412 MB', cachePercent: 40, intensity: '4.02 ops/B', hit: 92, share: '3.12%', status: 'ok' },
    { id: 'DB-002', runtime: '18.10ms', flops: '6.14', bytes: '2.4GB', cache: '850 MB', cachePercent: 75, intensity: '2.56 ops/B', hit: 88, share: '4.52%', status: 'ok' },
    { id: 'DB-004', runtime: '24.50ms', flops: '8.12', bytes: '4.2GB', cache: '1.8 GB', cachePercent: 95, intensity: '1.93 ops/B', hit: 42, share: '6.12%', status: 'warning' },
    { id: 'DB-005', runtime: '12.11ms', flops: '4.78', bytes: '1.15GB', cache: '405 MB', cachePercent: 38, intensity: '4.15 ops/B', hit: 95, share: '3.03%', status: 'ok' },
  ];

  return (
    <div className="p-8 space-y-8 animate-in fade-in duration-500">
      <div className="flex justify-between items-end mb-8">
        <div>
          <h2 className="font-headline text-3xl font-light text-primary mb-1">Decoder Blocks</h2>
          <p className="text-on-surface-variant font-mono text-xs uppercase tracking-widest">
            Active Inference Cluster: <span className="text-secondary">#L-04-A-BLOCKS</span>
          </p>
        </div>
        <div className="flex gap-4">
          <div className="bg-surface-container p-3 rounded-lg border-l-2 border-tertiary">
            <div className="text-[10px] text-on-surface-variant mb-1 uppercase tracking-tighter font-bold">Total Energy</div>
            <div className="text-xl font-headline font-bold text-tertiary">{totalEnergy} kWh</div>
          </div>
          <div className="bg-surface-container p-3 rounded-lg border-l-2 border-secondary">
            <div className="text-[10px] text-on-surface-variant mb-1 uppercase tracking-tighter font-bold">Peak Memory</div>
            <div className="text-xl font-headline font-bold text-secondary">{memoryUsedGB} GB</div>
          </div>
        </div>
      </div>

      <div className="space-y-4">
        <div className="grid grid-cols-12 px-6 py-2 text-[10px] font-mono uppercase tracking-widest text-on-surface-variant bg-surface-container/30 rounded-t-lg font-bold">
          <div className="col-span-1">ID</div>
          <div className="col-span-1">Runtime</div>
          <div className="col-span-1">FLOPs (T)</div>
          <div className="col-span-2">Bytes Moved</div>
          <div className="col-span-2">KV-Cache</div>
          <div className="col-span-2">Intensity</div>
          <div className="col-span-2">Cache Hit</div>
          <div className="col-span-1 text-right">Share %</div>
        </div>

        {blocks.map((block) => (
          <motion.div 
            key={block.id}
            whileHover={{ scale: 1.005 }}
            className={`bg-surface-container p-4 rounded-lg border transition-all group ${
              block.status === 'warning' ? 'bg-error/5 border-error/20 hover:border-error/50' : 'border-outline-variant/10 hover:border-primary/30'
            }`}
          >
            <div className="grid grid-cols-12 items-center">
              <div className="col-span-1 flex items-center gap-2">
                <span className={`font-headline font-bold ${block.status === 'warning' ? 'text-error' : 'text-primary'}`}>{block.id}</span>
                {block.status === 'warning' && <AlertTriangle className="w-3 h-3 text-error" />}
              </div>
              <div className="col-span-1">
                <div className={`text-lg font-headline font-bold leading-none ${block.status === 'warning' ? 'text-error' : 'text-white'}`}>
                  {block.runtime.replace('ms', '')}<span className="text-[10px] text-on-surface-variant ml-0.5">ms</span>
                </div>
              </div>
              <div className="col-span-1">
                <div className="text-lg font-headline font-bold text-white leading-none">{block.flops}</div>
              </div>
              <div className="col-span-2">
                <div className="text-lg font-headline font-bold text-white leading-none">
                  {block.bytes.replace('GB', '')}<span className="text-[10px] text-on-surface-variant ml-0.5">GB</span>
                </div>
              </div>
              <div className="col-span-2 pr-4">
                <div className="text-xs font-mono mb-1 text-on-surface-variant font-bold">{block.cache}</div>
                <div className="w-full bg-surface-container-highest h-1 rounded-full overflow-hidden">
                  <div className={`h-full ${block.status === 'warning' ? 'bg-error' : 'bg-secondary'}`} style={{ width: `${block.cachePercent}%` }}></div>
                </div>
              </div>
              <div className="col-span-2">
                <span className={`text-[10px] px-2 py-0.5 border rounded font-mono font-bold ${
                  block.status === 'warning' ? 'bg-error/10 text-error border-error/20' : 'bg-primary/10 text-primary border-primary/20'
                }`}>
                  {block.intensity}
                </span>
              </div>
              <div className="col-span-2 pr-4">
                <div className="flex items-center gap-2">
                  <span className={`text-xs font-bold ${block.hit < 50 ? 'text-error' : 'text-secondary'}`}>{block.hit}%</span>
                  <div className="flex-1 h-1 bg-surface-container-highest rounded-full overflow-hidden">
                    <div className={`h-full ${block.hit < 50 ? 'bg-error' : 'bg-secondary'}`} style={{ width: `${block.hit}%` }}></div>
                  </div>
                </div>
              </div>
              <div className="col-span-1 text-right">
                <span className={`text-lg font-headline font-bold ${block.status === 'warning' ? 'text-error' : 'text-secondary'}`}>{block.share}</span>
              </div>
            </div>
          </motion.div>
        ))}
      </div>

      <div className="mt-8 grid grid-cols-3 gap-6">
        <MetricCardSmall
          title="Inter-Block Latency"
          value={String(interBlockLatencyMs)}
          unit="ms"
          subtext="● DB-004 → DB-005 Path"
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

function MetricCardSmall({ title, value, unit, subtext, icon, color = "text-white" }: any) {
  return (
    <div className="bg-surface-container p-6 rounded-lg relative overflow-hidden group border border-outline-variant/10">
      <div className="absolute top-0 right-0 p-2 opacity-10 group-hover:opacity-20 transition-opacity">
        {icon}
      </div>
      <div className="text-[10px] text-on-surface-variant font-mono uppercase font-bold">{title}</div>
      <div className={`text-3xl font-headline font-bold mt-1 ${color}`}>
        {value}{unit && <span className="text-lg text-on-surface-variant ml-1">{unit}</span>}
      </div>
      <div className="text-[10px] text-outline mt-2 font-bold">{subtext}</div>
    </div>
  );
}
