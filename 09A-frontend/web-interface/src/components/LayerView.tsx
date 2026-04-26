import React from 'react';
import { 
  Layers, 
  ChevronDown, 
  ChevronRight, 
  CornerDownRight,
  Activity,
  Clock,
  Zap
} from 'lucide-react';
import { motion } from 'motion/react';
import { useAppState } from '@/src/controller/AppContext.tsx';

export default function LayerView() {
  const { state } = useAppState();
  const { latencyMs, memoryUsedGB, memoryTotalGB, computeEfficiencyPercent } = state;

  return (
    <div className="p-8 space-y-8 animate-in fade-in duration-500">
      <div className="flex justify-between items-end mb-8">
        <div>
          <h2 className="font-headline text-3xl font-light text-primary mb-1">Layer Operation Trace</h2>
          <p className="text-on-surface-variant font-mono text-xs uppercase tracking-widest">
            Active Layer Range: <span className="text-secondary">L24 - L26</span>
          </p>
        </div>
        <div className="flex gap-4">
          <div className="bg-surface-container p-3 rounded-lg border-l-2 border-primary">
            <div className="text-[10px] text-on-surface-variant mb-1 uppercase tracking-tighter font-bold">Inference Latency</div>
            <div className="text-xl font-headline font-bold text-primary">{latencyMs}ms</div>
          </div>
          <div className="bg-surface-container p-3 rounded-lg border-l-2 border-secondary">
            <div className="text-[10px] text-on-surface-variant mb-1 uppercase tracking-tighter font-bold">Compute Efficiency</div>
            <div className="text-xl font-headline font-bold text-secondary">{computeEfficiencyPercent}%</div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-12 gap-8">
        {/* Left Side: Granular Operation Trace */}
        <div className="col-span-12 lg:col-span-7">
          <section className="space-y-4">
            <div className="flex items-center gap-2 border-b border-outline-variant/20 pb-2">
              <Layers className="w-5 h-5 text-primary" />
              <h3 className="font-headline text-xl font-bold tracking-tight uppercase">Trace Execution Map</h3>
              <div className="ml-auto flex gap-2">
                <button className="px-3 py-1 bg-surface-container-high text-[10px] uppercase font-bold tracking-widest border border-outline-variant/30 rounded hover:bg-surface-container-highest transition-colors">Expand All</button>
              </div>
            </div>

            <div className="bg-surface-container rounded-lg overflow-hidden border border-outline-variant/10">
              <div className="p-4 space-y-2 font-mono">
                {/* Layer 24 Attention Block */}
                <div className="p-3 bg-surface-container-high border-l-4 border-primary/40 rounded flex items-center justify-between cursor-pointer hover:bg-surface-container-highest transition-colors">
                  <div className="flex items-center gap-3">
                    <ChevronDown className="w-4 h-4 text-outline" />
                    <span className="text-xs font-bold text-on-surface uppercase">L24_ATTENTION_BLOCK</span>
                  </div>
                  <div className="flex items-center gap-8">
                    <span className="text-xs text-secondary font-bold">8.42 GFLOPs</span>
                    <span className="text-xs text-primary font-bold">1.2ms</span>
                  </div>
                </div>

                {/* Children */}
                <div className="ml-6 space-y-1">
                  <div className="p-2 bg-surface-container-low border-l-2 border-outline-variant/30 flex items-center justify-between cursor-pointer hover:bg-surface-container transition-colors">
                    <div className="flex items-center gap-3">
                      <CornerDownRight className="w-4 h-4 text-outline" />
                      <span className="text-[11px] text-on-surface-variant font-medium">Q_PROJECTION_MATMUL</span>
                    </div>
                    <div className="flex items-center gap-8">
                      <span className="text-[11px] text-secondary/70">2.10 GFLOPs</span>
                      <span className="text-[11px] text-primary/70">0.4ms</span>
                    </div>
                  </div>

                  <div className="p-2 bg-primary/10 border-l-2 border-primary flex items-center justify-between cursor-pointer">
                    <div className="flex items-center gap-3">
                      <CornerDownRight className="w-4 h-4 text-primary" />
                      <span className="text-[11px] text-primary font-bold uppercase">SOFTMAX_KERNAL_NORM</span>
                    </div>
                    <div className="flex items-center gap-8">
                      <span className="text-[11px] text-secondary">0.08 GFLOPs</span>
                      <span className="text-[11px] text-primary">0.1ms</span>
                    </div>
                  </div>

                  <div className="p-2 bg-surface-container-low border-l-2 border-outline-variant/30 flex items-center justify-between cursor-pointer hover:bg-surface-container transition-colors">
                    <div className="flex items-center gap-3">
                      <CornerDownRight className="w-4 h-4 text-outline" />
                      <span className="text-[11px] text-on-surface-variant font-medium">V_VALUE_CONCAT</span>
                    </div>
                    <div className="flex items-center gap-8">
                      <span className="text-[11px] text-secondary/70">1.44 GFLOPs</span>
                      <span className="text-[11px] text-primary/70">0.3ms</span>
                    </div>
                  </div>
                </div>

                {/* Next Blocks */}
                <div className="p-3 bg-surface-container-high border-l-4 border-outline-variant/40 rounded flex items-center justify-between cursor-pointer mt-4 hover:bg-surface-container-highest transition-colors">
                  <div className="flex items-center gap-3">
                    <ChevronRight className="w-4 h-4 text-outline" />
                    <span className="text-xs font-bold text-on-surface uppercase">L24_MLP_FEED_FORWARD</span>
                  </div>
                  <div className="flex items-center gap-8">
                    <span className="text-xs text-secondary font-bold">12.1 GFLOPs</span>
                    <span className="text-xs text-primary font-bold">2.4ms</span>
                  </div>
                </div>

                <div className="p-3 bg-surface-container-high border-l-4 border-outline-variant/40 rounded flex items-center justify-between cursor-pointer hover:bg-surface-container-highest transition-colors">
                  <div className="flex items-center gap-3">
                    <ChevronRight className="w-4 h-4 text-outline" />
                    <span className="text-xs font-bold text-on-surface uppercase">L25_ATTENTION_BLOCK</span>
                  </div>
                  <div className="flex items-center gap-8">
                    <span className="text-xs text-secondary font-bold">8.42 GFLOPs</span>
                    <span className="text-xs text-primary font-bold">1.2ms</span>
                  </div>
                </div>
              </div>
            </div>
          </section>
        </div>

        {/* Right Side: Properties Panel */}
        <div className="col-span-12 lg:col-span-5 space-y-6">
          <section className="space-y-4">
            <div className="flex items-center gap-2 border-b border-outline-variant/20 pb-2">
              <Activity className="w-5 h-5 text-secondary" />
              <h3 className="font-headline text-xl font-bold tracking-tight uppercase">Operation Analysis</h3>
              <span className="ml-auto text-[10px] bg-secondary/10 text-secondary px-2 py-0.5 rounded-full border border-secondary/30 uppercase font-bold">Selected</span>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div className="bg-surface-container p-4 rounded-lg relative overflow-hidden group">
                <div className="absolute top-0 right-0 p-2 opacity-10 group-hover:opacity-20 transition-opacity">
                  <Clock className="w-12 h-12" />
                </div>
                <div className="text-[10px] text-on-surface-variant font-mono uppercase font-bold">Runtime</div>
                <div className="text-3xl font-headline font-bold text-white mt-1">0.124ms</div>
                <div className="w-full bg-surface-container-highest h-1.5 mt-3 rounded-full overflow-hidden">
                  <div className="bg-primary h-full" style={{ width: '25%' }}></div>
                </div>
                <div className="text-[10px] text-primary mt-1 font-bold">L24 Critical Path</div>
              </div>

              <div className="bg-surface-container p-4 rounded-lg relative overflow-hidden group">
                <div className="absolute top-0 right-0 p-2 opacity-10 group-hover:opacity-20 transition-opacity">
                  <Activity className="w-12 h-12" />
                </div>
                <div className="text-[10px] text-on-surface-variant font-mono uppercase font-bold">FLOPs</div>
                <div className="text-3xl font-headline font-bold text-white mt-1">82.4 <span className="text-lg text-on-surface-variant">M</span></div>
                <div className="flex items-center gap-2 mt-2">
                  <span className="text-secondary text-[10px] font-bold">↑ 4.2%</span>
                  <span className="text-[10px] text-on-surface-variant">vs. prev layer</span>
                </div>
              </div>
            </div>

            <div className="bg-surface-container p-6 rounded-lg border border-outline-variant/10">
              <div className="flex justify-between items-center mb-6">
                <h4 className="text-xs font-bold uppercase tracking-widest text-on-surface-variant flex items-center gap-2">
                  <span className="w-2 h-2 rounded-full bg-primary animate-pulse"></span>
                  Arithmetic Intensity
                </h4>
                <div className="text-[10px] font-mono text-outline">161.0 FLOPs/Byte</div>
              </div>
              <div className="h-40 w-full roofline-grid border-b border-l border-outline/30 relative">
                <svg className="absolute inset-0 w-full h-full">
                  <path d="M 0 160 L 100 80 L 400 80" fill="none" stroke="#3e4850" strokeDasharray="4 4" strokeWidth="2" />
                  <circle cx="180" cy="80" fill="#89ceff" r="6" className="drop-shadow-[0_0_8px_#89ceff]" />
                </svg>
                <div className="absolute bottom-2 right-2 text-[8px] text-outline font-mono">Memory Bound → Compute Bound</div>
              </div>
            </div>

            <div className="grid grid-cols-3 gap-4">
              <div className="bg-surface-container-low p-4 border border-outline-variant/10">
                <div className="text-[10px] text-on-surface-variant mb-2 font-bold uppercase">Bytes Moved</div>
                <div className="text-lg font-headline font-bold">512.2 <span className="text-xs text-on-surface-variant font-normal">KB</span></div>
              </div>
              <div className="bg-surface-container-low p-4 border border-outline-variant/10">
                <div className="text-[10px] text-on-surface-variant mb-2 font-bold uppercase">IPC</div>
                <div className="text-lg font-headline font-bold">3.42</div>
              </div>
              <div className="bg-surface-container-low p-4 border border-outline-variant/10">
                <div className="text-[10px] text-on-surface-variant mb-2 font-bold uppercase">Energy</div>
                <div className="text-lg font-headline font-bold text-tertiary">14.2 <span className="text-xs font-normal">mJ</span></div>
              </div>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div className="bg-surface-container p-4">
                <h4 className="text-[10px] font-bold text-on-surface-variant uppercase mb-4">Cache Behavior (L1/L2)</h4>
                <div className="flex items-end gap-2 mb-2">
                  <div className="text-2xl font-headline font-bold">98.2%</div>
                  <div className="text-[10px] text-secondary mb-1 font-bold">HIT RATE</div>
                </div>
                <div className="flex gap-1 h-2">
                  <div className="h-full bg-secondary" style={{ width: '98%' }}></div>
                  <div className="h-full bg-error" style={{ width: '2%' }}></div>
                </div>
              </div>
              <div className="bg-surface-container p-4 flex flex-col justify-center items-center relative">
                <svg className="w-20 h-20 -rotate-90">
                  <circle cx="40" cy="40" fill="transparent" r="32" stroke="#2d3449" strokeWidth="8" />
                  <circle cx="40" cy="40" fill="transparent" r="32" stroke="#89ceff" strokeDasharray="201" strokeDashoffset="30" strokeWidth="8" />
                </svg>
                <div className="absolute inset-0 flex flex-col items-center justify-center pt-2">
                  <span className="text-xs font-bold">85%</span>
                  <span className="text-[8px] text-on-surface-variant font-bold uppercase">Util</span>
                </div>
                <div className="mt-2 text-[10px] text-on-surface-variant uppercase font-bold">IPC Utilization</div>
              </div>
            </div>
          </section>
        </div>
      </div>
      
      {/* Contextual Footer */}
      <footer className="fixed bottom-0 left-64 right-0 bg-surface-container-lowest border-t border-outline-variant/10 px-6 h-8 flex items-center justify-between text-[9px] uppercase tracking-widest font-bold text-outline z-40">
        <div className="flex items-center gap-4">
          <span className="flex items-center gap-1"><span className="w-2 h-2 rounded-full bg-secondary"></span> System Online</span>
          <span>Cluster: kinetic-beta-04</span>
          <span>Kernel: synapse-v2.1</span>
        </div>
        <div className="flex items-center gap-4">
          <span className="text-primary">0% Error Rate</span>
          <span>Mem: {memoryUsedGB} / {memoryTotalGB} GB</span>
        </div>
      </footer>
    </div>
  );
}
