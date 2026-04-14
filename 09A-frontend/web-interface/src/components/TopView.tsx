import React from 'react';
import { 
  Info, 
  CheckCircle, 
  Timer, 
  Zap, 
  Activity, 
  Cpu, 
  AlertTriangle,
  MemoryStick,
  ArrowUpRight
} from 'lucide-react';
import { motion } from 'motion/react';

export default function TopView() {
  return (
    <div className="p-8 space-y-8 animate-in fade-in duration-500">
      {/* Header Section */}
      <header className="flex justify-between items-end">
        <div>
          <h1 className="font-headline text-3xl font-light text-primary mb-1">System Top View</h1>
          <p className="text-on-surface-variant font-mono text-xs uppercase tracking-widest">
            Real-time telemetry: <span className="text-secondary">GPT-Kinetic-4-v2.1</span>
          </p>
        </div>
        <div className="flex gap-4">
          <div className="bg-surface-container p-3 rounded-lg border-l-2 border-primary">
            <div className="text-[10px] text-on-surface-variant mb-1 uppercase tracking-tighter font-bold">Instance ID</div>
            <div className="text-xl font-headline font-bold text-primary">NODE-772-ALPH</div>
          </div>
          <div className="bg-surface-container p-3 rounded-lg border-l-2 border-secondary">
            <div className="text-[10px] text-on-surface-variant mb-1 uppercase tracking-tighter font-bold">Status</div>
            <div className="flex items-center gap-2 text-xl font-headline font-bold text-secondary">
              <span className="w-2 h-2 bg-secondary rounded-full animate-pulse"></span>
              SYNCHRONIZED
            </div>
          </div>
        </div>
      </header>

      {/* Primary Architecture & Metrics Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-6">
        {/* Left: Architecture Summary */}
        <div className="lg:col-span-4 space-y-6">
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
                <div className="text-2xl font-headline font-bold text-white mt-1">175.4 <span className="text-xs text-on-surface-variant">GB</span></div>
              </div>
              <div className="bg-surface-container-low p-4 rounded-lg border border-outline-variant/10">
                <div className="text-[10px] text-on-surface-variant font-mono uppercase font-bold">KV-Cache</div>
                <div className="text-2xl font-headline font-bold text-white mt-1">12.8 <span className="text-xs text-on-surface-variant">GB</span></div>
              </div>
            </div>
            <div className="mt-6">
              <div className="flex justify-between items-center text-[10px] text-on-surface-variant uppercase mb-2 font-bold">
                <span>Stability</span>
                <span className="text-secondary font-mono">99.2%</span>
              </div>
              <div className="h-1.5 bg-surface-container-lowest rounded-full overflow-hidden">
                <motion.div 
                  initial={{ width: 0 }}
                  animate={{ width: '99.2%' }}
                  transition={{ duration: 1, ease: "easeOut" }}
                  className="h-full bg-secondary" 
                />
              </div>
            </div>
          </section>

          {/* Health Monitor */}
          <section className="bg-surface-container p-6 rounded-lg border border-outline-variant/10">
            <div className="flex items-center gap-2 mb-6">
              <CheckCircle className="w-4 h-4 text-secondary" />
              <h4 className="text-xs font-bold uppercase tracking-widest">Health Monitor</h4>
            </div>
            <div className="space-y-4">
              <div className="flex justify-between items-center text-xs">
                <span className="text-on-surface-variant">Packet Loss</span>
                <span className="font-mono text-white">0.002%</span>
              </div>
              <div className="flex justify-between items-center text-xs">
                <span className="text-on-surface-variant">Thermal Status</span>
                <span className="text-secondary font-bold">OPTIMAL</span>
              </div>
            </div>
          </section>
        </div>

        {/* Right: Dynamic Metrics Bento */}
        <div className="lg:col-span-8 grid grid-cols-1 md:grid-cols-3 gap-6">
          {/* Row 1 */}
          <MetricCard 
            title="Total Runtime" 
            value="142:32:05" 
            icon={<Timer className="w-8 h-8" />} 
          />
          <MetricCard 
            title="Avg Tokens/s" 
            value="1,240" 
            trend="↑ 4.2% Peak" 
            icon={<Activity className="w-8 h-8" />} 
          />
          <MetricCard 
            title="Total Energy" 
            value="4.2" 
            unit="kW/h" 
            icon={<Zap className="w-8 h-8" />} 
            color="text-tertiary"
          />
          
          {/* Row 2 */}
          <MetricCard 
            title="Peak RSS Mem" 
            value="188.2" 
            unit="GB" 
            icon={<MemoryStick className="w-8 h-8" />} 
          />
          <MetricCard 
            title="Avg CPU Util" 
            value="84.5" 
            unit="%" 
            progress={84.5}
            icon={<Cpu className="w-8 h-8" />} 
          />
          <MetricCard 
            title="Cache Misses" 
            value="4.1" 
            unit="%" 
            icon={<AlertTriangle className="w-8 h-8 text-error" />} 
            isWarning
          />

          {/* Row 3 */}
          <div className="bg-surface-container p-6 rounded-lg md:col-span-1 border-l-2 border-primary">
            <div className="text-[10px] text-on-surface-variant font-mono uppercase font-bold">Input Tokens</div>
            <div className="text-2xl font-headline font-bold text-white mt-2">8.4 <span className="text-sm text-on-surface-variant">M</span></div>
          </div>
          <div className="bg-surface-container p-6 rounded-lg md:col-span-1 border-l-2 border-secondary">
            <div className="text-[10px] text-on-surface-variant font-mono uppercase font-bold">Output Tokens</div>
            <div className="text-2xl font-headline font-bold text-white mt-2">12.9 <span className="text-sm text-on-surface-variant">M</span></div>
          </div>
          <div className="bg-surface-container p-6 rounded-lg md:col-span-1 border-l-2 border-tertiary">
            <div className="text-[10px] text-on-surface-variant font-mono uppercase font-bold">Power Ribbon</div>
            <div className="flex items-center gap-1 mt-2">
              <div className="w-1 h-4 bg-tertiary/20"></div>
              <div className="w-1 h-6 bg-tertiary/40"></div>
              <div className="w-1 h-8 bg-tertiary/60"></div>
              <div className="w-1 h-4 bg-tertiary"></div>
              <span className="text-xs font-mono text-tertiary ml-2">422W</span>
            </div>
          </div>
        </div>
      </div>

      {/* Latency Profile */}
      <section className="bg-surface-container p-8 rounded-lg border border-outline-variant/10 overflow-hidden relative">
        <h4 className="text-xs font-bold uppercase tracking-widest text-on-surface-variant mb-8 flex items-center gap-2">
          <span className="w-2 h-2 rounded-full bg-primary animate-pulse"></span>
          Inference Latency Profile
        </h4>
        <div className="h-64 w-full relative">
          <svg className="w-full h-full" preserveAspectRatio="none" viewBox="0 0 800 200">
            <defs>
              <linearGradient id="latencyGradient" x1="0" x2="0" y1="0" y2="1">
                <stop offset="0%" stopColor="#89ceff" stopOpacity="0.2" />
                <stop offset="100%" stopColor="transparent" stopOpacity="0" />
              </linearGradient>
            </defs>
            <motion.path 
              initial={{ pathLength: 0 }}
              animate={{ pathLength: 1 }}
              transition={{ duration: 2, ease: "easeInOut" }}
              d="M0 150 Q 100 130, 200 160 T 400 100 T 600 50 T 800 120" 
              fill="none" 
              stroke="#89ceff" 
              strokeWidth="2" 
            />
            <motion.path 
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 1, duration: 1 }}
              d="M0 150 Q 100 130, 200 160 T 400 100 T 600 50 T 800 120 V 200 H 0 Z" 
              fill="url(#latencyGradient)" 
            />
          </svg>
          <div className="absolute bottom-4 left-4 flex gap-4 text-[10px] font-mono text-on-surface-variant">
            <span className="flex items-center gap-1"><span className="w-2 h-2 bg-primary"></span> ACTIVE THREADS (128)</span>
          </div>
        </div>
      </section>
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
