import React from 'react';
import { 
  Zap, 
  RefreshCw, 
  Clock, 
  Activity,
  AlertCircle
} from 'lucide-react';
import { motion } from 'motion/react';

export default function PhaseView() {
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
            <div className="text-xl font-headline font-bold text-tertiary">142.4ms</div>
          </div>
          <div className="bg-surface-container p-3 rounded-lg border-l-2 border-secondary">
            <div className="text-[10px] text-on-surface-variant mb-1 uppercase tracking-tighter font-bold">Throughput</div>
            <div className="text-xl font-headline font-bold text-secondary">84.2 tok/s</div>
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
          time="42.8ms"
          timePercent={30}
          flops="1.24 T"
          flopsTrend="↑ 12%"
          intensity="18.4 FLOPs/Byte"
          intensityPoint={{ x: 280, y: 80 }}
          bytesMoved="68.2 GB"
          ipc="3.82"
          energy="42.4 mJ"
          hitRate={94.2}
          matmul={85}
          primaryColor="bg-primary"
        />

        {/* DECODE PHASE */}
        <PhaseSection 
          title="PHASE 02: DECODE" 
          icon={<RefreshCw className="w-5 h-5 text-secondary" />}
          badge="MEMORY BOUND"
          badgeColor="text-secondary bg-secondary/10 border-secondary/30"
          time="99.6ms"
          timePercent={70}
          flops="0.18 T"
          flopsTrend="↓ 4%"
          intensity="0.42 FLOPs/Byte"
          intensityPoint={{ x: 20, y: 144 }}
          bytesMoved="428.1 GB"
          ipc="0.64"
          energy="182.8 mJ"
          hitRate={41.8}
          matmul={25}
          primaryColor="bg-secondary"
          isWarning
        />
      </div>

      {/* Global Resource Utilization Heatmap */}
      <div className="mt-8 bg-surface-container p-6 rounded-lg">
        <div className="flex justify-between items-center mb-4">
          <h4 className="text-xs font-bold uppercase tracking-widest">Core Utilization Profile</h4>
          <div className="flex gap-4 text-[10px] font-mono">
            <span className="flex items-center gap-1"><span className="w-2 h-2 bg-primary"></span> Prefill</span>
            <span className="flex items-center gap-1"><span className="w-2 h-2 bg-secondary"></span> Decode</span>
          </div>
        </div>
        <div className="grid grid-cols-16 gap-1 h-8">
          {[...Array(16)].map((_, i) => (
            <div 
              key={i} 
              className={`rounded-sm ${i < 8 ? 'bg-primary' : 'bg-secondary'}`} 
              style={{ opacity: 0.3 + Math.random() * 0.7 }}
            />
          ))}
        </div>
        <div className="flex justify-between mt-2 text-[8px] text-outline font-mono">
          <span>CORE_000</span>
          <span>CORE_128</span>
          <span>CORE_256</span>
        </div>
      </div>

      {/* Floating Alert */}
      <div className="fixed bottom-8 right-8 z-50 bg-surface-variant/40 backdrop-blur-xl p-4 rounded-xl border border-outline-variant/20 shadow-2xl w-80">
        <div className="flex items-center gap-3 mb-3">
          <div className="w-10 h-10 rounded-full bg-tertiary/20 flex items-center justify-center">
            <AlertCircle className="w-6 h-6 text-tertiary" />
          </div>
          <div>
            <div className="text-xs font-bold text-white uppercase">Phase 2 Bottleneck</div>
            <div className="text-[10px] text-on-surface-variant">HBM2e Bandwidth Saturation Detected</div>
          </div>
        </div>
        <p className="text-[10px] text-on-surface-variant leading-relaxed">
          The Decode phase is currently limited by weights-loading latency. Consider increasing KV-Cache quantization or reducing batch size to alleviate interconnect pressure.
        </p>
        <button className="mt-4 w-full py-1.5 bg-surface-container-highest text-primary text-[10px] font-bold uppercase tracking-widest hover:bg-primary hover:text-on-primary transition-all">
          Run Optimizer
        </button>
      </div>
    </div>
  );
}

function PhaseSection({ 
  title, icon, badge, badgeColor, time, timePercent, flops, flopsTrend, 
  intensity, intensityPoint, bytesMoved, ipc, energy, hitRate, matmul, primaryColor, isWarning 
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
          <div className={`text-[10px] mt-1 font-bold ${primaryColor.replace('bg-', 'text-')}`}>{timePercent}% of total inference</div>
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
            <div className="text-2xl font-headline font-bold">{hitRate}%</div>
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
