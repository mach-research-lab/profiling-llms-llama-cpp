import React from 'react';
import { 
  Brain, 
  Cpu, 
  Clock, 
  Activity, 
  Zap,
  AlertCircle
} from 'lucide-react';
import { motion } from 'motion/react';

export default function AttentionMLPView() {
  return (
    <div className="p-8 space-y-8 animate-in fade-in duration-500">
      <div className="flex justify-between items-end mb-8">
        <div>
          <h2 className="font-headline text-3xl font-light text-primary mb-1">Attention & MLP Analytics</h2>
          <p className="text-on-surface-variant font-mono text-xs uppercase tracking-widest">
            Decoder Block: <span className="text-secondary">#042-STABLE-KERNEL</span>
          </p>
        </div>
        <div className="flex gap-4">
          <div className="bg-surface-container p-3 rounded-lg border-l-2 border-primary">
            <div className="text-[10px] text-on-surface-variant mb-1 uppercase tracking-tighter font-bold">Block Latency</div>
            <div className="text-xl font-headline font-bold text-primary">1.24ms</div>
          </div>
          <div className="bg-surface-container p-3 rounded-lg border-l-2 border-secondary">
            <div className="text-[10px] text-on-surface-variant mb-1 uppercase tracking-tighter font-bold">VRAM Context</div>
            <div className="text-xl font-headline font-bold text-secondary">42.8 GB</div>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* ATTENTION MODULE */}
        <SubBlockSection 
          title="ATTENTION SUB-BLOCK"
          icon={<Brain className="w-5 h-5 text-primary fill-current" />}
          badge="OPTIMIZED"
          badgeColor="text-primary bg-primary/10 border-primary/30"
          runtime="0.52ms"
          runtimePercent={42}
          flops="12.4 T"
          flopsTrend="↑ 8%"
          intensity="14.7 FLOPs/Byte"
          intensityPoint={{ x: 240, y: 80 }}
          bytesMoved="842 MB"
          hitRate={92}
          energy="8.4/10"
          l1={92}
          l2={78}
          primaryColor="bg-primary"
        />

        {/* MLP MODULE */}
        <SubBlockSection 
          title="MLP SUB-BLOCK (FFN)"
          icon={<Cpu className="w-5 h-5 text-tertiary" />}
          badge="BOTTLENECK"
          badgeColor="text-tertiary bg-tertiary/10 border-tertiary/30"
          runtime="0.72ms"
          runtimePercent={58}
          flops="42.1 T"
          flopsTrend="↓ 12%"
          intensity="17.5 FLOPs/Byte"
          intensityPoint={{ x: 320, y: 80 }}
          bytesMoved="2.4 GB"
          hitRate={64}
          energy="4.2/10"
          l1={64}
          l2={41}
          primaryColor="bg-tertiary"
          isWarning
        />
      </div>

      <div className="mt-8 bg-surface-container p-6 rounded-lg">
        <div className="flex justify-between items-center mb-4">
          <h4 className="text-xs font-bold uppercase tracking-widest">Sub-Block Resource Saturation Heatmap</h4>
          <div className="flex gap-4 text-[10px] font-mono">
            <span className="flex items-center gap-1"><span className="w-2 h-2 bg-primary"></span> Attention Ops</span>
            <span className="flex items-center gap-1"><span className="w-2 h-2 bg-tertiary"></span> MLP Ops</span>
          </div>
        </div>
        <div className="grid grid-cols-12 gap-1 h-8">
          {[...Array(12)].map((_, i) => (
            <div 
              key={i} 
              className={`rounded-sm ${i < 5 ? 'bg-primary' : i === 5 ? 'bg-surface-container-highest' : 'bg-tertiary'}`} 
              style={{ opacity: 0.4 + Math.random() * 0.6 }}
            />
          ))}
        </div>
        <div className="flex justify-between mt-2 text-[8px] text-outline font-mono">
          <span>HEAD_00</span>
          <span>PROJECTION_LAYER</span>
          <span>FFN_EXPANSION</span>
        </div>
      </div>

      {/* Floating Alert */}
      <div className="fixed bottom-8 right-8 z-50 bg-surface-variant/40 backdrop-blur-xl p-4 rounded-xl border border-outline-variant/20 shadow-2xl w-80">
        <div className="flex items-center gap-3 mb-3">
          <div className="w-10 h-10 rounded-full bg-error/20 flex items-center justify-center">
            <AlertCircle className="w-6 h-6 text-error" />
          </div>
          <div>
            <div className="text-xs font-bold text-white uppercase">MLP Stagnation</div>
            <div className="text-[10px] text-on-surface-variant">L2 Cache Thrashing in Projection</div>
          </div>
        </div>
        <p className="text-[10px] text-on-surface-variant leading-relaxed">
          The MLP sub-block runtime is significantly degraded by L2 misses. Recommendation: Adjust tiling size for FFN expansion layers to improve spatial locality on H100 cores.
        </p>
        <button className="mt-4 w-full py-1.5 bg-surface-container-highest text-primary text-[10px] font-bold uppercase tracking-widest hover:bg-primary hover:text-on-primary transition-all">
          Apply Re-Tiling
        </button>
      </div>
    </div>
  );
}

function SubBlockSection({ 
  title, icon, badge, badgeColor, runtime, runtimePercent, flops, flopsTrend, 
  intensity, intensityPoint, bytesMoved, hitRate, energy, l1, l2, primaryColor, isWarning 
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
          <div className="text-[10px] text-on-surface-variant font-mono uppercase font-bold">Runtime</div>
          <div className="text-3xl font-headline font-bold text-white mt-1">{runtime}</div>
          <div className="w-full bg-surface-container-highest h-1.5 mt-3 rounded-full overflow-hidden">
            <motion.div 
              initial={{ width: 0 }}
              animate={{ width: `${runtimePercent}%` }}
              transition={{ duration: 1 }}
              className={`h-full ${primaryColor}`} 
            />
          </div>
          <div className={`text-[10px] mt-1 font-bold ${primaryColor.replace('bg-', 'text-')}`}>{runtimePercent}% of block runtime</div>
        </div>
        
        <div className="bg-surface-container p-4 rounded-lg relative overflow-hidden group">
          <div className="absolute top-0 right-0 p-2 opacity-10 group-hover:opacity-20 transition-opacity">
            <Activity className="w-12 h-12" />
          </div>
          <div className="text-[10px] text-on-surface-variant font-mono uppercase font-bold">FLOPs</div>
          <div className="text-3xl font-headline font-bold text-white mt-1">{flops}</div>
          <div className="flex items-center gap-2 mt-2">
            <span className={`${flopsTrend.includes('↑') ? 'text-secondary' : 'text-error'} text-[10px] font-bold`}>{flopsTrend}</span>
            <span className="text-[10px] text-on-surface-variant">vs. prev block</span>
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
              fill={primaryColor === 'bg-primary' ? '#89ceff' : '#ffb95f'} 
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
          <div className="text-[10px] text-on-surface-variant mb-2 font-bold uppercase">L1 Hit Rate</div>
          <div className={`text-lg font-headline font-bold ${hitRate > 80 ? 'text-secondary' : 'text-error'}`}>{hitRate}%</div>
        </div>
        <div className="bg-surface-container-low p-4 border border-outline-variant/10">
          <div className="text-[10px] text-on-surface-variant mb-2 font-bold uppercase">Energy Efficiency</div>
          <div className="text-lg font-headline font-bold text-tertiary">{energy}</div>
        </div>
      </div>

      <div className="bg-surface-container p-4">
        <h4 className="text-[10px] font-bold text-on-surface-variant uppercase mb-4">Cache Efficiency (L1/L2)</h4>
        <div className="space-y-3">
          <div className="flex items-center gap-4">
            <div className="w-8 text-[10px] text-on-surface-variant font-bold">L1</div>
            <div className="flex-1 h-2 bg-surface-container-highest rounded-full overflow-hidden">
              <div className={`h-full ${primaryColor}`} style={{ width: `${l1}%` }}></div>
            </div>
            <div className="text-[10px] font-mono font-bold">{l1}%</div>
          </div>
          <div className="flex items-center gap-4">
            <div className="w-8 text-[10px] text-on-surface-variant font-bold">L2</div>
            <div className="flex-1 h-2 bg-surface-container-highest rounded-full overflow-hidden">
              <div className={`h-full ${primaryColor}`} style={{ width: `${l2}%` }}></div>
            </div>
            <div className="text-[10px] font-mono font-bold">{l2}%</div>
          </div>
        </div>
      </div>
    </section>
  );
}
