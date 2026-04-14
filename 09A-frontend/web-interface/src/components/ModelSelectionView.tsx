import React from 'react';
import { 
  Box, 
  Cpu, 
  Zap, 
  CheckCircle, 
  ChevronRight,
  Database,
  Activity
} from 'lucide-react';
import { motion } from 'motion/react';

export default function ModelSelectionView() {
  const models = [
    { id: 'gpt-kinetic-4-v2.1', name: 'GPT-Kinetic-4-v2.1', type: 'Transformer', params: '175B', status: 'Active', latency: '42ms', energy: 'Low', color: 'text-primary' },
    { id: 'gpt-kinetic-4-v2.0', name: 'GPT-Kinetic-4-v2.0', type: 'Transformer', params: '175B', status: 'Standby', latency: '48ms', energy: 'Medium', color: 'text-outline' },
    { id: 'kinetic-light-v1', name: 'Kinetic-Light-v1', type: 'MoE', params: '32B', status: 'Offline', latency: '12ms', energy: 'Ultra-Low', color: 'text-outline' },
    { id: 'synapse-heavy-x', name: 'Synapse-Heavy-X', type: 'Dense', params: '1.2T', status: 'Offline', latency: '180ms', energy: 'High', color: 'text-outline' },
  ];

  return (
    <div className="p-8 space-y-8 animate-in fade-in duration-500">
      <header className="mb-8">
        <h2 className="font-headline text-3xl font-light text-primary mb-1">Model Selection</h2>
        <p className="text-on-surface-variant font-mono text-xs uppercase tracking-widest">
          Available Inference Engines: <span className="text-secondary">4 Registered</span>
        </p>
      </header>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {models.map((model) => (
          <motion.div 
            key={model.id}
            whileHover={{ scale: 1.01 }}
            className={`bg-surface-container p-6 rounded-xl border transition-all cursor-pointer group ${
              model.status === 'Active' ? 'border-primary shadow-[0_0_20px_rgba(137,206,255,0.1)]' : 'border-outline-variant/10 hover:border-outline-variant/30'
            }`}
          >
            <div className="flex justify-between items-start mb-6">
              <div className="flex items-center gap-4">
                <div className={`p-3 rounded-lg ${model.status === 'Active' ? 'bg-primary/10' : 'bg-surface-container-highest'}`}>
                  <Box className={`w-6 h-6 ${model.status === 'Active' ? 'text-primary' : 'text-outline'}`} />
                </div>
                <div>
                  <h3 className={`font-headline text-xl font-bold ${model.status === 'Active' ? 'text-white' : 'text-outline'}`}>{model.name}</h3>
                  <div className="text-[10px] text-on-surface-variant font-mono uppercase font-bold">{model.type} Architecture</div>
                </div>
              </div>
              <div className={`text-[10px] px-2 py-0.5 rounded-full border font-bold uppercase ${
                model.status === 'Active' ? 'bg-secondary/10 text-secondary border-secondary/30' : 'bg-surface-container-highest text-outline border-outline-variant/30'
              }`}>
                {model.status}
              </div>
            </div>

            <div className="grid grid-cols-3 gap-4 mb-6">
              <div className="space-y-1">
                <div className="text-[10px] text-on-surface-variant uppercase font-bold">Parameters</div>
                <div className="text-lg font-headline font-bold text-white">{model.params}</div>
              </div>
              <div className="space-y-1">
                <div className="text-[10px] text-on-surface-variant uppercase font-bold">Latency</div>
                <div className="text-lg font-headline font-bold text-white">{model.latency}</div>
              </div>
              <div className="space-y-1">
                <div className="text-[10px] text-on-surface-variant uppercase font-bold">Energy Profile</div>
                <div className={`text-lg font-headline font-bold ${model.energy === 'Low' ? 'text-secondary' : 'text-white'}`}>{model.energy}</div>
              </div>
            </div>

            <div className="flex items-center justify-between pt-4 border-t border-outline-variant/10">
              <div className="flex items-center gap-2">
                <Cpu className="w-4 h-4 text-outline" />
                <span className="text-[10px] text-on-surface-variant uppercase font-bold">Optimized for H100</span>
              </div>
              <button className={`flex items-center gap-2 text-[10px] font-bold uppercase tracking-widest transition-colors ${
                model.status === 'Active' ? 'text-primary' : 'text-outline group-hover:text-white'
              }`}>
                {model.status === 'Active' ? 'Configure Engine' : 'Activate Model'}
                <ChevronRight className="w-4 h-4" />
              </button>
            </div>
          </motion.div>
        ))}
      </div>

      <section className="bg-surface-container p-8 rounded-xl border border-outline-variant/10">
        <h3 className="font-headline text-xl font-bold text-white mb-6 flex items-center gap-2">
          <Activity className="w-5 h-5 text-primary" />
          Global Fleet Status
        </h3>
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
          <div className="space-y-4">
            <div className="flex justify-between items-center text-xs">
              <span className="text-on-surface-variant">Total Capacity</span>
              <span className="font-mono text-white">42.8 PFLOPS</span>
            </div>
            <div className="h-1.5 bg-surface-container-highest rounded-full overflow-hidden">
              <div className="bg-primary h-full w-3/4"></div>
            </div>
          </div>
          <div className="space-y-4">
            <div className="flex justify-between items-center text-xs">
              <span className="text-on-surface-variant">Fleet Utilization</span>
              <span className="font-mono text-white">68.2%</span>
            </div>
            <div className="h-1.5 bg-surface-container-highest rounded-full overflow-hidden">
              <div className="bg-secondary h-full w-2/3"></div>
            </div>
          </div>
          <div className="space-y-4">
            <div className="flex justify-between items-center text-xs">
              <span className="text-on-surface-variant">Energy Efficiency</span>
              <span className="font-mono text-white">94.1%</span>
            </div>
            <div className="h-1.5 bg-surface-container-highest rounded-full overflow-hidden">
              <div className="bg-tertiary h-full w-[94%]"></div>
            </div>
          </div>
        </div>
      </section>
    </div>
  );
}
