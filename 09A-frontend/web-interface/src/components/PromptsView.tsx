import React from 'react';
import { 
  MessageSquare, 
  Send, 
  History, 
  Trash2, 
  Copy,
  Zap,
  Terminal
} from 'lucide-react';
import { motion } from 'motion/react';

export default function PromptsView() {
  const history = [
    { id: 1, text: "Explain the concept of active inference in transformer models.", time: "2m ago", tokens: 42 },
    { id: 2, text: "Optimize the following CUDA kernel for H100 architecture...", time: "15m ago", tokens: 1240 },
    { id: 3, text: "What is the theoretical limit of arithmetic intensity for A100?", time: "1h ago", tokens: 18 },
  ];

  return (
    <div className="p-8 space-y-8 animate-in fade-in duration-500">
      <header className="mb-8">
        <h2 className="font-headline text-3xl font-light text-primary mb-1">Prompt Orchestrator</h2>
        <p className="text-on-surface-variant font-mono text-xs uppercase tracking-widest">
          Direct Interface: <span className="text-secondary">Low-Latency Path</span>
        </p>
      </header>

      <div className="grid grid-cols-12 gap-8">
        {/* Left: Input Area */}
        <div className="col-span-12 lg:col-span-8 space-y-6">
          <div className="bg-surface-container p-6 rounded-xl border border-outline-variant/10 relative">
            <div className="flex items-center gap-2 mb-4 text-xs font-bold uppercase tracking-widest text-on-surface-variant">
              <Terminal className="w-4 h-4 text-primary" />
              System Input
            </div>
            <textarea 
              className="w-full h-64 bg-surface-container-low border border-outline-variant/20 rounded-lg p-4 text-white font-mono text-sm focus:outline-none focus:border-primary/50 transition-colors resize-none"
              placeholder="Enter prompt for real-time inference analysis..."
            />
            <div className="flex justify-between items-center mt-4">
              <div className="flex gap-4">
                <div className="flex items-center gap-2 text-[10px] text-on-surface-variant font-bold uppercase">
                  <Zap className="w-3 h-3 text-secondary" />
                  Greedy Decoding
                </div>
                <div className="flex items-center gap-2 text-[10px] text-on-surface-variant font-bold uppercase">
                  <MessageSquare className="w-3 h-3 text-primary" />
                  Context: 8k
                </div>
              </div>
              <motion.button 
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                className="bg-primary text-on-primary px-6 py-2 rounded font-headline font-bold text-sm flex items-center gap-2 shadow-[0_0_20px_rgba(137,206,255,0.2)]"
              >
                EXECUTE INFERENCE
                <Send className="w-4 h-4" />
              </motion.button>
            </div>
          </div>

          <div className="bg-surface-container p-6 rounded-xl border border-outline-variant/10">
            <h3 className="text-xs font-bold uppercase tracking-widest text-on-surface-variant mb-4 flex items-center gap-2">
              <History className="w-4 h-4 text-primary" />
              Recent Trace History
            </h3>
            <div className="space-y-3">
              {history.map((item) => (
                <div key={item.id} className="flex items-center justify-between p-3 bg-surface-container-low border border-outline-variant/10 rounded group hover:border-primary/30 transition-all cursor-pointer">
                  <div className="flex items-center gap-4 overflow-hidden">
                    <span className="text-[10px] font-mono text-outline">{item.time}</span>
                    <p className="text-xs text-on-surface truncate">{item.text}</p>
                  </div>
                  <div className="flex items-center gap-4 pl-4">
                    <span className="text-[10px] font-mono text-secondary font-bold">{item.tokens} TOK</span>
                    <div className="flex gap-2 opacity-0 group-hover:opacity-100 transition-opacity">
                      <button className="text-outline hover:text-primary"><Copy className="w-3 h-3" /></button>
                      <button className="text-outline hover:text-error"><Trash2 className="w-3 h-3" /></button>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>

        {/* Right: Config Panel */}
        <div className="col-span-12 lg:col-span-4 space-y-6">
          <section className="bg-surface-container p-6 rounded-xl border border-outline-variant/10">
            <h3 className="text-xs font-bold uppercase tracking-widest text-on-surface-variant mb-6">Inference Parameters</h3>
            <div className="space-y-6">
              <div className="space-y-3">
                <div className="flex justify-between text-[10px] font-bold uppercase">
                  <span className="text-on-surface-variant">Temperature</span>
                  <span className="text-primary">0.7</span>
                </div>
                <div className="h-1 bg-surface-container-highest rounded-full">
                  <div className="bg-primary h-full w-[70%] relative">
                    <div className="absolute right-0 top-1/2 -translate-y-1/2 w-3 h-3 bg-white rounded-full shadow-lg"></div>
                  </div>
                </div>
              </div>
              <div className="space-y-3">
                <div className="flex justify-between text-[10px] font-bold uppercase">
                  <span className="text-on-surface-variant">Top-P</span>
                  <span className="text-primary">0.95</span>
                </div>
                <div className="h-1 bg-surface-container-highest rounded-full">
                  <div className="bg-primary h-full w-[95%] relative">
                    <div className="absolute right-0 top-1/2 -translate-y-1/2 w-3 h-3 bg-white rounded-full shadow-lg"></div>
                  </div>
                </div>
              </div>
              <div className="space-y-3">
                <div className="flex justify-between text-[10px] font-bold uppercase">
                  <span className="text-on-surface-variant">Max Tokens</span>
                  <span className="text-primary">2048</span>
                </div>
                <div className="h-1 bg-surface-container-highest rounded-full">
                  <div className="bg-primary h-full w-1/2 relative">
                    <div className="absolute right-0 top-1/2 -translate-y-1/2 w-3 h-3 bg-white rounded-full shadow-lg"></div>
                  </div>
                </div>
              </div>
            </div>
          </section>

          <section className="bg-surface-container p-6 rounded-xl border border-outline-variant/10">
            <h3 className="text-xs font-bold uppercase tracking-widest text-on-surface-variant mb-4">System Presets</h3>
            <div className="grid grid-cols-2 gap-3">
              <button className="p-3 bg-surface-container-low border border-outline-variant/20 rounded text-[10px] font-bold uppercase tracking-widest text-outline hover:text-primary hover:border-primary/50 transition-all">Creative</button>
              <button className="p-3 bg-primary/10 border border-primary/40 rounded text-[10px] font-bold uppercase tracking-widest text-primary">Precise</button>
              <button className="p-3 bg-surface-container-low border border-outline-variant/20 rounded text-[10px] font-bold uppercase tracking-widest text-outline hover:text-primary hover:border-primary/50 transition-all">Balanced</button>
              <button className="p-3 bg-surface-container-low border border-outline-variant/20 rounded text-[10px] font-bold uppercase tracking-widest text-outline hover:text-primary hover:border-primary/50 transition-all">Fast</button>
            </div>
          </section>
        </div>
      </div>
    </div>
  );
}
