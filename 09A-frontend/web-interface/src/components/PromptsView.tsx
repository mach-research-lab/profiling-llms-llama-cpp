import React, {useEffect} from 'react';
import { MessageSquare, Send, History, Trash2, Copy, Zap, Terminal, Cpu, Activity} from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';
import { useAppState } from '@/src/controller/AppContext.tsx';
import { fetchAndSetPapiEvents} from "@/src/controller/Controller.tsx";

//   Chat window style more
  //   Selection for papievents should be clear and maybe hidden
  //   Info page for that aswel
  //   Validation for papi event selection
  //   Reuse the roofline
// TODO:
// TODO:
// TODO:
// TODO:

export default function PromptsView() {
  const { state, set } = useAppState();
  const { contextLength, inferenceMessages: messages, availableHooks, maxTokens, papiEventsPerRun } = state;
  const [selectedPreset, setSelectedPreset] = React.useState<'general' | 'advanced'>('general');
  const [isRunning, setIsRunning] = React.useState(false);
  const [prompt, setPrompt] = React.useState('');
  const bottomRef = React.useRef<HTMLDivElement>(null);

  React.useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  useEffect(() => {
    fetchAndSetPapiEvents(set);
  }, []);

  const handleExecute = () => {
    if (!prompt.trim()) return;
    const userText = prompt.trim();
    setPrompt('');
    const withUser = [...messages, { role: 'user' as const, text: userText }];
    set('inferenceMessages', withUser);
    setIsRunning(true);
    setTimeout(() => {
      setIsRunning(false);
      set('hasRunInference', true);
      set('resultsUpdated', true);
      set('inferenceMessages', [...withUser, {
        role: 'assistant' as const,
        text: 'Inference complete. The arithmetic intensity of the SOFTMAX_KERNEL_NORM operation on an H100 at FP16 precision is approximately 161.0 FLOPs/Byte, placing it firmly in the compute-bound regime. Peak throughput was observed at 743 TFLOPS with a memory bandwidth utilization of 84.5%. Recommend enabling flash-attention v2 to reduce KV-cache pressure during the decode phase.',
      }]);
    }, 3000);
  };

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

      <AnimatePresence mode="wait">
        {isRunning && messages.length === 0 ? (
          <motion.div
            key="buffering"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.3 }}
            className="flex flex-col items-center justify-center py-32 gap-10"
          >
            {/* Spinning ring */}
            <div className="relative w-24 h-24">
              <motion.div
                animate={{ rotate: 360 }}
                transition={{ duration: 1.2, repeat: Infinity, ease: 'linear' }}
                className="absolute inset-0 rounded-full border-4 border-primary/20 border-t-primary"
              />
              <div className="absolute inset-0 flex items-center justify-center">
                <Cpu className="w-8 h-8 text-primary/60" />
              </div>
            </div>

            {/* Pulsing status line */}
            <div className="text-center space-y-2">
              <motion.p
                animate={{ opacity: [0.4, 1, 0.4] }}
                transition={{ duration: 1.5, repeat: Infinity, ease: 'easeInOut' }}
                className="font-headline text-xl text-primary tracking-tight"
              >
                Running Inference
              </motion.p>
              <p className="text-[10px] font-mono uppercase tracking-widest text-on-surface-variant">
                Processing prompt — please wait
              </p>
            </div>

            {/* Animated bar */}
            <div className="w-64 h-1 bg-surface-container-highest rounded-full overflow-hidden">
              <motion.div
                animate={{ x: ['-100%', '200%'] }}
                transition={{ duration: 1.4, repeat: Infinity, ease: 'easeInOut' }}
                className="w-1/2 h-full bg-primary rounded-full"
              />
            </div>
          </motion.div>
        ) : (
          <motion.div
            key="editor"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.3 }}
          >
      <div className="grid grid-cols-12 gap-8">
        {/* Left: Input Area / Chat Window */}
        <div className="col-span-12 lg:col-span-8">
          <AnimatePresence mode="wait">
            {messages.length === 0 ? (
              <motion.div
                key="input-view"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.2 }}
                className="space-y-6"
              >
                {/* System Input */}
                <div className="bg-surface-container p-6 rounded-xl border border-outline-variant/10 relative">
                  <div className="flex items-center gap-2 mb-4 text-xs font-bold uppercase tracking-widest text-on-surface-variant">
                    <Terminal className="w-4 h-4 text-primary" />
                    System Input
                  </div>
                  <textarea
                    value={prompt}
                    onChange={e => setPrompt(e.target.value)}
                    onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleExecute(); } }}
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
                        Context: {(contextLength / 1000).toFixed(0)}k
                      </div>
                    </div>
                    <motion.button
                      whileHover={{ scale: 1.02 }}
                      whileTap={{ scale: 0.98 }}
                      onClick={handleExecute}
                      disabled={!prompt.trim()}
                      className="bg-primary text-on-primary px-6 py-2 rounded font-headline font-bold text-sm flex items-center gap-2 shadow-[0_0_20px_rgba(137,206,255,0.2)] disabled:opacity-40 disabled:cursor-not-allowed disabled:shadow-none"
                    >
                      EXECUTE INFERENCE
                      <Send className="w-4 h-4" />
                    </motion.button>
                  </div>
                </div>

                {/* Recent Trace History */}
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
              </motion.div>
            ) : (
              <motion.div
                key="chat-view"
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.2 }}
                className="flex flex-col bg-surface-container rounded-xl border border-outline-variant/10 overflow-hidden"
                style={{ height: 'calc(100vh - 180px)', minHeight: '520px' }}
              >
                {/* Chat header */}
                <div className="flex items-center justify-between px-6 py-3 border-b border-outline-variant/10 bg-surface-container-highest shrink-0">
                  <div className="flex items-center gap-2 text-xs font-bold uppercase tracking-widest text-on-surface-variant">
                    <MessageSquare className="w-4 h-4 text-primary" />
                    Inference Session
                  </div>
                  <div className="flex items-center gap-2 text-[10px] text-on-surface-variant font-bold uppercase">
                    <MessageSquare className="w-3 h-3 text-primary" />
                    Context: {(contextLength / 1000).toFixed(0)}k
                  </div>
                </div>

                {/* Messages */}
                <div className="flex-1 overflow-y-auto p-6 space-y-6">
                  <AnimatePresence initial={false}>
                    {messages.map((msg, i) => (
                      <motion.div
                        key={i}
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        transition={{ duration: 0.25 }}
                        className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
                      >
                        {msg.role === 'assistant' && (
                          <div className="w-7 h-7 rounded-full bg-primary/10 border border-primary/30 flex items-center justify-center mr-3 mt-1 shrink-0">
                            <Cpu className="w-3.5 h-3.5 text-primary" />
                          </div>
                        )}
                        <div
                          className={`max-w-[80%] px-4 py-3 rounded-xl text-sm font-mono leading-relaxed ${
                            msg.role === 'user'
                              ? 'bg-primary/15 border border-primary/30 text-white rounded-br-sm'
                              : 'bg-surface-container-highest border border-outline-variant/20 text-on-surface rounded-bl-sm'
                          }`}
                        >
                          {msg.text}
                        </div>
                        {msg.role === 'user' && (
                          <div className="w-7 h-7 rounded-full bg-secondary/10 border border-secondary/30 flex items-center justify-center ml-3 mt-1 shrink-0">
                            <Terminal className="w-3.5 h-3.5 text-secondary" />
                          </div>
                        )}
                      </motion.div>
                    ))}
                  </AnimatePresence>

                  {/* Buffering indicator inside chat */}
                  <AnimatePresence>
                    {isRunning && (
                      <motion.div
                        initial={{ opacity: 0, y: 10 }}
                        animate={{ opacity: 1, y: 0 }}
                        exit={{ opacity: 0 }}
                        className="flex justify-start"
                      >
                        <div className="w-7 h-7 rounded-full bg-primary/10 border border-primary/30 flex items-center justify-center mr-3 mt-1 shrink-0">
                          <Cpu className="w-3.5 h-3.5 text-primary" />
                        </div>
                        <div className="bg-surface-container-highest border border-outline-variant/20 rounded-xl rounded-bl-sm px-4 py-3 flex items-center gap-3">
                          <motion.span
                            animate={{ opacity: [0.3, 1, 0.3] }}
                            transition={{ duration: 1.2, repeat: Infinity, ease: 'easeInOut' }}
                            className="text-[10px] font-mono uppercase tracking-widest text-on-surface-variant"
                          >
                            Processing
                          </motion.span>
                          <div className="flex gap-1">
                            {[0, 1, 2].map(d => (
                              <motion.div
                                key={d}
                                animate={{ opacity: [0.2, 1, 0.2] }}
                                transition={{ duration: 0.9, repeat: Infinity, delay: d * 0.2 }}
                                className="w-1.5 h-1.5 rounded-full bg-primary"
                              />
                            ))}
                          </div>
                        </div>
                      </motion.div>
                    )}
                  </AnimatePresence>
                  <div ref={bottomRef} />
                </div>

                {/* Compact input bar */}
                <div className="shrink-0 border-t border-outline-variant/10 px-4 py-3 flex items-end gap-3 bg-surface-container">
                  <textarea
                    value={prompt}
                    onChange={e => setPrompt(e.target.value)}
                    onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleExecute(); } }}
                    rows={1}
                    className="flex-1 bg-surface-container-low border border-outline-variant/20 rounded-lg px-4 py-2.5 text-white font-mono text-sm focus:outline-none focus:border-primary/50 transition-colors resize-none"
                    style={{ maxHeight: '120px', overflowY: 'auto' }}
                    placeholder="Continue the inference session..."
                    disabled={isRunning}
                  />
                  <motion.button
                    whileHover={{ scale: 1.04 }}
                    whileTap={{ scale: 0.96 }}
                    onClick={handleExecute}
                    disabled={!prompt.trim() || isRunning}
                    className="bg-primary text-on-primary p-2.5 rounded-lg shadow-[0_0_16px_rgba(137,206,255,0.2)] disabled:opacity-40 disabled:cursor-not-allowed disabled:shadow-none shrink-0"
                  >
                    <Send className="w-4 h-4" />
                  </motion.button>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
        </div>

        {/* Right: Config Panel */}
        <div className="col-span-12 lg:col-span-4 space-y-6">
          <section className="bg-surface-container p-6 rounded-xl border border-outline-variant/10">
            <style>{`
              .param-slider { appearance: none; height: 4px; border-radius: 9999px; outline: none; cursor: pointer; }
              .param-slider::-webkit-slider-thumb { appearance: none; width: 14px; height: 14px; border-radius: 50%; background: white; box-shadow: 0 1px 4px rgba(0,0,0,0.5); cursor: pointer; }
              .param-slider::-moz-range-thumb { width: 14px; height: 14px; border-radius: 50%; background: white; border: none; box-shadow: 0 1px 4px rgba(0,0,0,0.5); cursor: pointer; }
            `}</style>
            <h3 className="text-xs font-bold uppercase tracking-widest text-on-surface-variant mb-6">Inference Parameters</h3>
            <div className="space-y-6">

              <div className="space-y-3">
                <div className="flex justify-between text-[10px] font-bold uppercase">
                  <span className="text-on-surface-variant">Max Tokens</span>
                  <span className="text-primary">{maxTokens}</span>
                </div>
                <input
                  type="range" min={16} max={1056} step={16} value={maxTokens}
                  onChange={e => set('maxTokens', Number(e.target.value))}
                  className="param-slider w-full"
                  style={{ background: `linear-gradient(to right, #89ceff ${(maxTokens - 16) / (1056 - 16) * 100}%, rgba(255,255,255,0.08) ${(maxTokens - 16) / (1056 - 16) * 100}%)` }}
                />
              </div>
              <div className="space-y-3">
                <div className="flex justify-between text-[10px] font-bold uppercase">
                  <span className="text-on-surface-variant">Papi Events per run</span>
                  <span className="text-primary">{papiEventsPerRun}</span>
                </div>
                <input
                    type="range" min={1} max={10} step={1} value={papiEventsPerRun}
                    onChange={e => set('papiEventsPerRun', Number(e.target.value))}
                    className="param-slider w-full"
                    style={{ background: `linear-gradient(to right, #89ceff ${(papiEventsPerRun - 1) / (10 - 1) * 100}%, rgba(255,255,255,0.08) ${(papiEventsPerRun - 1) / (10 - 1) * 100}%)` }}
                />
              </div>
            </div>
          </section>

          <section className="bg-surface-container p-6 rounded-xl border border-outline-variant/10">
            <h3 className="text-xs font-bold uppercase tracking-widest text-on-surface-variant mb-4">System Presets</h3>
            <div className="grid grid-cols-2 gap-3">
              <button
                onClick={() => setSelectedPreset('general')}
                className={`p-3 rounded border text-[10px] font-bold uppercase tracking-widest transition-all ${
                  selectedPreset === 'general'
                    ? 'bg-primary/10 border-primary/40 text-primary'
                    : 'bg-surface-container-low border-outline-variant/20 text-outline hover:text-primary hover:border-primary/50'
                }`}
              >
                General Preset
              </button>
              <button
                onClick={() => setSelectedPreset('advanced')}
                className={`p-3 rounded border text-[10px] font-bold uppercase tracking-widest transition-all ${
                  selectedPreset === 'advanced'
                    ? 'bg-primary/10 border-primary/40 text-primary'
                    : 'bg-surface-container-low border-outline-variant/20 text-outline hover:text-primary hover:border-primary/50'
                }`}
              >
                Advanced
              </button>
            </div>
          </section>

          <AnimatePresence>
            {selectedPreset === 'advanced' && (
              <motion.section
                initial={{ opacity: 0, height: 0 }}
                animate={{ opacity: 1, height: 'auto' }}
                exit={{ opacity: 0, height: 0 }}
                transition={{ duration: 0.2 }}
                className="overflow-hidden"
              >
                <div className="bg-surface-container p-6 rounded-xl border border-outline-variant/10">
                  <h3 className="text-xs font-bold uppercase tracking-widest text-on-surface-variant mb-4">Available Papi Events</h3>
                  <div className="space-y-2">
                    {availableHooks.map(hook => (
                      <div key={hook.id} className="flex items-center justify-between p-3 bg-surface-container-low border border-outline-variant/10 rounded hover:border-primary/30 transition-all">
                        <span className="text-[10px] font-mono font-bold uppercase text-on-surface-variant">{hook.label}</span>
                        <span className="text-[10px] font-mono text-outline">{hook.id}</span>
                      </div>
                    ))}
                  </div>
                </div>
              </motion.section>
            )}
          </AnimatePresence>
        </div>
      </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
