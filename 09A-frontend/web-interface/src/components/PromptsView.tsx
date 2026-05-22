import React, {useEffect} from 'react';
import { MessageSquare, Send, History, Trash2, Copy, Zap, Terminal, Cpu, Activity, CheckCircle2} from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';
import { useAppState } from '@/src/controller/AppContext.tsx';
import { fetchAndSetPapiEvents} from "@/src/controller/Controller.tsx";

interface PromptsViewProps {
  onViewChange: (view: 'models' | 'prompts' | 'top' | 'phase' | 'decoder' | 'attention' | 'layer') => void;
}

export default function PromptsView({ onViewChange }: PromptsViewProps) {
  const { state, set } = useAppState();
  const { contextLength, inferenceMessages: messages, availableHooks, maxTokens, papiEventsPerRun } = state;
  const [selectedPreset, setSelectedPreset] = React.useState<'general' | 'advanced'>('general');
  const [selectedEvents, setSelectedEvents] = React.useState<string[]>([]);
  const [isRunning, setIsRunning] = React.useState(false);
  const [prompt, setPrompt] = React.useState('');
  const [sessionId, setSessionId] = React.useState<string | null>(null);
  const [chatDone, setChatDone] = React.useState(false);
  const [isProfiling, setIsProfiling] = React.useState(false);

  // Profiling Progress Popup states
  const [showProgressModal, setShowProgressModal] = React.useState(false);
  const [activeStep, setActiveStep] = React.useState('Initializing Profiler');
  const [progressPercent, setProgressPercent] = React.useState(0);
  const [profilingLogs, setProfilingLogs] = React.useState<string[]>([]);
  const terminalBottomRef = React.useRef<HTMLDivElement>(null);

  const esRef = React.useRef<EventSource | null>(null);
  const bottomRef = React.useRef<HTMLDivElement>(null);
  const messagesRef = React.useRef(messages);
  React.useEffect(() => { messagesRef.current = messages; }, [messages]);

  React.useEffect(() => {
    if (showProgressModal) {
      terminalBottomRef.current?.scrollIntoView({ behavior: 'smooth' });
    }
  }, [profilingLogs, showProgressModal]);

  const isLocked = sessionId !== null;

  React.useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  useEffect(() => {
    fetchAndSetPapiEvents(set);
  }, []);

  // Helper: filter raw stdout lines to extract only assistant response text
  const isNoiseLine = (line: string): boolean => {
    const trimmed = line.trim();
    if (!trimmed) return true;
    if (trimmed.startsWith('PAPI:')) return true;
    if (/^---\s*Turn\s+\d+/.test(trimmed)) return true;
    if (trimmed === 'User input processed.') return true;
    if (/^User \(or 'quit'/.test(trimmed)) return true;
    if (/^Re-packing/.test(trimmed)) return true;
    if (/^Final run count/.test(trimmed)) return true;
    if (/^Running group/.test(trimmed)) return true;
    if (/^Event /.test(trimmed)) return true;
    return false;
  };

  const extractAssistantText = (line: string): string | null => {
    // "Assistant: Hello! How can I help?" → "Hello! How can I help?"
    const match = line.match(/^Assistant:\s*(.*)/);
    if (match) return match[1];
    return null;
  };

  // First prompt — starts the chat
  const handleExecute = async () => {
    if (!prompt.trim()) return;
    const userText = prompt.trim();
    setPrompt('');
    const withUser = [...messagesRef.current, { role: 'user' as const, text: userText }];
    set('inferenceMessages', withUser);
    messagesRef.current = withUser;
    setIsRunning(true);
    setChatDone(false);

    try {
      const selectedModel = state.models.find(m => m.id === state.selectedModelId);
      const modelPath = selectedModel ? selectedModel.id : '';

      const res = await fetch('/api/run/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          model_path: modelPath,
          prompt: userText,
          n_predict: maxTokens,
          papi_events_per_run: papiEventsPerRun,
          custom_events: selectedPreset === 'advanced' && selectedEvents.length > 0 ? selectedEvents : null,
        }),
      });

      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.detail || 'Failed to start chat.');
      }

      const { sessionId: sid } = await res.json();
      setSessionId(sid);

      const es = new EventSource(`/api/run/stream/${sid}`);
      esRef.current = es;
      let assistantBuffer = '';

      es.onmessage = (e) => {
        const msg = JSON.parse(e.data) as { line?: string; chat_done?: boolean; done?: boolean; error?: string };

        if (msg.error) {
          es.close(); esRef.current = null;
          setSessionId(null); setIsRunning(false);
          const updated = [...messagesRef.current, { role: 'assistant' as const, text: `Error: ${msg.error}` }];
          set('inferenceMessages', updated); messagesRef.current = updated;
          return;
        }

        if (msg.chat_done) {
          // Chat process ended — keep sessionId alive for profiling
          es.close(); esRef.current = null;
          setIsRunning(false);
          setChatDone(true);
          // Flush remaining assistant buffer
          if (assistantBuffer.trim()) {
            const current = messagesRef.current;
            const last = current[current.length - 1];
            if (last?.role === 'assistant') {
              const updated = [...current.slice(0, -1), { role: 'assistant' as const, text: assistantBuffer.trim() }];
              set('inferenceMessages', updated); messagesRef.current = updated;
            }
          }
          return;
        }

        if (msg.line !== undefined) {
          const line = msg.line;

          if (line.trim() === '[TURN_DONE]') {
            setIsRunning(false);
            // Flush remaining assistant buffer
            if (assistantBuffer.trim()) {
              const current = messagesRef.current;
              const last = current[current.length - 1];
              if (last?.role === 'assistant') {
                const updated = [...current.slice(0, -1), { role: 'assistant' as const, text: assistantBuffer.trim() }];
                set('inferenceMessages', updated); messagesRef.current = updated;
              }
            }
            assistantBuffer = '';
            return;
          }

          // Skip noise lines
          if (isNoiseLine(line)) return;

          // Check if it's an assistant response line
          const assistantText = extractAssistantText(line);
          if (assistantText !== null) {
            // Start or continue the assistant response
            assistantBuffer = assistantText;
            const current = messagesRef.current;
            const last = current[current.length - 1];
            let updated: typeof current;
            if (last?.role === 'assistant') {
              updated = [...current.slice(0, -1), { role: 'assistant' as const, text: assistantBuffer }];
            } else {
              updated = [...current, { role: 'assistant' as const, text: assistantBuffer }];
            }
            set('inferenceMessages', updated); messagesRef.current = updated;
          } else if (assistantBuffer) {
            // Continuation of assistant text (multi-line responses)
            assistantBuffer += '\n' + line;
            const current = messagesRef.current;
            const last = current[current.length - 1];
            if (last?.role === 'assistant') {
              const updated = [...current.slice(0, -1), { role: 'assistant' as const, text: assistantBuffer }];
              set('inferenceMessages', updated); messagesRef.current = updated;
            }
          }
        }
      };

      es.onerror = () => {
        es.close(); esRef.current = null;
        setIsRunning(false);
      };
    } catch (error: any) {
      console.error(error);
      setIsRunning(false); setSessionId(null);
      const updated = [...messagesRef.current, { role: 'assistant' as const, text: `Error: ${error.message || error}` }];
      set('inferenceMessages', updated); messagesRef.current = updated;
    }
  };

  // Follow-up prompts — sent to stdin of the running process
  const handleFollowUp = async () => {
    if (!prompt.trim() || !sessionId) return;
    const userText = prompt.trim();
    setPrompt('');
    const updated = [...messagesRef.current, { role: 'user' as const, text: userText }];
    set('inferenceMessages', updated);
    messagesRef.current = updated;
    setIsRunning(true);

    try {
      await fetch(`/api/run/prompt/${sessionId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ prompt: userText }),
      });
    } catch (err: any) {
      console.error(err);
      setIsRunning(false);
    }
  };

  // Run profiling — ends the chat and runs the remaining profiling views
  const handleProfile = async () => {
    if (!sessionId) return;
    setIsProfiling(true);
    setShowProgressModal(true);
    setActiveStep('Initializing PAPI events and model layers...');
    setProgressPercent(5);
    setProfilingLogs([]);

    const statusMsg = [...messagesRef.current, { role: 'assistant' as const, text: '⏳ Running PAPI profiling across all views... This may take a minute.' }];
    set('inferenceMessages', statusMsg); messagesRef.current = statusMsg;

    try {
      const res = await fetch(`/api/run/profile/${sessionId}`, { method: 'POST' });
      if (!res.ok) {
        throw new Error('Failed to start profiling.');
      }

      // Open new EventSource for the profiling progress SSE stream
      const es = new EventSource(`/api/run/stream/${sessionId}`);
      
      es.onmessage = async (event) => {
        try {
          const msg = JSON.parse(event.data);
          
          if (msg.line !== undefined) {
            const line = msg.line;
            setProfilingLogs((prev) => [...prev, line]);
            
            if (line.includes('Phase View group')) {
              setActiveStep(`Running Phase View Profiling (${line.split('group ')[1]?.split(' ')[0] || ''})`);
              setProgressPercent(15);
            } else if (line.includes('Starting Top View')) {
              setActiveStep('Running Top View Global Telemetry Analysis');
              setProgressPercent(40);
            } else if (line.includes('Starting Decoder Block View')) {
              setActiveStep('Running Layer-by-Layer Decoder Block Analysis');
              setProgressPercent(60);
            } else if (line.includes('Starting Tensor-Op View')) {
              setActiveStep('Measuring Custom PAPI Events & Core Tensor Operations');
              setProgressPercent(80);
            } else if (line.includes('Re-packing') || line.includes('Final run count')) {
              setActiveStep('Consolidating hardware event parameters');
            }
          }

          if (msg.done) {
            es.close();
            // Profiling complete!
            setProgressPercent(100);
            setActiveStep('Profiling Complete! Aggregating database telemetry...');
            setIsProfiling(false);
            setSessionId(null);
            setChatDone(false);
            set('hasRunInference', true);
            set('resultsUpdated', true);
            const { fetchAndSetResults } = await import('@/src/controller/Controller.tsx');
            await fetchAndSetResults(set);
            const final = [...messagesRef.current.slice(0, -1), { role: 'assistant' as const, text: '✅ Profiling complete! Redirecting you to the Top View results...' }];
            set('inferenceMessages', final); messagesRef.current = final;
            
            setTimeout(() => {
              setShowProgressModal(false);
              onViewChange('top');
            }, 1500);
          }

          if (msg.error) {
            es.close();
            setActiveStep(`Error: ${msg.error}`);
            setProgressPercent(0);
            const updated = [...messagesRef.current, { role: 'assistant' as const, text: `Profiling error: ${msg.error}` }];
            set('inferenceMessages', updated); messagesRef.current = updated;
            setTimeout(() => setShowProgressModal(false), 3000);
            setIsProfiling(false);
            setSessionId(null);
            setChatDone(false);
          }
        } catch (parseErr) {
          // skip unparseable lines
        }
      };

      es.onerror = (err) => {
        console.error('EventSource connection error:', err);
      };

    } catch (error: any) {
      console.error(error);
      const updated = [...messagesRef.current, { role: 'assistant' as const, text: `Profiling error: ${error.message || error}` }];
      set('inferenceMessages', updated); messagesRef.current = updated;
      setActiveStep(`Error: ${error.message || error}`);
      setProgressPercent(0);
      setTimeout(() => setShowProgressModal(false), 3000);
      setIsProfiling(false);
      setSessionId(null);
      setChatDone(false);
    }
  };

  // Route to the right handler depending on whether a run is active
  const onSend = () => (sessionId && !chatDone) ? handleFollowUp() : handleExecute();

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
                    onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); onSend(); } }}
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
                      onClick={onSend}
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
                <div className="shrink-0 border-t border-outline-variant/10 px-4 py-3 flex flex-col gap-3 bg-surface-container">
                  {sessionId && !isProfiling && (
                    <motion.button
                      initial={{ opacity: 0, y: 10 }}
                      animate={{ opacity: 1, y: 0 }}
                      whileHover={{ scale: 1.02 }}
                      whileTap={{ scale: 0.98 }}
                      onClick={handleProfile}
                      disabled={isRunning}
                      className="w-full bg-secondary text-on-secondary py-3 rounded-lg font-headline font-bold text-sm flex items-center justify-center gap-2 shadow-[0_0_20px_rgba(137,206,255,0.15)] disabled:opacity-40 disabled:cursor-not-allowed disabled:shadow-none"
                    >
                      <Zap className="w-4 h-4" />
                      End Conversation & Run PAPI Profiling
                    </motion.button>
                  )}
                  {isProfiling && (
                    <div className="w-full text-center py-3 text-secondary font-mono text-xs uppercase tracking-widest animate-pulse">
                      ⏳ Profiling in progress...
                    </div>
                  )}
                  <div className="flex items-end gap-3">
                    <textarea
                      value={prompt}
                      onChange={e => setPrompt(e.target.value)}
                      onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); onSend(); } }}
                      rows={1}
                      className="flex-1 bg-surface-container-low border border-outline-variant/20 rounded-lg px-4 py-2.5 text-white font-mono text-sm focus:outline-none focus:border-primary/50 transition-colors resize-none"
                      style={{ maxHeight: '120px', overflowY: 'auto' }}
                      placeholder={!sessionId ? "Start a new conversation..." : "Continue the inference session..."}
                      disabled={isRunning || isProfiling}
                    />
                    <motion.button
                      whileHover={{ scale: 1.04 }}
                      whileTap={{ scale: 0.96 }}
                      onClick={onSend}
                      disabled={!prompt.trim() || isRunning || isProfiling}
                      className="bg-primary text-on-primary p-2.5 rounded-lg shadow-[0_0_16px_rgba(137,206,255,0.2)] disabled:opacity-40 disabled:cursor-not-allowed disabled:shadow-none shrink-0"
                    >
                      <Send className="w-4 h-4" />
                    </motion.button>
                  </div>
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
                  type="range" min={1} max={10000} step={1} value={maxTokens}
                  onChange={e => set('maxTokens', Number(e.target.value))}
                  disabled={isLocked}
                  className="param-slider w-full disabled:opacity-50 disabled:cursor-not-allowed"
                  style={{ background: `linear-gradient(to right, #89ceff ${(maxTokens - 1) / (10000 - 1) * 100}%, rgba(255,255,255,0.08) ${(maxTokens - 1) / (10000 - 1) * 100}%)` }}
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
                    disabled={isLocked}
                    className="param-slider w-full disabled:opacity-50 disabled:cursor-not-allowed"
                    style={{ background: `linear-gradient(to right, #89ceff ${(papiEventsPerRun - 1) / (10 - 1) * 100}%, rgba(255,255,255,0.08) ${(papiEventsPerRun - 1) / (10 - 1) * 100}%)` }}
                />
              </div>
            </div>
          </section>

          <section className="bg-surface-container p-6 rounded-xl border border-outline-variant/10">
            <h3 className="text-xs font-bold uppercase tracking-widest text-on-surface-variant mb-4">System Presets</h3>
            <div className="grid grid-cols-2 gap-3">
              <button
                disabled={isLocked}
                onClick={() => setSelectedPreset('general')}
                className={`p-3 rounded border text-[10px] font-bold uppercase tracking-widest transition-all ${
                  selectedPreset === 'general'
                    ? 'bg-primary/10 border-primary/40 text-primary'
                    : 'bg-surface-container-low border-outline-variant/20 text-outline hover:text-primary hover:border-primary/50'
                } disabled:opacity-50 disabled:cursor-not-allowed`}
              >
                General Preset
              </button>
              <button
                disabled={isLocked}
                onClick={() => setSelectedPreset('advanced')}
                className={`p-3 rounded border text-[10px] font-bold uppercase tracking-widest transition-all ${
                  selectedPreset === 'advanced'
                    ? 'bg-primary/10 border-primary/40 text-primary'
                    : 'bg-surface-container-low border-outline-variant/20 text-outline hover:text-primary hover:border-primary/50'
                } disabled:opacity-50 disabled:cursor-not-allowed`}
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
                  <div className="space-y-2 max-h-[300px] overflow-y-auto pr-1">
                    {availableHooks.map(hook => {
                      const isSelected = selectedEvents.includes(hook.id);
                      return (
                        <div
                          key={hook.id}
                          onClick={() => {
                            if (isLocked) return;
                            setSelectedEvents(prev =>
                              prev.includes(hook.id)
                                ? prev.filter(id => id !== hook.id)
                                : [...prev, hook.id]
                            );
                          }}
                          className={`flex items-center justify-between p-3 rounded border transition-all cursor-pointer select-none ${
                            isLocked ? 'opacity-60 cursor-not-allowed' : ''
                          } ${
                            isSelected
                              ? 'bg-primary/10 border-primary/40 text-primary shadow-[0_0_12px_rgba(137,206,255,0.1)]'
                              : 'bg-surface-container-low border-outline-variant/10 hover:border-primary/30 text-outline hover:text-primary'
                          }`}
                        >
                          <span className={`text-[10px] font-mono font-bold uppercase ${isSelected ? 'text-primary' : 'text-on-surface-variant'}`}>
                            {hook.label}
                          </span>
                          <span className="text-[10px] font-mono text-outline">{hook.id}</span>
                        </div>
                      );
                    })}
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

      <AnimatePresence>
        {showProgressModal && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 flex items-center justify-center bg-background/80 backdrop-blur-md p-4"
          >
            <motion.div
              initial={{ scale: 0.95, y: 20 }}
              animate={{ scale: 1, y: 0 }}
              exit={{ scale: 0.95, y: 20 }}
              transition={{ type: 'spring', duration: 0.5 }}
              className="w-full max-w-2xl bg-surface-container border border-primary/20 rounded-2xl shadow-[0_0_50px_rgba(137,206,255,0.15)] overflow-hidden"
            >
              {/* Header */}
              <div className="bg-surface-container-highest border-b border-outline-variant/10 px-6 py-4 flex items-center gap-3">
                <div className="w-10 h-10 rounded-full bg-primary/10 border border-primary/30 flex items-center justify-center animate-pulse">
                  <Activity className="w-5 h-5 text-primary animate-spin" style={{ animationDuration: '3s' }} />
                </div>
                <div>
                  <h3 className="font-headline font-bold text-base text-white">Hardware Profiling Active</h3>
                  <p className="text-[10px] font-mono uppercase tracking-widest text-primary font-bold">PAPI Counter Acquisition Layer</p>
                </div>
              </div>

              {/* Body */}
              <div className="p-6 space-y-6">
                {/* Active Step Indicator */}
                <div className="space-y-2">
                  <div className="flex justify-between items-center text-xs">
                    <span className="font-mono text-outline font-bold uppercase tracking-wider">Active Stage</span>
                    <span className="font-mono text-primary font-bold">{progressPercent}%</span>
                  </div>
                  <div className="font-headline text-lg text-white font-bold tracking-tight">
                    {activeStep}
                  </div>

                  {/* Custom Progress Bar */}
                  <div className="w-full h-2 bg-surface-container-highest rounded-full overflow-hidden relative border border-outline-variant/10">
                    <motion.div
                      initial={{ width: 0 }}
                      animate={{ width: `${progressPercent}%` }}
                      transition={{ duration: 0.3 }}
                      className="h-full bg-gradient-to-r from-primary to-secondary rounded-full shadow-[0_0_12px_rgba(137,206,255,0.4)]"
                    />
                  </div>
                </div>

                {/* Subtitle / Explainer */}
                <p className="text-xs text-on-surface-variant leading-relaxed">
                  Executing sequential multibatch inference passes to sample event groups across multiple execution paths. Hardware counters are bound dynamically to isolate arithmetic intensity, data caching misses, and tensor compute ratios.
                </p>

                {/* Live Console Terminal */}
                <div className="space-y-2">
                  <div className="flex items-center gap-2 text-[10px] font-mono font-bold uppercase text-outline tracking-wider">
                    <Terminal className="w-3.5 h-3.5 text-primary animate-pulse" />
                    Real-time Telemetry Stream
                  </div>
                  <div className="h-48 bg-black/40 border border-outline-variant/15 rounded-lg p-4 font-mono text-xs text-on-surface overflow-y-auto space-y-1.5 scrollbar-thin select-text">
                    {profilingLogs.length === 0 ? (
                      <div className="text-outline italic animate-pulse">Waiting for telemetry logs...</div>
                    ) : (
                      profilingLogs.map((log, index) => {
                        let textClass = "text-on-surface-variant";
                        if (log.startsWith('=====') || log.startsWith('Running group')) {
                          textClass = "text-primary font-bold";
                        } else if (log.startsWith('Warning')) {
                          textClass = "text-yellow-400";
                        } else if (log.startsWith('Error')) {
                          textClass = "text-error font-bold";
                        }
                        return (
                          <div key={index} className={textClass}>
                            {log}
                          </div>
                        );
                      })
                    )}
                    <div ref={terminalBottomRef} />
                  </div>
                </div>
              </div>

              {/* Footer */}
              <div className="bg-surface-container-highest border-t border-outline-variant/10 px-6 py-4 flex justify-between items-center text-[10px] font-mono text-outline">
                <span>DO NOT NAVIGATE OR CLOSE BROWSER</span>
                <span className="animate-pulse text-secondary">ACTIVE SAMPLING</span>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
