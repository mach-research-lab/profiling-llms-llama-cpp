import React, { useState } from 'react';
import { useAppState, AppState } from '../context/AppContext';

// ─── tiny field components ────────────────────────────────────────────────────

function SliderField({
  label, field, min, max, step = 1, format,
}: {
  label: string;
  field: keyof AppState;
  min: number; max: number; step?: number;
  format?: (v: number) => string;
}) {
  const { state, set } = useAppState();
  const value = state[field] as number;
  const display = format ? format(value) : String(value);
  return (
    <div className="mb-3">
      <div className="flex justify-between items-baseline mb-1">
        <span className="text-xs text-on-surface-variant">{label}</span>
        <span className="text-xs font-mono text-primary">{display}</span>
      </div>
      <input
        type="range" min={min} max={max} step={step}
        value={value}
        onChange={e => set(field, parseFloat(e.target.value) as AppState[typeof field])}
        className="w-full h-1 rounded-full appearance-none cursor-pointer accent-primary bg-surface-container-high"
      />
    </div>
  );
}

function NumberField({
  label, field, step = 1,
}: {
  label: string; field: keyof AppState; step?: number;
}) {
  const { state, set } = useAppState();
  const value = state[field] as number;
  return (
    <div className="mb-2 flex items-center gap-2">
      <span className="text-xs text-on-surface-variant flex-1 truncate">{label}</span>
      <input
        type="number" step={step} value={value}
        onChange={e => set(field, parseFloat(e.target.value) as AppState[typeof field])}
        className="w-24 text-right text-xs font-mono bg-surface-container-high border border-outline-variant rounded px-2 py-1 text-on-surface focus:outline-none focus:border-primary"
      />
    </div>
  );
}

function TextField({ label, field }: { label: string; field: keyof AppState }) {
  const { state, set } = useAppState();
  return (
    <div className="mb-2 flex items-center gap-2">
      <span className="text-xs text-on-surface-variant flex-1 truncate">{label}</span>
      <input
        type="text" value={state[field] as string}
        onChange={e => set(field, e.target.value as AppState[typeof field])}
        className="w-32 text-right text-xs font-mono bg-surface-container-high border border-outline-variant rounded px-2 py-1 text-on-surface focus:outline-none focus:border-primary"
      />
    </div>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  const [open, setOpen] = useState(true);
  return (
    <div className="mb-1">
      <button
        onClick={() => setOpen(o => !o)}
        className="w-full flex items-center justify-between py-2 text-xs font-semibold uppercase tracking-widest text-on-surface-variant hover:text-on-surface transition-colors"
      >
        <span>{title}</span>
        <span className="text-base leading-none">{open ? '−' : '+'}</span>
      </button>
      {open && (
        <div className="pb-1">
          {children}
        </div>
      )}
      <div className="border-t border-outline-variant/50 mt-1" />
    </div>
  );
}

// ─── main panel ───────────────────────────────────────────────────────────────

export default function ControlPanel() {
  const { reset } = useAppState();
  const [open, setOpen] = useState(false);

  return (
    <>
      {/* toggle button — fixed bottom-right */}
      <button
        onClick={() => setOpen(o => !o)}
        title="Open control panel"
        className="fixed bottom-6 right-6 z-50 w-12 h-12 rounded-full bg-primary text-on-primary shadow-lg flex items-center justify-center text-xl hover:brightness-110 transition-all"
        style={{ boxShadow: '0 4px 24px rgba(137,206,255,0.25)' }}
      >
        {open ? '✕' : '⚙'}
      </button>

      {/* panel */}
      <div
        className={`
          fixed top-16 right-0 bottom-0 z-40 w-72
          bg-surface border-l border-outline-variant
          overflow-y-auto transition-transform duration-300 ease-in-out
          ${open ? 'translate-x-0' : 'translate-x-full'}
        `}
        style={{ scrollbarWidth: 'thin' }}
      >
        {/* header */}
        <div className="sticky top-0 bg-surface z-10 px-4 py-3 border-b border-outline-variant flex items-center justify-between">
          <span className="text-sm font-semibold text-on-surface tracking-wide">Control Panel</span>
          <button
            onClick={reset}
            className="text-xs text-on-surface-variant hover:text-error border border-outline-variant hover:border-error rounded px-2 py-0.5 transition-colors"
          >
            Reset
          </button>
        </div>

        <div className="px-4 py-3">
          {/* ── Model ── */}
          <Section title="Model">
            <TextField label="Model name" field="modelName" />
            <NumberField label="Parameters (B)" field="parameterCount" />
            <SliderField label="Context length" field="contextLength" min={512} max={131072} step={512} format={v => `${(v/1024).toFixed(1)}k`} />
            <SliderField label="Batch size" field="batchSize" min={1} max={128} />
          </Section>

          {/* ── Performance ── */}
          <Section title="Performance">
            <SliderField label="Tokens / sec" field="tokensPerSecond" min={0} max={10000} step={10} format={v => v.toLocaleString()} />
            <SliderField label="Latency (ms)" field="latencyMs" min={1} max={500} />
            <SliderField label="Throughput (GB/s)" field="throughputGBs" min={0} max={2000} step={5} />
            <SliderField label="Memory used (GB)" field="memoryUsedGB" min={0} max={640} step={1} />
            <SliderField label="Memory total (GB)" field="memoryTotalGB" min={1} max={640} step={1} />
          </Section>

          {/* ── Attention / MLP ── */}
          <Section title="Attention / MLP">
            <SliderField label="Attention heads" field="numHeads" min={1} max={128} />
            <SliderField label="Head dim" field="headDim" min={32} max={256} step={32} />
            <SliderField label="MLP expansion" field="mlpExpansion" min={1} max={16} />
            <SliderField label="Dropout" field="attentionDropout" min={0} max={1} step={0.01} format={v => v.toFixed(2)} />
          </Section>

          {/* ── Layers / Training ── */}
          <Section title="Layers / Training">
            <SliderField label="Num layers" field="numLayers" min={1} max={160} />
            <NumberField label="Learning rate" field="learningRate" step={0.0001} />
            <NumberField label="Warmup steps" field="warmupSteps" step={100} />
            <SliderField label="Steps done" field="trainingStepsDone" min={0} max={500000} step={500} format={v => v.toLocaleString()} />
            <SliderField label="Steps total" field="trainingStepsTotal" min={1000} max={500000} step={1000} format={v => v.toLocaleString()} />
            <SliderField label="Loss" field="lossValue" min={0} max={10} step={0.01} format={v => v.toFixed(2)} />
          </Section>

          {/* ── Phase progress ── */}
          <Section title="Phase Progress (%)">
            <SliderField label="Prefill" field="prefillProgress" min={0} max={100} />
            <SliderField label="Decode" field="decodeProgress" min={0} max={100} />
            <SliderField label="KV-Cache" field="kvcacheProgress" min={0} max={100} />
          </Section>

          {/* ── Roofline / Compute ── */}
          <Section title="Roofline / Compute">
            <SliderField label="Arith. intensity (FLOP/B)" field="arithmeticIntensity" min={0} max={2000} step={5} />
            <SliderField label="Peak TFLOPS" field="peakTFLOPS" min={1} max={4000} step={10} />
            <SliderField label="Achieved TFLOPS" field="achievedTFLOPS" min={0} max={4000} step={10} />
          </Section>
        </div>
      </div>

      {/* dim overlay when open on narrow screens */}
      {open && (
        <div
          className="fixed inset-0 z-30 bg-black/30 lg:hidden"
          onClick={() => setOpen(false)}
        />
      )}
    </>
  );
}
