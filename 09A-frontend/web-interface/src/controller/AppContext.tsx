import React, { createContext, useContext, useState } from 'react';
import { Model } from '../types';

export interface AppState {
  // Model / hardware
  modelName: string;
  parameterCount: number;       // billions
  contextLength: number;        // tokens
  batchSize: number;
  totalRuntime: number;

  // Performance metrics
  tokensPerSecond: number;
  latencyMs: number;
  throughputGBs: number;
  memoryUsedGB: number;
  memoryTotalGB: number;

  // Attention / MLP
  numHeads: number;
  headDim: number;
  mlpExpansion: number;         // multiplier, e.g. 4
  attentionDropout: number;     // 0–1

  // Layer / training
  numLayers: number;
  learningRate: number;
  warmupSteps: number;
  trainingStepsDone: number;
  trainingStepsTotal: number;
  lossValue: number;

  // Phase progress (0–100)
  prefillProgress: number;
  decodeProgress: number;
  kvcacheProgress: number;

  // Roofline / compute
  arithmeticIntensity: number;  // FLOP/byte
  peakTFLOPS: number;
  achievedTFLOPS: number;

  // Energy
  totalEnergy: number;

  // Model summary / health
  modelSizeGB: number;
  kvCacheGB: number;
  stabilityPercent: number;
  packetLossPercent: number;
  inputTokensM: number;
  outputTokensM: number;
  powerWatts: number;
  cpuUtilPercent: number;
  cacheMissPercent: number;

  // Phase – prefill detail
  prefillTimeMs: number;
  prefillTimePercent: number;
  prefillFlopsTrillion: number;
  prefillFlopsTrendPct: number;   // positive = up, negative = down
  prefillIntensity: number;       // FLOPs/Byte
  prefillBytesMovedGB: number;
  prefillIPC: number;
  prefillEnergyMJ: number;
  prefillHitRate: number;
  prefillMatmulPct: number;

  // Phase – decode detail
  decodeTimeMs: number;
  decodeTimePercent: number;
  decodeFlopsTrillion: number;
  decodeFlopsTrendPct: number;
  decodeIntensity: number;
  decodeBytesMovedGB: number;
  decodeIPC: number;
  decodeEnergyMJ: number;
  decodeHitRate: number;
  decodeMatmulPct: number;

  // Decoder block
  interBlockLatencyMs: number;
  parallelismFactor: number;
  ioWaitState: string;

  // Attention sub-block
  blockLatencyMs: number;
  attentionRuntimeMs: number;
  attentionRuntimePct: number;
  attentionFlopsTrillion: number;
  attentionFlopsTrendPct: number;
  attentionIntensity: number;
  attentionBytesMoved: string;
  attentionHitRate: number;
  attentionEnergyScore: number;
  attentionL1: number;
  attentionL2: number;

  // MLP sub-block
  mlpRuntimeMs: number;
  mlpRuntimePct: number;
  mlpFlopsTrillion: number;
  mlpFlopsTrendPct: number;
  mlpIntensity: number;
  mlpBytesMoved: string;
  mlpHitRate: number;
  mlpEnergyScore: number;
  mlpL1: number;
  mlpL2: number;

  // Layer view
  computeEfficiencyPercent: number;

  // Fleet (ModelSelection)
  fleetCapacityPFLOPS: number;
  fleetUtilizationPercent: number;
  fleetEnergyEfficiencyPercent: number;

  // Available models
  models: Model[];
  selectedModelId: string;

  // Hooks
  availableHooks: { id: string; label: string }[];

  // Session state
  hasRunInference: boolean;
  resultsUpdated: boolean;
  inferenceMessages: { role: 'user' | 'assistant'; text: string }[];
}

const defaultState: AppState = {
  modelName: 'LLaMA-3 70B',
  parameterCount: 70,
  contextLength: 4096,
  batchSize: 8,
  totalRuntime: 112.4,

  tokensPerSecond: 1240,
  latencyMs: 38,
  throughputGBs: 142,
  memoryUsedGB: 48,
  memoryTotalGB: 80,

  numHeads: 32,
  headDim: 128,
  mlpExpansion: 4,
  attentionDropout: 0.1,

  numLayers: 80,
  learningRate: 0.0003,
  warmupSteps: 2000,
  trainingStepsDone: 14500,
  trainingStepsTotal: 50000,
  lossValue: 1.82,

  prefillProgress: 72,
  decodeProgress: 45,
  kvcacheProgress: 61,

  arithmeticIntensity: 312,
  peakTFLOPS: 989,
  achievedTFLOPS: 743,

  totalEnergy: 4.2,

  // Model summary / health
  modelSizeGB: 175.4,
  kvCacheGB: 12.8,
  stabilityPercent: 99.2,
  packetLossPercent: 0.002,
  inputTokensM: 8.4,
  outputTokensM: 12.9,
  powerWatts: 422,
  cpuUtilPercent: 84.5,
  cacheMissPercent: 4.1,

  // Phase – prefill detail
  prefillTimeMs: 42.8,
  prefillTimePercent: 30,
  prefillFlopsTrillion: 1.24,
  prefillFlopsTrendPct: 12,
  prefillIntensity: 18.4,
  prefillBytesMovedGB: 68.2,
  prefillIPC: 3.82,
  prefillEnergyMJ: 42.4,
  prefillHitRate: 94.2,
  prefillMatmulPct: 85,

  // Phase – decode detail
  decodeTimeMs: 99.6,
  decodeTimePercent: 70,
  decodeFlopsTrillion: 0.18,
  decodeFlopsTrendPct: -4,
  decodeIntensity: 0.42,
  decodeBytesMovedGB: 428.1,
  decodeIPC: 0.64,
  decodeEnergyMJ: 182.8,
  decodeHitRate: 41.8,
  decodeMatmulPct: 25,

  // Decoder block
  interBlockLatencyMs: 42,
  parallelismFactor: 8,
  ioWaitState: 'HIGH',

  // Attention sub-block
  blockLatencyMs: 1.24,
  attentionRuntimeMs: 0.52,
  attentionRuntimePct: 42,
  attentionFlopsTrillion: 12.4,
  attentionFlopsTrendPct: 8,
  attentionIntensity: 14.7,
  attentionBytesMoved: '842 MB',
  attentionHitRate: 92,
  attentionEnergyScore: 8.4,
  attentionL1: 92,
  attentionL2: 78,

  // MLP sub-block
  mlpRuntimeMs: 0.72,
  mlpRuntimePct: 58,
  mlpFlopsTrillion: 42.1,
  mlpFlopsTrendPct: -12,
  mlpIntensity: 17.5,
  mlpBytesMoved: '2.4 GB',
  mlpHitRate: 64,
  mlpEnergyScore: 4.2,
  mlpL1: 64,
  mlpL2: 41,

  // Layer view
  computeEfficiencyPercent: 92.4,

  // Fleet (ModelSelection)
  fleetCapacityPFLOPS: 42.8,
  fleetUtilizationPercent: 68.2,
  fleetEnergyEfficiencyPercent: 94.1,

  // Hooks
  availableHooks: [
    { id: 'pre_tokenize',   label: 'Pre-Tokenize'   },
    { id: 'post_tokenize',  label: 'Post-Tokenize'  },
    { id: 'pre_inference',  label: 'Pre-Inference'  },
    { id: 'post_inference', label: 'Post-Inference' },
    { id: 'on_sample',      label: 'On Sample'      },
    { id: 'on_logits',      label: 'On Logits'      },
  ],

  // Available models
  selectedModelId: '',
  models: [
    { id: 'gpt-kinetic-4-v2.1', name: 'GPT-Kinetic-4-v2.1', type: 'Transformer', params: '175B', status: 'Active',  latency: '42ms',  energy: 'Low'       },
    { id: 'gpt-kinetic-4-v2.0', name: 'GPT-Kinetic-4-v2.0', type: 'Transformer', params: '175B', status: 'Standby', latency: '48ms',  energy: 'Medium'    },
    { id: 'kinetic-light-v1',   name: 'Kinetic-Light-v1',   type: 'MoE',         params: '32B',  status: 'Offline', latency: '12ms',  energy: 'Ultra-Low' },
    { id: 'synapse-heavy-x',    name: 'Synapse-Heavy-X',    type: 'Dense',        params: '1.2T', status: 'Offline', latency: '180ms', energy: 'High'      },
  ],

  // Session state
  hasRunInference: false,
  resultsUpdated: false,
  inferenceMessages: [],
};

interface AppContextValue {
  state: AppState;
  set: <K extends keyof AppState>(key: K, value: AppState[K]) => void;
  reset: () => void;
}

const AppContext = createContext<AppContextValue | null>(null);

export function AppProvider({ children }: { children: React.ReactNode }) {
  const [state, setState] = useState<AppState>(defaultState);

  const set = <K extends keyof AppState>(key: K, value: AppState[K]) => {
    setState(prev => ({ ...prev, [key]: value }));
  };

  const reset = () => setState(defaultState);

  return (
    <AppContext.Provider value={{ state, set, reset }}>
      {children}
    </AppContext.Provider>
  );
}

export function useAppState() {
  const ctx = useContext(AppContext);
  if (!ctx) throw new Error('useAppState must be used inside <AppProvider>');
  return ctx;
}
