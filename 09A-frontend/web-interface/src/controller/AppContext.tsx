import React, { createContext, useContext, useState } from 'react';
import {Model, PapiEvent} from '../types';

export interface AppState {
  // Model / hardware
  modelName: string;
  parameterCount: number;       // billions
  contextLength: number;        // tokens
  batchSize: number;
  totalRuntimeS: number;

  // Performance metrics
  tokensPerSecond: number;
  latencyS: number;
  throughputBs: number;
  memoryUsedBytes: number;
  memoryTotalBytes: number;

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
  peakFLOPS: number;            // raw FLOPS/s
  achievedFLOPS: number;        // raw FLOPS/s
  memBwBs: number;              // raw B/s
  ridgePoint: number;
  totalFLOPs: number;
  dramBytes: number;
  hwCpuModel: string;
  hwCores: number;
  hwBaseGHz: number;
  hwBoostGHz: number;
  hwAvgGHz: number;
  hwISA: string;
  hwFlopsPerCycle: number;

  // Energy
  totalEnergy: number;

  // Model summary / health
  modelSizeBytes: number;
  kvCacheBytes: number;
  kvCapacityBytes: number;
  kvUsedBytes: number;
  kvTokensCapacity: number;
  kvTokensUsed: number;
  kvUtilPercent: number;
  packetLossPercent: number;
  inputTokens: number;
  outputTokens: number;
  powerWatts: number;
  cpuUtilPercent: number;
  cacheMissPercent: number;

  // Phase – prefill detail
  prefillTimeS: number;
  prefillTimePercent: number;
  prefillFLOPs: number;           // raw FLOPs
  prefillFlopsTrendPct: number;   // positive = up, negative = down
  prefillIntensity: number;       // FLOPs/Byte
  prefillBytesMoved: number;      // raw bytes
  prefillIPC: number;
  prefillEnergyJ: number;
  prefillHitRate: number;
  prefillMatmulPct: number;

  // Phase – decode detail
  decodeTimeS: number;
  decodeTimePercent: number;
  decodeFLOPs: number;            // raw FLOPs
  decodeFlopsTrendPct: number;
  decodeIntensity: number;
  decodeBytesMoved: number;       // raw bytes
  decodeIPC: number;
  decodeEnergyJ: number;
  decodeHitRate: number;
  decodeMatmulPct: number;

  // Decoder block
  interBlockLatencyS: number;
  parallelismFactor: number;
  ioWaitState: string;

  // Attention sub-block
  blockLatencyS: number;
  attentionRuntimeS: number;
  attentionRuntimePct: number;
  attentionFLOPs: number;         // raw FLOPs
  attentionFlopsTrendPct: number;
  attentionIntensity: number;
  attentionBytesMoved: number;    // raw bytes
  attentionHitRate: number;
  attentionEnergyScore: number;
  attentionL1: number;
  attentionL2: number;
  attentionIPC: number;
  attentionFLOPsPerS: number;     // raw FLOPs/s
  attentionL1Misses: number;
  attentionL2Misses: number;
  attentionL3Misses: number;
  attentionL3Accesses: number;

  // MLP sub-block
  mlpRuntimeS: number;
  mlpRuntimePct: number;
  mlpFLOPs: number;               // raw FLOPs
  mlpFlopsTrendPct: number;
  mlpIntensity: number;
  mlpBytesMoved: number;          // raw bytes
  mlpHitRate: number;
  mlpEnergyScore: number;
  mlpL1: number;
  mlpL2: number;
  mlpIPC: number;
  mlpFLOPsPerS: number;           // raw FLOPs/s
  mlpL1Misses: number;
  mlpL2Misses: number;
  mlpL3Misses: number;
  mlpL3Accesses: number;

  // Decoder block list (raw decode blocks from JSON)
  decoderBlockList: any[];

  // Layer view
  computeEfficiencyPercent: number;

  // Available models
  models: Model[];
  selectedModelId: string;

  // Hooks
  availableHooks: PapiEvent[];

  // Energy breakdown (μJ → J)
  energyPsysJ: number;
  energyPkgJ: number;
  energyCoresJ: number;

  // PAPI cache misses
  papiL1Misses: number;
  papiL2Misses: number;
  papiL3Misses: number;

  // Inference run config
  maxTokens: number;
  papiEventsPerRun: number;

  // Display
  decimalPrecision: number;

  // Session state
  hasRunInference: boolean;
  resultsUpdated: boolean;
  inferenceMessages: { role: 'user' | 'assistant'; text: string }[];
  selectedBlockLabel: string;
}

const defaultState: AppState = {
  modelName: 'LLaMA-3 70B',
  parameterCount: 70,
  contextLength: 4096,
  batchSize: 8,
  totalRuntimeS: 0.1124,

  tokensPerSecond: 1240,
  latencyS: 0.038,
  throughputBs: 142e9,
  memoryUsedBytes: 48e9,
  memoryTotalBytes: 80e9,

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
  peakFLOPS: 989e12,
  achievedFLOPS: 743e12,
  memBwBs: 30e9,
  ridgePoint: 16.64,
  totalFLOPs: 0,
  dramBytes: 0,
  hwCpuModel: '',
  hwCores: 0,
  hwBaseGHz: 0,
  hwBoostGHz: 0,
  hwAvgGHz: 0,
  hwISA: '',
  hwFlopsPerCycle: 0,
  totalEnergy: 4.2,

  // Model summary / health / TopViewData
  modelSizeBytes: 175.4e9,
  kvCacheBytes: 12.8e9,
  kvCapacityBytes: 4e9,
  kvUsedBytes: 0,
  kvTokensCapacity: 0,
  kvTokensUsed: 0,
  kvUtilPercent: 0,
  packetLossPercent: 0.002,
  inputTokens: 0,
  outputTokens: 0,
  powerWatts: 422,
  cpuUtilPercent: 84.5,
  cacheMissPercent: 4.1,

  // Phase – prefill detail
  prefillTimeS: 0.0428,
  prefillTimePercent: 30,
  prefillFLOPs: 1.24e12,
  prefillFlopsTrendPct: 12,
  prefillIntensity: 18.4,
  prefillBytesMoved: 68.2e9,
  prefillIPC: 3.82,
  prefillEnergyJ: 42.4e-3,
  prefillHitRate: 94.2,
  prefillMatmulPct: 85,

  // Phase – decode detail
  decodeTimeS: 0.0996,
  decodeTimePercent: 70,
  decodeFLOPs: 0.18e12,
  decodeFlopsTrendPct: -4,
  decodeIntensity: 0.42,
  decodeBytesMoved: 428.1e9,
  decodeIPC: 0.64,
  decodeEnergyJ: 182.8e-3,
  decodeHitRate: 41.8,
  decodeMatmulPct: 25,

  // Decoder block
  interBlockLatencyS: 0.042,
  parallelismFactor: 8,
  ioWaitState: 'HIGH',

  // Attention sub-block
  blockLatencyS: 0.00124,
  attentionRuntimeS: 0.00052,
  attentionRuntimePct: 42,
  attentionFLOPs: 12.4e12,
  attentionFlopsTrendPct: 8,
  attentionIntensity: 14.7,
  attentionBytesMoved: 842e6,
  attentionHitRate: 92,
  attentionEnergyScore: 8.4,
  attentionL1: 92,
  attentionL2: 78,
  attentionIPC: 0,
  attentionFLOPsPerS: 0,
  attentionL1Misses: 0,
  attentionL2Misses: 0,
  attentionL3Misses: 0,
  attentionL3Accesses: 0,

  // MLP sub-block
  mlpRuntimeS: 0.00072,
  mlpRuntimePct: 58,
  mlpFLOPs: 42.1e12,
  mlpFlopsTrendPct: -12,
  mlpIntensity: 17.5,
  mlpBytesMoved: 2.4e9,
  mlpHitRate: 64,
  mlpEnergyScore: 4.2,
  mlpL1: 64,
  mlpL2: 41,
  mlpIPC: 0,
  mlpFLOPsPerS: 0,
  mlpL1Misses: 0,
  mlpL2Misses: 0,
  mlpL3Misses: 0,
  mlpL3Accesses: 0,

  // Decoder block list
  decoderBlockList: [],

  // Layer view
  computeEfficiencyPercent: 92.4,

  // Hooks, default values that are overridden in controller
  availableHooks: [
    { id: 'pre_tokenize',   label: 'Pre-Tokenize'   },
    { id: 'post_tokenize',  label: 'Post-Tokenize'  },
    { id: 'pre_inference',  label: 'Pre-Inference'  },
    { id: 'post_inference', label: 'Post-Inference' },
    { id: 'on_sample',      label: 'On Sample'      },
    { id: 'on_logits',      label: 'On Logits'      },
  ],

  // Available models, default values that are overridden in controller
  selectedModelId: '',
  models: [
    { id: 'gpt-kinetic-4-v2.1', name: 'GPT-Kinetic-4-v2.1', type: 'Transformer', params: '175B', status: 'Active',  latency: '42ms',  energy: 'Low'       },
    { id: 'gpt-kinetic-4-v2.0', name: 'GPT-Kinetic-4-v2.0', type: 'Transformer', params: '175B', status: 'Standby', latency: '48ms',  energy: 'Medium'    },
    { id: 'kinetic-light-v1',   name: 'Kinetic-Light-v1',   type: 'MoE',         params: '32B',  status: 'Offline', latency: '12ms',  energy: 'Ultra-Low' },
    { id: 'synapse-heavy-x',    name: 'Synapse-Heavy-X',    type: 'Dense',        params: '1.2T', status: 'Offline', latency: '180ms', energy: 'High'      },
  ],

  // Energy breakdown
  energyPsysJ: 0,
  energyPkgJ: 0,
  energyCoresJ: 0,

  // PAPI cache misses
  papiL1Misses: 0,
  papiL2Misses: 0,
  papiL3Misses: 0,

  // Inference run config
  maxTokens: 528,
  papiEventsPerRun: 1,

  // Display
  decimalPrecision: 2,

  // Session state
  hasRunInference: false,
  resultsUpdated: false,
  inferenceMessages: [],
  selectedBlockLabel: '',
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
