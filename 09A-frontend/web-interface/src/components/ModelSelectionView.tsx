import React, { useEffect } from 'react';
import { Box, Cpu, ChevronRight, AlertTriangle } from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';
import { useAppState } from '@/src/controller/AppContext.tsx';
import { View } from '../types';
import { fetchAndSetModels } from "@/src/controller/Controller.tsx";

interface ModelSelectionViewProps {
  onViewChange: (view: View) => void;
}

export default function ModelSelectionView({ onViewChange }: ModelSelectionViewProps) {
  const { state, set } = useAppState();
  const { models, selectedModelId } = state;
  const selectedModel = models.find(m => m.id === selectedModelId);
  const [pendingModelId, setPendingModelId] = React.useState<string | null>(null);

  //Updates the appstate with the models from the db
  useEffect(() => {
    fetchAndSetModels(set);
  }, []);


  // Lets the selected model be chosen in the app context
  const handleModelClick = (modelId: string) => {
    if (modelId === selectedModelId) return;
    if (state.hasRunInference) {
      setPendingModelId(modelId);
    } else {
      set('selectedModelId', modelId);
      set('modelName', models.find(m => m.id === modelId)?.name ?? modelId);
    }
  };

  const confirmSwitch = () => {
    if (!pendingModelId) return;
    set('selectedModelId', pendingModelId);
    set('modelName', models.find(m => m.id === pendingModelId)?.name ?? pendingModelId);
    set('inferenceMessages', []);
    set('hasRunInference', false);
    set('resultsUpdated', false);
    setPendingModelId(null);
  };

  const cancelSwitch = () => setPendingModelId(null);

  const pendingModel = models.find(m => m.id === pendingModelId);

  return (
    <div className="p-8 space-y-8 animate-in fade-in duration-500">
      <header className="mb-8">
        <h2 className="font-headline text-3xl font-light text-primary mb-1">Model Selection</h2>
        <p className="text-on-surface-variant font-mono text-xs uppercase tracking-widest">
          Available Models: <span className="text-secondary">{models.length} Registered</span>
        </p>
      </header>

      {/* Confirmation modal */}
      <AnimatePresence>
        {pendingModelId && (
          <motion.div
            key="confirm-overlay"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="fixed inset-0 z-50 flex items-center justify-center bg-background/70 backdrop-blur-sm"
          >
            <motion.div
              initial={{ opacity: 0, scale: 0.95, y: 8 }}
              animate={{ opacity: 1, scale: 1, y: 0 }}
              exit={{ opacity: 0, scale: 0.95, y: 8 }}
              transition={{ duration: 0.2 }}
              className="bg-surface-container border border-outline-variant/20 rounded-2xl p-8 max-w-md w-full mx-6 shadow-2xl"
            >
              <div className="flex items-center gap-3 mb-4">
                <div className="p-2 rounded-lg bg-amber-500/10 border border-amber-500/20">
                  <AlertTriangle className="w-5 h-5 text-amber-400" />
                </div>
                <h3 className="font-headline text-lg font-bold text-white">Switch Model?</h3>
              </div>
              <p className="text-sm text-on-surface-variant leading-relaxed mb-2">
                You're switching to <span className="text-white font-bold">{pendingModel?.name}</span>.
              </p>
              <p className="text-sm text-on-surface-variant leading-relaxed mb-8">
                Your current prompt history and results will be cleared and you will need to run a new inference. If you wish to save the results,
                export the metrics using the "Export metrics" button on the results panel.
              </p>

              <div className="flex gap-3 justify-end">
                <motion.button
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  onClick={cancelSwitch}
                  className="px-5 py-2 rounded border border-outline-variant/30 text-xs font-bold uppercase tracking-widest text-on-surface-variant hover:text-white hover:border-outline-variant/60 transition-colors"
                >
                  Cancel
                </motion.button>
                <motion.button
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  onClick={confirmSwitch}
                  className="px-5 py-2 rounded bg-primary text-on-primary text-xs font-bold uppercase tracking-widest shadow-[0_0_16px_rgba(137,206,255,0.2)]"
                >
                  Switch Model
                </motion.button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {models.map((model) => (
          <motion.div
            key={model.id}
            whileHover={{ scale: 1.01 }}
            onClick={() => handleModelClick(model.id)}
            className={`bg-surface-container p-6 rounded-xl border transition-all cursor-pointer group ${model.id === selectedModelId
              ? 'border-primary shadow-[0_0_20px_rgba(137,206,255,0.1)]'
              : 'border-outline-variant/10 hover:border-outline-variant/30'
              }`}
          >
            <div className="flex justify-between items-start mb-6">
              <div className="flex items-center gap-4">
                <div className={`p-3 rounded-lg ${model.id === selectedModelId ? 'bg-primary/10' : 'bg-surface-container-highest'}`}>
                  <Box className={`w-6 h-6 ${model.id === selectedModelId ? 'text-primary' : 'text-outline'}`} />
                </div>
                <div>
                  <h3 className={`font-headline text-xl font-bold ${model.id === selectedModelId ? 'text-white' : 'text-outline'}`}>{model.name}</h3>
                  <div className="text-[10px] text-on-surface-variant font-mono uppercase font-bold">{model.type} Architecture</div>
                </div>
              </div>
              <div className={`text-[10px] px-2 py-0.5 rounded-full border font-bold uppercase ${model.status === 'Active' ? 'bg-secondary/10 text-secondary border-secondary/30' : 'bg-surface-container-highest text-outline border-outline-variant/30'
                }`}>
                {model.status}
              </div>
            </div>

            <div className="grid grid-cols-1 gap-4 mb-6">
              <div className="space-y-1">
                <div className="text-[10px] text-on-surface-variant uppercase font-bold">Parameters</div>
                <div className="text-lg font-headline font-bold text-white">{model.params}</div>
              </div>
            </div>

            <div className="flex items-center justify-between pt-4 border-t border-outline-variant/10">
              <button className={`flex items-center gap-2 text-[10px] font-bold uppercase tracking-widest transition-colors ${model.id === selectedModelId ? 'text-primary' : 'text-outline group-hover:text-white'
                }`}>
                {model.id === selectedModelId ? 'Selected' : 'Select Model'}
                <ChevronRight className="w-4 h-4" />
              </button>
            </div>
          </motion.div>
        ))}
      </div>

      <AnimatePresence>
        {selectedModel && (
          <motion.div
            initial={{ opacity: 0, y: 16 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 16 }}
            transition={{ duration: 0.2 }}
            className="flex items-center justify-between bg-surface-container border border-primary/30 rounded-xl px-6 py-4"
          >
            <div className="text-sm text-on-surface-variant">
              Selected: <span className="text-white font-bold">{selectedModel.name}</span>
            </div>
            <motion.button
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              onClick={() => onViewChange('prompts')}
              className="flex items-center gap-2 bg-primary text-on-primary px-6 py-2 rounded font-headline font-bold text-sm shadow-[0_0_20px_rgba(137,206,255,0.2)]"
            >
              Continue to Prompts
              <ChevronRight className="w-4 h-4" />
            </motion.button>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
