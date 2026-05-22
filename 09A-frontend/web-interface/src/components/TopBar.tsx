import React from 'react';
import { ChevronRight, RotateCcw } from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';
import { View } from '../types';
import { useAppState } from '../controller/AppContext';

interface TopBarProps {
  currentView: View;
  onViewChange: (view: View) => void;
}

export default function TopBar({ currentView, onViewChange }: TopBarProps) {
  const { state, set } = useAppState();
  const modelSelected = !!state.selectedModelId;
  const resultsUnlocked = state.hasRunInference;
  const resultsUpdated = state.resultsUpdated;

  const handleNewRun = () => {
    fetch('/api/run/cancel', { method: 'POST' }).catch((err) => console.error("Failed to cancel active runs:", err));
    set('inferenceMessages', []);
    set('hasRunInference', false);
    set('resultsUpdated', false);
    onViewChange('models');
  };

  return (
    <header className="fixed top-0 left-0 right-0 z-50 bg-background/80 backdrop-blur-md flex items-center w-full px-6 h-16 border-b border-outline-variant/10">
      <div className="flex items-center justify-between w-full">
        <div className="text-xl font-bold tracking-tighter text-primary font-headline">
          LLama Cpp Analytics
        </div>
        
        <nav className="flex items-center gap-12 absolute left-1/2 -translate-x-1/2">
          <button
            onClick={() => onViewChange('models')}
            className={`font-headline text-lg tracking-tight transition-all duration-200 pb-1 border-b-2 ${
              currentView === 'models' ? 'text-primary border-primary' : 'text-outline hover:text-primary border-transparent'
            }`}
          >
            Model Selection
          </button>
          <ChevronRight className={`w-4 h-4 shrink-0 transition-opacity ${modelSelected ? 'text-outline/40' : 'text-outline/20'}`} />
          <button
            onClick={() => modelSelected && onViewChange('prompts')}
            disabled={!modelSelected}
            title={!modelSelected ? 'Select a model first' : undefined}
            className={`font-headline text-lg tracking-tight transition-all duration-200 pb-1 border-b-2 ${
              !modelSelected
                ? 'text-outline/30 border-transparent cursor-not-allowed'
                : currentView === 'prompts'
                  ? 'text-primary border-primary'
                  : 'text-outline hover:text-primary border-transparent'
            }`}
          >
            Prompts
          </button>
          <ChevronRight className={`w-4 h-4 shrink-0 transition-opacity ${resultsUnlocked ? 'text-outline/40' : 'text-outline/20'}`} />
          <button
            onClick={() => {
              if (!resultsUnlocked) return;
              set('resultsUpdated', false);
              onViewChange('top');
            }}
            disabled={!resultsUnlocked}
            title={!resultsUnlocked ? 'Run an inference first' : undefined}
            className={`relative font-headline text-lg tracking-tight transition-all duration-200 pb-1 border-b-2 ${
              !resultsUnlocked
                ? 'text-outline/30 border-transparent cursor-not-allowed'
                : ['top', 'phase', 'decoder', 'attention', 'layer'].includes(currentView)
                  ? 'text-primary border-primary'
                  : 'text-outline hover:text-primary border-transparent'
            }`}
          >
            Results
            <AnimatePresence>
              {resultsUpdated && (
                <motion.span
                   key="dot"
                  initial={{ scale: 0, opacity: 0 }}
                  animate={{ scale: 1, opacity: 1 }}
                  exit={{ scale: 0, opacity: 0 }}
                  transition={{ duration: 0.2 }}
                  className="absolute -top-1 -right-3 flex items-center justify-center"
                >
                  <motion.span
                    animate={{ opacity: [1, 0.3, 1] }}
                    transition={{ duration: 1.2, repeat: Infinity, ease: 'easeInOut' }}
                    className="w-2 h-2 rounded-full bg-secondary block"
                  />
                </motion.span>
              )}
            </AnimatePresence>
          </button>
        </nav>

        <div className="flex justify-end items-center">
          {modelSelected && (
            <motion.button
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.98 }}
              onClick={handleNewRun}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg border border-outline-variant/30 text-xs font-bold uppercase tracking-widest text-outline hover:text-white hover:border-outline-variant/60 hover:bg-surface-container-high transition-all"
            >
              <RotateCcw className="w-3.5 h-3.5" />
              New Run
            </motion.button>
          )}
        </div>
      </div>
    </header>
  );
}
