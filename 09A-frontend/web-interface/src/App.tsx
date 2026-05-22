import React, { useState } from 'react';
import { View } from './types';
import { AppProvider, useAppState } from './controller/AppContext';
import Sidebar from './components/Sidebar';
import TopBar from './components/TopBar';
import TopView from './components/TopView';
import PhaseView from './components/PhaseView';
import DecoderBlockView from './components/DecoderBlockView';
import AttentionMLPView from './components/AttentionMLPView';
import LayerView from './components/LayerView';
import ModelSelectionView from './components/ModelSelectionView';
import PromptsView from './components/PromptsView';
import { fetchAndSetResults } from './controller/Controller';
import { useEffect } from 'react';

function AppContent() {
  const [currentView, setCurrentView] = useState<View>(() => {
    const saved = localStorage.getItem('currentView');
    return (saved as View) || 'models';
  });

  const { state, set } = useAppState();

  // Load hasRunInference and fetch results if an active run exists
  useEffect(() => {
    const saved = localStorage.getItem('hasRunInference');
    if (saved === 'true') {
      set('hasRunInference', true);
      fetchAndSetResults(set).catch((err) => console.log("Stale or missing metrics on boot:", err));
    } else {
      fetch('/api/run/cancel', { method: 'POST' }).catch(() => {});
    }
  }, []);

  // Persist hasRunInference state changes
  useEffect(() => {
    localStorage.setItem('hasRunInference', String(state.hasRunInference));
  }, [state.hasRunInference]);

  // Persist current view on change
  useEffect(() => {
    localStorage.setItem('currentView', currentView);
  }, [currentView]);

  const renderView = () => {
    switch (currentView) {
      case 'top':
        return <TopView />;
      case 'phase':
        return <PhaseView />;
      case 'decoder':
        return <DecoderBlockView onViewChange={setCurrentView} />;
      case 'attention':
        return <AttentionMLPView />;
      case 'layer':
        return <LayerView />;
      case 'models':
        return <ModelSelectionView onViewChange={setCurrentView} />;
      case 'prompts':
        return <PromptsView onViewChange={setCurrentView} />;
      default:
        return <ModelSelectionView onViewChange={setCurrentView} />;
    }
  };

  const hideSidebar = currentView === 'prompts' || currentView === 'models';

  return (
    <div className="min-h-screen bg-background text-on-surface selection:bg-primary/30 selection:text-primary">
      {/* Background Pattern */}
      <div className="fixed inset-0 data-grid-pattern opacity-20 pointer-events-none z-0"></div>

      <TopBar currentView={currentView} onViewChange={setCurrentView} />
      {!hideSidebar && <Sidebar currentView={currentView} onViewChange={setCurrentView} />}

      <main className="pl-64 pt-16 min-h-screen relative z-10">
        <div className="max-w-[1600px] mx-auto">
          {renderView()}
        </div>
      </main>
    </div>
  );
}

export default function App() {
  return (
    <AppProvider>
      <AppContent />
    </AppProvider>
  );
}
