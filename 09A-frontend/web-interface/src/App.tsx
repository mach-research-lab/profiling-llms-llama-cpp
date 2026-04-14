import React, { useState } from 'react';
import { View } from './types';
import Sidebar from './components/Sidebar';
import TopBar from './components/TopBar';
import TopView from './components/TopView';
import PhaseView from './components/PhaseView';
import DecoderBlockView from './components/DecoderBlockView';
import AttentionMLPView from './components/AttentionMLPView';
import LayerView from './components/LayerView';
import ModelSelectionView from './components/ModelSelectionView';
import PromptsView from './components/PromptsView';

export default function App() {
  const [currentView, setCurrentView] = useState<View>('top');

  const renderView = () => {
    switch (currentView) {
      case 'top':
        return <TopView />;
      case 'phase':
        return <PhaseView />;
      case 'decoder':
        return <DecoderBlockView />;
      case 'attention':
        return <AttentionMLPView />;
      case 'layer':
        return <LayerView />;
      case 'models':
        return <ModelSelectionView />;
      case 'prompts':
        return <PromptsView />;
      default:
        return <TopView />;
    }
  };

  return (
    <div className="min-h-screen bg-background text-on-surface selection:bg-primary/30 selection:text-primary">
      {/* Background Pattern */}
      <div className="fixed inset-0 data-grid-pattern opacity-20 pointer-events-none z-0"></div>
      
      <TopBar currentView={currentView} onViewChange={setCurrentView} />
      <Sidebar currentView={currentView} onViewChange={setCurrentView} />

      <main className="pl-64 pt-16 min-h-screen relative z-10">
        <div className="max-w-[1600px] mx-auto">
          {renderView()}
        </div>
      </main>
    </div>
  );
}
