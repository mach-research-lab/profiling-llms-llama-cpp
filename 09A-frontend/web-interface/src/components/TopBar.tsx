import React from 'react';
import { Settings, Bell, User } from 'lucide-react';
import { View } from '../types';

interface TopBarProps {
  currentView: View;
  onViewChange: (view: View) => void;
}

export default function TopBar({ currentView, onViewChange }: TopBarProps) {
  return (
    <header className="fixed top-0 left-0 right-0 z-50 bg-background/80 backdrop-blur-md flex items-center w-full px-6 h-16 border-b border-outline-variant/10">
      <div className="flex items-center justify-between w-full">
        <div className="text-xl font-bold tracking-tighter text-primary font-headline">
          Kinetic Intelligence Console
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
          <button 
            onClick={() => onViewChange('prompts')}
            className={`font-headline text-lg tracking-tight transition-all duration-200 pb-1 border-b-2 ${
              currentView === 'prompts' ? 'text-primary border-primary' : 'text-outline hover:text-primary border-transparent'
            }`}
          >
            Prompts
          </button>
          <button 
            className="font-headline text-lg tracking-tight text-outline hover:text-primary transition-all duration-200 pb-1 border-b-2 border-transparent"
          >
            Results
          </button>
        </nav>

        <div className="flex items-center gap-4">
          <button className="text-outline hover:text-primary transition-colors">
            <Settings className="w-5 h-5" />
          </button>
          <button className="text-outline hover:text-primary transition-colors">
            <Bell className="w-5 h-5" />
          </button>
          <div className="w-8 h-8 rounded-full bg-surface-container-highest flex items-center justify-center border border-outline-variant/20 overflow-hidden">
            <img 
              src="https://picsum.photos/seed/user/100/100" 
              alt="User" 
              className="w-full h-full object-cover"
              referrerPolicy="no-referrer"
            />
          </div>
        </div>
      </div>
    </header>
  );
}
