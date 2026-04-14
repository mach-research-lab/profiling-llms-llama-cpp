import React from 'react';
import { 
  LayoutDashboard, 
  GitBranch, 
  Grid3X3, 
  Brain, 
  Layers, 
  Download, 
  FileText, 
  HelpCircle,
  Zap
} from 'lucide-react';
import { View } from '../types';
import { motion } from 'motion/react';

interface SidebarProps {
  currentView: View;
  onViewChange: (view: View) => void;
}

export default function Sidebar({ currentView, onViewChange }: SidebarProps) {
  const navItems = [
    { id: 'top' as View, label: 'Top View', icon: LayoutDashboard },
    { id: 'phase' as View, label: 'Phase View', icon: GitBranch },
    { id: 'decoder' as View, label: 'Decoder Block', icon: Grid3X3 },
    { id: 'attention' as View, label: 'Attention/MLP', icon: Brain },
    { id: 'layer' as View, label: 'Layer View', icon: Layers },
  ];

  return (
    <aside className="fixed left-0 top-0 h-full flex flex-col z-40 bg-surface-container w-64 shadow-2xl shadow-black/40 font-headline text-sm uppercase tracking-widest">
      <div className="p-6 flex items-center gap-3">
        <div className="w-8 h-8 rounded bg-primary flex items-center justify-center">
          <Zap className="text-on-primary w-5 h-5 fill-current" />
        </div>
        <div>
          <div className="text-lg font-black text-white leading-none">GPT-Kinetic-4</div>
          <div className="text-[10px] text-primary/70 tracking-tighter">Active Inference</div>
        </div>
      </div>

      <nav className="flex-1 px-4 space-y-1 mt-4">
        {navItems.map((item) => (
          <button
            key={item.id}
            onClick={() => onViewChange(item.id)}
            className={`w-full flex items-center gap-3 px-4 py-3 transition-all text-left ${
              currentView === item.id 
                ? 'text-primary font-bold bg-surface-container-highest border-r-4 border-primary' 
                : 'text-outline hover:bg-surface-container-high'
            }`}
          >
            <item.icon className="w-5 h-5" />
            <span>{item.label}</span>
          </button>
        ))}
      </nav>

      <div className="p-4 mt-auto space-y-4">
        <motion.button 
          whileHover={{ scale: 0.98 }}
          whileTap={{ scale: 0.95 }}
          className="w-full py-2 bg-primary text-on-primary font-bold text-xs flex items-center justify-center gap-2 rounded-sm shadow-[0_0_15px_rgba(137,206,255,0.3)]"
        >
          <Download className="w-4 h-4" />
          EXPORT METRICS
        </motion.button>
        
        <div className="space-y-1">
          <a href="#" className="flex items-center gap-3 px-4 py-2 text-outline hover:text-primary transition-colors text-[10px]">
            <FileText className="w-4 h-4" />
            <span>Docs</span>
          </a>
          <a href="#" className="flex items-center gap-3 px-4 py-2 text-outline hover:text-primary transition-colors text-[10px]">
            <HelpCircle className="w-4 h-4" />
            <span>Support</span>
          </a>
        </div>
      </div>
    </aside>
  );
}
