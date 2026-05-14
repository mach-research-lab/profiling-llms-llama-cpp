import React from 'react';
import {
  LayoutDashboard,
  GitBranch,
  Grid3X3,
  Brain,
  Layers,
  Download,
  FileText,
  Zap,
  FolderOpen,
  X,
  FileJson,
  FileSpreadsheet,
} from 'lucide-react';
import { View } from '../types';
import { motion, AnimatePresence } from 'motion/react';
import { useAppState } from '../controller/AppContext';

const FORMATS = [
  { id: 'json', label: 'JSON',  desc: 'Structured key-value data',    icon: FileJson        },
  { id: 'csv',  label: 'CSV',   desc: 'Spreadsheet-compatible rows',   icon: FileSpreadsheet },
  { id: 'txt',  label: 'TXT',   desc: 'Plain text summary',            icon: FileText        },
];

interface SidebarProps {
  currentView: View;
  onViewChange: (view: View) => void;
}

export default function Sidebar({ currentView, onViewChange }: SidebarProps) {
  const { state, set } = useAppState();
  const [showExport, setShowExport] = React.useState(false);
  const [exportFilename, setExportFilename] = React.useState('llama-metrics');
  const [exportLocation, setExportLocation] = React.useState('');
  const [exportFormat, setExportFormat] = React.useState('json');

  const handleBrowse = async () => {
    try {
      // @ts-ignore — File System Access API
      const dir = await window.showDirectoryPicker();
      setExportLocation(dir.name);
    } catch {
      // user cancelled
    }
  };

  const handleExport = () => {
    // Placeholder: wire up real file-write logic here
    setShowExport(false);
  };

  const { selectedBlockLabel } = state;

  const navItems = [
    { id: 'top' as View, label: 'Top View', icon: LayoutDashboard },
    { id: 'phase' as View, label: 'Phase View', icon: GitBranch },
    { id: 'decoder' as View, label: 'Decoder Block', icon: Grid3X3 },
    { id: 'attention' as View, label: 'Attention/MLP', icon: Brain },
    // { id: 'layer' as View, label: 'Layer View', icon: Layers },
  ];

  return (
    <>
    {/* Export modal — rendered outside the sidebar so it can cover the full viewport */}
    <AnimatePresence>
      {showExport && (
        <motion.div
          key="export-overlay"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.2 }}
          className="fixed inset-0 z-50 flex items-center justify-center bg-background/70 backdrop-blur-sm"
          onClick={e => { if (e.target === e.currentTarget) setShowExport(false); }}
        >
          <motion.div
            initial={{ opacity: 0, scale: 0.95, y: 10 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.95, y: 10 }}
            transition={{ duration: 0.2 }}
            className="bg-surface-container border border-outline-variant/20 rounded-2xl p-8 w-full max-w-lg mx-6 shadow-2xl"
          >
            {/* Modal header */}
            <div className="flex items-center justify-between mb-8">
              <div className="flex items-center gap-3">
                <div className="p-2 rounded-lg bg-primary/10 border border-primary/20">
                  <Download className="w-5 h-5 text-primary" />
                </div>
                <div>
                  <h3 className="font-headline text-lg font-bold text-white">Export Metrics</h3>
                  <p className="text-[10px] font-mono uppercase tracking-widest text-on-surface-variant">Configure output file</p>
                </div>
              </div>
              <button
                onClick={() => setShowExport(false)}
                className="text-outline hover:text-white transition-colors p-1"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            <div className="space-y-6">
              {/* Filename */}
              <div className="space-y-2">
                <label className="text-[10px] font-bold uppercase tracking-widest text-on-surface-variant">
                  Filename
                </label>
                <div className="flex items-center bg-surface-container-low border border-outline-variant/20 rounded-lg overflow-hidden focus-within:border-primary/50 transition-colors">
                  <input
                    type="text"
                    value={exportFilename}
                    onChange={e => setExportFilename(e.target.value)}
                    className="flex-1 bg-transparent px-4 py-3 text-sm text-white font-mono focus:outline-none"
                    placeholder="llama-metrics"
                  />
                  <span className="px-4 text-xs font-mono text-outline border-l border-outline-variant/20 py-3">
                    .{exportFormat}
                  </span>
                </div>
              </div>

              {/* File location */}
              <div className="space-y-2">
                <label className="text-[10px] font-bold uppercase tracking-widest text-on-surface-variant">
                  Save Location
                </label>
                <div className="flex gap-2">
                  <div className="flex-1 flex items-center bg-surface-container-low border border-outline-variant/20 rounded-lg focus-within:border-primary/50 transition-colors">
                    <input
                      type="text"
                      value={exportLocation}
                      onChange={e => setExportLocation(e.target.value)}
                      className="flex-1 bg-transparent px-4 py-3 text-sm text-white font-mono focus:outline-none"
                      placeholder="Select a folder..."
                    />
                  </div>
                  <motion.button
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                    onClick={handleBrowse}
                    className="flex items-center gap-2 px-4 py-3 bg-surface-container-highest border border-outline-variant/20 rounded-lg text-xs font-bold uppercase tracking-widest text-on-surface-variant hover:text-white hover:border-outline-variant/50 transition-colors"
                  >
                    <FolderOpen className="w-4 h-4" />
                    Browse
                  </motion.button>
                </div>
                <p className="text-[10px] text-outline font-mono">
                  Leave blank to save to your default Downloads folder.
                </p>
              </div>

              {/* Format */}
              <div className="space-y-2">
                <label className="text-[10px] font-bold uppercase tracking-widest text-on-surface-variant">
                  Export Format
                </label>
                <div className="grid grid-cols-3 gap-3">
                  {FORMATS.map(fmt => {
                    const Icon = fmt.icon;
                    const active = exportFormat === fmt.id;
                    return (
                      <button
                        key={fmt.id}
                        onClick={() => setExportFormat(fmt.id)}
                        className={`flex flex-col items-start gap-2 p-4 rounded-lg border text-left transition-all ${
                          active
                            ? 'bg-primary/10 border-primary/40'
                            : 'bg-surface-container-low border-outline-variant/20 hover:border-outline-variant/50'
                        }`}
                      >
                        <Icon className={`w-5 h-5 ${active ? 'text-primary' : 'text-outline'}`} />
                        <div>
                          <div className={`text-xs font-bold uppercase tracking-widest ${active ? 'text-primary' : 'text-on-surface-variant'}`}>
                            {fmt.label}
                          </div>
                          <div className="text-[10px] text-outline font-mono mt-0.5 normal-case">{fmt.desc}</div>
                        </div>
                      </button>
                    );
                  })}
                </div>
              </div>
            </div>

            {/* Actions */}
            <div className="flex gap-3 justify-end mt-8">
              <motion.button
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                onClick={() => setShowExport(false)}
                className="px-5 py-2 rounded border border-outline-variant/30 text-xs font-bold uppercase tracking-widest text-on-surface-variant hover:text-white hover:border-outline-variant/60 transition-colors"
              >
                Cancel
              </motion.button>
              <motion.button
                whileHover={{ scale: 1.02 }}
                whileTap={{ scale: 0.98 }}
                onClick={handleExport}
                disabled={!exportFilename.trim()}
                className="flex items-center gap-2 px-5 py-2 rounded bg-primary text-on-primary text-xs font-bold uppercase tracking-widest shadow-[0_0_16px_rgba(137,206,255,0.2)] disabled:opacity-40 disabled:cursor-not-allowed disabled:shadow-none"
              >
                <Download className="w-4 h-4" />
                Export
              </motion.button>
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>

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
        {navItems.map((item) => {
          if (item.id === 'attention' && !selectedBlockLabel) return null;
          const isAttention = item.id === 'attention';
          const isActive = currentView === item.id;

          return (
            <button
              key={item.id}
              onClick={() => {
                if (item.id !== 'attention') set('selectedBlockLabel', '');
                onViewChange(item.id);
              }}
              className={`w-full flex items-center gap-3 py-3 transition-all text-left ${
                isAttention ? 'pl-8 pr-4' : 'px-4'
              } ${
                isActive
                  ? 'text-primary font-bold bg-surface-container-highest border-r-4 border-primary'
                  : 'text-outline hover:bg-surface-container-high'
              }`}
            >
              <item.icon className={`${isAttention ? 'w-4 h-4' : 'w-5 h-5'}`} />
              <span className={isAttention ? 'text-[11px]' : ''}>{item.label}</span>
            </button>
          );
        })}
      </nav>

      <div className="p-4 mt-auto space-y-4">
        <div className="space-y-1">
          <label className="text-[10px] font-bold uppercase tracking-widest text-on-surface-variant px-1">
            Decimal Precision
          </label>
          <select
            value={state.decimalPrecision}
            onChange={e => set('decimalPrecision', Number(e.target.value))}
            className="w-full bg-surface-container-high border border-outline-variant/20 text-white text-xs font-mono px-3 py-2 rounded focus:outline-none focus:border-primary/50"
          >
            {[0, 1, 2, 3, 4].map(n => (
              <option key={n} value={n}>{n} decimal{n !== 1 ? 's' : ''}</option>
            ))}
          </select>
        </div>

        <motion.button
          whileHover={{ scale: 0.98 }}
          whileTap={{ scale: 0.95 }}
          onClick={() => setShowExport(true)}
          className="w-full py-2 bg-primary text-on-primary font-bold text-xs flex items-center justify-center gap-2 rounded-sm shadow-[0_0_15px_rgba(137,206,255,0.3)]"
        >
          <Download className="w-4 h-4" />
          EXPORT METRICS
        </motion.button>
      </div>
    </aside>
    </>
  );
}
