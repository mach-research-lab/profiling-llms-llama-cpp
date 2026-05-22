import React from 'react';
import { ChevronDown } from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';

// ─── Types ────────────────────────────────────────────────────────────────────

export interface HeatmapRow {
  label: string;
  /** Normalised 0..1 values used for the colour scale */
  values: number[];
  /** Optional raw (un-normalised) values used for the cell label in 'raw' display mode */
  rawValues?: number[];
}

export interface HeatmapTab {
  label: string;
  rows: HeatmapRow[];
}

export interface HeatmapProps {
  /** Column headers (x-axis) */
  stages: string[];
  /** Either a single dataset or multiple tabs the user can toggle between */
  tabs: HeatmapTab[];
  title?: string;
  description?: string;
  /** Max cell width in px — height is always equal (square). Default 40 */
  cellSize?: number;
  /** Start collapsed. Default false */
  defaultCollapsed?: boolean;
  /**
   * How to render the value label inside each cell.
   * - 'percent'  (default) — multiply by 100, show "{n}%"
   * - 'raw'      — show the raw value as-is, formatted by `formatValue`
   */
  displayMode?: 'percent' | 'raw';
  /** Called with the raw cell value (0..1 for percent, any range for raw).
   *  Only used when displayMode === 'raw'. Defaults to one decimal place. */
  formatValue?: (v: number) => string;
}

// ─── Colour scale ─────────────────────────────────────────────────────────────
// Read exact values from CSS custom properties so the heatmap always uses
// the same colour as the rest of the site, with no risk of drift.

function readCSSColor(variable: string): [number, number, number] {
  const raw = getComputedStyle(document.documentElement)
    .getPropertyValue(variable).trim();
  const hex = raw.startsWith('#') ? raw.slice(1) : raw;
  return [parseInt(hex.slice(0, 2), 16), parseInt(hex.slice(2, 4), 16), parseInt(hex.slice(4, 6), 16)];
}

function buildStops(): Array<[number, number, number]> {
  return [
    readCSSColor('--color-surface-container'), // 0.00
    readCSSColor('--color-primary'),            // 0.40
    readCSSColor('--color-secondary'),          // 0.70
    readCSSColor('--color-tertiary'),           // 1.00
  ];
}

const HEAT_POS = [0, 0.4, 0.7, 1.0];

function makeHeatColor(stops: Array<[number, number, number]>) {
  return function heatColor(t: number): string {
    let i = HEAT_POS.length - 2;
    for (let j = 0; j < HEAT_POS.length - 1; j++) {
      if (t <= HEAT_POS[j + 1]) { i = j; break; }
    }
    const lo = stops[i];
    const hi = stops[i + 1];
    const f  = (t - HEAT_POS[i]) / (HEAT_POS[i + 1] - HEAT_POS[i]);
    return `rgb(${Math.round(lo[0] + (hi[0] - lo[0]) * f)},${Math.round(lo[1] + (hi[1] - lo[1]) * f)},${Math.round(lo[2] + (hi[2] - lo[2]) * f)})`;
  };
}

function makeLegendGradient(stops: Array<[number, number, number]>) {
  return `linear-gradient(to right, ${
    HEAT_POS.map((p, i) => `rgb(${stops[i].join(',')}) ${p * 100}%`).join(', ')
  })`;
}

// ─── Component ────────────────────────────────────────────────────────────────

export default function Heatmap({ stages, tabs, title, description, cellSize = 40, defaultCollapsed = false, displayMode = 'percent', formatValue }: HeatmapProps) {
  const [activeTab, setActiveTab]   = React.useState(0);
  const [collapsed, setCollapsed]   = React.useState(defaultCollapsed);
  const [hoveredRow, setHoveredRow] = React.useState<number | null>(null);
  const [hoveredCol, setHoveredCol] = React.useState<number | null>(null);

  const stops         = React.useMemo(() => buildStops(), []);
  const heatColor     = React.useMemo(() => makeHeatColor(stops), [stops]);
  const legendGradient = React.useMemo(() => makeLegendGradient(stops), [stops]);

  const rows = tabs[activeTab]?.rows ?? [];

  const tabAccentClass = (idx: number) => {
    const active = [
      'bg-primary text-on-primary',
      'bg-secondary text-on-secondary',
      'bg-tertiary text-on-tertiary',
    ];
    return active[idx % active.length];
  };

  const colTemplate = `repeat(${stages.length}, minmax(0, ${cellSize}px))`;

  return (
    <div className="bg-surface-container p-6 rounded-lg">
      {/* Header — click anywhere to collapse */}
      <div
        className="flex justify-between items-center cursor-pointer select-none"
        onClick={() => setCollapsed(c => !c)}
      >
        <div>
          {title && <h4 className="text-xs font-bold uppercase tracking-widest">{title}</h4>}
          {!collapsed && description && <p className="text-[10px] text-outline font-mono mt-1">{description}</p>}
        </div>

        <div className="flex items-center gap-3">
          {/* Tab toggle — stop propagation so clicking tabs doesn't collapse */}
          {!collapsed && tabs.length > 1 && (
            <div className="flex gap-1 text-[10px] font-bold uppercase tracking-widest">
              {tabs.map((tab, idx) => (
                <button
                  key={tab.label}
                  onClick={e => { e.stopPropagation(); setActiveTab(idx); }}
                  className={`px-3 py-1 transition-colors ${
                    idx === 0 ? 'rounded-l' : idx === tabs.length - 1 ? 'rounded-r' : ''
                  } ${
                    activeTab === idx
                      ? tabAccentClass(idx)
                      : 'bg-surface-container-high text-on-surface-variant hover:bg-surface-container-highest'
                  }`}
                >
                  {tab.label}
                </button>
              ))}
            </div>
          )}

          <ChevronDown
            className="w-4 h-4 text-outline transition-transform duration-300"
            style={{ transform: collapsed ? 'rotate(-90deg)' : 'rotate(0deg)' }}
          />
        </div>
      </div>

      {/* Collapsible body */}
      <AnimatePresence initial={false}>
        {!collapsed && (
          <motion.div
            key="body"
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: 'auto', opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.25 }}
            className="overflow-hidden"
          >
      <div className="mt-6 flex gap-3 overflow-hidden">
        {/* Y-axis labels */}
        <div className="flex flex-col py-0.5 border-r border-outline-variant/10 pr-2 gap-2" style={{ minWidth: '95px' }}>
          {rows.map((row, ri) => (
            <div
              key={row.label}
              className="text-right pr-2 leading-none flex items-center justify-end font-mono font-bold transition-all duration-150"
              style={{
                fontSize:    hoveredRow === ri ? '11px' : '9px',
                color:       hoveredRow === ri ? '#fff' : '#88929b',
                height:      `${cellSize}px`,
              }}
            >
              {row.label}
            </div>
          ))}
        </div>

        {/* Cells & X-axis scrollable view container */}
        <div className="flex-1 overflow-x-auto scrollbar-thin pb-2">
          <div className="space-y-2 min-w-max">
            {rows.map((row, ri) => (
              <div
                key={row.label}
                className="grid gap-2"
                style={{ gridTemplateColumns: colTemplate }}
              >
                {row.values.map((val, ci) => {
                  // `val` is always the normalised 0..1 value used for colour.
                  const v = Math.min(Math.max(val, 0), 1);

                  // Raw value for the label (falls back to normalised if not supplied).
                  const rawVal = row.rawValues ? row.rawValues[ci] : val;

                  // Label shown inside the cell.
                  // In 'raw' mode we need real rawValues to format meaningfully;
                  // rows that only have normalised values fall back to percent.
                  let cellLabel: string;
                  if (displayMode === 'raw') {
                    cellLabel = row.rawValues ? (formatValue ? formatValue(rawVal) : rawVal.toFixed(1)) : '';
                  } else {
                    const pct = Math.round(v * 100);
                    cellLabel = pct > 0 ? `${pct}%` : '';
                  }
                  // In raw mode with real data always show the label even on near-black cells;
                  // in percent mode keep the threshold so empty (0%) cells stay clean.
                  const showLabel = cellLabel !== '' && (displayMode === 'raw' && row.rawValues ? true : v >= 0.02);

                  // Dynamically calculate readable font size that scales with cell size and label length
                  const labelLen = cellLabel.length;
                  const baseSize = cellSize * 0.28; // e.g. 40px cell -> 11.2px, 60px cell -> 16.8px
                  const fontScale = labelLen > 5 ? 0.65 : labelLen > 4 ? 0.75 : labelLen > 3 ? 0.9 : 1.0;
                  const calculatedSize = Math.max(Math.round(baseSize * fontScale), 8);
                  const isHovered = hoveredRow === ri && hoveredCol === ci;
                  const finalFontSize = isHovered ? Math.max(Math.round(calculatedSize * 1.35), 11) : calculatedSize;

                  return (
                    <div
                      key={ci}
                      title={`${row.label} @ ${stages[ci]}: ${cellLabel || '0'}`}
                      className="rounded-sm transition-all duration-150 cursor-default flex items-center justify-center overflow-hidden"
                      style={{
                        background:  v < 0.02 ? `rgb(${stops[0].join(',')})` : heatColor(v),
                        aspectRatio: '1',
                        width:       `${cellSize}px`,
                        height:      `${cellSize}px`,
                      }}
                      onMouseEnter={() => { setHoveredRow(ri); setHoveredCol(ci); }}
                      onMouseLeave={() => { setHoveredRow(null); setHoveredCol(null); }}
                    >
                      {showLabel && (
                        <span
                          className="transition-all duration-150 select-none text-center px-0.5 overflow-hidden text-ellipsis whitespace-nowrap"
                          style={{
                            fontSize:   `${finalFontSize}px`,
                            letterSpacing: '-0.04em',
                            fontFamily: 'var(--font-mono)',
                            fontWeight: 700,
                            color:      '#fff',
                            lineHeight: 1.1,
                            textShadow: '0 0 3px rgba(0,0,0,0.9), 0 1px 2px rgba(0,0,0,0.8)',
                            maxWidth:   '94%',
                          }}
                        >
                          {cellLabel}
                        </span>
                      )}
                    </div>
                  );
                })}
              </div>
            ))}

            {/* X-axis labels */}
            <div className="grid mt-2 gap-2" style={{ gridTemplateColumns: colTemplate }}>
            {stages.map((s, ci) => (
              <div
                key={s}
                className="font-mono text-center leading-tight transition-all duration-150"
                style={{
                  fontSize: hoveredCol === ci ? '11px' : '9px',
                  color:    hoveredCol === ci ? '#fff' : '#88929b',
                }}
              >
                {s}
              </div>
            ))}
          </div>
        </div>
      </div>
      </div>

      {/* Legend */}
      <div className="flex items-center gap-3 mt-5">
        <span className="text-[9px] text-outline font-mono uppercase">Low</span>
        <div className="flex-1 h-2 rounded-full" style={{ background: legendGradient }} />
        <span className="text-[9px] text-outline font-mono uppercase">High</span>
      </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
