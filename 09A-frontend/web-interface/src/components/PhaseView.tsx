import React from 'react';
import { 
  Zap, 
  RefreshCw, 
  Clock, 
  Activity,
  AlertCircle
} from 'lucide-react';
import { motion } from 'motion/react';
import { useAppState } from '@/src/controller/AppContext.tsx';
import { fmt, fmtSI } from '@/src/controller/Controller.tsx';
import Heatmap from '@/src/components/Heatmap.tsx';
import RooflineChart from '@/src/components/RooflineChart';

export default function PhaseView() {
  const { state } = useAppState();
  const {
    latencyS, tokensPerSecond,
    prefillTimeS, prefillTimePercent, prefillFLOPs, prefillFlopsTrendPct,
    prefillIntensity, prefillBytesMoved, prefillIPC, prefillEnergyJ,
    prefillHitRate, prefillMatmulPct, prefillOpTypeShare,
    prefillCoreUtilPercent, prefillCoreThreads, prefillEnergyCoresJ, prefillEnergyPkgJ, prefillEnergyPsysJ, prefillAvgPowerPkgW,
    decodeTimeS, decodeTimePercent, decodeFLOPs, decodeFlopsTrendPct,
    decodeIntensity, decodeBytesMoved, decodeIPC, decodeEnergyJ,
    decodeHitRate, decodeMatmulPct, decodeOpTypeShare,
    decodeCoreUtilPercent, decodeCoreThreads, decodeEnergyCoresJ, decodeEnergyPkgJ, decodeEnergyPsysJ, decodeAvgPowerPkgW,
    decimalPrecision, prefillRooflineOI, prefillRooflineAchievedGFLOPS,
    decodeRooflineOI, decodeRooflineAchievedGFLOPS,
  } = state;

  const LAYER_HEATMAP_OPTIONS = [
    { label: 'Time Heatmap', value: 'time' },
    { label: 'Memory Heatmap', value: 'memory' },
    { label: 'IPC Heatmap', value: 'ipc' },
  ];

  const PHASE_HEATMAP_TYPES = [
    { label: 'Phase Runtime', value: 'op-share' },
    { label: 'Phase Memory', value: 'op-share-memory' },
    { label: 'Phase IPC', value: 'op-share-ipc' },
  ];

  const [selectedLayerHeatmap, setSelectedLayerHeatmap] = React.useState('time');

  const f = (n: number) => fmt(n, decimalPrecision);
  const si = (n: number, unit: string) => fmtSI(n, unit, decimalPrecision);
  const trend = (pct: number) => `${pct >= 0 ? '↑' : '↓'} ${Math.abs(pct)}%`;

  // Determine standard metric formatting suffixes for values depending on selection state
  const getFormatFn = (kind: string) => {
    if (kind.includes('time'))   return (v: number) => fmtSI(v, 'us', decimalPrecision);
    if (kind.includes('memory')) return (v: number) => fmtSI(v, 'B', decimalPrecision);
    if (kind.includes('ipc'))    return (v: number) => fmt(v, decimalPrecision);
    return (v: number) => fmtSI(v, '', decimalPrecision);
  };

  return (
    <div className="p-8 space-y-8 animate-in fade-in duration-500">
      {/* Header Status */}
      <div className="flex justify-between items-end mb-8">
        <div>
          <h2 className="font-headline text-3xl font-light text-primary mb-1">Phase Comparative Analytics</h2>
        </div>
        <div className="flex gap-4">
          <div className="bg-surface-container p-3 rounded-lg border-l-2 border-secondary">
            <div className="text-[10px] text-on-surface-variant mb-1 uppercase tracking-tighter font-bold">Throughput</div>
            <div className="text-xl font-headline font-bold text-secondary">{f(tokensPerSecond)} tok/s</div>
          </div>
        </div>
      </div>

      {/* Side-by-Side Comparison Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* PREFILL PHASE */}
        <PhaseSection
          title="PHASE 01: PREFILL"
          icon={<Zap className="w-5 h-5 text-primary fill-current" />}
          time={si(prefillTimeS, 's')}
          timePercent={prefillTimePercent}
          f={f}
          flops={si(prefillFLOPs, 'FLOPs')}
          flopsNum={prefillFLOPs}
          flopsTrend={trend(prefillFlopsTrendPct)}
          intensity={`${f(prefillIntensity)} FLOPs/Byte`}
          intensityNum={prefillIntensity}
          intensityPoint={{ x: 280, y: 80 }}
          bytesMoved={si(prefillBytesMoved, 'B')}
          ipc={f(prefillIPC)}
          energy={si(prefillEnergyJ, 'J')}
          hitRate={prefillHitRate}
          matmul={prefillMatmulPct}
          opShare={prefillOpTypeShare}
          coreUtilization={prefillCoreUtilPercent}
          coreThreads={prefillCoreThreads}
          energyCores={prefillEnergyCoresJ}
          energyPkg={prefillEnergyPkgJ}
          energyPsys={prefillEnergyPsysJ}
          avgPowerPkgW={prefillAvgPowerPkgW}
          primaryColor="bg-primary"
          rooflineOI={prefillRooflineOI}
          rooflineAchievedGFLOPS={prefillRooflineAchievedGFLOPS}
        />

        {/* DECODE PHASE */}
        <PhaseSection 
          title="PHASE 02: DECODE" 
          icon={<RefreshCw className="w-5 h-5 text-secondary" />}
          time={si(decodeTimeS, 's')}
          timePercent={decodeTimePercent}
          f={f}
          flops={si(decodeFLOPs, 'FLOPs')}
          flopsNum={decodeFLOPs}
          flopsTrend={trend(decodeFlopsTrendPct)}
          intensity={`${f(decodeIntensity)} FLOPs/Byte`}
          intensityNum={decodeIntensity}
          intensityPoint={{ x: 20, y: 144 }}
          bytesMoved={si(decodeBytesMoved, 'B')}
          ipc={f(decodeIPC)}
          energy={si(decodeEnergyJ, 'J')}
          hitRate={decodeHitRate}
          matmul={decodeMatmulPct}
          opShare={decodeOpTypeShare}
          coreUtilization={decodeCoreUtilPercent}
          coreThreads={decodeCoreThreads}
          energyCores={decodeEnergyCoresJ}
          energyPkg={decodeEnergyPkgJ}
          energyPsys={decodeEnergyPsysJ}
          avgPowerPkgW={decodeAvgPowerPkgW}
          primaryColor="bg-secondary"
          isWarning
          rooflineOI={decodeRooflineOI}
          rooflineAchievedGFLOPS={decodeRooflineAchievedGFLOPS}
        />
      </div>

      <div className="space-y-6">
        {/* Layer Matrices Module */}
        <div className="bg-surface-container p-6 rounded-lg border border-outline-variant/10">
          <div className="flex flex-col sm:flex-row sm:items-end sm:justify-between gap-4 mb-6">
            <div>
              <h3 className="text-sm font-bold uppercase tracking-widest text-on-surface-variant">Layer Heatmaps</h3>
              <p className="text-[10px] text-outline mt-2">Show runtime, memory, or IPC intensity across transformer layers.</p>
            </div>
            <label className="block text-sm">
              <span className="text-[10px] uppercase tracking-widest text-on-surface-variant">Heatmap type</span>
              <select
                value={selectedLayerHeatmap}
                onChange={e => setSelectedLayerHeatmap(e.target.value)}
                className="mt-2 w-full rounded-lg border border-outline/30 bg-surface-container p-2 text-sm text-on-surface"
              >
                {LAYER_HEATMAP_OPTIONS.map(option => (
                  <option key={option.value} value={option.value}>{option.label}</option>
                ))}
              </select>
            </label>
          </div>

          <Heatmap
            title={`${selectedLayerHeatmap.toUpperCase()} PROFILE MAP`}
            description="Operation type (Y) × Layer stage (X)"
            stages={[]}
            cellSize={40}
            displayMode="raw"
            autoFetch={true}
            useLayerHeatmapEndpoint={true}
            heatmapKind={selectedLayerHeatmap}
            fetchTopN={20}
            formatValue={getFormatFn(selectedLayerHeatmap)}
            tabs={[]}
          />
        </div>

        {/* Phase Comparison Metrics Grid Module */}
        <div className="bg-surface-container p-6 rounded-lg border border-outline-variant/10">
          <div className="mb-6">
            <h3 className="text-sm font-bold uppercase tracking-widest text-on-surface-variant">Phase Compare Heatmaps</h3>
            <p className="text-[10px] text-outline mt-2">Compare runtime, memory, and IPC across prefill and decode phases.</p>
          </div>

          <div className="grid grid-cols-1 xl:grid-cols-3 gap-4">
            {PHASE_HEATMAP_TYPES.map(type => (
              <div key={type.value} className="rounded-lg p-0">
                <h4 className="text-xs font-bold uppercase tracking-widest text-on-surface-variant mb-3">{type.label}</h4>
                <Heatmap
                  title={type.label}
                  description=""
                  stages={[]}
                  cellSize={36}
                  displayMode="raw"
                  autoFetch={true}
                  useLayerHeatmapEndpoint={true}
                  heatmapKind={type.value}
                  fetchTopN={15}
                  formatValue={getFormatFn(type.value)}
                  tabs={[]}
                />
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}

// ─── Subcomponent Layout Section ─────────────────────────────────────────────

function PhaseSection({
  title, icon, time, timePercent, flops, flopsTrend,
  intensity, intensityNum, bytesMoved, ipc, energy, hitRate, opShare,
  coreUtilization, coreThreads, energyCores, energyPkg, energyPsys, avgPowerPkgW,
  primaryColor, f, rooflineOI, rooflineAchievedGFLOPS,
}: any) {
  const coreThreadList = coreThreads ?? [];
  const { state } = useAppState();
  const peakGF = (state.peakFLOPS || 0) / 1e9;
  const ridge = state.ridgePoint || 1;

  return (
    <section className="space-y-6">
      <div className="flex items-center gap-2 border-b border-outline-variant/20 pb-2">
        {icon}
        <h3 className="font-headline text-xl font-bold tracking-tight">{title}</h3>
      </div>
      
      <div className="grid grid-cols-2 gap-4">
        <div className="bg-surface-container p-4 rounded-lg relative overflow-hidden group">
          <div className="absolute top-0 right-0 p-2 opacity-10 group-hover:opacity-20 transition-opacity">
            <Clock className="w-12 h-12" />
          </div>
          <div className="text-[10px] text-on-surface-variant font-mono uppercase font-bold">Time Spent</div>
          <div className="text-3xl font-headline font-bold text-white mt-1">{time}</div>
          <div className="w-full bg-surface-container-highest h-1.5 mt-3 rounded-full overflow-hidden">
            <motion.div 
              initial={{ width: 0 }}
              animate={{ width: `${timePercent}%` }}
              transition={{ duration: 1 }}
              className={`h-full ${primaryColor}`} 
            />
          </div>
          <div className={`text-[10px] mt-1 font-bold ${primaryColor.replace('bg-', 'text-')}`}>{f(timePercent)}% of total inference</div>
        </div>
        
        <div className="bg-surface-container p-4 rounded-lg relative overflow-hidden group">
          <div className="absolute top-0 right-0 p-2 opacity-10 group-hover:opacity-20 transition-opacity">
            <Activity className="w-12 h-12" />
          </div>
          <div className="text-[10px] text-on-surface-variant font-mono uppercase font-bold">FLOPs</div>
          <div className="text-3xl font-headline font-bold text-white mt-1">{flops}</div>
        </div>
      </div>

      <div className="bg-surface-container p-6 rounded-lg border border-outline-variant/10">
        <div className="flex justify-between items-center mb-4">
          <h4 className="text-xs font-bold uppercase tracking-widest text-on-surface-variant flex items-center gap-2">
            <span className={`w-2 h-2 rounded-full animate-pulse ${primaryColor}`}></span>
            Arithmetic Intensity
          </h4>
          <div className="text-[10px] font-mono text-outline">{intensity}</div>
        </div>
        <RooflineChart
          size="compact"
          height={200}
          data={{
            arithmeticIntensity: rooflineOI ?? intensityNum ?? 0,
            achievedGFLOPS:      rooflineAchievedGFLOPS ?? 0,
            peakGFLOPS:          peakGF,
            memBwGBs:            (state.memBwBs ?? 0) / 1e9,
            ridgePoint:          ridge,
          }}
          dotColor={primaryColor === 'bg-primary' ? '#89ceff' : '#4edea3'}
          className="w-full h-40"
        />
      </div>

      <div className="grid grid-cols-3 gap-4">
        <div className="bg-surface-container-low p-4 border border-outline-variant/10">
          <div className="text-[10px] text-on-surface-variant mb-2 font-bold uppercase">Bytes Moved</div>
          <div className="text-lg font-headline font-bold">{bytesMoved}</div>
        </div>
        <div className="bg-surface-container-low p-4 border border-outline-variant/10">
          <div className="text-[10px] text-on-surface-variant mb-2 font-bold uppercase">IPC</div>
          <div className="text-lg font-headline font-bold">{ipc}</div>
        </div>
        <div className="bg-surface-container-low p-4 border border-outline-variant/10">
          <div className="text-[10px] text-on-surface-variant mb-2 font-bold uppercase">Energy</div>
          <div className="text-lg font-headline font-bold text-tertiary">{energy}</div>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-4">
        <div className="bg-surface-container p-4 rounded-lg border border-outline-variant/10">
          <div className="flex items-center justify-between mb-3">
            <div className="text-[10px] font-bold uppercase tracking-widest text-on-surface-variant">Core Utilization</div>
            <div className="text-[10px] text-on-surface-variant">Average + active thread</div>
          </div>
          <div className="text-2xl font-headline font-bold mb-2">{f(coreUtilization)}%</div>
          <div className="h-2 w-full rounded-full bg-surface-container-highest overflow-hidden mb-4">
            <div className="h-full bg-secondary rounded-full" style={{ width: `${coreUtilization}%` }} />
          </div>
          <div className="space-y-2 text-[11px] text-on-surface-variant">
            {coreThreadList.length > 0 ? (
              coreThreadList.map((item: any) => (
                <div key={`${item.socket}-${item.core}-${item.thread}`} className="flex items-center justify-between gap-4">
                  <div>
                    <div className="font-bold text-on-surface">{`${item.socket.replace(/_/g, ' ')} / ${item.core.replace(/_/g, ' ')}`}</div>
                    <div className="text-[10px] text-on-surface-variant">{item.thread.replace(/_/g, ' ')}</div>
                  </div>
                  <div className="font-bold">{f(item.utilizationPct)}%</div>
                </div>
              ))
            ) : (
              <div className="text-[10px] text-on-surface-variant">No core utilization breakdown available.</div>
            )}
          </div>
        </div>

        <div className="bg-surface-container p-4 rounded-lg border border-outline-variant/10">
          <div className="flex items-center justify-between mb-3">
            <div className="text-[10px] font-bold uppercase tracking-widest text-on-surface-variant">Energy Breakdown</div>
            <div className="text-[10px] text-on-surface-variant">Joules / Watts</div>
          </div>
          <div className="space-y-2 text-[11px]">
            <div className="flex items-center justify-between">
              <span className="text-on-surface-variant">Package</span>
              <span className="font-bold text-white">{fmtSI(energyPkg, 'J')}</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-on-surface-variant">Cores</span>
              <span className="font-bold text-white">{fmtSI(energyCores, 'J')}</span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-on-surface-variant">PSYS</span>
              <span className="font-bold text-white">{fmtSI(energyPsys, 'J')}</span>
            </div>
            <div className="flex items-center justify-between pt-2 border-t border-outline-variant/10">
              <span className="text-on-surface-variant">Avg package power</span>
              <span className="font-bold text-white">{f(avgPowerPkgW)} W</span>
            </div>
          </div>
        </div>
      </div>

      <div className="space-y-4">
        <div className="bg-surface-container p-4 rounded-lg border border-outline-variant/10">
          <h4 className="text-[10px] font-bold text-on-surface-variant uppercase mb-4">LLC Efficiency</h4>
          <div className="flex items-end gap-2 mb-2">
            <div className="text-2xl font-headline font-bold">{f(hitRate)}%</div>
            <div className={`text-[10px] mb-1 font-bold ${hitRate > 50 ? 'text-secondary' : 'text-secondary'}`}>HIT RATE</div>
          </div>
          <div className="flex gap-1 h-2">
            <div className="h-full bg-secondary" style={{ width: `${hitRate}%` }}></div>
            <div className="h-full bg-error" style={{ width: `${100 - hitRate}%` }}></div>
          </div>
        </div>

        <div className="bg-surface-container p-4 rounded-lg border border-outline-variant/10">
          <div className="flex items-center justify-between mb-4">
            <div className="text-[10px] font-bold uppercase tracking-widest text-on-surface-variant">Op Type Share</div>
          </div>
          <div className="space-y-4">
            {opShare && opShare.length > 0 ? (
              opShare.map((item: any) => (
                <div key={item.label} className="space-y-1">
                  <div className="flex items-center justify-between text-[10px] font-bold uppercase tracking-tight text-on-surface">
                    <span>{item.label.replace(/_/g, ' ')}</span>
                    <span>{f(item.timeSharePct)}%</span>
                  </div>
                  <div className="h-2 w-full rounded-full bg-surface-container-highest overflow-hidden">
                    <div
                      className={`${primaryColor} h-full rounded-full`}
                      style={{ width: `${item.timeSharePct}%` }}
                    />
                  </div>
                </div>
              ))
            ) : (
              <div className="text-[10px] text-on-surface-variant">No op type share data available.</div>
            )}
          </div>
        </div>
      </div>
    </section>
  );
}