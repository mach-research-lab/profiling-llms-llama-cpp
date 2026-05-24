/**
 * roofline.ts — shared roofline chart helpers
 */

export interface RooflineData {
  arithmeticIntensity: number; // FLOPs / Byte  (x-axis)
  achievedGFLOPS:      number; // achieved GFLOPS/s (dot y-value)
  peakGFLOPS:          number; // hardware peak GFLOPS (flat ceiling)
  memBwGBs:            number; // memory bandwidth in GB/s
  ridgePoint:          number; // FLOPs/Byte at the ridge
}

export interface RooflineSVGSpec {
  width:     number;
  height:    number;
  padLeft:   number;
  padBottom: number;
  padTop:    number;
  padRight:  number;
}

export const DEFAULT_SVG_SPEC: RooflineSVGSpec = {
  width: 500, height: 240,
  padLeft: 56, padBottom: 28, padTop: 20, padRight: 16,
};

export const COMPACT_SVG_SPEC: RooflineSVGSpec = {
  width: 400, height: 160,
  padLeft: 50, padBottom: 24, padTop: 12, padRight: 12,
};

function safeLog10(v: number): number {
  return Math.log10(Math.max(v, 1e-30));
}

export interface RooflineRanges {
  logXMin: number; logXMax: number;
  logYMin: number; logYMax: number;
}

export function computeRanges(d: RooflineData): RooflineRanges {
  const logRidge = safeLog10(Math.max(d.ridgePoint, 1e-6));
  const logPeak  = safeLog10(Math.max(d.peakGFLOPS, 1e-6));
  const logOI    = safeLog10(Math.max(d.arithmeticIntensity, 1e-6));

  // X: always show ridge point, OI dot, and 1 decade either side
  const xPoints  = [logRidge, logOI];
  const xCenter  = (Math.min(...xPoints) + Math.max(...xPoints)) / 2;
  const xHalf    = Math.max(Math.abs(logRidge - logOI) / 2 + 1.2, 1.5);
  const logXMin  = Math.floor(xCenter - xHalf);
  const logXMax  = Math.ceil(xCenter  + xHalf);

  // Y: always show peak and at least 3 decades of range, cap at 4 decades
  const logYMax  = Math.ceil(logPeak + 0.3);
  const logYMin  = logYMax - 4;          // fixed 4-decade window — avoids tick explosion

  return { logXMin, logXMax, logYMin, logYMax };
}

export function makeMappers(ranges: RooflineRanges, spec: RooflineSVGSpec) {
  const { logXMin, logXMax, logYMin, logYMax } = ranges;
  const plotW = spec.width  - spec.padLeft - spec.padRight;
  const plotH = spec.height - spec.padTop  - spec.padBottom;

  const mapX = (lx: number) =>
    spec.padLeft + ((lx - logXMin) / (logXMax - logXMin)) * plotW;

  // inverted: larger GFLOPS → smaller y pixel (higher on screen)
  const mapY = (ly: number) =>
    spec.padTop + plotH - ((ly - logYMin) / (logYMax - logYMin)) * plotH;

  return { mapX, mapY, plotW, plotH };
}

export interface RooflinePaths {
  memRoofPath:     string;
  computeRoofPath: string;
  ceilingPath:     string;
}

export function buildPaths(
  d: RooflineData,
  ranges: RooflineRanges,
  spec: RooflineSVGSpec,
): RooflinePaths {
  const { mapX, mapY } = makeMappers(ranges, spec);
  const { logXMin, logXMax } = ranges;

  const logBw    = safeLog10(Math.max(d.memBwGBs,   1e-6));
  const logRidge = safeLog10(Math.max(d.ridgePoint,  1e-6));
  const logPeak  = safeLog10(Math.max(d.peakGFLOPS,  1e-6));

  // Memory roof: performance(x) = memBw * x  →  log(perf) = log(memBw) + log(x)
  // So at logX, logPerf = logBw + logX
  const memStartX = mapX(logXMin);
  const memStartY = mapY(logBw + logXMin);   // ← correct slope formula
  const ridgeX    = mapX(logRidge);
  const ridgeY    = mapY(logPeak);

  // Clamp memStartY to plot area so the line doesn't start outside the SVG
  const plotBottom = spec.height - spec.padBottom;
  const clampedMemStartY = Math.min(memStartY, plotBottom);

  const memRoofPath     = `M ${memStartX} ${clampedMemStartY} L ${ridgeX} ${ridgeY}`;
  const computeRoofPath = `M ${ridgeX} ${ridgeY} L ${mapX(logXMax)} ${ridgeY}`;
  const ceilingPath     = `${memRoofPath} L ${mapX(logXMax)} ${ridgeY}`;

  return { memRoofPath, computeRoofPath, ceilingPath };
}

export interface RooflineDot {
  x: number; y: number;
  isMemoryBound: boolean;
  clipped: boolean;
}

export function computeDot(
  d: RooflineData,
  ranges: RooflineRanges,
  spec: RooflineSVGSpec,
): RooflineDot {
  const { mapX, mapY } = makeMappers(ranges, spec);
  const logOI  = safeLog10(Math.max(d.arithmeticIntensity, 1e-6));
  const logAch = safeLog10(Math.max(d.achievedGFLOPS,      1e-6));

  const rawX = mapX(logOI);
  const rawY = mapY(logAch);

  const minX = spec.padLeft;
  const maxX = spec.width  - spec.padRight;
  const minY = spec.padTop;
  const maxY = spec.height - spec.padBottom;

  const x = Math.min(Math.max(rawX, minX), maxX);
  const y = Math.min(Math.max(rawY, minY), maxY);

  return {
    x, y,
    isMemoryBound: d.arithmeticIntensity < d.ridgePoint,
    clipped: rawX !== x || rawY !== y,
  };
}

export function xTicks(ranges: RooflineRanges, spec: RooflineSVGSpec) {
  const { mapX } = makeMappers(ranges, spec);
  const ticks: { x: number; label: string }[] = [];
  for (let v = Math.ceil(ranges.logXMin); v <= Math.floor(ranges.logXMax); v++) {
    const val = Math.pow(10, v);
    const label = val >= 1 ? `${val.toFixed(0)}` : `${val.toFixed(2)}`;
    ticks.push({ x: mapX(v), label });
  }
  return ticks;
}

export function yTicks(ranges: RooflineRanges, spec: RooflineSVGSpec) {
  const { mapY } = makeMappers(ranges, spec);
  const ticks: { y: number; label: string }[] = [];
  for (let v = Math.ceil(ranges.logYMin); v <= Math.floor(ranges.logYMax); v++) {
    const val = Math.pow(10, v);
    const label = val >= 1000 ? `${(val / 1000).toFixed(0)}T`
                : val >= 1    ? `${val.toFixed(0)}G`
                :               `${val.toFixed(2)}G`;
    ticks.push({ y: mapY(v), label });
  }
  return ticks;
}

export interface ComputedRoofline {
  ranges:     RooflineRanges;
  paths:      RooflinePaths;
  dot:        RooflineDot;
  xt:         ReturnType<typeof xTicks>;
  yt:         ReturnType<typeof yTicks>;
  plotLeft:   number;
  plotBottom: number;
  plotRight:  number;
  plotTop:    number;
  hasData:    boolean;
}

export function computeRoofline(
  d: RooflineData,
  spec: RooflineSVGSpec = DEFAULT_SVG_SPEC,
): ComputedRoofline {
  const hasData = d.peakGFLOPS > 0 && d.memBwGBs > 0;

  // Safe fallback when no profiling data yet
  if (!hasData) {
    return {
      ranges:     { logXMin: -2, logXMax: 3, logYMin: -1, logYMax: 3 },
      paths:      { memRoofPath: '', computeRoofPath: '', ceilingPath: '' },
      dot:        { x: 0, y: 0, isMemoryBound: true, clipped: false },
      xt:         [],
      yt:         [],
      plotLeft:   spec.padLeft,
      plotBottom: spec.height - spec.padBottom,
      plotRight:  spec.width  - spec.padRight,
      plotTop:    spec.padTop,
      hasData:    false,
    };
  }

  const ranges = computeRanges(d);
  return {
    ranges,
    paths:      buildPaths(d, ranges, spec),
    dot:        computeDot(d, ranges, spec),
    xt:         xTicks(ranges, spec),
    yt:         yTicks(ranges, spec),
    plotLeft:   spec.padLeft,
    plotBottom: spec.height - spec.padBottom,
    plotRight:  spec.width  - spec.padRight,
    plotTop:    spec.padTop,
    hasData:    true,
  };
}