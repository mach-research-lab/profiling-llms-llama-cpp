import React from 'react';
import { motion } from 'motion/react';
import {
  computeRoofline,
  RooflineData,
  RooflineSVGSpec,
  DEFAULT_SVG_SPEC,
  COMPACT_SVG_SPEC,
} from '@/src/utils/roofline';

interface RooflineChartProps {
  data:       RooflineData;
  size?:      'default' | 'compact';
  dotColor?:  string;
  showAxes?:  boolean;
  className?: string;
  dotLabel?:  string;
  height?:     number;
}

function RooflineChartInner({
  data,
  size      = 'default',
  dotColor,
  showAxes  = true,
  className = '',
  dotLabel,
  height,
}: RooflineChartProps) {
  const spec: RooflineSVGSpec = size === 'compact' ? COMPACT_SVG_SPEC : DEFAULT_SVG_SPEC;
  const { paths, dot, xt, yt, plotLeft, plotBottom, plotRight, plotTop, hasData } =
    computeRoofline(data, spec);

  const fill = dotColor ?? (dot.isMemoryBound ? '#89ceff' : '#4edea3');

  return (
    <div
      className={`relative border border-outline/20 rounded-lg bg-surface-container-low ${className}`}
      style={{ width: '100%', height: height ?? spec.height }}
    >
      <svg
        viewBox={`0 0 ${spec.width} ${spec.height}`}
        width="100%"
        height="100%"
        className="absolute inset-0"
        preserveAspectRatio="xMidYMid meet"
      >
        {/* Grid lines */}
        {showAxes && yt.map(({ y }) => (
          <line key={`yg-${y}`} x1={plotLeft} x2={plotRight} y1={y} y2={y}
                stroke="#2a3340" strokeWidth={0.5} />
        ))}
        {showAxes && xt.map(({ x }) => (
          <line key={`xg-${x}`} x1={x} x2={x} y1={plotTop} y2={plotBottom}
                stroke="#2a3340" strokeWidth={0.5} />
        ))}

        {/* Memory slope */}
        {hasData && (
          <path d={paths.memRoofPath}
                fill="none" stroke="#4a5a6a" strokeWidth={1.5} strokeDasharray="6 3" />
        )}
        {/* Compute ceiling */}
        {hasData && (
          <path d={paths.computeRoofPath}
                fill="none" stroke="#4a5a6a" strokeWidth={1.5} strokeDasharray="6 3" />
        )}

        {/* X-axis ticks + labels */}
        {showAxes && xt.map(({ x, label }) => (
          <g key={`xt-${x}`}>
            <line x1={x} y1={plotBottom} x2={x} y2={plotBottom + 4}
                  stroke="#3e4850" strokeWidth={1} />
            <text x={x} y={plotBottom + 14}
                  fontFamily="monospace" fontSize={9} fill="#6a7a8a" textAnchor="middle">
              {label}
            </text>
          </g>
        ))}

        {/* Y-axis ticks + labels */}
        {showAxes && yt.map(({ y, label }) => (
          <g key={`yt-${y}`}>
            <line x1={plotLeft - 4} y1={y} x2={plotLeft} y2={y}
                  stroke="#3e4850" strokeWidth={1} />
            <text x={plotLeft - 6} y={y + 3}
                  fontFamily="monospace" fontSize={9} fill="#6a7a8a" textAnchor="end">
              {label}
            </text>
          </g>
        ))}

        {/* Axis labels */}
        {showAxes && (
          <>
            <text x={(plotLeft + plotRight) / 2} y={spec.height - 2}
                  fontFamily="monospace" fontSize={9} fill="#4a5a6a" textAnchor="middle">
              Arithmetic Intensity (FLOPs/Byte)
            </text>
            <text
              x={10} y={(plotTop + plotBottom) / 2}
              fontFamily="monospace" fontSize={9} fill="#4a5a6a" textAnchor="middle"
              transform={`rotate(-90, 10, ${(plotTop + plotBottom) / 2})`}
            >
              GFLOPS/s
            </text>
          </>
        )}

        {/* Achieved dot — only when we have real data */}
        {hasData && data.achievedGFLOPS > 0 && (
          <>
            <line x1={dot.x} x2={dot.x} y1={dot.y} y2={plotBottom}
                  stroke={fill} strokeWidth={0.8} strokeDasharray="3 3" opacity={0.35} />
            <line x1={plotLeft} x2={dot.x} y1={dot.y} y2={dot.y}
                  stroke={fill} strokeWidth={0.8} strokeDasharray="3 3" opacity={0.35} />
            <motion.circle
              key={`dot-${dot.x.toFixed(1)}-${dot.y.toFixed(1)}`}
              initial={{ opacity: 0, r: 0 }}
              animate={{ opacity: 1, r: 6 }}
              transition={{ duration: 0.5, ease: 'easeOut' }}
              cx={dot.x} cy={dot.y} fill={fill}
              style={{ filter: `drop-shadow(0 0 6px ${fill})` }}
            />
            {dotLabel && (
              <text
                x={Math.min(dot.x + 10, plotRight - 80)}
                y={Math.max(dot.y - 8, plotTop + 10)}
                fontFamily="monospace" fontSize={9} fill={fill}
              >
                {dotLabel}
              </text>
            )}
          </>
        )}

        {/* Bound label */}
        {hasData && (
          <text x={plotLeft + 4} y={plotTop + 12}
                fontFamily="monospace" fontSize={10} fontWeight="bold"
                fill={dot.isMemoryBound ? '#f28b82' : '#4edea3'}>
            {dot.isMemoryBound ? 'MEMORY BOUND' : 'COMPUTE BOUND'}
          </text>
        )}

        {/* No-data state */}
        {!hasData && (
          <text x={(plotLeft + plotRight) / 2} y={(plotTop + plotBottom) / 2}
                fontFamily="monospace" fontSize={11} fill="#4a5a6a" textAnchor="middle">
            No profiling data
          </text>
        )}
      </svg>

      <div className="absolute bottom-1 right-2 text-[8px] text-outline font-mono pointer-events-none">
        Memory Bound → Compute Bound
      </div>
    </div>
  );
}

// Error boundary so a chart crash never whites out the page
class RooflineErrorBoundary extends React.Component<
  { children: React.ReactNode; className?: string; height?: number },
  { hasError: boolean }
> {
  state = { hasError: false };
  static getDerivedStateFromError() { return { hasError: true }; }
  render() {
    if (this.state.hasError) {
      return (
        <div
          className={`flex items-center justify-center border border-outline/20 rounded ${this.props.className ?? ''}`}
          style={{ height: this.props.height ?? 160 }}
        >
          <span className="text-[10px] text-outline font-mono">Chart error</span>
        </div>
      );
    }
    return this.props.children;
  }
}

export default function RooflineChart(props: RooflineChartProps) {
  const spec = props.size === 'compact' ? COMPACT_SVG_SPEC : DEFAULT_SVG_SPEC;
  return (
    <RooflineErrorBoundary className={props.className} height={spec.height}>
      <RooflineChartInner {...props} />
    </RooflineErrorBoundary>
  );
}
