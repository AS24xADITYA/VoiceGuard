import React from 'react';

export interface RiskGaugeProps {
  value: number; // 0.0 to 1.0
  thresholds?: {
    moderate: number;
    high: number;
  };
  size?: 'sm' | 'md' | 'lg';
  showLabel?: boolean;
}

export const RiskGauge: React.FC<RiskGaugeProps> = ({
  value,
  thresholds = { moderate: 0.35, high: 0.70 },
  size = 'md',
  showLabel = true,
}) => {
  // Clamp value
  const clamped = Math.max(0, Math.min(1, value));
  const percentage = Math.round(clamped * 100);

  // Dimensions
  const config = {
    sm: { radius: 28, stroke: 5, svgSize: 70, textSize: 'text-xs' },
    md: { radius: 44, stroke: 7, svgSize: 104, textSize: 'text-xl' },
    lg: { radius: 64, stroke: 9, svgSize: 150, textSize: 'text-3xl' },
  }[size];

  const circumference = 2 * Math.PI * config.radius;
  const strokeDashoffset = circumference - clamped * circumference;

  let colorVar = 'var(--risk-low)';
  if (clamped >= thresholds.high) {
    colorVar = 'var(--risk-high)';
  } else if (clamped >= thresholds.moderate) {
    colorVar = 'var(--risk-moderate)';
  }

  return (
    <div className="relative inline-flex flex-col items-center justify-center">
      <svg
        width={config.svgSize}
        height={config.svgSize}
        className="transform -rotate-90"
        aria-hidden="true"
      >
        {/* Background Track */}
        <circle
          cx={config.svgSize / 2}
          cy={config.svgSize / 2}
          r={config.radius}
          fill="none"
          stroke="var(--bg-overlay)"
          strokeWidth={config.stroke}
        />
        {/* Progress Arc */}
        <circle
          cx={config.svgSize / 2}
          cy={config.svgSize / 2}
          r={config.radius}
          fill="none"
          stroke={colorVar}
          strokeWidth={config.stroke}
          strokeDasharray={circumference}
          strokeDashoffset={strokeDashoffset}
          strokeLinecap="round"
          className="transition-all duration-700 ease-out"
        />
      </svg>

      {showLabel && (
        <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none">
          <span className={`font-mono font-bold tracking-tight text-text-primary ${config.textSize}`}>
            {percentage}%
          </span>
          <span className="text-[10px] uppercase font-mono text-text-tertiary">risk</span>
        </div>
      )}
    </div>
  );
};
