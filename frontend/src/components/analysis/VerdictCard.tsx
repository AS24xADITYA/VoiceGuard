import React from 'react';
import {
  CheckCircle2,
  AlertTriangle,
  AlertOctagon,
  HelpCircle,
  ShieldAlert,
  Info,
} from 'lucide-react';
import { VerdictType, AppliedOverride } from '../../types/api';
import { RiskGauge } from '../ui/RiskGauge';
import { Card } from '../ui/Card';

export interface VerdictCardProps {
  verdict: VerdictType;
  probability: number; // 0.0 - 1.0
  confidence: number;
  summary: string;
  guidance: string;
  reasons: string[];
  overrides?: AppliedOverride[];
}

export const VerdictCard: React.FC<VerdictCardProps> = ({
  verdict,
  probability,
  confidence,
  summary,
  guidance,
  reasons,
  overrides = [],
}) => {
  const getVerdictMeta = () => {
    switch (verdict) {
      case 'LOW':
        return {
          title: 'LOW RISK',
          sub: 'Authentic Speech Characteristics',
          textColor: 'text-risk-low',
          bgBanner: 'bg-risk-low-bg',
          borderColor: 'border-risk-low/30',
          Icon: CheckCircle2,
        };
      case 'MODERATE':
        return {
          title: 'MODERATE RISK',
          sub: 'Ambiguous or Mixed Indicators',
          textColor: 'text-risk-moderate',
          bgBanner: 'bg-risk-moderate-bg',
          borderColor: 'border-risk-moderate/30',
          Icon: AlertTriangle,
        };
      case 'HIGH':
        return {
          title: 'HIGH RISK',
          sub: 'Synthetic Speech or Scam Patterns Detected',
          textColor: 'text-risk-high',
          bgBanner: 'bg-risk-high-bg',
          borderColor: 'border-risk-high/30',
          Icon: AlertOctagon,
        };
      case 'INCONCLUSIVE':
      default:
        return {
          title: 'INCONCLUSIVE',
          sub: 'Audio Quality Below Threshold',
          textColor: 'text-risk-unknown',
          bgBanner: 'bg-bg-overlay',
          borderColor: 'border-border-default',
          Icon: HelpCircle,
        };
    }
  };

  const meta = getVerdictMeta();
  const Icon = meta.Icon;

  return (
    <Card
      className={`border-2 ${meta.borderColor} ${meta.bgBanner} relative overflow-hidden transition-all duration-300`}
      aria-live="assertive"
    >
      <div className="p-6 md:p-8 flex flex-col md:flex-row items-start md:items-center justify-between gap-6">
        {/* Left: Verdict and Details */}
        <div className="flex-1 space-y-4">
          <div className="flex items-center gap-3">
            <div className={`p-2.5 rounded-lg bg-bg-surface border ${meta.borderColor}`}>
              <Icon className={`w-7 h-7 ${meta.textColor}`} />
            </div>
            <div>
              <div className="flex items-center gap-3">
                <h1 className={`text-2xl md:text-3xl font-bold font-mono tracking-tight ${meta.textColor}`}>
                  {meta.title}
                </h1>
                <span className="text-xs font-mono uppercase px-2.5 py-0.5 rounded bg-bg-surface border border-border-subtle text-text-secondary">
                  Confidence: {Math.round(confidence * 100)}%
                </span>
              </div>
              <p className="text-sm font-medium text-text-secondary mt-0.5">{meta.sub}</p>
            </div>
          </div>

          {/* Plain Language Summary & Guidance */}
          <div className="space-y-2 pt-1">
            <p className="text-base text-text-primary font-normal leading-relaxed">{summary}</p>
            <div className="p-3.5 rounded-md bg-bg-surface/80 border border-border-subtle flex items-start gap-2.5">
              <span className="text-xs uppercase font-mono font-bold text-accent shrink-0 mt-0.5">
                Recommended Action:
              </span>
              <p className="text-sm text-text-primary font-medium">{guidance}</p>
            </div>
          </div>

          {/* Contributing Reason Chips */}
          {reasons.length > 0 && (
            <div className="flex flex-wrap items-center gap-2 pt-1">
              <span className="text-xs font-mono text-text-tertiary">Primary Factors:</span>
              {reasons.map((reason, idx) => (
                <span
                  key={idx}
                  className="px-2.5 py-1 rounded-md text-xs font-mono bg-bg-surface border border-border-default text-text-secondary"
                >
                  {reason}
                </span>
              ))}
            </div>
          )}

          {/* Applied Overrides Banner (if any) */}
          {overrides.length > 0 && (
            <div className="p-3 rounded-md bg-bg-elevated border border-accent/30 text-xs text-text-secondary space-y-1">
              <div className="flex items-center gap-1.5 font-semibold text-accent">
                <ShieldAlert className="w-3.5 h-3.5" />
                <span>Decision Override Applied ({overrides.map((o) => o.rule).join(', ')})</span>
              </div>
              {overrides.map((o, idx) => (
                <p key={idx} className="text-text-tertiary">
                  {o.reason}
                </p>
              ))}
            </div>
          )}
        </div>

        {/* Right: Radial Calibrated Risk Gauge */}
        <div className="flex flex-col items-center justify-center p-4 rounded-xl bg-bg-surface/60 border border-border-subtle shrink-0 self-stretch md:self-auto min-w-[180px]">
          <RiskGauge value={probability} size="lg" />
          <div className="mt-3 text-center">
            <span className="text-xs font-mono text-text-secondary block">Calibrated Risk</span>
            <span className="text-[10px] text-text-tertiary flex items-center justify-center gap-1 mt-0.5">
              <Info className="w-3 h-3" />
              Empirically calibrated probability
            </span>
          </div>
        </div>
      </div>

      {/* Persistent Legal Footnote */}
      <div className="px-6 md:px-8 py-2.5 bg-bg-base/40 border-t border-border-subtle/50 text-xs text-text-tertiary flex items-center justify-between">
        <span>An automated risk assessment, not a determination of fraud.</span>
        <span className="font-mono text-[11px] hidden sm:inline">VoiceGuard Pipeline</span>
      </div>
    </Card>
  );
};
