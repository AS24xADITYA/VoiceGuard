import { BarChart3 } from 'lucide-react';
import { FeatureContribution } from '../../types/api';
import { Card } from '../ui/Card';

export interface ContributionChartProps {
  contributions: FeatureContribution[];
  maxItems?: number;
}

const FEATURE_LABELS: Record<string, string> = {
  acoustic_spoof_prob: 'Acoustic synthetic speech markers',
  scam_prob: 'Scam & extortion language in transcript',
  vad_speech_ratio: 'Voiced speech duration consistency',
  snr_db: 'Background noise & signal quality',
  spectral_flatness: 'High-frequency spectral dispersion',
  clipping_rate: 'Audio sample saturation & clipping',
  challenge_consistency: 'Challenge acoustic consistency delta',
};

export const ContributionChart: React.FC<ContributionChartProps> = ({
  contributions = [],
  maxItems = 5,
}) => {
  const validContributions = (contributions || [])
    .filter((c) => c != null)
    .map((c: any) => {
      const rawVal = typeof c.contribution === 'number' ? c.contribution : Number(c.contribution ?? c.value ?? 0);
      return {
        feature: c.feature || 'unknown',
        label: c.label || (c.feature ? FEATURE_LABELS[c.feature] : undefined),
        contribution: isNaN(rawVal) ? 0 : rawVal,
      };
    });

  // Sort by absolute contribution and take top maxItems
  const sorted = [...validContributions]
    .sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution))
    .slice(0, maxItems);

  // Maximum absolute value for normalization
  const maxAbs = Math.max(0.1, ...sorted.map((s) => Math.abs(s.contribution)));

  return (
    <Card className="p-6 bg-bg-surface border-border-default space-y-4">
      <div className="flex items-center justify-between pb-3 border-b border-border-subtle">
        <div>
          <h3 className="text-base font-semibold text-text-primary font-mono flex items-center gap-2">
            <BarChart3 className="w-4 h-4 text-accent" />
            <span>Signal Contribution Weights</span>
          </h3>
          <p className="text-xs text-text-tertiary mt-0.5">
            Directional impact of primary features on the calibrated risk assessment
          </p>
        </div>
      </div>

      {sorted.length > 0 ? (
        <div className="space-y-4 pt-1">
          {sorted.map((item, idx) => {
            const isRiskIncreasing = item.contribution >= 0;
            const widthPct = Math.min(100, Math.round((Math.abs(item.contribution) / maxAbs) * 100));
            const plainLabel = item.label || FEATURE_LABELS[item.feature] || item.feature;

            return (
              <div key={idx} className="space-y-1.5">
                <div className="flex items-center justify-between text-xs">
                  <span className="font-medium text-text-primary">{plainLabel}</span>
                  <div className="font-mono flex items-center gap-1.5">
                    <span
                      className={`text-[11px] font-semibold ${
                        isRiskIncreasing ? 'text-risk-high' : 'text-risk-low'
                      }`}
                    >
                      {isRiskIncreasing ? '+' : ''}
                      {item.contribution.toFixed(3)}
                    </span>
                    <span className="text-[10px] text-text-tertiary uppercase">
                      ({isRiskIncreasing ? 'Increases risk' : 'Reduces risk'})
                    </span>
                  </div>
                </div>

                {/* Diverging Bar Track */}
                <div className="w-full h-2 rounded-full bg-bg-elevated overflow-hidden relative">
                  <div
                    className={`h-full rounded-full transition-all duration-500 ${
                      isRiskIncreasing ? 'bg-risk-high' : 'bg-risk-low'
                    }`}
                    style={{ width: `${widthPct}%` }}
                  />
                </div>
              </div>
            );
          })}
        </div>
      ) : (
        <div className="p-6 text-center text-text-tertiary text-xs">
          No feature attribution breakdown available for this evaluation.
        </div>
      )}
    </Card>
  );
};
