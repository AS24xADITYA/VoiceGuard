import React from 'react';
import { Card, CardHeader, CardTitle, CardContent } from '../ui/Card';
import { AlertTriangle, CheckCircle2, XCircle } from 'lucide-react';

export interface SignalCardProps {
  title: string;
  score: number; // 0.0 - 1.0
  scoreLabel: string;
  status: 'available' | 'unavailable' | 'degraded';
  reason?: string;
  children?: React.ReactNode;
}

export const SignalCard: React.FC<SignalCardProps> = ({
  title,
  score,
  scoreLabel,
  status,
  reason,
  children,
}) => {
  const percentage = Math.round(score * 100);

  const getScoreColor = () => {
    if (score >= 0.7) return 'bg-risk-high text-risk-high';
    if (score >= 0.35) return 'bg-risk-moderate text-risk-moderate';
    return 'bg-risk-low text-risk-low';
  };

  const getStatusBadge = () => {
    switch (status) {
      case 'available':
        return (
          <span className="inline-flex items-center gap-1 text-[11px] font-mono text-risk-low">
            <CheckCircle2 className="w-3 h-3" />
            Active
          </span>
        );
      case 'degraded':
        return (
          <span className="inline-flex items-center gap-1 text-[11px] font-mono text-risk-moderate">
            <AlertTriangle className="w-3 h-3" />
            Degraded
          </span>
        );
      case 'unavailable':
      default:
        return (
          <span className="inline-flex items-center gap-1 text-[11px] font-mono text-text-tertiary">
            <XCircle className="w-3 h-3" />
            Not Available
          </span>
        );
    }
  };

  return (
    <Card className="flex flex-col h-full bg-bg-surface border-border-default hover:border-border-strong transition-colors">
      <CardHeader className="flex flex-row items-center justify-between pb-3">
        <div>
          <CardTitle className="text-sm font-semibold tracking-wide uppercase font-mono text-text-secondary">
            {title}
          </CardTitle>
          <div className="mt-0.5">{getStatusBadge()}</div>
        </div>

        {status !== 'unavailable' ? (
          <div className="text-right">
            <div className="text-xl font-bold font-mono text-text-primary">{percentage}%</div>
            <div className="text-[10px] uppercase font-mono text-text-tertiary">{scoreLabel}</div>
          </div>
        ) : null}
      </CardHeader>

      <CardContent className="flex-1 flex flex-col justify-between pt-0 space-y-4">
        {/* Horizontal score bar */}
        {status !== 'unavailable' && (
          <div className="w-full bg-bg-overlay rounded-full h-2 overflow-hidden">
            <div
              className={`h-full rounded-full transition-all duration-500 ${getScoreColor().split(' ')[0]}`}
              style={{ width: `${percentage}%` }}
            />
          </div>
        )}

        {/* Degraded / Unavailable reason */}
        {reason && (
          <div className="p-2.5 rounded bg-bg-elevated border border-border-subtle text-xs text-text-secondary">
            {reason}
          </div>
        )}

        {/* Children details (e.g. sparklines, chips, badges) */}
        <div className="flex-1 flex flex-col justify-end space-y-3">{children}</div>
      </CardContent>
    </Card>
  );
};
