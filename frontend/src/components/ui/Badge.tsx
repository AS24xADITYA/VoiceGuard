import React from 'react';
import { CheckCircle2, AlertTriangle, AlertOctagon, HelpCircle } from 'lucide-react';
import { VerdictType } from '../../types/api';

export interface BadgeProps extends React.HTMLAttributes<HTMLSpanElement> {
  variant?: 'low' | 'moderate' | 'high' | 'unknown' | 'accent' | 'subtle';
  size?: 'sm' | 'md';
}

export const Badge: React.FC<BadgeProps> = ({
  className = '',
  variant = 'subtle',
  size = 'md',
  children,
  ...props
}) => {
  const sizeClasses = size === 'sm' ? 'px-2 py-0.5 text-[11px] gap-1' : 'px-2.5 py-1 text-xs gap-1.5';

  const variantClasses = {
    low: 'bg-risk-low-bg text-risk-low border-risk-low/20',
    moderate: 'bg-risk-moderate-bg text-risk-moderate border-risk-moderate/20',
    high: 'bg-risk-high-bg text-risk-high border-risk-high/20',
    unknown: 'bg-bg-overlay text-text-tertiary border-border-default',
    accent: 'bg-accent-glow text-accent border-accent/20',
    subtle: 'bg-bg-elevated text-text-secondary border-border-subtle',
  }[variant];

  return (
    <span
      className={`inline-flex items-center font-medium rounded-full border ${sizeClasses} ${variantClasses} ${className}`}
      {...props}
    >
      {children}
    </span>
  );
};

export const VerdictBadge: React.FC<{ verdict: VerdictType | string; size?: 'sm' | 'md' }> = ({
  verdict,
  size = 'md',
}) => {
  switch (verdict) {
    case 'LOW':
      return (
        <Badge variant="low" size={size}>
          <CheckCircle2 className="w-3.5 h-3.5 shrink-0" />
          <span>Low Risk</span>
        </Badge>
      );
    case 'MODERATE':
      return (
        <Badge variant="moderate" size={size}>
          <AlertTriangle className="w-3.5 h-3.5 shrink-0" />
          <span>Moderate Risk</span>
        </Badge>
      );
    case 'HIGH':
      return (
        <Badge variant="high" size={size}>
          <AlertOctagon className="w-3.5 h-3.5 shrink-0" />
          <span>High Risk</span>
        </Badge>
      );
    case 'INCONCLUSIVE':
    default:
      return (
        <Badge variant="unknown" size={size}>
          <HelpCircle className="w-3.5 h-3.5 shrink-0" />
          <span>Inconclusive</span>
        </Badge>
      );
  }
};
