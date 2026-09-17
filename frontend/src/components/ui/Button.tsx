import React from 'react';
import { Loader2 } from 'lucide-react';

export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: 'primary' | 'secondary' | 'outline' | 'ghost' | 'danger';
  size?: 'sm' | 'md' | 'lg';
  loading?: boolean;
}

export const Button = React.forwardRef<HTMLButtonElement, ButtonProps>(
  ({ className = '', variant = 'primary', size = 'md', loading = false, disabled, children, ...props }, ref) => {
    const baseClasses =
      'inline-flex items-center justify-center font-medium transition-all duration-150 rounded-md focus:outline-none focus-visible:ring-2 focus-visible:ring-accent disabled:opacity-50 disabled:cursor-not-allowed select-none active:scale-[0.98]';

    const sizeClasses = {
      sm: 'px-2.5 py-1.5 text-xs gap-1.5',
      md: 'px-4 py-2 text-sm gap-2',
      lg: 'px-5 py-2.5 text-base gap-2.5',
    }[size];

    const variantClasses = {
      primary: 'bg-accent text-text-inverse hover:bg-accent/90 shadow-sm font-semibold',
      secondary: 'bg-bg-elevated text-text-primary hover:bg-bg-overlay border border-border-default',
      outline: 'bg-transparent text-text-secondary hover:text-text-primary hover:bg-bg-surface border border-border-default',
      ghost: 'bg-transparent text-text-secondary hover:text-text-primary hover:bg-bg-surface',
      danger: 'bg-risk-high text-text-inverse hover:bg-risk-high/90 shadow-sm font-semibold',
    }[variant];

    return (
      <button
        ref={ref}
        disabled={disabled || loading}
        className={`${baseClasses} ${sizeClasses} ${variantClasses} ${className}`}
        {...props}
      >
        {loading && <Loader2 className="w-4 h-4 animate-spin shrink-0" />}
        {children}
      </button>
    );
  }
);

Button.displayName = 'Button';
