import React from 'react';
import { AlertCircle } from 'lucide-react';
import { Link } from 'react-router-dom';
import { Button } from './ui/Button';

interface ErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
}

interface ErrorBoundaryProps {
  children: React.ReactNode;
  fallback?: React.ReactNode;
}

export class ErrorBoundary extends React.Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = { hasError: false, error: null };
  }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, info: React.ErrorInfo) {
    console.error('[ErrorBoundary] Caught render error:', error, info.componentStack);
  }

  render() {
    if (this.state.hasError) {
      if (this.props.fallback) return this.props.fallback;
      return (
        <div className="py-16 max-w-xl mx-auto text-center space-y-4">
          <div className="p-6 rounded-xl bg-risk-high-bg border border-risk-high/30 text-risk-high">
            <AlertCircle className="w-8 h-8 mx-auto mb-3" />
            <p className="font-semibold text-base">Page Rendering Error</p>
            <p className="text-xs mt-2 text-text-secondary font-mono">
              {this.state.error?.message || 'An unexpected error occurred while rendering this page.'}
            </p>
          </div>
          <Link to="/analyze">
            <Button variant="secondary">Submit a New Analysis</Button>
          </Link>
        </div>
      );
    }
    return this.props.children;
  }
}
