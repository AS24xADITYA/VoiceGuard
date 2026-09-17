import React from 'react';
import { Link } from 'react-router-dom';
import { Shield, AlertCircle, ExternalLink, Code } from 'lucide-react';
import { COPY } from '../../i18n/en';

export const Footer: React.FC = () => {
  return (
    <footer className="w-full border-t border-border-subtle bg-bg-surface/50 mt-auto py-8">
      <div className="mx-auto max-w-7xl px-4 sm:px-6 lg:px-8 space-y-6">
        {/* Scope banner per 09 §4.1 / §10 */}
        <div className="flex items-start gap-3 p-4 rounded-lg bg-bg-elevated border border-border-subtle text-xs text-text-secondary leading-relaxed">
          <AlertCircle className="w-4 h-4 text-accent mt-0.5 shrink-0" />
          <div>
            <span className="font-semibold text-text-primary block sm:inline">Scope Notice: </span>
            VoiceGuard analyses audio you upload or record. It does not monitor or intercept live phone calls.
            Assessments are probabilistic and do not constitute legal determinations of identity or fraud.
          </div>
        </div>

        {/* Links & Attribution */}
        <div className="flex flex-col sm:flex-row items-center justify-between gap-4 pt-4 border-t border-border-subtle text-xs text-text-tertiary">
          <div className="flex items-center gap-2">
            <Shield className="w-4 h-4 text-accent" />
            <span className="font-mono font-medium text-text-secondary">VoiceGuard</span>
            <span>&copy; {new Date().getFullYear()} — Multi-Signal Deepfake Voice & Scam Intelligence</span>
          </div>

          <div className="flex flex-wrap items-center gap-4">
            <Link to="/about" className="hover:text-text-primary transition-colors">
              Methodology & Limitations
            </Link>
            <a
              href="/docs"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-1 hover:text-text-primary transition-colors"
            >
              <span>API Docs</span>
              <ExternalLink className="w-3 h-3" />
            </a>
            <a
              href="https://github.com"
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-1 hover:text-text-primary transition-colors"
            >
              <Code className="w-3 h-3" />
              <span>Source Repository</span>
            </a>
          </div>
        </div>
      </div>
    </footer>
  );
};
