import React from 'react';
import { Link } from 'react-router-dom';
import {
  Shield,
  Mic,
  FileAudio,
  Activity,
  Zap,
  ArrowRight,
  AlertTriangle,
  Layers,
  CheckCircle2,
  Info,
} from 'lucide-react';
import { Button } from '../components/ui/Button';
import { Card } from '../components/ui/Card';

export const LandingPage: React.FC = () => {
  return (
    <div className="space-y-16 py-6 sm:py-10">
      {/* 1. Hero Section */}
      <section className="text-center max-w-3xl mx-auto space-y-6">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-accent-glow border border-accent/20 text-accent text-xs font-mono">
          <Shield className="w-3.5 h-3.5" />
          <span>Multi-Signal Voice Synthesis & Extortion Defense</span>
        </div>

        <h1 className="text-4xl sm:text-5xl font-extrabold tracking-tight text-text-primary">
          Deepfake Voice &amp; Scam Intelligence
        </h1>

        <p className="text-base sm:text-lg text-text-secondary leading-relaxed max-w-2xl mx-auto">
          An automated security pipeline uniting convolutional acoustic artifact detection,
          multilingual extortion language analysis, and interactive challenge verification.
        </p>

        <div className="flex flex-col sm:flex-row items-center justify-center gap-3 pt-2">
          <Link to="/analyze">
            <Button size="lg" className="w-full sm:w-auto gap-2">
              <Mic className="w-4 h-4" />
              <span>Analyse Audio</span>
              <ArrowRight className="w-4 h-4 ml-1" />
            </Button>
          </Link>
          <Link to="/about">
            <Button variant="secondary" size="lg" className="w-full sm:w-auto">
              How It Works
            </Button>
          </Link>
        </div>
      </section>

      {/* 2. Scope Notice (Above the fold per 09 §4.1) */}
      <section className="max-w-4xl mx-auto">
        <div className="p-4 sm:p-5 rounded-xl bg-bg-surface border border-accent/30 flex items-start gap-3.5 text-sm text-text-secondary shadow-card">
          <AlertTriangle className="w-5 h-5 text-accent shrink-0 mt-0.5" />
          <div>
            <span className="font-semibold text-text-primary block sm:inline">
              Mandatory Scope Notice:{' '}
            </span>
            VoiceGuard analyses audio you upload or record. It does not monitor or intercept live phone calls.
            Assessments are probabilistic and do not constitute definitive legal proof of identity or fabrication.
          </div>
        </div>
      </section>

      {/* 3. Pipeline Diagram (Static, crisp SVG per 09 §4.1) */}
      <section className="max-w-5xl mx-auto space-y-6">
        <div className="text-center space-y-2">
          <h2 className="text-2xl font-bold font-mono tracking-tight text-text-primary">
            Convergent Multi-Signal Architecture
          </h2>
          <p className="text-sm text-text-tertiary max-w-xl mx-auto">
            Independent signals converge on a calibrated Bayesian fusion engine to prevent single-point failures.
          </p>
        </div>

        <div className="p-6 rounded-2xl bg-bg-surface border border-border-default overflow-x-auto">
          <svg
            viewBox="0 0 900 240"
            className="w-full min-w-[700px] text-text-primary select-none"
            fill="none"
          >
            {/* Input Node */}
            <rect x="20" y="90" width="120" height="60" rx="8" fill="var(--bg-elevated)" stroke="var(--border-default)" strokeWidth="1.5" />
            <text x="80" y="118" textAnchor="middle" fill="var(--text-primary)" fontSize="12" fontWeight="600" fontFamily="var(--font-mono)">INPUT AUDIO</text>
            <text x="80" y="134" textAnchor="middle" fill="var(--text-tertiary)" fontSize="10" fontFamily="var(--font-mono)">16kHz Mono</text>

            {/* Connecting lines */}
            <path d="M 140 120 L 220 50" stroke="var(--border-strong)" strokeWidth="1.5" strokeDasharray="3 3" />
            <path d="M 140 120 L 220 120" stroke="var(--border-strong)" strokeWidth="1.5" strokeDasharray="3 3" />
            <path d="M 140 120 L 220 190" stroke="var(--border-strong)" strokeWidth="1.5" strokeDasharray="3 3" />

            {/* Branch 1: Acoustic */}
            <rect x="220" y="20" width="200" height="60" rx="8" fill="var(--bg-elevated)" stroke="var(--border-default)" strokeWidth="1.5" />
            <text x="320" y="46" textAnchor="middle" fill="var(--text-primary)" fontSize="12" fontWeight="600" fontFamily="var(--font-mono)">ACOUSTIC BRANCH</text>
            <text x="320" y="62" textAnchor="middle" fill="var(--accent)" fontSize="10" fontFamily="var(--font-mono)">Log-Mel CNN + Grad-CAM</text>

            {/* Branch 2: Linguistic */}
            <rect x="220" y="90" width="200" height="60" rx="8" fill="var(--bg-elevated)" stroke="var(--border-default)" strokeWidth="1.5" />
            <text x="320" y="116" textAnchor="middle" fill="var(--text-primary)" fontSize="12" fontWeight="600" fontFamily="var(--font-mono)">LINGUISTIC BRANCH</text>
            <text x="320" y="132" textAnchor="middle" fill="var(--accent)" fontSize="10" fontFamily="var(--font-mono)">Whisper + XLM-R Intent</text>

            {/* Branch 3: Challenge */}
            <rect x="220" y="160" width="200" height="60" rx="8" fill="var(--bg-elevated)" stroke="var(--border-default)" strokeWidth="1.5" />
            <text x="320" y="186" textAnchor="middle" fill="var(--text-primary)" fontSize="12" fontWeight="600" fontFamily="var(--font-mono)">CHALLENGE BRANCH</text>
            <text x="320" y="202" textAnchor="middle" fill="var(--accent)" fontSize="10" fontFamily="var(--font-mono)">Interactive Verification</text>

            {/* Convergence to Fusion */}
            <path d="M 420 50 L 520 120" stroke="var(--border-strong)" strokeWidth="1.5" />
            <path d="M 420 120 L 520 120" stroke="var(--border-strong)" strokeWidth="1.5" />
            <path d="M 420 190 L 520 120" stroke="var(--border-strong)" strokeWidth="1.5" />

            {/* Fusion Node */}
            <rect x="520" y="80" width="160" height="80" rx="10" fill="var(--bg-overlay)" stroke="var(--accent)" strokeWidth="2" />
            <text x="600" y="115" textAnchor="middle" fill="var(--accent)" fontSize="13" fontWeight="bold" fontFamily="var(--font-mono)">FUSION ENGINE</text>
            <text x="600" y="133" textAnchor="middle" fill="var(--text-secondary)" fontSize="10" fontFamily="var(--font-mono)">Calibrated Logistic +</text>
            <text x="600" y="146" textAnchor="middle" fill="var(--text-secondary)" fontSize="10" fontFamily="var(--font-mono)">Safety Overrides</text>

            {/* Output to Verdict */}
            <path d="M 680 120 L 760 120" stroke="var(--border-strong)" strokeWidth="1.5" />

            {/* Final Verdict Node */}
            <rect x="760" y="90" width="120" height="60" rx="8" fill="var(--bg-elevated)" stroke="var(--risk-low)" strokeWidth="1.5" />
            <text x="820" y="118" textAnchor="middle" fill="var(--text-primary)" fontSize="12" fontWeight="600" fontFamily="var(--font-mono)">VERDICT</text>
            <text x="820" y="134" textAnchor="middle" fill="var(--risk-low)" fontSize="10" fontFamily="var(--font-mono)">Calibrated Risk %</text>
          </svg>
        </div>
      </section>

      {/* 4. Four Signal Cards per 09 §4.1 */}
      <section className="max-w-5xl mx-auto space-y-6">
        <h2 className="text-xl font-bold font-mono tracking-tight text-text-primary text-center">
          Comprehensive Analysis Modalities
        </h2>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <Card className="p-5 bg-bg-surface border-border-default space-y-2">
            <div className="w-8 h-8 rounded-lg bg-bg-elevated flex items-center justify-center text-accent border border-border-subtle">
              <Activity className="w-4 h-4" />
            </div>
            <h3 className="text-sm font-semibold font-mono text-text-primary">Acoustic CNN</h3>
            <p className="text-xs text-text-secondary leading-relaxed">
              Detects vocoder phase discontinuities, spectral tilt anomalies, and high-frequency synthesis cutoffs.
            </p>
          </Card>

          <Card className="p-5 bg-bg-surface border-border-default space-y-2">
            <div className="w-8 h-8 rounded-lg bg-bg-elevated flex items-center justify-center text-accent border border-border-subtle">
              <FileAudio className="w-4 h-4" />
            </div>
            <h3 className="text-sm font-semibold font-mono text-text-primary">Linguistic Intent</h3>
            <p className="text-xs text-text-secondary leading-relaxed">
              Transcribes multilingual audio and classifies extortion tactics, artificial urgency, and financial coercion.
            </p>
          </Card>

          <Card className="p-5 bg-bg-surface border-border-default space-y-2">
            <div className="w-8 h-8 rounded-lg bg-bg-elevated flex items-center justify-center text-accent border border-border-subtle">
              <Zap className="w-4 h-4" />
            </div>
            <h3 className="text-sm font-semibold font-mono text-text-primary">Interactive Challenge</h3>
            <p className="text-xs text-text-secondary leading-relaxed">
              Issues physiological pitch and phonetic challenges that real-time voice conversion algorithms cannot synthesize.
            </p>
          </Card>

          <Card className="p-5 bg-bg-surface border-border-default space-y-2">
            <div className="w-8 h-8 rounded-lg bg-bg-elevated flex items-center justify-center text-accent border border-border-subtle">
              <Layers className="w-4 h-4" />
            </div>
            <h3 className="text-sm font-semibold font-mono text-text-primary">Grad-CAM Salience</h3>
            <p className="text-xs text-text-secondary leading-relaxed">
              Visualizes exact time-frequency spectrogram regions that influenced the model&apos;s classification decision.
            </p>
          </Card>
        </div>
      </section>

      {/* 5. Limitations Strip (Honest statements from 01 §6 per 09 §4.1) */}
      <section className="max-w-4xl mx-auto">
        <div className="p-6 rounded-xl bg-bg-elevated border border-border-subtle space-y-3">
          <div className="flex items-center gap-2 text-xs font-mono font-semibold uppercase text-accent">
            <Info className="w-4 h-4" />
            <span>Documented System Limitations (PRD §6)</span>
          </div>
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 text-xs text-text-secondary">
            <div className="p-3 rounded bg-bg-surface border border-border-subtle">
              <strong className="text-text-primary block mb-1">Acoustic Codec Degradation</strong>
              Standard cellular codecs (AMR-WB, Opus 6kbps) compress high frequencies and can increase false positive rates.
            </div>
            <div className="p-3 rounded bg-bg-surface border border-border-subtle">
              <strong className="text-text-primary block mb-1">Non-Real-Time Architecture</strong>
              Pipeline processes whole files asynchronously to achieve calibrated accuracy; it does not monitor active cellular calls.
            </div>
            <div className="p-3 rounded bg-bg-surface border border-border-subtle">
              <strong className="text-text-primary block mb-1">Novel Voice Cloners</strong>
              Zero-shot acoustic models unseen during training may produce lower confidence scores, triggering challenge protocols.
            </div>
          </div>
        </div>
      </section>
    </div>
  );
};
