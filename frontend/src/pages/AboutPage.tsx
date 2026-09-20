import React, { useState, useEffect } from 'react';
import {
  Shield,
  Activity,
  Layers,
  FileAudio,
  Zap,
  AlertTriangle,
  Lock,
  ExternalLink,
  BookOpen,
} from 'lucide-react';
import { Card } from '../components/ui/Card';
import { api } from '../api/endpoints';
import { SystemMetrics } from '../types/api';

export const AboutPage: React.FC = () => {
  const [metrics, setMetrics] = useState<SystemMetrics | null>(null);

  useEffect(() => {
    api.system.metrics()
      .then(setMetrics)
      .catch(() => {
        setMetrics(null);
      });
  }, []);

  return (
    <div className="max-w-4xl mx-auto space-y-12 py-6">
      {/* Title */}
      <div className="space-y-3">
        <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-bg-surface border border-border-default text-xs font-mono text-accent">
          <BookOpen className="w-3.5 h-3.5" />
          <span>Scientific Methodology &amp; Engineering Specification</span>
        </div>
        <h1 className="text-3xl sm:text-4xl font-bold font-mono tracking-tight text-text-primary">
          About VoiceGuard
        </h1>
        <p className="text-sm sm:text-base text-text-secondary leading-relaxed">
          A multi-signal AI pipeline designed to detect acoustic deepfakes, assess extortion language
          tactics, and provide interpretable attribution evidence without making uncalibrated claims.
        </p>
      </div>

      {/* 1. Architecture Overview */}
      <section className="space-y-4">
        <h2 className="text-xl font-bold font-mono text-text-primary flex items-center gap-2">
          <Activity className="w-5 h-5 text-accent" />
          <span>1. Architecture &amp; Methodology</span>
        </h2>
        <p className="text-xs sm:text-sm text-text-secondary leading-relaxed">
          VoiceGuard treats synthetic speech detection as a multi-modal, adversarial forensic problem.
          Rather than relying solely on a single black-box audio classifier, the system decouples analysis
          into three orthogonal branches:
        </p>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 pt-2">
          <Card className="p-4 bg-bg-surface border-border-default space-y-2">
            <div className="text-accent font-mono text-xs font-bold uppercase">Acoustic CNN</div>
            <p className="text-xs text-text-secondary leading-relaxed">
              Log-mel spectrogram features processed via EfficientNet-B0 trained on ASVspoof 2019/2021
              and In-the-Wild datasets to detect phase discontinuities, spectral roll-off anomalies, and
              neural vocoder artifacts.
            </p>
          </Card>

          <Card className="p-4 bg-bg-surface border-border-default space-y-2">
            <div className="text-accent font-mono text-xs font-bold uppercase">Linguistic Intent</div>
            <p className="text-xs text-text-secondary leading-relaxed">
              Multilingual transcription powered by Whisper with hallucination guards, coupled with an
              XLM-RoBERTa dual-head classifier detecting 8 extortion tactics. Production-tier accuracy
              in English, Hindi, and Tamil; Marathi and Bengali are designated as experimental/degraded
              due to ASR phoneme drift under telephone speech.
            </p>
          </Card>

          <Card className="p-4 bg-bg-surface border-border-default space-y-2">
            <div className="text-accent font-mono text-xs font-bold uppercase">Challenge–Response</div>
            <p className="text-xs text-text-secondary leading-relaxed">
              Interactive physiological challenge protocol that prompts randomized pitch modulations and
              phonetic shifts. Feature deltas are benchmarked against human biological ranges.
            </p>
          </Card>
        </div>
      </section>

      {/* 2. Published Evaluation Benchmarks */}
      <section className="space-y-4">
        <h2 className="text-xl font-bold font-mono text-text-primary flex items-center gap-2">
          <Layers className="w-5 h-5 text-accent" />
          <span>2. Published Evaluation Metrics</span>
        </h2>
        <div className="p-3 rounded-lg bg-amber-500/10 border border-amber-500/20 text-amber-400 text-xs font-mono">
          ⚠️ Model training in progress — evaluation metrics pending real evaluation pass.
        </div>
        <p className="text-xs text-text-secondary">
          Empirical results across standardized benchmarks will populate automatically once genuine evaluation scripts complete.
        </p>

        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 pt-1 font-mono">
          <div className="p-4 rounded-xl bg-bg-surface border border-border-default">
            <span className="text-text-tertiary text-[11px] block">In-Domain EER (ASVspoof)</span>
            <span className="text-xl font-bold text-text-primary mt-1 block">
              {metrics?.model_performance?.acoustic_eer_in_domain != null
                ? `${(metrics.model_performance.acoustic_eer_in_domain * 100).toFixed(1)}%`
                : 'PENDING'}
            </span>
            <span className="text-[10px] text-text-tertiary">Equal Error Rate</span>
          </div>

          <div className="p-4 rounded-xl bg-bg-surface border border-border-default">
            <span className="text-text-tertiary text-[11px] block">Out-of-Domain EER</span>
            <span className="text-xl font-bold text-text-primary mt-1 block">
              {metrics?.model_performance?.acoustic_eer_out_of_domain != null
                ? `${(metrics.model_performance.acoustic_eer_out_of_domain * 100).toFixed(1)}%`
                : 'PENDING'}
            </span>
            <span className="text-[10px] text-text-tertiary">In-the-Wild generalization</span>
          </div>

          <div className="p-4 rounded-xl bg-bg-surface border border-border-default">
            <span className="text-text-tertiary text-[11px] block">Scam Intent F1</span>
            <span className="text-xl font-bold text-text-primary mt-1 block">
              {metrics?.model_performance?.scam_macro_f1 != null
                ? metrics.model_performance.scam_macro_f1.toFixed(3)
                : 'PENDING'}
            </span>
            <span className="text-[10px] text-text-tertiary">Macro-averaged F1</span>
          </div>

          <div className="p-4 rounded-xl bg-bg-surface border border-border-default">
            <span className="text-text-tertiary text-[11px] block">Calibration ECE</span>
            <span className="text-xl font-bold text-text-primary mt-1 block">
              {metrics?.model_performance?.fusion_ece != null
                ? metrics.model_performance.fusion_ece.toFixed(3)
                : 'PENDING'}
            </span>
            <span className="text-[10px] text-text-tertiary">Expected Calibration Error</span>
          </div>
        </div>
      </section>

      {/* 3. System Limitations (Verbatim from 01 §6) */}
      <section className="space-y-4">
        <h2 className="text-xl font-bold font-mono text-text-primary flex items-center gap-2">
          <AlertTriangle className="w-5 h-5 text-accent" />
          <span>3. Known System Limitations</span>
        </h2>
        <div className="space-y-3 text-xs sm:text-sm text-text-secondary">
          <div className="p-4 rounded-xl bg-bg-surface border border-border-default space-y-1">
            <strong className="text-text-primary block font-mono">1. Codec Compression Artifacts</strong>
            <p>
              Low-bitrate cellular telephony codecs (such as AMR-NB and aggressive Opus compression)
              severely attenuate high-frequency harmonics (&gt;4 kHz), which can mimic the acoustic
              fingerprints of neural vocoders and cause false positives.
            </p>
          </div>

          <div className="p-4 rounded-xl bg-bg-surface border border-border-default space-y-1">
            <strong className="text-text-primary block font-mono">2. Zero-Shot Generative Models</strong>
            <p>
              Diffusion-based voice synthesizers and continuous-time vocoders that were absent from the
              training corpus can produce smoother spectral representations that acoustic models assign
              lower confidence to, triggering the fallback to linguistic and challenge branches.
            </p>
          </div>

          <div className="p-4 rounded-xl bg-bg-surface border border-border-default space-y-1">
            <strong className="text-text-primary block font-mono">3. Audio Length Requirements</strong>
            <p>
              Recordings shorter than 3 seconds or with voice activity ratios below 40% cannot be
              reliably assessed by the acoustic CNN. Such samples are flagged as INCONCLUSIVE rather than
              falsely categorized.
            </p>
          </div>

          <div className="p-4 rounded-xl bg-bg-surface border border-border-default space-y-1">
            <strong className="text-text-primary block font-mono">4. Non-Interception Boundary</strong>
            <p>
              VoiceGuard does not tap, monitor, or intercept active PSTN/GSM telephone calls. It operates
              strictly on audio explicitly uploaded or recorded through client interfaces.
            </p>
          </div>

          <div className="p-4 rounded-xl bg-bg-surface border border-amber-500/30 space-y-1">
            <strong className="text-amber-400 block font-mono">5. Bengali &amp; Marathi ASR Degradation (per 13 §7.3)</strong>
            <p>
              Empirical speech-to-text evaluation (Condition S3) demonstrates severe transcription breakdown
              for Bengali (F1 0.0769) and Marathi (F1 0.2143). Whisper ASR produces phonetic script
              transliteration and word drops on telephone audio, significantly suppressing tactic recall.
              Scam detection for Bengali and Marathi is currently classified as <strong>experimental / degraded</strong>,
              whereas English (F1 0.8889), Hindi (F1 0.7805), and Tamil (F1 0.8095) achieve production-grade reliability.
            </p>
          </div>
        </div>
      </section>

      {/* 4. Privacy & Data Handling */}
      <section className="space-y-4">
        <h2 className="text-xl font-bold font-mono text-text-primary flex items-center gap-2">
          <Lock className="w-5 h-5 text-accent" />
          <span>4. Privacy &amp; Data Provenance</span>
        </h2>
        <div className="p-5 rounded-xl bg-bg-surface border border-border-default space-y-3 text-xs sm:text-sm text-text-secondary leading-relaxed">
          <p>
            Audio data submitted to VoiceGuard is stored in encrypted local/S3 partitions and bound to the
            originating user account with strict row-level authorization.
          </p>
          <p>
            An automated retention service purges temporary analyses and audio artifacts according to the
            configurable window (default: 30 days). Users can export or irreversibly delete their records at
            any time from their account settings.
          </p>
        </div>
      </section>
    </div>
  );
};
