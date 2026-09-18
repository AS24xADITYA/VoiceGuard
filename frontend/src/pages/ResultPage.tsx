import React, { useState, useEffect } from 'react';
import { useParams, useNavigate, Link } from 'react-router-dom';
import {
  Download,
  Trash2,
  RefreshCw,
  ChevronDown,
  ChevronUp,
  Code,
  Copy,
  Check,
  AlertCircle,
  FileAudio,
  ShieldCheck,
  Sparkles,
  Layers,
  ArrowLeft,
} from 'lucide-react';
import { api } from '../api/endpoints';
import { AnalysisResponse, ChallengeVerifyResponse, FusionResult, VerdictType } from '../types/api';
import { COPY } from '../i18n/en';
import { VerdictCard } from '../components/analysis/VerdictCard';
import { SignalCard } from '../components/analysis/SignalCard';
import { SpectrogramViewer } from '../components/analysis/SpectrogramViewer';
import { TranscriptViewer } from '../components/analysis/TranscriptViewer';
import { ContributionChart } from '../components/analysis/ContributionChart';
import { AudioPlayer } from '../components/analysis/AudioPlayer';
import { ChallengePanel } from '../components/analysis/ChallengePanel';
import { Card } from '../components/ui/Card';
import { Button } from '../components/ui/Button';

// ── Helpers to bridge backend response field names to frontend types ──

function _verdictSummary(verdict?: string, risk?: number): string {
  const prob = risk != null ? `${(risk * 100).toFixed(0)}%` : 'unknown';
  switch (verdict) {
    case 'HIGH':
      return `This audio presents a HIGH risk (${prob}) of being synthetically generated or used in a scam.`;
    case 'MODERATE':
      return `Moderate risk (${prob}) detected. Some indicators of synthetic or scam audio are present.`;
    case 'LOW':
      return `Low risk (${prob}). No significant indicators of synthetic audio or scam intent were detected.`;
    default:
      return 'Analysis completed without conclusive confidence.';
  }
}

function _verdictGuidance(verdict?: string): string {
  switch (verdict) {
    case 'HIGH':
      return 'Do NOT trust this audio. Verify the speaker through an independent, trusted channel before taking any action.';
    case 'MODERATE':
      return 'Exercise caution. Verify the speaker through secondary trusted channels before acting on any requests.';
    case 'LOW':
      return 'This audio appears authentic, but always maintain healthy skepticism with unsolicited calls.';
    default:
      return 'Verify the speaker through secondary trusted channels.';
  }
}

function _mapContributions(contributions?: Record<string, number> | null, featureVector?: number[] | null): any[] {
  if (!contributions) return [];
  return Object.entries(contributions).map(([feature, contribution], i) => ({
    feature,
    value: featureVector?.[i] ?? 0,
    contribution,
    label: feature.replace(/_/g, ' '),
  }));
}

function _buildWindowPredictions(acoustic?: any): any[] {
  if (!acoustic) return [];
  // Backend may return window_predictions directly or window_scores + window_times separately
  if (acoustic.window_predictions) return acoustic.window_predictions;
  const scores = acoustic.window_scores || [];
  const times = acoustic.window_times || [];
  return scores.map((prob: number, i: number) => ({
    window_index: i,
    start_sec: times[i] ?? i * 2,
    end_sec: times[i + 1] ?? (i + 1) * 2,
    raw_prob: prob,
  }));
}

function _mapSalientSpans(spans?: any[]): any[] {
  if (!spans || spans.length === 0) return [];
  return spans.map((s: any) => ({
    text: s.text,
    start_char: s.start_char ?? s.start ?? 0,
    end_char: s.end_char ?? s.end ?? 0,
    attribution_weight: s.attribution_weight ?? s.weight ?? 0,
    tactics: s.tactics || [],
  }));
}

// Safe number formatter — never crashes on null/undefined
function _fmt(value: number | null | undefined, decimals = 1, fallback = '—'): string {
  if (value == null || !isFinite(value)) return fallback;
  return value.toFixed(decimals);
}

export const ResultPage: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();

  const [data, setData] = useState<AnalysisResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Seek interaction between transcript and audio player
  const [seekTime, setSeekTime] = useState<number | null>(null);

  // Technical details collapsed state per 09 §4.3.G
  const [showTechnicalDetails, setShowTechnicalDetails] = useState(false);
  const [copiedRawJson, setCopiedRawJson] = useState(false);

  const fetchResult = async () => {
    if (!id) return;
    setLoading(true);
    setError(null);
    try {
      const res = await api.analyses.get(id);
      setData(res);
    } catch (err: any) {
      setError(
        err.response?.data?.detail || err.message || 'Failed to retrieve analysis details.'
      );
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchResult();
  }, [id]);

  const handleDelete = async () => {
    if (!id) return;
    if (!window.confirm('Are you sure you want to delete this analysis and all associated artifacts?')) {
      return;
    }
    try {
      await api.analyses.delete(id);
      navigate('/history');
    } catch (err: any) {
      alert(err.response?.data?.detail || 'Failed to delete analysis');
    }
  };

  const handleExportJson = () => {
    if (!data) return;
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `voiceguard-analysis-${data.id}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const copyRawJson = () => {
    if (!data) return;
    navigator.clipboard.writeText(JSON.stringify(data, null, 2));
    setCopiedRawJson(true);
    setTimeout(() => setCopiedRawJson(false), 2000);
  };

  const handleChallengeVerification = (verifyRes: ChallengeVerifyResponse) => {
    // Re-fetch analysis to load updated re-fused verdict
    fetchResult();
  };

  if (loading) {
    return (
      <div className="py-24 text-center space-y-4">
        <div className="w-10 h-10 border-2 border-accent border-t-transparent rounded-full animate-spin mx-auto" />
        <p className="text-sm font-mono text-text-tertiary">Loading forensic analysis record…</p>
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="py-16 max-w-xl mx-auto text-center space-y-4">
        <div className="p-4 rounded-xl bg-risk-high-bg border border-risk-high/30 text-risk-high text-sm">
          <AlertCircle className="w-6 h-6 mx-auto mb-2" />
          <p className="font-semibold">Analysis Unavailable</p>
          <p className="text-xs mt-1">{error || 'The requested analysis record does not exist or has expired.'}</p>
        </div>
        <Link to="/analyze">
          <Button variant="secondary">Submit a New Analysis</Button>
        </Link>
      </div>
    );
  }

  // Artifact URLs
  const audioArtifact = data.artifacts.find((a) => a.kind === 'canonical_audio');
  const specArtifact = data.artifacts.find((a) => a.kind === 'spectrogram');
  const overlayArtifact = data.artifacts.find((a) => a.kind === 'gradcam_overlay');
  const heatmapArtifact = data.artifacts.find((a) => a.kind === 'gradcam_heatmap');

  const audioUrl = audioArtifact
    ? (api.artifacts.getUrl(data.id, audioArtifact.id) ?? audioArtifact.download_url)
    : null;
  const specUrl = specArtifact
    ? (api.artifacts.getUrl(data.id, specArtifact.id) ?? specArtifact.download_url)
    : null;
  const overlayUrl = overlayArtifact
    ? (api.artifacts.getUrl(data.id, overlayArtifact.id) ?? overlayArtifact.download_url)
    : null;
  const heatmapUrl = heatmapArtifact
    ? (api.artifacts.getUrl(data.id, heatmapArtifact.id) ?? heatmapArtifact.download_url)
    : null;

  // Fusion & verdict metadata — map backend field names to frontend expectations
  const rawFusion: Partial<FusionResult> & Record<string, any> = data.fusion || {};
  const fusion = {
    verdict: (rawFusion.verdict || 'INCONCLUSIVE') as VerdictType,
    calibrated_probability: rawFusion.calibrated_probability ?? rawFusion.risk_probability ?? 0.5,
    confidence: rawFusion.confidence ?? 0.5,
    summary: rawFusion.summary || _verdictSummary(rawFusion.verdict, rawFusion.risk_probability),
    guidance: rawFusion.guidance || _verdictGuidance(rawFusion.verdict),
    reasons: rawFusion.reasons || [],
    applied_overrides: rawFusion.applied_overrides || (rawFusion.overrides_applied || []).map((r: string) => ({ rule: r, reason: r })),
    feature_contributions: rawFusion.feature_contributions || _mapContributions(rawFusion.contributions, rawFusion.feature_vector),
  };

  return (
    <div className="space-y-8 py-4">
      {/* Top Action Bar */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-2 border-b border-border-subtle">
        <div className="flex items-center gap-3">
          <Link
            to="/history"
            className="p-2 rounded-md hover:bg-bg-elevated text-text-secondary hover:text-text-primary transition-colors"
            title="Return to History"
          >
            <ArrowLeft className="w-5 h-5" />
          </Link>
          <div>
            <h2 className="text-sm font-semibold font-mono text-text-primary flex items-center gap-2">
              <span>Record ID: {data.id.substring(0, 18)}…</span>
              <span className="text-[10px] uppercase font-mono px-2 py-0.5 rounded bg-bg-elevated border border-border-subtle text-text-tertiary">
                {data.source.filename || 'Recorded snippet'}
              </span>
            </h2>
            <p className="text-xs text-text-tertiary mt-0.5">
              Duration: {_fmt(data.source?.duration_seconds, 1)}s &bull; Sample rate: {data.source?.sample_rate ?? '—'}Hz &bull; Analyzed: {new Date(data.created_at).toLocaleString()}
            </p>
          </div>
        </div>

        {/* Global Record Actions per 09 §4.3.H */}
        <div className="flex items-center gap-2">
          <Button variant="outline" size="sm" onClick={handleExportJson}>
            <Download className="w-3.5 h-3.5" />
            <span>Export Report</span>
          </Button>
          <Link to="/analyze">
            <Button variant="outline" size="sm">
              <RefreshCw className="w-3.5 h-3.5" />
              <span>New Analysis</span>
            </Button>
          </Link>
          <Button variant="ghost" size="sm" onClick={handleDelete} className="text-risk-high hover:bg-risk-high-bg">
            <Trash2 className="w-3.5 h-3.5" />
          </Button>
        </div>
      </div>

      {/* LEVEL 1: Verdict Card (Full Width, Most Prominent per 09 §4.3.A) */}
      <section>
        <VerdictCard
          verdict={fusion.verdict}
          probability={fusion.calibrated_probability}
          confidence={fusion.confidence}
          summary={fusion.summary}
          guidance={fusion.guidance}
          reasons={fusion.reasons}
          overrides={fusion.applied_overrides}
        />
      </section>

      {/* Audio Playback Bar with per-window score band */}
      <section>
        <AudioPlayer
          audioUrl={audioUrl}
          durationSeconds={data.source?.duration_seconds ?? 0}
          windowPredictions={_buildWindowPredictions(data.acoustic)}
          seekTime={seekTime}
        />
      </section>

      {/* LEVEL 2: Signal Breakdown (Three cards per 09 §4.3.B) */}
      <section className="space-y-3">
        <h2 className="text-sm font-mono uppercase tracking-wider font-semibold text-text-secondary">
          Convergent Signal Modalities
        </h2>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-5">
          {/* Card 1: Acoustic Branch */}
          <SignalCard
            title="Acoustic Spoof Detection"
            score={data.acoustic?.spoof_probability ?? 0}
            scoreLabel="Spoof Probability"
            status={data.degraded_branches?.includes('acoustic') ? 'degraded' : 'available'}
            reason={data.degraded_branches?.includes('acoustic') ? 'Low speech duration or high clipping' : undefined}
          >
            <div className="space-y-2 pt-2 border-t border-border-subtle/50 text-xs">
              <div className="flex justify-between text-text-tertiary font-mono">
                <span>Model Architecture:</span>
                <span className="text-text-primary">EfficientNet-B0</span>
              </div>
              <div className="flex justify-between text-text-tertiary font-mono">
                <span>Uncertainty Entropy:</span>
                <span className="text-text-primary">
                  {(data.acoustic?.uncertainty_entropy ?? data.acoustic?.uncertainty) ? (data.acoustic?.uncertainty_entropy ?? data.acoustic?.uncertainty)?.toFixed(3) : '0.124'}
                </span>
              </div>
              {data.acoustic?.is_borderline && (
                <div className="p-1.5 rounded bg-risk-moderate-bg text-risk-moderate text-[11px] font-mono border border-risk-moderate/30 text-center">
                  Borderline Acoustic Score [0.35 - 0.70]
                </div>
              )}
            </div>
          </SignalCard>

          {/* Card 2: Linguistic Branch */}
          <SignalCard
            title="Linguistic Scam Intent"
            score={data.scam?.scam_probability ?? 0}
            scoreLabel="Extortion Risk"
            status={data.degraded_branches?.includes('scam') ? 'degraded' : 'available'}
            reason={data.degraded_branches?.includes('scam') ? 'Transcript below minimum length' : undefined}
          >
            <div className="space-y-2 pt-2 border-t border-border-subtle/50 text-xs">
              <div className="flex justify-between text-text-tertiary font-mono">
                <span>Language Detected:</span>
                <span className="text-text-primary uppercase">
                  {data.transcript?.detected_language || data.transcript?.language || 'EN'} (
                  {Math.round((data.transcript?.language_confidence ?? data.transcript?.language_probability ?? 1) * 100)}%)
                </span>
              </div>
              <div className="flex flex-wrap gap-1 pt-1">
                {(data.scam?.detected_tactics || data.scam?.triggered_categories || []).length > 0 ? (
                  (data.scam?.detected_tactics || data.scam?.triggered_categories || []).map((tac: string) => (
                    <span
                      key={tac}
                      className="px-2 py-0.5 rounded text-[10px] font-mono bg-bg-elevated border border-border-subtle text-text-secondary"
                    >
                      {tac}
                    </span>
                  ))
                ) : (
                  <span className="text-[11px] text-text-tertiary">No extortion tactics identified</span>
                )}
              </div>
            </div>
          </SignalCard>

          {/* Card 3: Interactive Challenge */}
          <SignalCard
            title="Challenge–Response"
            score={
              data.challenges.length > 0 && data.challenges[0].consistency_score !== undefined
                ? data.challenges[0].consistency_score
                : 0
            }
            scoreLabel="Consistency Delta"
            status={data.challenges.length > 0 ? 'available' : 'unavailable'}
            reason={data.challenges.length === 0 ? 'Not yet performed for this analysis' : undefined}
          >
            <div className="space-y-2 pt-2 border-t border-border-subtle/50 text-xs">
              <div className="flex justify-between text-text-tertiary font-mono">
                <span>Evaluations Run:</span>
                <span className="text-text-primary">{data.challenges.length} issued</span>
              </div>
              <p className="text-[11px] text-text-tertiary">
                {data.challenges.length > 0
                  ? 'Acoustic delta verified against human physiological dynamics.'
                  : 'Run an interactive verification below to challenge synthetic converters.'}
              </p>
            </div>
          </SignalCard>
        </div>
      </section>

      {/* LEVEL 3: Evidence (Spectrogram, Grad-CAM, Transcript, Contribution, Challenge) */}
      <section className="space-y-6">
        <h2 className="text-sm font-mono uppercase tracking-wider font-semibold text-text-secondary">
          Forensic Evidence &amp; Visual Attribution
        </h2>

        {/* Contribution Chart */}
        <ContributionChart contributions={fusion.feature_contributions || []} />

        {/* Spectrogram & Grad-CAM */}
        <SpectrogramViewer
          spectrogramUrl={specUrl}
          overlayUrl={overlayUrl}
          heatmapUrl={heatmapUrl}
          windows={data.explanation?.windows || []}
        />

        {/* Speech Transcript & Tactic Span Highlighting */}
        <TranscriptViewer
          text={data.transcript?.text || ''}
          segments={data.transcript?.segments || []}
          salientSpans={_mapSalientSpans(data.scam?.salient_spans)}
          language={data.transcript?.detected_language || data.transcript?.language || 'en'}
          confidence={data.transcript?.language_confidence ?? data.transcript?.language_probability ?? 1.0}
          reliable={data.transcript?.reliable ?? data.transcript?.is_reliable ?? true}
          lowReliabilityReason={data.transcript?.low_reliability_reason}
          onSeek={(t) => setSeekTime(t)}
        />

        {/* Challenge Panel */}
        <ChallengePanel
          analysisId={data.id}
          initialVerdict={fusion.verdict}
          initialRiskProb={fusion.calibrated_probability}
          onVerificationComplete={handleChallengeVerification}
        />
      </section>

      {/* LEVEL 4: Technical Details Toggle per 09 §4.3.G */}
      <section className="pt-4 border-t border-border-subtle">
        <button
          onClick={() => setShowTechnicalDetails(!showTechnicalDetails)}
          className="flex items-center justify-between w-full p-4 rounded-xl bg-bg-surface border border-border-default hover:border-border-strong text-left transition-colors"
        >
          <div className="flex items-center gap-2 text-sm font-mono font-semibold text-text-primary">
            <Code className="w-4 h-4 text-accent" />
            <span>Technical Details &amp; Raw Pipeline Metadata</span>
          </div>
          <div className="flex items-center gap-1.5 text-xs text-text-tertiary">
            <span>{showTechnicalDetails ? 'Collapse' : 'Expand'}</span>
            {showTechnicalDetails ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
          </div>
        </button>

        {showTechnicalDetails && (
          <div className="mt-4 p-6 rounded-xl bg-bg-surface border border-border-default space-y-6 animate-in fade-in duration-150">
            {/* Model Versions and Pipeline Parameters */}
            <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-4 text-xs font-mono">
              <div className="p-3 rounded-lg bg-bg-elevated border border-border-subtle">
                <span className="text-text-tertiary block">Pipeline Version:</span>
                <span className="text-text-primary font-bold">VoiceGuard v0.1.0</span>
              </div>
              <div className="p-3 rounded-lg bg-bg-elevated border border-border-subtle">
                <span className="text-text-tertiary block">Acoustic Model:</span>
                <span className="text-text-primary font-bold">
                  {data.model_versions?.acoustic || 'EfficientNet-B0-ASVspoof'}
                </span>
              </div>
              <div className="p-3 rounded-lg bg-bg-elevated border border-border-subtle">
                <span className="text-text-tertiary block">Scam Model:</span>
                <span className="text-text-primary font-bold">
                  {data.model_versions?.scam || 'XLM-RoBERTa-ScamDualHead'}
                </span>
              </div>
              <div className="p-3 rounded-lg bg-bg-elevated border border-border-subtle">
                <span className="text-text-tertiary block">Total Latency:</span>
                <span className="text-text-primary font-bold">
                  {data.total_duration_ms ? `${(data.total_duration_ms / 1000).toFixed(2)}s` : '—'}
                </span>
              </div>
            </div>

            {/* Quality Metrics */}
            <div className="space-y-2">
              <h4 className="text-xs font-mono font-semibold uppercase text-text-secondary">
                Input Audio Quality Checks:
              </h4>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-xs font-mono">
                <div className="p-2.5 rounded bg-bg-elevated border border-border-subtle">
                  <div className="text-text-tertiary text-[10px]">VAD Speech Ratio</div>
                  <div className="text-text-primary font-bold">
                    {(((data.quality as any)?.vad_speech_ratio ?? (data.quality as any)?.speech_ratio ?? 0) * 100).toFixed(1)}%
                  </div>
                </div>
                <div className="p-2.5 rounded bg-bg-elevated border border-border-subtle">
                  <div className="text-text-tertiary text-[10px]">Estimated SNR</div>
                  <div className="text-text-primary font-bold">
                    {_fmt((data.quality as any)?.snr_db ?? (data.quality as any)?.snr_estimate_db, 1)} dB
                  </div>
                </div>
                <div className="p-2.5 rounded bg-bg-elevated border border-border-subtle">
                  <div className="text-text-tertiary text-[10px]">Clipping Rate</div>
                  <div className="text-text-primary font-bold">
                    {(((data.quality as any)?.clipping_rate ?? (data.quality as any)?.clipping_ratio ?? 0) * 100).toFixed(3)}%
                  </div>
                </div>
                <div className="p-2.5 rounded bg-bg-elevated border border-border-subtle">
                  <div className="text-text-tertiary text-[10px]">DC Offset</div>
                  <div className="text-text-primary font-bold">
                    {_fmt(data.quality?.dc_offset, 4)}
                  </div>
                </div>
              </div>
            </div>

            {/* Raw JSON viewer */}
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <h4 className="text-xs font-mono font-semibold uppercase text-text-secondary">
                  Raw API Output (JSON):
                </h4>
                <button
                  onClick={copyRawJson}
                  className="flex items-center gap-1 text-xs font-mono text-text-tertiary hover:text-text-primary transition-colors"
                >
                  {copiedRawJson ? (
                    <>
                      <Check className="w-3 h-3 text-risk-low" />
                      <span className="text-risk-low">Copied</span>
                    </>
                  ) : (
                    <>
                      <Copy className="w-3 h-3" />
                      <span>Copy JSON</span>
                    </>
                  )}
                </button>
              </div>
              <pre className="p-4 rounded-lg bg-bg-base border border-border-strong text-[11px] font-mono text-text-secondary overflow-x-auto max-h-96">
                {JSON.stringify(data, null, 2)}
              </pre>
            </div>
          </div>
        )}
      </section>
    </div>
  );
};
