import React, { useState } from 'react';
import {
  ArrowRight,
  CheckCircle2,
  XCircle,
  AlertTriangle,
  Loader2,
  Activity,
} from 'lucide-react';
import {
  ChallengeIssueResponse,
  ChallengeVerifyResponse,
  VerdictType,
} from '../../types/api';
import { api } from '../../api/endpoints';
import { Card } from '../ui/Card';
import { Button } from '../ui/Button';
import { VerdictBadge } from '../ui/Badge';
import { AudioRecorder } from './AudioRecorder';

export interface ChallengePanelProps {
  analysisId: string;
  initialVerdict?: VerdictType;
  initialRiskProb?: number;
  onVerificationComplete?: (verifyRes: ChallengeVerifyResponse) => void;
}

export const ChallengePanel: React.FC<ChallengePanelProps> = ({
  analysisId,
  initialVerdict = 'MODERATE',
  initialRiskProb = 0.5,
  onVerificationComplete,
}) => {
  const [step, setStep] = useState<'idle' | 'brief' | 'perform' | 'evaluating' | 'result'>('idle');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [challengeData, setChallengeData] = useState<ChallengeIssueResponse | null>(null);
  const [verifyResult, setVerifyResult] = useState<ChallengeVerifyResponse | null>(null);

  const startChallengeFlow = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await api.challenge.issue(analysisId);
      setChallengeData(res);
      setStep('brief');
    } catch (err: any) {
      setError(err.response?.data?.detail || err.message || 'Failed to issue challenge');
    } finally {
      setLoading(false);
    }
  };

  const handleAudioRecorded = async (blob: Blob) => {
    if (!challengeData) return;
    setStep('evaluating');
    setError(null);
    try {
      const verifyRes = await api.challenge.verify(
        analysisId,
        challengeData.id,
        blob,
        'challenge_response.wav'
      );
      setVerifyResult(verifyRes);
      setStep('result');
      if (onVerificationComplete) {
        onVerificationComplete(verifyRes);
      }
    } catch (err: any) {
      setError(err.response?.data?.detail || err.message || 'Failed to verify response');
      setStep('brief');
    }
  };

  return (
    <Card className="p-6 bg-bg-surface border-border-default space-y-6">
      {/* Title Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-4 border-b border-border-subtle">
        <div className="flex items-center gap-3">
          <div className="p-2.5 rounded-lg bg-bg-elevated border border-border-subtle text-accent">
            <Activity className="w-5 h-5" />
          </div>
          <div>
            <h3 className="text-base font-semibold text-text-primary font-mono flex items-center gap-2">
              <span>Interactive Challenge–Response Verification</span>
            </h3>
            <p className="text-xs text-text-tertiary mt-0.5">
              Live acoustic modulation test to challenge real-time voice conversion artifacts
            </p>
          </div>
        </div>

        {step === 'idle' && (
          <Button onClick={startChallengeFlow} loading={loading} size="sm">
            <span>Issue Challenge</span>
          </Button>
        )}
      </div>

      {error && (
        <div className="p-3.5 rounded-lg bg-risk-high-bg border border-risk-high/30 text-xs text-risk-high flex items-start gap-2">
          <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5" />
          <span>{error}</span>
        </div>
      )}

      {/* Step 1: Idle introduction */}
      {step === 'idle' && (
        <div className="p-6 rounded-lg bg-bg-elevated/40 border border-border-subtle text-center space-y-4">
          <p className="text-sm text-text-secondary max-w-xl mx-auto leading-relaxed">
            When speech manipulation indicators are borderline or ambiguous, an interactive challenge
            prompts the speaker to produce randomized acoustic variations (such as pitch shifts,
            whispering, or phonetic tongue twisters) that real-time voice conversion models struggle to
            render faithfully.
          </p>
          <Button onClick={startChallengeFlow} loading={loading} variant="primary">
            Start Challenge Verification
          </Button>
        </div>
      )}

      {/* Step 2: Brief */}
      {step === 'brief' && challengeData && (
        <div className="space-y-5">
          <div className="p-4 rounded-lg bg-bg-elevated border border-border-subtle space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-xs font-mono uppercase text-accent font-semibold">
                Challenge Directive: {challengeData.challenge_type.replace('_', ' ')}
              </span>
              <span className="text-xs font-mono text-text-tertiary">
                Target: ~{Number(challengeData.expected_duration_s ?? 0).toFixed(1)}s
              </span>
            </div>

            <div className="text-lg font-mono font-bold text-text-primary p-4 rounded-md bg-bg-surface border border-border-default text-center">
              &ldquo;{challengeData.prompt_text}&rdquo;
            </div>

            {challengeData.instructions.length > 0 && (
              <div className="space-y-1.5 pt-2">
                <span className="text-xs font-mono text-text-tertiary">Instructions:</span>
                <ul className="text-xs text-text-secondary space-y-1 list-disc list-inside">
                  {challengeData.instructions.map((inst, i) => (
                    <li key={i}>{inst}</li>
                  ))}
                </ul>
              </div>
            )}
          </div>

          <div className="flex justify-end gap-3">
            <Button variant="ghost" onClick={() => setStep('idle')}>
              Cancel
            </Button>
            <Button variant="primary" onClick={() => setStep('perform')}>
              <span>Ready to Record</span>
              <ArrowRight className="w-4 h-4 ml-1" />
            </Button>
          </div>
        </div>
      )}

      {/* Step 3: Perform Recording */}
      {step === 'perform' && challengeData && (
        <div className="space-y-4">
          <div className="p-3 rounded-md bg-accent-glow border border-accent/20 text-xs text-accent">
            Speak the prompted phrase clearly: <strong>{challengeData.prompt_text}</strong>
          </div>
          <AudioRecorder
            maxDurationSeconds={challengeData.expected_duration_s + 5}
            onComplete={handleAudioRecorded}
          />
        </div>
      )}

      {/* Step 4: Evaluating */}
      {step === 'evaluating' && (
        <div className="py-12 flex flex-col items-center justify-center space-y-4 text-center">
          <Loader2 className="w-8 h-8 text-accent animate-spin" />
          <div className="space-y-1">
            <h4 className="text-sm font-semibold font-mono text-text-primary">
              Analyzing Challenge Compliance & Acoustic Deltas
            </h4>
            <p className="text-xs text-text-tertiary">
              Comparing feature differentials against human physiological bounds…
            </p>
          </div>
        </div>
      )}

      {/* Step 5: Result (BEFORE vs AFTER) */}
      {step === 'result' && verifyResult && (
        <div className="space-y-6 animate-in fade-in duration-200">
          {/* CRITICAL: Before vs After Verdict Comparison per 09 §4.4 */}
          <div className="p-5 rounded-xl bg-bg-elevated border border-border-default space-y-4">
            <div className="text-xs font-mono font-semibold uppercase tracking-wider text-accent">
              Multi-Signal Fusion: Before vs After Challenge
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {/* BEFORE */}
              <div className="p-4 rounded-lg bg-bg-surface border border-border-subtle space-y-2">
                <span className="text-[11px] uppercase font-mono text-text-tertiary block">
                  Original Assessment
                </span>
                <div className="flex items-center justify-between">
                  <VerdictBadge verdict={initialVerdict} />
                  <span className="text-sm font-mono font-bold text-text-primary">
                    {Math.round(initialRiskProb * 100)}% Risk
                  </span>
                </div>
                <p className="text-xs text-text-tertiary pt-1">
                  Evaluated prior to active acoustic challenge.
                </p>
              </div>

              {/* AFTER */}
              <div className="p-4 rounded-lg bg-bg-surface border border-accent/40 space-y-2">
                <span className="text-[11px] uppercase font-mono text-accent block">
                  Re-Fused Assessment (Updated)
                </span>
                <div className="flex items-center justify-between">
                  <VerdictBadge verdict={verifyResult.new_verdict || initialVerdict} />
                  <span className="text-sm font-mono font-bold text-text-primary">
                    {verifyResult.new_risk_probability !== null && verifyResult.new_risk_probability !== undefined
                      ? `${Math.round(verifyResult.new_risk_probability * 100)}% Risk`
                      : `${Math.round(initialRiskProb * 100)}% Risk`}
                  </span>
                </div>
                <p className="text-xs text-accent pt-1">
                  {verifyResult.passed
                    ? 'Acoustic modulation consistent with human physiology.'
                    : 'Modulation showed anomalies or failed compliance.'}
                </p>
              </div>
            </div>
          </div>

          {/* Verification Metrics Details */}
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-3 text-xs">
            <div className="p-3 rounded-md bg-bg-elevated border border-border-subtle">
              <span className="text-text-tertiary font-mono block">Consistency Score:</span>
              <span className="text-base font-mono font-bold text-text-primary">
                {(Number(verifyResult.consistency_score ?? 0) * 100).toFixed(1)}%
              </span>
            </div>
            <div className="p-3 rounded-md bg-bg-elevated border border-border-subtle">
              <span className="text-text-tertiary font-mono block">Phrase Match:</span>
              <span className="text-base font-mono font-bold text-text-primary">
                {verifyResult.compliance?.phrase_matched ? (
                  <span className="text-risk-low flex items-center gap-1">
                    <CheckCircle2 className="w-4 h-4" /> Passed
                  </span>
                ) : (
                  <span className="text-risk-high flex items-center gap-1">
                    <XCircle className="w-4 h-4" /> Not Compliant
                  </span>
                )}
              </span>
            </div>
            <div className="p-3 rounded-md bg-bg-elevated border border-border-subtle">
              <span className="text-text-tertiary font-mono block">Verification Status:</span>
              <span className="text-base font-mono font-bold text-text-primary">
                {verifyResult.passed ? (
                  <span className="text-risk-low">Valid Delta</span>
                ) : (
                  <span className="text-risk-moderate">Inconsistent</span>
                )}
              </span>
            </div>
          </div>

          {/* Feature Deltas */}
          {Object.keys(verifyResult.feature_deltas || {}).length > 0 && (
            <div className="p-3.5 rounded-lg bg-bg-elevated border border-border-subtle space-y-2">
              <span className="text-xs font-mono font-semibold text-text-secondary uppercase">
                Acoustic Feature Deltas:
              </span>
              <div className="grid grid-cols-2 sm:grid-cols-3 gap-2 text-xs font-mono">
                {Object.entries(verifyResult.feature_deltas).map(([feat, delta]) => (
                  <div key={feat} className="p-2 rounded bg-bg-surface border border-border-subtle">
                    <div className="text-text-tertiary text-[10px]">{feat}</div>
                    <div className="text-text-primary font-bold">
                      {Number(delta ?? 0) > 0 ? '+' : ''}
                      {Number(delta ?? 0).toFixed(3)}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          )}

          <div className="flex justify-end pt-2">
            <Button variant="secondary" onClick={() => setStep('idle')}>
              Done
            </Button>
          </div>
        </div>
      )}
    </Card>
  );
};
