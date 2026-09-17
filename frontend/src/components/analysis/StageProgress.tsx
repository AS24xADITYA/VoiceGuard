import React from 'react';
import { CheckCircle2, Loader2, Circle } from 'lucide-react';
import { StageName } from '../../types/api';

export interface StageItem {
  key: StageName;
  label: string;
}

export const STAGES_ORDER: StageItem[] = [
  { key: 'AUDIO_PREPARATION', label: 'Audio preparation & validation' },
  { key: 'ACOUSTIC_INFERENCE', label: 'Acoustic deepfake CNN analysis' },
  { key: 'TRANSCRIPTION', label: 'Speech transcription (Whisper)' },
  { key: 'SCAM_DETECTION', label: 'Multilingual scam-intent classification' },
  { key: 'FUSION', label: 'Calibrated Bayesian fusion' },
  { key: 'EXPLANATION', label: 'Generating Grad-CAM attribution' },
];

export interface StageProgressProps {
  currentStage?: StageName | null;
  progressPct?: number;
  stageTimings?: Record<string, number>; // in seconds or ms
}

export const StageProgress: React.FC<StageProgressProps> = ({
  currentStage = 'AUDIO_PREPARATION',
  progressPct = 0,
  stageTimings = {},
}) => {
  const currentIndex = STAGES_ORDER.findIndex((s) => s.key === currentStage);
  const activeIdx = currentIndex === -1 ? 0 : currentIndex;

  return (
    <div
      className="p-6 rounded-xl bg-bg-surface border border-border-default space-y-5"
      aria-live="polite"
    >
      <div className="flex items-center justify-between pb-2 border-b border-border-subtle">
        <div>
          <h3 className="text-sm font-semibold uppercase tracking-wider font-mono text-text-primary">
            Analysis Pipeline In Progress
          </h3>
          <p className="text-xs text-text-tertiary mt-0.5">
            Real-time stage execution and signal synchronization
          </p>
        </div>
        <div className="text-right font-mono">
          <span className="text-lg font-bold text-accent">{progressPct}%</span>
        </div>
      </div>

      {/* Progress Bar */}
      <div className="w-full bg-bg-elevated rounded-full h-1.5 overflow-hidden">
        <div
          className="bg-accent h-full rounded-full transition-all duration-300 ease-out"
          style={{ width: `${Math.max(5, progressPct)}%` }}
        />
      </div>

      {/* Stages List */}
      <div className="space-y-3 pt-1">
        {STAGES_ORDER.map((stage, idx) => {
          const isCompleted = idx < activeIdx || currentStage === 'COMPLETE';
          const isActive = idx === activeIdx && currentStage !== 'COMPLETE';

          const timing = stageTimings[stage.key];

          return (
            <div
              key={stage.key}
              className={`flex items-center justify-between py-1.5 px-3 rounded-md transition-colors ${
                isActive
                  ? 'bg-bg-elevated border border-accent/30 text-text-primary'
                  : isCompleted
                  ? 'text-text-secondary'
                  : 'text-text-tertiary opacity-60'
              }`}
            >
              <div className="flex items-center gap-3">
                {isCompleted ? (
                  <CheckCircle2 className="w-4 h-4 text-risk-low shrink-0" />
                ) : isActive ? (
                  <Loader2 className="w-4 h-4 text-accent animate-spin shrink-0" />
                ) : (
                  <Circle className="w-4 h-4 text-border-strong shrink-0" />
                )}
                <span className={`text-sm ${isActive ? 'font-medium text-accent' : ''}`}>
                  {stage.label}
                </span>
              </div>

              <div className="font-mono text-xs">
                {timing !== undefined ? (
                  <span className="text-text-tertiary">
                    {timing < 10 ? timing.toFixed(2) : Math.round(timing)} s
                  </span>
                ) : isActive ? (
                  <span className="text-accent text-[11px] uppercase tracking-wider animate-pulse">
                    Running…
                  </span>
                ) : isCompleted ? (
                  <span className="text-risk-low text-[11px]">done</span>
                ) : (
                  <span className="text-text-tertiary">—</span>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
};
