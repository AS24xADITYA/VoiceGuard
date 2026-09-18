import React, { useState } from 'react';
import {
  FileText,
  Copy,
  Check,
  Globe,
  AlertTriangle,
  Play,
  Sparkles,
} from 'lucide-react';
import { TranscriptSegment, SalientSpan } from '../../types/api';
import { Card } from '../ui/Card';

export interface TranscriptViewerProps {
  text?: string;
  segments?: TranscriptSegment[];
  salientSpans?: SalientSpan[];
  language?: string;
  confidence?: number;
  reliable?: boolean;
  lowReliabilityReason?: string | null;
  onSeek?: (timestampSeconds: number) => void;
}

export const TranscriptViewer: React.FC<TranscriptViewerProps> = ({
  text = '',
  segments = [],
  salientSpans = [],
  language = 'en',
  confidence = 1.0,
  reliable = true,
  lowReliabilityReason,
  onSeek,
}) => {
  const [copied, setCopied] = useState(false);
  const [showSegmentDetails, setShowSegmentDetails] = useState(true);

  const handleCopy = () => {
    if (!text) return;
    navigator.clipboard.writeText(text);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const formatTimestamp = (s: number) => {
    const min = Math.floor(s / 60);
    const sec = Math.floor(s % 60);
    const ms = Math.floor((s % 1) * 10);
    return `${min}:${sec < 10 ? '0' : ''}${sec}.${ms}`;
  };

  return (
    <Card className="p-6 bg-bg-surface border-border-default space-y-4">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 pb-4 border-b border-border-subtle">
        <div className="flex items-center gap-3">
          <div className="p-2 rounded-md bg-bg-elevated text-accent border border-border-subtle">
            <FileText className="w-5 h-5" />
          </div>
          <div>
            <h3 className="text-base font-semibold text-text-primary font-mono flex items-center gap-2">
              <span>Speech Transcript</span>
              <span className="text-xs font-mono font-normal uppercase px-2 py-0.5 rounded bg-bg-elevated border border-border-subtle text-text-secondary flex items-center gap-1">
                <Globe className="w-3 h-3 text-accent" />
                {language.toUpperCase()} ({Math.round(confidence * 100)}% conf)
              </span>
            </h3>
            <p className="text-xs text-text-tertiary mt-0.5">
              Automated transcription with scam-tactic token attribution
            </p>
          </div>
        </div>

        {/* Action Controls */}
        <div className="flex items-center gap-2 self-start sm:self-auto">
          <button
            onClick={() => setShowSegmentDetails(!showSegmentDetails)}
            className="px-2.5 py-1 rounded text-xs font-medium text-text-secondary hover:text-text-primary bg-bg-elevated border border-border-subtle transition-colors"
          >
            {showSegmentDetails ? 'Show Plain Text' : 'Show Timestamps'}
          </button>
          <button
            onClick={handleCopy}
            className="flex items-center gap-1.5 px-2.5 py-1 rounded text-xs font-medium text-text-secondary hover:text-text-primary bg-bg-elevated border border-border-subtle transition-colors"
            title="Copy transcript to clipboard"
          >
            {copied ? (
              <>
                <Check className="w-3.5 h-3.5 text-risk-low" />
                <span className="text-risk-low">Copied</span>
              </>
            ) : (
              <>
                <Copy className="w-3.5 h-3.5" />
                <span>Copy</span>
              </>
            )}
          </button>
        </div>
      </div>

      {/* Low Reliability Warning Banner per 09 §4.3.D */}
      {!reliable && (
        <div className="p-3.5 rounded-lg bg-risk-moderate-bg border border-risk-moderate/30 flex items-start gap-2.5 text-xs text-risk-moderate">
          <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5" />
          <div>
            <span className="font-semibold">Transcript Reliability Notice: </span>
            {lowReliabilityReason ||
              'Audio quality or background noise reduced transcription confidence. Linguistic signal weight has been calibrated accordingly.'}
          </div>
        </div>
      )}

      {/* Salient Tactics Span Chips */}
      {salientSpans.length > 0 && (
        <div className="p-3 rounded-lg bg-bg-elevated border border-border-subtle space-y-2">
          <div className="flex items-center gap-1.5 text-xs font-mono font-semibold text-accent">
            <Sparkles className="w-3.5 h-3.5" />
            <span>Highlighted Salient Spans (Scam Language Attribution):</span>
          </div>
          <div className="flex flex-wrap gap-2">
            {salientSpans.map((span, idx) => (
              <span
                key={idx}
                className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded text-xs font-mono bg-bg-surface border border-risk-high/30 text-risk-high"
                title={`Attribution weight: ${Number(span.attribution_weight ?? 0).toFixed(3)} | Tactics: ${(span.tactics || []).join(', ')}`}
              >
                <span>&ldquo;{span.text}&rdquo;</span>
                <span className="text-[10px] uppercase font-sans font-semibold px-1 rounded bg-risk-high-bg text-risk-high">
                  {(span.tactics || [])[0] || 'risk'}
                </span>
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Transcript Text Body */}
      {text ? (
        showSegmentDetails && segments.length > 0 ? (
          /* Segment list with clickable seek timestamps */
          <div className="space-y-2 max-h-80 overflow-y-auto pr-1">
            {segments.map((seg) => (
              <div
                key={seg.id}
                className="group flex items-start gap-3 p-2 rounded-md hover:bg-bg-elevated/70 transition-colors border border-transparent hover:border-border-subtle"
              >
                <button
                  onClick={() => onSeek && onSeek(seg.start)}
                  className="flex items-center gap-1 px-1.5 py-0.5 rounded font-mono text-[11px] bg-bg-surface border border-border-subtle text-text-tertiary group-hover:text-accent group-hover:border-accent/40 shrink-0 mt-0.5"
                  title="Click to seek audio"
                >
                  <Play className="w-2.5 h-2.5 fill-current" />
                  <span>{formatTimestamp(seg.start)}</span>
                </button>
                <p className="text-sm text-text-primary leading-relaxed flex-1 font-normal">
                  {seg.text}
                </p>
                <span className="text-[10px] font-mono text-text-tertiary shrink-0 mt-1">
                  {Math.round(seg.confidence * 100)}%
                </span>
              </div>
            ))}
          </div>
        ) : (
          /* Plain paragraph text */
          <div className="p-4 rounded-lg bg-bg-elevated/40 border border-border-subtle text-sm text-text-primary leading-relaxed">
            {text}
          </div>
        )
      ) : (
        <div className="p-8 text-center text-text-tertiary text-sm">
          No speech transcript available for this recording.
        </div>
      )}
    </Card>
  );
};
