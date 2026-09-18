import { useState } from 'react';
import { Layers, Sliders, AlertCircle } from 'lucide-react';
import { AxisExtents, PeakRegion, ExplanationWindow } from '../../types/api';
import { COPY } from '../../i18n/en';
import { Card } from '../ui/Card';

export interface SpectrogramViewerProps {
  spectrogramUrl?: string | null;
  overlayUrl?: string | null;
  heatmapUrl?: string | null;
  windows?: ExplanationWindow[];
  defaultOpacity?: number;
}

export const SpectrogramViewer: React.FC<SpectrogramViewerProps> = ({
  spectrogramUrl,
  overlayUrl,
  heatmapUrl,
  windows = [],
  defaultOpacity = 0.45,
}) => {
  const [activeTab, setActiveTab] = useState<'overlay' | 'spectrogram' | 'heatmap'>('overlay');
  const [opacity, setOpacity] = useState<number>(defaultOpacity);
  const [selectedWindowIdx, setSelectedWindowIdx] = useState<number>(0);

  const activeWindow = windows[selectedWindowIdx] || windows[0];

  const rawExtents = (activeWindow as any)?.axis_extents || (windows[0] as any)?.axis_extents || {};
  const axisExtents: AxisExtents = {
    time_start_s: Number(rawExtents.time_start_s ?? rawExtents.t_min ?? 0.0),
    time_end_s: Number(rawExtents.time_end_s ?? rawExtents.t_max ?? 4.0),
    freq_min_hz: Number(rawExtents.freq_min_hz ?? rawExtents.f_min ?? 0),
    freq_max_hz: Number(rawExtents.freq_max_hz ?? rawExtents.f_max ?? 8000),
  };

  const rawPeaks: any[] = activeWindow?.peak_regions || [];
  const peakRegions: PeakRegion[] = rawPeaks.map((r: any) => ({
    time_start_s: Number(r.time_start_s ?? r.t_start_s ?? 0),
    time_end_s: Number(r.time_end_s ?? r.t_end_s ?? 0),
    freq_start_hz: Number(r.freq_start_hz ?? r.f_low_hz ?? 0),
    freq_end_hz: Number(r.freq_end_hz ?? r.f_high_hz ?? 0),
    intensity: Number(r.intensity ?? r.mean_weight ?? 0),
    description: r.description || '',
  }));

  // Alt text generation per 09 §7
  const altText =
    peakRegions.length > 0
      ? `Spectrogram with model attention concentrated at ${peakRegions
          .map((r) => `${r.time_start_s.toFixed(1)}–${r.time_end_s.toFixed(1)}s in the ${(r.freq_start_hz / 1000).toFixed(1)}–${(r.freq_end_hz / 1000).toFixed(1)}kHz range`)
          .join(', ')}.`
      : 'Audio spectrogram and Grad-CAM attention heatmap.';

  return (
    <Card className="p-6 bg-bg-surface border-border-default space-y-5">
      {/* Header with View Controls */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-4 border-b border-border-subtle">
        <div>
          <h3 className="text-base font-semibold text-text-primary flex items-center gap-2 font-mono">
            <Layers className="w-4 h-4 text-accent" />
            <span>Acoustic Salience & Grad-CAM Spectrogram</span>
          </h3>
          <p className="text-xs text-text-tertiary mt-0.5">
            Visual attribution of neural network activations across time and frequency
          </p>
        </div>

        {/* View Mode Toggle Buttons */}
        <div className="flex items-center gap-1.5 p-1 bg-bg-elevated rounded-lg border border-border-default self-start sm:self-auto">
          {(['overlay', 'spectrogram', 'heatmap'] as const).map((mode) => (
            <button
              key={mode}
              onClick={() => setActiveTab(mode)}
              className={`px-3 py-1 rounded text-xs font-medium capitalize transition-colors ${
                activeTab === mode
                  ? 'bg-accent text-text-inverse font-semibold'
                  : 'text-text-secondary hover:text-text-primary hover:bg-bg-overlay'
              }`}
            >
              {mode}
            </button>
          ))}
        </div>
      </div>

      {/* Multiple Windows Selector (if available) */}
      {windows.length > 1 && (
        <div className="flex items-center gap-2 overflow-x-auto pb-1">
          <span className="text-xs font-mono text-text-tertiary shrink-0">Analysis Window:</span>
          {windows.map((w, idx) => (
            <button
              key={idx}
              onClick={() => setSelectedWindowIdx(idx)}
              className={`px-2.5 py-1 rounded text-xs font-mono shrink-0 transition-colors border ${
                selectedWindowIdx === idx
                  ? 'bg-accent-glow border-accent text-accent font-semibold'
                  : 'bg-bg-elevated border-border-subtle text-text-secondary hover:bg-bg-overlay'
              }`}
            >
              Window #{w.window_index ?? idx} ({Number((w as any).start_time_s ?? (w as any).t_start_s ?? 0).toFixed(1)}s – {Number((w as any).end_time_s ?? (w as any).t_end_s ?? 0).toFixed(1)}s)
            </button>
          ))}
        </div>
      )}

      {/* Opacity Control for Overlay */}
      {activeTab === 'overlay' && (
        <div className="flex items-center gap-3 p-2.5 rounded-md bg-bg-elevated border border-border-subtle text-xs">
          <Sliders className="w-4 h-4 text-text-tertiary shrink-0" />
          <span className="text-text-secondary font-mono">Heatmap Opacity:</span>
          <input
            type="range"
            min="0"
            max="1"
            step="0.05"
            value={opacity}
            onChange={(e) => setOpacity(parseFloat(e.target.value))}
            className="flex-1 accent-accent h-1.5 bg-bg-overlay rounded-lg cursor-pointer"
            aria-label="Grad-CAM Heatmap Opacity"
          />
          <span className="font-mono text-text-primary w-10 text-right">
            {Math.round(opacity * 100)}%
          </span>
        </div>
      )}

      {/* Visual Canvas Area with React-Drawn Crisp Axes */}
      <div className="relative rounded-lg overflow-hidden border border-border-strong bg-bg-base">
        {/* Y-Axis (Frequency) Labels */}
        <div className="absolute left-2 top-2 bottom-6 flex flex-col justify-between text-[10px] font-mono text-text-secondary pointer-events-none z-20">
          <span className="bg-bg-base/80 px-1 rounded shadow-sm">
            {(axisExtents.freq_max_hz / 1000).toFixed(1)} kHz
          </span>
          <span className="bg-bg-base/80 px-1 rounded shadow-sm">
            {((axisExtents.freq_max_hz + axisExtents.freq_min_hz) / 2000).toFixed(1)} kHz
          </span>
          <span className="bg-bg-base/80 px-1 rounded shadow-sm">
            {(axisExtents.freq_min_hz / 1000).toFixed(1)} kHz
          </span>
        </div>

        {/* X-Axis (Time) Labels */}
        <div className="absolute left-14 right-2 bottom-1 flex justify-between text-[10px] font-mono text-text-secondary pointer-events-none z-20">
          <span className="bg-bg-base/80 px-1 rounded shadow-sm">
            {axisExtents.time_start_s.toFixed(1)}s
          </span>
          <span className="bg-bg-base/80 px-1 rounded shadow-sm">
            {((axisExtents.time_start_s + axisExtents.time_end_s) / 2).toFixed(1)}s
          </span>
          <span className="bg-bg-base/80 px-1 rounded shadow-sm">
            {axisExtents.time_end_s.toFixed(1)}s
          </span>
        </div>

        {/* Image Containers */}
        <div className="relative w-full aspect-[2/1] min-h-[260px] flex items-center justify-center bg-bg-surface">
          {spectrogramUrl ? (
            <>
              {/* Base Spectrogram */}
              <img
                src={spectrogramUrl}
                alt={altText}
                className="absolute inset-0 w-full h-full object-cover"
              />

              {/* Heatmap / Overlay Layer */}
              {activeTab === 'heatmap' && heatmapUrl && (
                <img
                  src={heatmapUrl}
                  alt="Grad-CAM activation heatmap"
                  className="absolute inset-0 w-full h-full object-cover z-10"
                />
              )}

              {activeTab === 'overlay' && (
                <img
                  src={overlayUrl || heatmapUrl || spectrogramUrl}
                  alt={altText}
                  style={{ opacity: overlayUrl ? 1.0 : opacity }}
                  className="absolute inset-0 w-full h-full object-cover z-10 pointer-events-none transition-opacity duration-150"
                />
              )}
            </>
          ) : (
            <div className="flex flex-col items-center justify-center p-8 text-center text-text-tertiary">
              <Layers className="w-8 h-8 mb-2 opacity-50" />
              <p className="text-sm font-medium">Spectrogram rendering not available for this recording</p>
              <p className="text-xs text-text-tertiary mt-1">Audio might be below minimum duration or unanalyzed</p>
            </div>
          )}
        </div>
      </div>

      {/* Peak Regions Text Summary per 09 §4.3.E */}
      {peakRegions.length > 0 && (
        <div className="p-3.5 rounded-lg bg-bg-elevated border border-border-subtle space-y-2">
          <div className="text-xs font-mono font-semibold text-accent uppercase tracking-wider">
            Attributed Salience Regions:
          </div>
          <div className="space-y-1.5">
            {peakRegions.map((region, idx) => (
              <div key={idx} className="flex items-start gap-2 text-xs text-text-secondary">
                <span className="w-1.5 h-1.5 rounded-full bg-accent mt-1.5 shrink-0" />
                <span>
                  Model attention concentrated at{' '}
                  <strong className="text-text-primary font-mono font-medium">
                    {region.time_start_s.toFixed(2)}–{region.time_end_s.toFixed(2)}s
                  </strong>{' '}
                  in the{' '}
                  <strong className="text-text-primary font-mono font-medium">
                    {(region.freq_start_hz / 1000).toFixed(1)}–{(region.freq_end_hz / 1000).toFixed(1)} kHz
                  </strong>{' '}
                  band (intensity: {Math.round(region.intensity * 100)}%).
                </span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Mandatory Disclaimer per 05 §6.5 & 09 §4.3.E (Verbatim, non-collapsible, non-truncated) */}
      <div className="p-3.5 rounded-lg bg-bg-elevated/60 border border-border-subtle flex items-start gap-2.5 text-xs text-text-tertiary leading-relaxed">
        <AlertCircle className="w-4 h-4 text-accent shrink-0 mt-0.5" />
        <p>{COPY.disclaimers.gradCam}</p>
      </div>
    </Card>
  );
};
