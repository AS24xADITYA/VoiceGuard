import React, { useRef, useState, useEffect } from 'react';
import { Play, Pause, RotateCcw, Volume2 } from 'lucide-react';
import { WindowPrediction } from '../../types/api';
import { Card } from '../ui/Card';

export interface AudioPlayerProps {
  audioUrl?: string | null;
  durationSeconds?: number;
  windowPredictions?: WindowPrediction[];
  seekTime?: number | null;
}

export const AudioPlayer: React.FC<AudioPlayerProps> = ({
  audioUrl,
  durationSeconds = 0,
  windowPredictions = [],
  seekTime = null,
}) => {
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(durationSeconds);

  useEffect(() => {
    if (seekTime !== null && audioRef.current) {
      audioRef.current.currentTime = seekTime;
      setCurrentTime(seekTime);
    }
  }, [seekTime]);

  const togglePlay = () => {
    if (!audioRef.current) return;
    if (isPlaying) {
      audioRef.current.pause();
      setIsPlaying(false);
    } else {
      audioRef.current.play();
      setIsPlaying(true);
    }
  };

  const handleTimeUpdate = () => {
    if (audioRef.current) {
      setCurrentTime(audioRef.current.currentTime);
    }
  };

  const handleLoadedMetadata = () => {
    if (audioRef.current) {
      setDuration(audioRef.current.duration || durationSeconds);
    }
  };

  const handleSeek = (e: React.MouseEvent<HTMLDivElement>) => {
    if (!audioRef.current || duration === 0) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const pos = (e.clientX - rect.left) / rect.width;
    const target = pos * duration;
    audioRef.current.currentTime = target;
    setCurrentTime(target);
  };

  const handleRestart = () => {
    if (!audioRef.current) return;
    audioRef.current.currentTime = 0;
    setCurrentTime(0);
  };

  const formatTime = (sec: number) => {
    const m = Math.floor(sec / 60);
    const s = Math.floor(sec % 60);
    return `${m}:${s < 10 ? '0' : ''}${s}`;
  };

  const progressPct = duration > 0 ? (currentTime / duration) * 100 : 0;

  return (
    <Card className="p-5 bg-bg-surface border-border-default space-y-3">
      {audioUrl && (
        <audio
          ref={audioRef}
          src={audioUrl}
          onTimeUpdate={handleTimeUpdate}
          onLoadedMetadata={handleLoadedMetadata}
          onEnded={() => setIsPlaying(false)}
        />
      )}

      {/* Controls Row */}
      <div className="flex items-center justify-between gap-4">
        <div className="flex items-center gap-2">
          <button
            onClick={togglePlay}
            disabled={!audioUrl}
            className="w-10 h-10 rounded-full bg-accent text-text-inverse flex items-center justify-center hover:bg-accent/90 transition-all shadow-sm focus:outline-none focus-visible:ring-2 focus-visible:ring-accent disabled:opacity-50"
            aria-label={isPlaying ? 'Pause audio' : 'Play audio'}
          >
            {isPlaying ? <Pause className="w-4 h-4 fill-current" /> : <Play className="w-4 h-4 fill-current ml-0.5" />}
          </button>
          <button
            onClick={handleRestart}
            disabled={!audioUrl}
            className="p-2 rounded-md text-text-secondary hover:text-text-primary hover:bg-bg-elevated transition-colors disabled:opacity-50"
            aria-label="Restart audio"
          >
            <RotateCcw className="w-4 h-4" />
          </button>
          <div className="text-xs font-mono text-text-secondary ml-2">
            <span>{formatTime(currentTime)}</span>
            <span className="text-text-tertiary"> / </span>
            <span className="text-text-tertiary">{formatTime(duration)}</span>
          </div>
        </div>

        <div className="flex items-center gap-1.5 text-xs text-text-tertiary font-mono">
          <Volume2 className="w-4 h-4 text-text-secondary" />
          <span>Canonical 16kHz PCM</span>
        </div>
      </div>

      {/* Scrubber & Waveform Track */}
      <div
        onClick={handleSeek}
        className="relative h-12 bg-bg-elevated rounded-lg cursor-pointer overflow-hidden border border-border-subtle group"
      >
        {/* Fake / preview waveform bars */}
        <div className="absolute inset-0 flex items-center justify-around px-2 pointer-events-none opacity-40 group-hover:opacity-60 transition-opacity">
          {Array.from({ length: 48 }).map((_, i) => {
            const height = 20 + Math.sin(i * 0.4) * 16 + (i % 3) * 6;
            return (
              <div
                key={i}
                className="w-1 rounded-full bg-text-secondary"
                style={{ height: `${Math.min(100, height)}%` }}
              />
            );
          })}
        </div>

        {/* Progress Fill */}
        <div
          className="absolute inset-y-0 left-0 bg-accent/20 border-r-2 border-accent transition-all duration-75 pointer-events-none"
          style={{ width: `${progressPct}%` }}
        />

        {/* Per-Window Acoustic Risk Score Bands */}
        {windowPredictions.length > 0 && duration > 0 && (
          <div className="absolute bottom-0 inset-x-0 h-1.5 flex pointer-events-none">
            {windowPredictions.map((win, idx) => {
              const leftPct = (win.start_sec / duration) * 100;
              const widthPct = ((win.end_sec - win.start_sec) / duration) * 100;
              let bg = 'bg-risk-low';
              if (win.raw_prob >= 0.7) bg = 'bg-risk-high';
              else if (win.raw_prob >= 0.35) bg = 'bg-risk-moderate';

              return (
                <div
                  key={idx}
                  className={`absolute h-full ${bg} opacity-80`}
                  style={{ left: `${leftPct}%`, width: `${widthPct}%` }}
                  title={`Window ${win.window_index}: ${Math.round(win.raw_prob * 100)}% risk`}
                />
              );
            })}
          </div>
        )}
      </div>

      {windowPredictions.length > 0 && (
        <div className="flex items-center justify-between text-[11px] text-text-tertiary font-mono pt-1">
          <span>Window Risk Timeline (bottom indicator bar)</span>
          <div className="flex items-center gap-3">
            <span className="flex items-center gap-1">
              <span className="w-2 h-2 rounded-full bg-risk-low" /> Low
            </span>
            <span className="flex items-center gap-1">
              <span className="w-2 h-2 rounded-full bg-risk-moderate" /> Mod
            </span>
            <span className="flex items-center gap-1">
              <span className="w-2 h-2 rounded-full bg-risk-high" /> High
            </span>
          </div>
        </div>
      )}
    </Card>
  );
};
