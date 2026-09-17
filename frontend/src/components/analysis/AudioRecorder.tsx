import React, { useState, useRef, useEffect } from 'react';
import { Mic, Square, RotateCcw, AlertTriangle, Play, Pause, Check } from 'lucide-react';
import { Button } from '../ui/Button';

export interface AudioRecorderProps {
  maxDurationSeconds?: number;
  onComplete: (blob: Blob, durationS: number) => void;
}

export const AudioRecorder: React.FC<AudioRecorderProps> = ({
  maxDurationSeconds = 300,
  onComplete,
}) => {
  const [isRecording, setIsRecording] = useState(false);
  const [recordedBlob, setRecordedBlob] = useState<Blob | null>(null);
  const [recordingDuration, setRecordingDuration] = useState(0);
  const [micPermissionError, setMicPermissionError] = useState<string | null>(null);
  const [volumeLevel, setVolumeLevel] = useState<number>(0);

  // Playback state for preview
  const [isPlayingPreview, setIsPlayingPreview] = useState(false);
  const audioPreviewRef = useRef<HTMLAudioElement | null>(null);

  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioChunksRef = useRef<Blob[]>([]);
  const timerIntervalRef = useRef<number | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const animationFrameRef = useRef<number | null>(null);

  useEffect(() => {
    return () => {
      cleanup();
    };
  }, []);

  const cleanup = () => {
    if (timerIntervalRef.current) clearInterval(timerIntervalRef.current);
    if (animationFrameRef.current) cancelAnimationFrame(animationFrameRef.current);
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
    }
    if (audioContextRef.current && audioContextRef.current.state !== 'closed') {
      audioContextRef.current.close();
    }
  };

  const startRecording = async () => {
    setMicPermissionError(null);
    audioChunksRef.current = [];
    setRecordedBlob(null);
    setRecordingDuration(0);

    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          channelCount: 1,
          sampleRate: 16000,
          echoCancellation: true,
          noiseSuppression: false,
        },
      });
      streamRef.current = stream;

      // Setup Web Audio AnalyserNode for live level meter
      const audioCtx = new (window.AudioContext || (window as any).webkitAudioContext)();
      audioContextRef.current = audioCtx;
      const analyser = audioCtx.createAnalyser();
      analyser.fftSize = 256;
      analyserRef.current = analyser;

      const source = audioCtx.createMediaStreamSource(stream);
      source.connect(analyser);

      // Volume monitoring loop
      const dataArray = new Uint8Array(analyser.frequencyBinCount);
      const updateMeter = () => {
        if (!analyserRef.current) return;
        analyserRef.current.getByteFrequencyData(dataArray);
        let sum = 0;
        for (let i = 0; i < dataArray.length; i++) {
          sum += dataArray[i];
        }
        const avg = sum / dataArray.length;
        setVolumeLevel(Math.min(100, Math.round((avg / 128) * 100)));
        animationFrameRef.current = requestAnimationFrame(updateMeter);
      };
      updateMeter();

      // Check supported MIME types
      let mimeType = 'audio/webm;codecs=opus';
      if (!MediaRecorder.isTypeSupported(mimeType)) {
        mimeType = 'audio/mp4';
        if (!MediaRecorder.isTypeSupported(mimeType)) {
          mimeType = ''; // Let browser choose default
        }
      }

      const recorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
      mediaRecorderRef.current = recorder;

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) {
          audioChunksRef.current.push(e.data);
        }
      };

      recorder.onstop = () => {
        const finalBlob = new Blob(audioChunksRef.current, {
          type: mimeType || 'audio/wav',
        });
        setRecordedBlob(finalBlob);
      };

      recorder.start(250); // Slice every 250ms
      setIsRecording(true);

      const startTime = Date.now();
      timerIntervalRef.current = window.setInterval(() => {
        const elapsed = (Date.now() - startTime) / 1000;
        setRecordingDuration(elapsed);

        if (elapsed >= maxDurationSeconds) {
          stopRecording();
        }
      }, 100);
    } catch (err: any) {
      setMicPermissionError(
        'Microphone access was denied or is unavailable. Please grant microphone permissions to record.'
      );
      cleanup();
    }
  };

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop();
      setIsRecording(false);
      if (timerIntervalRef.current) clearInterval(timerIntervalRef.current);
      if (animationFrameRef.current) cancelAnimationFrame(animationFrameRef.current);
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((t) => t.stop());
      }
    }
  };

  const handleReset = () => {
    cleanup();
    setRecordedBlob(null);
    setRecordingDuration(0);
    setIsRecording(false);
    setIsPlayingPreview(false);
  };

  const handleConfirm = () => {
    if (recordedBlob) {
      onComplete(recordedBlob, recordingDuration);
    }
  };

  const togglePreviewPlay = () => {
    if (!audioPreviewRef.current) return;
    if (isPlayingPreview) {
      audioPreviewRef.current.pause();
      setIsPlayingPreview(false);
    } else {
      audioPreviewRef.current.play();
      setIsPlayingPreview(true);
    }
  };

  const formatTimer = (sec: number) => {
    const m = Math.floor(sec / 60);
    const s = Math.floor(sec % 60);
    return `${m}:${s < 10 ? '0' : ''}${s}`;
  };

  return (
    <div className="p-6 rounded-xl bg-bg-surface border border-border-default space-y-6">
      {micPermissionError && (
        <div className="p-3.5 rounded-lg bg-risk-high-bg border border-risk-high/30 text-xs text-risk-high flex items-start gap-2">
          <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5" />
          <span>{micPermissionError}</span>
        </div>
      )}

      {/* Main Recording Interface */}
      {!recordedBlob ? (
        <div className="flex flex-col items-center justify-center py-8 space-y-6 text-center">
          {/* Animated circular recording button */}
          <div className="relative">
            {isRecording && (
              <div
                className="absolute -inset-3 rounded-full bg-risk-high/20 animate-ping pointer-events-none"
                style={{ transform: `scale(${1 + volumeLevel / 150})` }}
              />
            )}
            <button
              onClick={isRecording ? stopRecording : startRecording}
              className={`w-20 h-20 rounded-full flex items-center justify-center transition-all duration-200 shadow-md ${
                isRecording
                  ? 'bg-risk-high text-text-inverse scale-105'
                  : 'bg-accent text-text-inverse hover:scale-105'
              }`}
              aria-label={isRecording ? 'Stop recording' : 'Start recording'}
            >
              {isRecording ? <Square className="w-7 h-7 fill-current" /> : <Mic className="w-8 h-8" />}
            </button>
          </div>

          <div className="space-y-1">
            <div className="text-2xl font-mono font-bold tracking-tight text-text-primary">
              {formatTimer(recordingDuration)}
            </div>
            <p className="text-xs text-text-tertiary">
              {isRecording
                ? 'Recording in progress… Speak clearly into your microphone'
                : 'Click the microphone to start recording'}
            </p>
          </div>

          {/* Live Level Meter Bar */}
          {isRecording && (
            <div className="w-64 space-y-1">
              <div className="w-full bg-bg-elevated h-2 rounded-full overflow-hidden border border-border-subtle">
                <div
                  className="h-full bg-accent transition-all duration-75"
                  style={{ width: `${volumeLevel}%` }}
                />
              </div>
              <div className="flex justify-between text-[10px] font-mono text-text-tertiary">
                <span>Input Level</span>
                <span>{volumeLevel}%</span>
              </div>
            </div>
          )}

          {/* Time limit warning at 280s */}
          {recordingDuration >= 280 && (
            <div className="text-xs text-risk-moderate flex items-center gap-1.5 font-mono">
              <AlertTriangle className="w-3.5 h-3.5" />
              <span>Approaching 5-minute maximum limit</span>
            </div>
          )}
        </div>
      ) : (
        /* Preview and action confirmation state */
        <div className="space-y-5 py-4">
          <div className="p-4 rounded-lg bg-bg-elevated border border-border-subtle flex items-center justify-between">
            <div className="flex items-center gap-3">
              <button
                onClick={togglePreviewPlay}
                className="w-10 h-10 rounded-full bg-accent text-text-inverse flex items-center justify-center hover:bg-accent/90"
              >
                {isPlayingPreview ? <Pause className="w-4 h-4 fill-current" /> : <Play className="w-4 h-4 fill-current ml-0.5" />}
              </button>
              <div>
                <div className="text-sm font-semibold text-text-primary">Recorded Audio</div>
                <div className="text-xs font-mono text-text-tertiary">
                  Duration: {formatTimer(recordingDuration)} | Size: {Math.round(recordedBlob.size / 1024)} KB
                </div>
              </div>
            </div>

            <audio
              ref={audioPreviewRef}
              src={URL.createObjectURL(recordedBlob)}
              onEnded={() => setIsPlayingPreview(false)}
            />

            <button
              onClick={handleReset}
              className="flex items-center gap-1.5 text-xs text-text-secondary hover:text-risk-high transition-colors"
            >
              <RotateCcw className="w-3.5 h-3.5" />
              <span>Record Again</span>
            </button>
          </div>

          <div className="flex items-center justify-end gap-3 pt-2">
            <Button variant="secondary" onClick={handleReset}>
              Discard
            </Button>
            <Button variant="primary" onClick={handleConfirm} className="gap-2">
              <Check className="w-4 h-4" />
              <span>Use This Recording</span>
            </Button>
          </div>
        </div>
      )}
    </div>
  );
};
