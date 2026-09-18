import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  UploadCloud,
  Mic,
  FileAudio,
  AlertTriangle,
  Clock,
  Cpu,
  ArrowRight,
  CheckCircle2,
  X,
} from 'lucide-react';
import { Tabs } from '../components/ui/Tabs';
import { Button } from '../components/ui/Button';
import { Card } from '../components/ui/Card';
import { StageProgress } from '../components/analysis/StageProgress';
import { AudioRecorder } from '../components/analysis/AudioRecorder';
import { api } from '../api/endpoints';
import { StageName, AnalysisPollResponse } from '../types/api';

const MAX_FILE_SIZE_BYTES = 50 * 1024 * 1024; // 50MB
const MAX_DURATION_SECONDS = 300; // 5 minutes
const ALLOWED_EXTENSIONS = ['.wav', '.mp3', '.m4a', '.flac', '.ogg', '.webm'];

export const AnalyzePage: React.FC = () => {
  const navigate = useNavigate();
  const [activeTab, setActiveTab] = useState<'upload' | 'record'>('upload');

  // Device & processing expectation info
  const [deviceInfo, setDeviceInfo] = useState<string>('cuda');
  const [expectedTimeText, setExpectedTimeText] = useState<string>('about 15 seconds');

  // File state
  const [file, setFile] = useState<File | null>(null);
  const [fileDuration, setFileDuration] = useState<number | null>(null);
  const [label, setLabel] = useState<string>('');
  const [isDragOver, setIsDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement | null>(null);

  // Validation / Error state
  const [validationError, setValidationError] = useState<string | null>(null);

  // Analysis pipeline execution state
  const [analysisId, setAnalysisId] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [pollState, setPollState] = useState<AnalysisPollResponse | null>(null);
  const [stageTimings, setStageTimings] = useState<Record<string, number>>({});

  // Fetch device info on mount to set processing time expectation per 09 §4.2
  useEffect(() => {
    api.system.health()
      .then((health) => {
        setDeviceInfo(health.device || 'cpu');
        if (health.device && health.device.toLowerCase().includes('cuda')) {
          setExpectedTimeText('about 15 seconds (GPU accelerated)');
        } else {
          setExpectedTimeText('up to a minute (CPU mode)');
        }
      })
      .catch(() => {
        setExpectedTimeText('about 30 seconds');
      });
  }, []);

  // Polling loop
  useEffect(() => {
    if (!analysisId) return;

    let timer: NodeJS.Timeout;
    let startTime = Date.now();

    const poll = async () => {
      try {
        const res = await api.analyses.poll(analysisId);
        setPollState(res);

        // Update elapsed timing for current stage
        const elapsedSec = (Date.now() - startTime) / 1000;
        if (res.stage) {
          setStageTimings((prev) => ({
            ...prev,
            [res.stage as string]: elapsedSec,
          }));
        }

        if (res.status === 'COMPLETE') {
          navigate(`/analysis/${analysisId}`);
        } else if (res.status === 'FAILED' || res.status === 'ERROR') {
          setValidationError('Analysis pipeline failed to complete. Please try another audio file.');
          setIsSubmitting(false);
          setAnalysisId(null);
        } else {
          timer = setTimeout(poll, 1000);
        }
      } catch (err: any) {
        setValidationError(err.response?.data?.detail || 'Failed to poll analysis progress.');
        setIsSubmitting(false);
      }
    };

    poll();

    return () => {
      if (timer) clearTimeout(timer);
    };
  }, [analysisId, navigate]);

  // Client-side file validation and audio duration probe
  const handleFileSelection = (selectedFile: File) => {
    setValidationError(null);
    setFile(null);
    setFileDuration(null);

    // Extension check
    const ext = '.' + (selectedFile.name.split('.').pop()?.toLowerCase() || '');
    if (!ALLOWED_EXTENSIONS.includes(ext)) {
      setValidationError(
        `Unsupported audio format (${ext}). Supported formats: ${ALLOWED_EXTENSIONS.join(', ')}.`
      );
      return;
    }

    // Size check
    if (selectedFile.size > MAX_FILE_SIZE_BYTES) {
      setValidationError('File size exceeds the 50 MB limit.');
      return;
    }

    // Probe duration client-side immediately per 09 §4.2
    const audioEl = document.createElement('audio');
    const objectUrl = URL.createObjectURL(selectedFile);
    audioEl.src = objectUrl;

    audioEl.onloadedmetadata = () => {
      URL.revokeObjectURL(objectUrl);
      if (audioEl.duration > MAX_DURATION_SECONDS) {
        setValidationError(
          `Audio duration (${Math.round(audioEl.duration)}s) exceeds 5-minute maximum limit.`
        );
      } else {
        setFileDuration(audioEl.duration);
        setFile(selectedFile);
      }
    };

    audioEl.onerror = () => {
      URL.revokeObjectURL(objectUrl);
      setValidationError('Could not decode audio header. The file may be corrupt.');
    };
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragOver(false);
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFileSelection(e.dataTransfer.files[0]);
    }
  };

  const handleSubmitAnalysis = async (fileToUpload?: File | Blob) => {
    const target = fileToUpload || file;
    if (!target) {
      setValidationError('Please select or record an audio file first.');
      return;
    }

    setIsSubmitting(true);
    setValidationError(null);

    try {
      const formData = new FormData();
      formData.append('file', target, file?.name || 'recorded_audio.wav');
      if (label) formData.append('label', label);

      const res = await api.analyses.create(formData);
      setAnalysisId(res.id);
    } catch (err: any) {
      setIsSubmitting(false);
      setValidationError(
        err.response?.data?.detail || err.message || 'Failed to submit audio for analysis.'
      );
    }
  };

  const handleRecordedAudio = (blob: Blob, durationS: number) => {
    setFileDuration(durationS);
    handleSubmitAnalysis(blob);
  };

  return (
    <div className="max-w-3xl mx-auto space-y-8 py-6">
      {/* Title & Introduction */}
      <div className="space-y-2 text-center">
        <h1 className="text-3xl font-bold font-mono tracking-tight text-text-primary">
          Submit Audio for Analysis
        </h1>
        <p className="text-sm text-text-secondary max-w-xl mx-auto">
          Upload a recorded call, voicemail, or audio snippet. The pipeline inspects acoustic synthesis
          artifacts and scans the transcript for extortion tactics.
        </p>
      </div>

      {/* When processing, show real-time StageProgress instead of upload forms */}
      {isSubmitting || analysisId ? (
        <div className="space-y-6 animate-in fade-in duration-200">
          <StageProgress
            currentStage={pollState?.stage || 'AUDIO_PREPARATION'}
            progressPct={pollState?.progress_pct || 10}
            stageTimings={stageTimings}
          />
          <div className="p-4 rounded-lg bg-bg-surface border border-border-subtle flex items-center justify-between text-xs text-text-tertiary">
            <div className="flex items-center gap-2">
              <Cpu className="w-4 h-4 text-accent" />
              <span>Processing on {deviceInfo.toUpperCase()}</span>
            </div>
            <div className="flex items-center gap-2">
              <Clock className="w-4 h-4 text-accent" />
              <span>Estimated wait: {expectedTimeText}</span>
            </div>
          </div>
        </div>
      ) : (
        /* Upload & Record Tabs */
        <div className="space-y-6">
          <Tabs
            tabs={[
              { id: 'upload', label: 'Upload File', icon: UploadCloud },
              { id: 'record', label: 'Record Microphone', icon: Mic },
            ]}
            activeTab={activeTab}
            onChange={(id) => {
              setActiveTab(id as any);
              setValidationError(null);
            }}
          />

          {validationError && (
            <div className="p-4 rounded-lg bg-risk-high-bg border border-risk-high/30 text-xs text-risk-high flex items-start gap-2.5">
              <AlertTriangle className="w-4 h-4 shrink-0 mt-0.5" />
              <div className="flex-1">{validationError}</div>
              <button onClick={() => setValidationError(null)}>
                <X className="w-4 h-4 text-risk-high hover:opacity-75" />
              </button>
            </div>
          )}

          {activeTab === 'upload' ? (
            <div className="space-y-6">
              {/* Dropzone per 09 §4.2: dashed border, accent on drag-over, 280px min height */}
              <div
                onDragOver={(e) => {
                  e.preventDefault();
                  setIsDragOver(true);
                }}
                onDragLeave={() => setIsDragOver(false)}
                onDrop={handleDrop}
                onClick={() => fileInputRef.current?.click()}
                className={`relative min-h-[280px] rounded-xl border-2 border-dashed flex flex-col items-center justify-center p-8 text-center cursor-pointer transition-all duration-200 ${
                  isDragOver
                    ? 'border-accent bg-accent-glow'
                    : file
                    ? 'border-border-strong bg-bg-surface'
                    : 'border-border-default hover:border-border-strong bg-bg-surface/50'
                }`}
              >
                <input
                  ref={fileInputRef}
                  type="file"
                  accept=".wav,.mp3,.m4a,.flac,.ogg,.webm"
                  className="hidden"
                  onChange={(e) => {
                    if (e.target.files && e.target.files.length > 0) {
                      handleFileSelection(e.target.files[0]);
                    }
                  }}
                />

                {!file ? (
                  <div className="space-y-4">
                    <div className="w-14 h-14 rounded-2xl bg-bg-elevated border border-border-default flex items-center justify-center text-accent mx-auto">
                      <UploadCloud className="w-7 h-7" />
                    </div>
                    <div>
                      <p className="text-base font-medium text-text-primary">
                        Drag and drop audio file here, or browse
                      </p>
                      <p className="text-xs text-text-tertiary mt-1">
                        Accepted: WAV, MP3, M4A, FLAC, OGG, WebM
                      </p>
                      <p className="text-xs text-text-tertiary">
                        Limits: Maximum 50 MB &bull; Up to 5 minutes duration
                      </p>
                    </div>
                  </div>
                ) : (
                  <div className="space-y-3">
                    <div className="w-12 h-12 rounded-xl bg-bg-elevated border border-border-default flex items-center justify-center text-accent mx-auto">
                      <FileAudio className="w-6 h-6" />
                    </div>
                    <div>
                      <div className="text-sm font-semibold text-text-primary font-mono">
                        {file.name}
                      </div>
                      <div className="text-xs font-mono text-text-tertiary mt-1 flex items-center justify-center gap-2">
                        <span>{(file.size / (1024 * 1024)).toFixed(2)} MB</span>
                        <span>&bull;</span>
                        <span>
                          {fileDuration ? `${fileDuration.toFixed(1)}s duration` : 'Probed'}
                        </span>
                      </div>
                    </div>
                    <p className="text-xs text-accent">Click or drag another file to replace</p>
                  </div>
                )}
              </div>

              {/* Optional Label input */}
              <div className="space-y-1.5">
                <label className="text-xs font-mono text-text-secondary block" htmlFor="label">
                  Analysis Reference / Label (optional)
                </label>
                <input
                  id="label"
                  type="text"
                  value={label}
                  onChange={(e) => setLabel(e.target.value)}
                  placeholder="e.g. Voicemail from unknown bank representative"
                  className="w-full px-3.5 py-2 rounded-md bg-bg-surface border border-border-default text-text-primary text-sm focus:outline-none focus:border-accent font-sans"
                />
              </div>

              {/* Action Button and Estimated Time Note */}
              <div className="p-4 rounded-lg bg-bg-elevated border border-border-subtle flex flex-col sm:flex-row sm:items-center justify-between gap-4">
                <div className="flex items-center gap-2 text-xs text-text-secondary">
                  <Clock className="w-4 h-4 text-accent shrink-0" />
                  <span>Expected processing time: {expectedTimeText}</span>
                </div>

                <Button
                  onClick={() => handleSubmitAnalysis()}
                  disabled={!file}
                  className="w-full sm:w-auto"
                >
                  <span>Analyse Audio</span>
                  <ArrowRight className="w-4 h-4 ml-1" />
                </Button>
              </div>
            </div>
          ) : (
            /* Record tab */
            <div className="space-y-4">
              <AudioRecorder onComplete={handleRecordedAudio} maxDurationSeconds={300} />
            </div>
          )}
        </div>
      )}
    </div>
  );
};
