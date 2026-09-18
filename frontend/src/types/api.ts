/**
 * TypeScript API type definitions for VoiceGuard frontend.
 * Conforms to 07-DATA-MODELS-AND-SCHEMA.md, 08-API-SPECIFICATION.md, and 09-FRONTEND-UIUX-SPEC.md.
 */

export type VerdictType = 'LOW' | 'MODERATE' | 'HIGH' | 'INCONCLUSIVE';

export type AnalysisStatus = 'QUEUED' | 'RUNNING' | 'COMPLETE' | 'FAILED' | 'ERROR';

export type StageName =
  | 'AUDIO_PREPARATION'
  | 'ACOUSTIC_INFERENCE'
  | 'TRANSCRIPTION'
  | 'SCAM_DETECTION'
  | 'FUSION'
  | 'EXPLANATION'
  | 'COMPLETE';

export interface AppliedOverride {
  rule: string;
  previous_score?: number;
  new_score?: number;
  reason: string;
}

export interface FeatureContribution {
  feature: string;
  value: number;
  contribution: number;
  label: string;
}

export interface QualityMetrics {
  vad_speech_ratio: number;
  snr_db: number;
  clipping_rate: number;
  dc_offset: number;
  quality_passed: boolean;
  notes: string[];
}

export interface WindowPrediction {
  window_index: number;
  start_sec: number;
  end_sec: number;
  raw_prob: number;
}

export interface AcousticResult {
  spoof_probability: number;
  uncertainty_entropy?: number;
  uncertainty?: number;
  is_borderline: boolean;
  degraded?: boolean;
  window_predictions?: WindowPrediction[];
  window_scores?: number[];
  window_times?: number[];
  spectrogram_path?: string;
  model_version?: string;
}

export interface TranscriptSegment {
  id: number;
  start: number;
  end: number;
  text: string;
  confidence: number;
}

export interface TranscriptResult {
  text: string;
  detected_language?: string;
  language?: string;
  language_confidence?: number;
  language_probability?: number;
  segments?: TranscriptSegment[];
  reliable?: boolean;
  is_reliable?: boolean;
  low_reliability_reason?: string | null;
  degraded?: boolean;
  mean_confidence?: number;
  word_count?: number;
  hallucination_flags?: string[];
  language_supported?: boolean;
}

export interface SalientSpan {
  text: string;
  start_char: number;
  end_char: number;
  attribution_weight: number;
  tactics: string[];
}

export interface ScamResult {
  scam_probability: number;
  detected_tactics?: string[];
  triggered_categories?: string[];
  category_scores?: Record<string, number>;
  tactic_confidences?: Record<string, number>;
  salient_spans: SalientSpan[];
  degraded?: boolean;
}

export interface FusionResult {
  verdict: VerdictType;
  calibrated_probability?: number;
  risk_probability?: number;
  raw_fusion_score?: number;
  confidence: number;
  summary?: string;
  guidance?: string;
  reasons: string[];
  applied_overrides?: AppliedOverride[];
  overrides_applied?: string[];
  feature_contributions?: FeatureContribution[];
  contributions?: Record<string, number>;
  feature_vector?: number[];
}

export interface AxisExtents {
  time_start_s: number;
  time_end_s: number;
  freq_min_hz: number;
  freq_max_hz: number;
}

export interface PeakRegion {
  time_start_s: number;
  time_end_s: number;
  freq_start_hz: number;
  freq_end_hz: number;
  intensity: number;
  description: string;
}

export interface ExplanationWindow {
  window_index: number;
  start_time_s: number;
  end_time_s: number;
  peak_regions: PeakRegion[];
  axis_extents: AxisExtents;
  heatmap_path?: string;
  overlay_path?: string;
}

export interface ExplanationResult {
  windows: ExplanationWindow[];
  disclaimer: string;
}

export interface AnalysisArtifact {
  id: string;
  kind: 'canonical_audio' | 'spectrogram' | 'gradcam_heatmap' | 'gradcam_overlay';
  download_url: string;
  content_type: string;
  size_bytes: number;
  sha256: string;
}

export interface AnalysisSourceMeta {
  filename?: string | null;
  source_type: string;
  duration_seconds: number;
  sample_rate: number;
  file_size_bytes: number;
}

export interface ChallengeSummary {
  id: string;
  challenge_type: string;
  status: string;
  consistency_score?: number;
  passed?: boolean;
}

export interface AnalysisResponse {
  id: string;
  status: AnalysisStatus;
  created_at: string;
  completed_at?: string | null;
  total_duration_ms?: number | null;
  source: AnalysisSourceMeta;
  quality: QualityMetrics;
  acoustic?: AcousticResult | null;
  transcript?: TranscriptResult | null;
  scam?: ScamResult | null;
  fusion?: FusionResult | null;
  explanation?: ExplanationResult | null;
  challenges: ChallengeSummary[];
  artifacts: AnalysisArtifact[];
  model_versions: Record<string, string>;
  degraded_branches: string[];
  deduplicated: boolean;
}

export interface AnalysisPollResponse {
  id: string;
  status: AnalysisStatus;
  stage?: StageName | null;
  progress_pct: number;
  stage_label?: string | null;
  elapsed_ms?: number | null;
  estimated_remaining_ms?: number | null;
}

export interface AnalysisCreateResponse {
  id: string;
  status: AnalysisStatus;
  stage?: StageName | null;
  progress_pct: number;
  created_at: string;
  poll_url: string;
  deduplicated: boolean;
}

export interface AnalysisSummaryItem {
  id: string;
  status: AnalysisStatus;
  created_at: string;
  original_filename?: string | null;
  duration_seconds: number;
  verdict?: VerdictType | null;
  risk_probability?: number | null;
  detected_language?: string | null;
  is_borderline: boolean;
}

export interface AnalysisListResponse {
  items: AnalysisSummaryItem[];
  total: number;
  page: number;
  page_size: number;
}

export interface ChallengeIssueResponse {
  id: string;
  analysis_id: string;
  challenge_type: string;
  prompt_text: string;
  expected_phrase?: string | null;
  expected_duration_s: number;
  instructions: string[];
  issued_at: string;
  expires_at: string;
}

export interface ChallengeVerifyResponse {
  challenge_id: string;
  analysis_id: string;
  status: string;
  consistency_score: number;
  passed: boolean;
  compliance: {
    phrase_matched?: boolean;
    transcription?: string;
    expected_phrase?: string;
  };
  feature_deltas: Record<string, number>;
  notes: string[];
  re_fused: boolean;
  new_verdict?: VerdictType | null;
  new_risk_probability?: number | null;
}

export interface HealthComponent {
  loaded: boolean;
  warm: boolean;
  version: string;
  load_ms: number;
  error?: string | null;
}

export interface HealthResponse {
  status: 'healthy' | 'degraded' | 'unhealthy';
  version: string;
  pipeline_version: string;
  uptime_seconds: number;
  device: string;
  components: Record<string, HealthComponent>;
  queue: {
    running: number;
    max_concurrency: number;
    available_slots: number;
  };
}

export interface SystemMetrics {
  total_analyses: number;
  verdict_distribution: Record<VerdictType, number>;
  average_duration_ms: number;
  model_performance: {
    acoustic_eer_in_domain: number;
    acoustic_eer_out_of_domain: number;
    scam_macro_f1: number;
    fusion_ece: number;
  };
}
