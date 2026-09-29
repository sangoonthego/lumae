/** Canonical frontend domain model for Lumae API schema 1.0. */
export type ModelStage = "baseline" | "stage_a" | "stage_b" | "stage_c";
export type Provenance = "MODEL_OUTPUT" | "DERIVED" | "HEURISTIC";
export type AnalysisErrorCode =
  | "VIDEO_TOO_LONG"
  | "UNSUPPORTED_VIDEO"
  | "INVALID_QUERY"
  | "MODEL_UNAVAILABLE"
  | "ANALYSIS_TIMEOUT"
  | "MALFORMED_RESPONSE"
  | "UNKNOWN_ERROR";
export type QueryIntent = "hook_candidate" | "demo_moment" | "benefit_moment" | "custom";
export type MockScenario = "standard" | "confident_hook" | "no_predictions" | "video_too_long" | "malformed_response";

export interface QueryDefinition {
  id: string;
  query: string;
  intent: QueryIntent;
}

export interface TemporalWindow {
  start: number;
  end: number;
  confidence: number;
}

export interface VideoMetadata {
  filename: string;
  durationSeconds: number;
  width: number;
  height: number;
  codec: string;
}

export interface QueryPrediction {
  queryId: string;
  intent: QueryIntent;
  query: string;
  windows: TemporalWindow[];
  /** Immutable model output. May contain negative values. */
  rawSaliencyScores: number[];
  saliencyClipDurationSeconds: number;
}

export interface ModelMetadata {
  id: string;
  displayName?: string;
  baseModel?: string;
  stage: ModelStage;
  temporalResolutionSeconds?: number;
  maxVideoDurationSeconds?: number;
  status?: string;
}

export interface HookInterpretation {
  provenance: "HEURISTIC";
  status: "detected" | "not_confidently_detected";
  candidate: TemporalWindow | null;
}

export interface TemporalInterpretation<P extends "MODEL_OUTPUT" | "DERIVED" = "MODEL_OUTPUT" | "DERIVED"> {
  provenance: P;
  start: number;
  end: number;
  confidence?: number;
}

export interface LumaeInterpretation {
  hook: HookInterpretation;
  demoMoment: TemporalInterpretation<"MODEL_OUTPUT"> | null;
  resultMoment: TemporalInterpretation<"MODEL_OUTPUT"> | null;
  coreActionRegion: TemporalInterpretation<"DERIVED"> | null;
}

export interface BenchmarkMetrics {
  dataset: string;
  MR_R1_0_5: number;
  MR_R1_0_7: number;
  MR_mAP: number;
  MR_mAP_0_5: number;
  MR_mAP_0_75: number;
  MR_short_mAP: number;
  MR_middle_mAP: number;
  MR_long_mAP: number;
  HL_VeryGood_mAP: number;
  HL_VeryGood_HIT1: number;
}

export interface ModelInfoResponse {
  schemaVersion: "1.0";
  model: ModelMetadata;
  benchmark: BenchmarkMetrics;
}

export interface AnalysisResponse {
  schemaVersion: "1.0";
  analysisId: string;
  model: Pick<ModelMetadata, "id" | "stage">;
  video: VideoMetadata;
  predictions: QueryPrediction[];
  interpretation: LumaeInterpretation;
}

export interface LumaeApiError {
  schemaVersion: "1.0";
  error: { code: AnalysisErrorCode; message: string; retryable: boolean };
}

export const presets: QueryDefinition[] = [
  { id: "hook", intent: "hook_candidate", query: "The product is first introduced at the beginning of the video." },
  { id: "demo", intent: "demo_moment", query: "The creator demonstrates how the product works." },
  { id: "result", intent: "benefit_moment", query: "The creator shows what was collected after using the product." },
];
