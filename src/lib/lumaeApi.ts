import type {
  AnalysisErrorCode, AnalysisResponse, BenchmarkMetrics, LumaeApiError,
  LumaeInterpretation, MockScenario, ModelInfoResponse, ModelMetadata,
  ModelStage, QueryDefinition, QueryIntent, QueryPrediction, TemporalInterpretation,
  TemporalWindow, VideoMetadata,
} from "./analysisTypes";
import { createMockAnalysisWire, mockModelInfoWire } from "./mockAnalysis";

export class LumaeApiException extends Error {
  constructor(public readonly code: AnalysisErrorCode, message: string, public readonly retryable: boolean) {
    super(message);
    this.name = "LumaeApiException";
  }
}

const stages: ModelStage[] = ["baseline", "stage_a", "stage_b", "stage_c"];
const intents: QueryIntent[] = ["hook_candidate", "demo_moment", "benefit_moment", "custom"];
const errorCodes: AnalysisErrorCode[] = ["VIDEO_TOO_LONG", "UNSUPPORTED_VIDEO", "INVALID_QUERY", "MODEL_UNAVAILABLE", "ANALYSIS_TIMEOUT", "MALFORMED_RESPONSE", "UNKNOWN_ERROR"];
const malformed = (): never => { throw new LumaeApiException("MALFORMED_RESPONSE", "The API returned a malformed schema 1.0 response.", false); };
const object = (value: unknown): Record<string, unknown> => value !== null && typeof value === "object" && !Array.isArray(value) ? value as Record<string, unknown> : malformed();
const string = (value: unknown): string => typeof value === "string" ? value : malformed();
const number = (value: unknown): number => typeof value === "number" && Number.isFinite(value) ? value : malformed();
const array = (value: unknown): unknown[] => Array.isArray(value) ? value : malformed();
const optionalString = (value: unknown): string | undefined => value === undefined ? undefined : string(value);
const optionalNumber = (value: unknown): number | undefined => value === undefined ? undefined : number(value);
const version = (value: unknown): "1.0" => value === "1.0" ? "1.0" : malformed();
const stage = (value: unknown): ModelStage => stages.includes(value as ModelStage) ? value as ModelStage : malformed();
const intent = (value: unknown): QueryIntent => intents.includes(value as QueryIntent) ? value as QueryIntent : malformed();

export function parseTemporalWindow(value: unknown): TemporalWindow {
  const wire = object(value);
  const start = number(wire.start), end = number(wire.end), confidence = number(wire.confidence);
  if (start < 0 || end <= start || confidence < 0 || confidence > 1) malformed();
  return { start, end, confidence };
}

function parseModel(value: unknown): ModelMetadata {
  const wire = object(value);
  return {
    id: string(wire.id), stage: stage(wire.stage),
    displayName: optionalString(wire.display_name), baseModel: optionalString(wire.base_model),
    temporalResolutionSeconds: optionalNumber(wire.temporal_resolution_seconds),
    maxVideoDurationSeconds: optionalNumber(wire.max_video_duration_seconds),
    status: optionalString(wire.status),
  };
}

export function parseModelInfoResponse(value: unknown): ModelInfoResponse {
  const wire = object(value);
  const benchmark = object(wire.benchmark);
  const metrics: BenchmarkMetrics = {
    dataset: string(benchmark.dataset),
    MR_R1_0_5: number(benchmark.MR_R1_0_5), MR_R1_0_7: number(benchmark.MR_R1_0_7),
    MR_mAP: number(benchmark.MR_mAP), MR_mAP_0_5: number(benchmark.MR_mAP_0_5),
    MR_mAP_0_75: number(benchmark.MR_mAP_0_75), MR_short_mAP: number(benchmark.MR_short_mAP),
    MR_middle_mAP: number(benchmark.MR_middle_mAP), MR_long_mAP: number(benchmark.MR_long_mAP),
    HL_VeryGood_mAP: number(benchmark.HL_VeryGood_mAP), HL_VeryGood_HIT1: number(benchmark.HL_VeryGood_HIT1),
  };
  return { schemaVersion: version(wire.schema_version), model: parseModel(wire.model), benchmark: metrics };
}

function parseVideo(value: unknown): VideoMetadata {
  const wire = object(value);
  const durationSeconds = number(wire.duration_seconds), width = number(wire.width), height = number(wire.height);
  if (durationSeconds <= 0 || width <= 0 || height <= 0) malformed();
  return { filename: string(wire.filename), durationSeconds, width, height, codec: string(wire.codec) };
}

function parsePrediction(value: unknown): QueryPrediction {
  const wire = object(value);
  const saliencyClipDurationSeconds = number(wire.saliency_clip_duration_seconds);
  if (saliencyClipDurationSeconds <= 0) malformed();
  return {
    queryId: string(wire.query_id), intent: intent(wire.intent), query: string(wire.query),
    windows: array(wire.windows).map(parseTemporalWindow),
    rawSaliencyScores: array(wire.raw_saliency_scores).map(number),
    saliencyClipDurationSeconds,
  };
}

function parseTemporalInterpretation<P extends "MODEL_OUTPUT" | "DERIVED">(value: unknown, provenance: P): TemporalInterpretation<P> | null {
  if (value === null) return null;
  const wire = object(value);
  const start = number(wire.start), end = number(wire.end);
  if (wire.provenance !== provenance || start < 0 || end <= start) malformed();
  const confidence = wire.confidence === undefined ? undefined : number(wire.confidence);
  if (confidence !== undefined && (confidence < 0 || confidence > 1)) malformed();
  return { provenance, start, end, ...(confidence === undefined ? {} : { confidence }) };
}

function parseInterpretation(value: unknown): LumaeInterpretation {
  const wire = object(value), hook = object(wire.hook);
  if (hook.provenance !== "HEURISTIC" || (hook.status !== "detected" && hook.status !== "not_confidently_detected")) malformed();
  const candidate = hook.candidate === null ? null : parseTemporalWindow(hook.candidate);
  if ((hook.status === "detected") !== (candidate !== null)) malformed();
  return {
    hook: { provenance: "HEURISTIC", status: hook.status as "detected" | "not_confidently_detected", candidate },
    demoMoment: parseTemporalInterpretation(wire.demo_moment, "MODEL_OUTPUT"),
    resultMoment: parseTemporalInterpretation(wire.result_moment, "MODEL_OUTPUT"),
    coreActionRegion: parseTemporalInterpretation(wire.core_action_region, "DERIVED"),
  };
}

export function parseAnalysisResponse(value: unknown): AnalysisResponse {
  const wire = object(value), model = object(wire.model);
  return {
    schemaVersion: version(wire.schema_version), analysisId: string(wire.analysis_id),
    model: { id: string(model.id), stage: stage(model.stage) },
    video: parseVideo(wire.video), predictions: array(wire.predictions).map(parsePrediction),
    interpretation: parseInterpretation(wire.interpretation),
  };
}

export function parseLumaeApiError(value: unknown): LumaeApiError {
  const wire = object(value), error = object(wire.error);
  if (!errorCodes.includes(error.code as AnalysisErrorCode) || typeof error.retryable !== "boolean") malformed();
  return { schemaVersion: version(wire.schema_version), error: { code: error.code as AnalysisErrorCode, message: string(error.message), retryable: error.retryable as boolean } };
}

export interface ClientOptions { mode: "mock" | "real"; scenario?: MockScenario; signal?: AbortSignal }
export interface AnalyzeRequest extends ClientOptions {
  video: File;
  queries: QueryDefinition[];
  videoMetadata: Omit<VideoMetadata, "filename">;
  maxVideoDurationSeconds?: number;
}

function apiBase(): string {
  const value = process.env.NEXT_PUBLIC_LUMAE_API_BASE_URL?.trim();
  if (!value) throw new LumaeApiException("MODEL_UNAVAILABLE", "Analysis backend is not configured. Set NEXT_PUBLIC_LUMAE_API_BASE_URL or enable Demo Data Mode.", true);
  return value.replace(/\/$/, "");
}

async function jsonResponse(response: Response): Promise<unknown> {
  try { return await response.json(); } catch { return malformed(); }
}

function unwrapError(value: unknown): void {
  if (value && typeof value === "object" && "error" in value) {
    const parsed = parseLumaeApiError(value);
    throw new LumaeApiException(parsed.error.code, parsed.error.message, parsed.error.retryable);
  }
}

async function realRequest(path: string, init: RequestInit = {}, timeoutMs = 30_000): Promise<unknown> {
  const base = apiBase();
  const controller = new AbortController();
  if (init.signal?.aborted) controller.abort();
  const cancel = () => controller.abort();
  init.signal?.addEventListener("abort", cancel, { once: true });
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  let response: Response;
  try { response = await fetch(`${base}${path}`, { ...init, signal: controller.signal }); }
  catch {
    throw new LumaeApiException(controller.signal.aborted ? "ANALYSIS_TIMEOUT" : "MODEL_UNAVAILABLE",
      controller.signal.aborted ? "Analysis timed out or was cancelled." : "Analysis backend is offline. The runtime may have expired.", true);
  } finally { clearTimeout(timer); init.signal?.removeEventListener("abort", cancel); }
  const value = await jsonResponse(response);
  unwrapError(value);
  if (!response.ok) throw new LumaeApiException("UNKNOWN_ERROR", `API request failed (${response.status}).`, response.status >= 500);
  return value;
}

function analysisTimeoutMs(): number {
  const configured = Number(process.env.NEXT_PUBLIC_LUMAE_ANALYSIS_TIMEOUT_MS);
  return Number.isFinite(configured) && configured > 0 ? configured : 180_000;
}

export async function getModelInfo({ mode, signal }: ClientOptions): Promise<ModelInfoResponse> {
  const raw = mode === "mock" ? mockModelInfoWire : await realRequest("/api/v1/model", { signal });
  return parseModelInfoResponse(raw);
}

export async function analyzeVideo({ video, queries, videoMetadata, maxVideoDurationSeconds = 150, mode, scenario = "standard", signal }: AnalyzeRequest): Promise<AnalysisResponse> {
  if (!queries.length || queries.some((query) => !query.query.trim())) throw new LumaeApiException("INVALID_QUERY", "Enter at least one non-empty query.", false);
  if (videoMetadata.durationSeconds > maxVideoDurationSeconds && scenario !== "video_too_long") throw new LumaeApiException("VIDEO_TOO_LONG", `Video duration exceeds the ${maxVideoDurationSeconds} second model limit.`, false);
  let raw: unknown;
  if (mode === "mock") {
    await new Promise<void>((resolve) => setTimeout(resolve, 900));
    raw = createMockAnalysisWire({ video: { filename: video.name, ...videoMetadata }, queries, scenario });
    unwrapError(raw);
  } else {
    const form = new FormData();
    form.set("video", video);
    form.set("queries", JSON.stringify(queries));
    raw = await realRequest("/api/v1/analyze", { method: "POST", body: form, signal }, analysisTimeoutMs());
  }
  return parseAnalysisResponse(raw);
}

/** Local sample only; no model request or external upload occurs here. */
export async function loadBundledSampleVideo(): Promise<File> {
  const response = await fetch("/lumae-demo.mp4");
  if (!response.ok) throw new LumaeApiException("UNKNOWN_ERROR", "Bundled sample video is unavailable.", false);
  return new File([await response.blob()], "lumae-demo.mp4", { type: "video/mp4" });
}
