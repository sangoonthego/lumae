import type { MockScenario, QueryDefinition, VideoMetadata } from "./analysisTypes";

/** Wire-format fixtures. Mock and HTTP responses pass through the same decoder. */
export const mockModelInfoWire = {
  schema_version: "1.0",
  model: {
    id: "moment-detr-clip-baseline",
    display_name: "Moment-DETR CLIP-only",
    base_model: "Moment-DETR",
    stage: "baseline",
    temporal_resolution_seconds: 2,
    max_video_duration_seconds: 150,
    status: "research_demo",
  },
  benchmark: {
    dataset: "QVHighlights",
    MR_R1_0_5: 53.23,
    MR_R1_0_7: 34.00,
    MR_mAP: 30.58,
    MR_mAP_0_5: 54.80,
    MR_mAP_0_75: 29.02,
    MR_short_mAP: 3.11,
    MR_middle_mAP: 29.42,
    MR_long_mAP: 41.27,
    HL_VeryGood_mAP: 35.51,
    HL_VeryGood_HIT1: 55.87,
  },
} as const;

const demoSaliency = [-0.8359, -0.4944, 0.2634, -0.3079, 0.5986, 0.6387, 0.5044, 0.3943, -0.6152, -0.6299];
const resultSaliency = [-0.72, -0.45, -0.33, -0.15, 0.28, 0.55, 0.73, 0.94, 0.67, 0.31];
const hookSaliency = [0.12, 0.31, 0.18, -0.11, -0.43, -0.54, -0.62, -0.64, -0.48, -0.5];

interface MockRequest {
  video: VideoMetadata;
  queries: QueryDefinition[];
  scenario: MockScenario;
}

export function createMockAnalysisWire({ video, queries, scenario }: MockRequest): unknown {
  if (scenario === "video_too_long") return {
    schema_version: "1.0",
    error: { code: "VIDEO_TOO_LONG", message: "Video duration exceeds the 150 second model limit.", retryable: false },
  };
  if (scenario === "malformed_response") return { schema_version: "1.0", predictions: "invalid" };

  const factor = Math.min(1, video.durationSeconds / 19.133);
  const scoreCount = Math.max(1, Math.min(demoSaliency.length, Math.ceil(video.durationSeconds / 2)));
  const window = (start: number, end: number, confidence: number) => ({ start: start * factor, end: end * factor, confidence });
  const noPredictions = scenario === "no_predictions";
  const confidentHook = scenario === "confident_hook";
  const hookCandidate = confidentHook ? window(0.4, 3.2, 0.8421) : null;
  const demoMoment = noPredictions ? null : { provenance: "MODEL_OUTPUT", ...window(7.7, 15.46, 0.9275) };
  const resultMoment = noPredictions ? null : { provenance: "MODEL_OUTPUT", ...window(8.0, 17.06, 0.9990) };
  const coreActionRegion = noPredictions ? null : { provenance: "DERIVED", start: 8.0 * factor, end: 15.46 * factor };

  return {
    schema_version: "1.0",
    analysis_id: `analysis_demo_${scenario}`,
    model: { id: mockModelInfoWire.model.id, stage: mockModelInfoWire.model.stage },
    video: {
      filename: video.filename,
      duration_seconds: video.durationSeconds,
      width: video.width,
      height: video.height,
      codec: video.codec,
    },
    predictions: queries.map(({ id, intent, query }) => ({
      query_id: id,
      intent,
      query,
      windows: noPredictions ? [] : intent === "hook_candidate"
        ? [confidentHook ? window(0.4, 3.2, 0.8421) : window(0.4, 3.2, 0.1842), window(4.1, 7.3, 0.0731)]
        : intent === "benefit_moment"
          ? [window(8.0, 17.06, 0.9990), window(15.5, 18.4, 0.1204)]
          : intent === "demo_moment"
            ? [window(7.7, 15.46, 0.9275), window(3.26, 7.4, 0.0213), window(9.72, 16.19, 0.0188)]
            : [window(9.1, 16.4, 0.7312), window(4.3, 6.2, 0.1142)],
      raw_saliency_scores: (intent === "hook_candidate" ? hookSaliency
        : intent === "benefit_moment" ? resultSaliency : demoSaliency).slice(0, scoreCount),
      saliency_clip_duration_seconds: 2,
    })),
    interpretation: {
      hook: { provenance: "HEURISTIC", status: confidentHook ? "detected" : "not_confidently_detected", candidate: hookCandidate },
      demo_moment: demoMoment,
      result_moment: resultMoment,
      core_action_region: coreActionRegion,
    },
  };
}
