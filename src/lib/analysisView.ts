import type { AnalysisResponse, QueryPrediction, TemporalWindow } from "./analysisTypes";

export function normalizeSaliency(raw: readonly number[]): number[] {
  if (!raw.length) return [];
  const min = Math.min(...raw), max = Math.max(...raw), range = max - min;
  return raw.map(value => range === 0 ? 0 : (value - min) / range);
}

export function saliencyPoints(prediction: QueryPrediction) {
  const normalized = normalizeSaliency(prediction.rawSaliencyScores);
  return prediction.rawSaliencyScores.map((raw, index) => ({
    time: index * prediction.saliencyClipDurationSeconds,
    raw,
    normalized: normalized[index],
  }));
}

export function selectPrediction(analysis: AnalysisResponse | null, queryId: string): QueryPrediction | null {
  return analysis?.predictions.find(prediction => prediction.queryId === queryId) ?? null;
}

export function visibleTimelineWindows(analysis: AnalysisResponse): Array<{ prediction: QueryPrediction; window: TemporalWindow }> {
  return analysis.predictions.flatMap(prediction => {
    if (prediction.intent === "hook_candidate" && analysis.interpretation.hook.candidate === null) return [];
    return prediction.windows.slice(0, 1).map(window => ({ prediction, window }));
  });
}
