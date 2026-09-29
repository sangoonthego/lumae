"use client";

import { useCallback, useEffect, useState } from "react";
import type { AnalysisResponse, MockScenario, ModelInfoResponse, QueryDefinition, VideoMetadata } from "./analysisTypes";
import { analyzeVideo, getModelInfo, LumaeApiException } from "./lumaeApi";

export const analysisStages = [
  "Validating video", "Extracting features", "Encoding query",
  "Running temporal inference", "Building saliency timeline", "Interpreting moments",
] as const;

export function useVideoAnalysis(mode: "mock" | "real", scenario: MockScenario) {
  const [modelInfo, setModelInfo] = useState<ModelInfoResponse | null>(null);
  const [modelError, setModelError] = useState<LumaeApiException | null>(null);
  const [analysis, setAnalysis] = useState<AnalysisResponse | null>(null);
  const [error, setError] = useState<LumaeApiException | null>(null);
  const [busy, setBusy] = useState(false);
  const [stage, setStage] = useState(0);

  useEffect(() => {
    const controller = new AbortController();
    void getModelInfo({ mode, signal: controller.signal }).then(info => {
      if (!controller.signal.aborted) { setModelInfo(info); setModelError(null); }
    }).catch(caught => {
      if (!controller.signal.aborted) { setModelInfo(null); setModelError(caught as LumaeApiException); }
    });
    return () => controller.abort();
  }, [mode]);

  useEffect(() => {
    if (!busy) return;
    const timer = window.setInterval(() => setStage(previous => Math.min(analysisStages.length - 1, previous + 1)), 150);
    return () => window.clearInterval(timer);
  }, [busy]);

  const reset = useCallback(() => { setAnalysis(null); setError(null); setBusy(false); setStage(0); }, []);
  const run = useCallback(async (video: File, videoMetadata: Omit<VideoMetadata, "filename">, queries: QueryDefinition[]) => {
    setError(null); setStage(0); setBusy(true);
    try {
      const result = await analyzeVideo({ video, videoMetadata, queries, mode, scenario, maxVideoDurationSeconds: modelInfo?.model.maxVideoDurationSeconds });
      setAnalysis(result);
      return result;
    } catch (caught) {
      const apiError = caught instanceof LumaeApiException ? caught
        : new LumaeApiException("UNKNOWN_ERROR", "Analysis failed. Please retry.", true);
      setError(apiError);
      return null;
    } finally { setBusy(false); }
  }, [mode, scenario, modelInfo]);

  return { modelInfo, modelError, analysis, error, busy, stage, run, reset };
}
