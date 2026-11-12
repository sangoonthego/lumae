"use client";

import React, { useState, useEffect, useRef } from "react";
import { useRouter } from "next/navigation";
import { AppHeader } from "@/components/layout/AppHeader";
import { AppFooter } from "@/components/layout/AppFooter";
import { ProcessingVideoPreview } from "@/components/processing/ProcessingVideoPreview";
import { QueryContextCard } from "@/components/processing/QueryContextCard";
import { ProcessingStagesList } from "@/components/processing/ProcessingStagesList";
import { INITIAL_PROCESSING_STAGES, ProcessingStage } from "@/lib/mockData";
import { useAnalysis } from "@/context/AnalysisContext";

export default function ProcessingPage() {
  const router = useRouter();
  const { video, query } = useAnalysis();

  const [stages, setStages] = useState<ProcessingStage[]>(() =>
    INITIAL_PROCESSING_STAGES.map((s, idx) => ({
      ...s,
      status: idx === 0 ? "running" : "pending",
    }))
  );
  const [stageProgress, setStageProgress] = useState<number>(0);
  const [elapsedSeconds, setElapsedSeconds] = useState<number>(0);
  const [isCompleted, setIsCompleted] = useState<boolean>(false);

  const simulationActiveRef = useRef<boolean>(true);

  // Elapsed simulation timer
  useEffect(() => {
    const timer = setInterval(() => {
      setElapsedSeconds((prev) => +(prev + 0.1).toFixed(1));
    }, 100);

    return () => clearInterval(timer);
  }, []);

  // Sequential Stage Progression Engine
  useEffect(() => {
    simulationActiveRef.current = true;

    let progressInterval: NodeJS.Timeout | null = null;
    let stageTimeout: NodeJS.Timeout | null = null;

    const runStage = (index: number) => {
      if (!simulationActiveRef.current) return;

      if (index >= INITIAL_PROCESSING_STAGES.length) {
        setIsCompleted(true);
        // Clean transition to results
        setTimeout(() => {
          if (simulationActiveRef.current) {
            router.push("/analyze/results");
          }
        }, 800);
        return;
      }

      setStageProgress(0);

      // Advance stages state
      setStages((prev) =>
        prev.map((s, idx) => {
          if (idx < index) return { ...s, status: "completed" };
          if (idx === index) return { ...s, status: "running" };
          return { ...s, status: "pending" };
        })
      );

      const stageDuration = INITIAL_PROCESSING_STAGES[index].durationMs;
      const stepInterval = 40;
      const progressIncrement = 100 / (stageDuration / stepInterval);

      progressInterval = setInterval(() => {
        setStageProgress((prev) => {
          const next = prev + progressIncrement;
          return next >= 100 ? 100 : next;
        });
      }, stepInterval);

      stageTimeout = setTimeout(() => {
        if (progressInterval) clearInterval(progressInterval);
        setStageProgress(100);

        setStages((prev) =>
          prev.map((s, idx) => (idx === index ? { ...s, status: "completed" } : s))
        );

        runStage(index + 1);
      }, stageDuration);
    };

    runStage(0);

    return () => {
      simulationActiveRef.current = false;
      if (progressInterval) clearInterval(progressInterval);
      if (stageTimeout) clearInterval(stageTimeout);
    };
  }, [router]);

  const handleSkip = () => {
    simulationActiveRef.current = false;
    router.push("/analyze/results");
  };

  const runningStageIndex = stages.findIndex((s) => s.status === "running");
  const currentStepIndex =
    runningStageIndex >= 0 ? runningStageIndex : isCompleted ? stages.length - 1 : 0;
  const currentStepNumber = Math.min(stages.length, currentStepIndex + 1);

  return (
    <div className="relative min-h-screen flex flex-col bg-white text-[#111111] overflow-x-hidden">
      <AppHeader />

      <main className="relative z-10 flex-1 max-w-[1320px] w-full mx-auto px-4 sm:px-6 lg:px-8 py-8 sm:py-10">
        {/* Responsive Grid Layout */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-10 lg:gap-14 items-start">
          {/* Mobile-Only Header (on small screens, show title first) */}
          <div className="lg:hidden flex flex-col gap-3">
            <div className="flex items-center justify-between gap-3">
              <h1 className="text-2xl sm:text-3xl font-black tracking-tight text-[#111111]">
                Finding your <span className="text-[#FE2C55]">moment...</span>
              </h1>

              {/* Status Badge */}
              <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-[#FFEBEF] text-[#FE2C55] text-xs font-bold border border-[#FE2C55]/20 flex-shrink-0">
                <span className="w-2 h-2 rounded-full bg-[#FE2C55] animate-ping" />
                <span>{isCompleted ? "Completed" : "Analyzing"}</span>
              </div>
            </div>
            <p className="text-xs text-[#6B7280]">
              Analyzing the video to locate the most relevant time interval.
            </p>
          </div>

          {/* Left Column: Video Context & Query Card (approx 38-40% width on desktop) */}
          <div className="lg:col-span-5 flex flex-col gap-5 w-full">
            <ProcessingVideoPreview video={video} />
            <QueryContextCard query={query} />
          </div>

          {/* Right Column: Processing Header & Stages (approx 60-62% width) */}
          <div className="lg:col-span-7 flex flex-col gap-6 w-full">
            {/* Desktop Processing Header */}
            <div className="hidden lg:flex items-center justify-between gap-4">
              <div>
                <h1 className="text-3xl sm:text-4xl font-black tracking-tight text-[#111111] leading-tight">
                  Finding your <span className="text-[#FE2C55]">moment...</span>
                </h1>
                <p className="text-sm text-[#6B7280] mt-1">
                  Analyzing the video to locate the most relevant time interval.
                </p>
              </div>

              {/* Status Indicators: Step Counter + Elapsed Time + Analyzing Badge */}
              <div className="flex items-center gap-2 flex-shrink-0">
                {/* Step Counter Chip */}
                <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-[#F8F9FA] border border-[#E5E7EB] text-xs font-mono font-bold text-[#111111] shadow-2xs">
                  <span>Step {currentStepNumber} of {stages.length}</span>
                </div>

                {/* Elapsed Time Chip */}
                <div className="inline-flex items-center gap-1.5 px-3 py-1 rounded-full bg-[#F8F9FA] border border-[#E5E7EB] text-xs font-mono font-bold text-[#111111] shadow-2xs">
                  <svg
                    width="12"
                    height="12"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="#FE2C55"
                    strokeWidth="2.5"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    aria-hidden="true"
                  >
                    <circle cx="12" cy="12" r="10" />
                    <polyline points="12 6 12 12 16 14" />
                  </svg>
                  <span>{elapsedSeconds.toFixed(1)}s</span>
                </div>

                {/* Analysis Status Pill */}
                <div className="inline-flex items-center gap-1.5 px-3.5 py-1 rounded-full bg-[#FFEBEF] text-[#FE2C55] text-xs font-bold border border-[#FE2C55]/20 shadow-2xs">
                  <span className="w-2 h-2 rounded-full bg-[#FE2C55] animate-ping" />
                  <span>{isCompleted ? "Completed" : "Analyzing"}</span>
                </div>
              </div>
            </div>

            {/* Compact Processing Stepper (with expanded active stage and single progress bar) */}
            <ProcessingStagesList
              stages={stages}
              stageProgress={stageProgress}
            />

            {/* Bottom Demo Controls */}
            <div className="flex items-center justify-between text-xs text-[#6B7280] pt-1 px-1">
              <span className="flex items-center gap-1.5">
                <span className="w-1.5 h-1.5 rounded-full bg-[#16A34A]" />
                Demo mode • Moment-DETR & CLIP ViT-B/32
              </span>

              <button
                type="button"
                onClick={handleSkip}
                className="text-[#111111] hover:text-[#FE2C55] font-semibold underline underline-offset-4 cursor-pointer transition-colors"
              >
                Skip simulation →
              </button>
            </div>
          </div>
        </div>
      </main>

      <AppFooter />
    </div>
  );
}
