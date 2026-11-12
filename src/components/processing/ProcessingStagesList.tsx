"use client";

import React from "react";
import { ProcessingStage } from "@/lib/mockData";

interface ProcessingStagesListProps {
  stages: ProcessingStage[];
  stageProgress: number; // 0 to 100 for current active stage
}

export function ProcessingStagesList({
  stages,
  stageProgress,
}: ProcessingStagesListProps) {
  return (
    <div className="w-full bg-white rounded-3xl p-5 sm:p-6 border border-[#E5E7EB] shadow-xs flex flex-col gap-2.5 transition-all">
      {stages.map((stage) => {
        const isCompleted = stage.status === "completed";
        const isRunning = stage.status === "running";
        const isPending = stage.status === "pending";

        // Dynamic frame calculation for Stage 2
        let dynamicSubtext = stage.activeDetails || "";
        if (stage.id === 2 && isRunning) {
          const currentFrame = Math.min(
            13,
            Math.max(1, Math.floor((stageProgress / 100) * 13) + 1)
          );
          dynamicSubtext = `Frame ${currentFrame} of 13 · Sampling at 0.5 FPS`;
        }

        return (
          <div
            key={stage.id}
            className={`transition-all duration-300 rounded-2xl ${
              isRunning
                ? "bg-[rgba(254,44,85,0.06)] border border-[rgba(254,44,85,0.20)] p-4 sm:p-5 shadow-2xs"
                : "bg-transparent p-3 sm:px-4 sm:py-2.5"
            }`}
          >
            {/* Active / Running Stage: Expanded */}
            {isRunning && (
              <div className="flex flex-col gap-3">
                <div className="flex items-start justify-between gap-3">
                  <div className="flex items-center gap-3">
                    {/* Pink Spinner Ring */}
                    <div className="w-5 h-5 rounded-full border-2 border-[#FE2C55]/20 border-t-[#FE2C55] animate-spin flex-shrink-0" />
                    <h3 className="text-sm sm:text-base font-bold text-[#FE2C55]">
                      {stage.title}
                    </h3>
                  </div>

                  {/* Percentage Chip */}
                  <span className="text-xs font-mono font-bold text-[#FE2C55] bg-white px-2 py-0.5 rounded-full border border-[rgba(254,44,85,0.25)] shadow-2xs">
                    {Math.round(stageProgress)}%
                  </span>
                </div>

                {/* Description */}
                {stage.description && (
                  <p className="text-xs text-[#111111]/85 font-medium pl-8 leading-relaxed">
                    {stage.description}
                  </p>
                )}

                {/* Progress Bar & Dynamic Details */}
                <div className="pl-8 flex flex-col gap-2 pt-1">
                  <div className="w-full h-2 bg-[#FE2C55]/15 rounded-full overflow-hidden">
                    <div
                      className="h-full bg-[#FE2C55] rounded-full transition-all duration-150 ease-out"
                      style={{ width: `${stageProgress}%` }}
                    />
                  </div>

                  {dynamicSubtext && (
                    <div className="flex items-center justify-between text-[11px] font-mono text-[#FE2C55]/90 font-medium">
                      <span>{dynamicSubtext}</span>
                      <span>Moment-DETR Pipeline</span>
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* Completed Stage: Compact Row */}
            {isCompleted && (
              <div className="flex items-center justify-between gap-3">
                <div className="flex items-center gap-3 min-w-0">
                  <div className="w-5 h-5 rounded-full bg-[#DCFCE7] text-[#16A34A] flex items-center justify-center flex-shrink-0 shadow-2xs">
                    <svg
                      width="12"
                      height="12"
                      viewBox="0 0 24 24"
                      fill="none"
                      stroke="currentColor"
                      strokeWidth="3.5"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                      aria-hidden="true"
                    >
                      <polyline points="20 6 9 17 4 12" />
                    </svg>
                  </div>

                  <span className="text-sm font-semibold text-[#111111] truncate">
                    {stage.title}
                  </span>
                </div>

                {/* Completed summary or meta */}
                {stage.completedSummary && (
                  <span className="text-xs font-mono text-[#6B7280] hidden sm:inline flex-shrink-0">
                    {stage.completedSummary}
                  </span>
                )}
              </div>
            )}

            {/* Pending Stage: Minimal Compact Row */}
            {isPending && (
              <div className="flex items-center gap-3">
                <div className="w-5 h-5 rounded-full border-2 border-[#E5E7EB] bg-white flex-shrink-0" />
                <span className="text-sm font-medium text-[#6B7280]">
                  {stage.title}
                </span>
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
