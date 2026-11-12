"use client";

import React, { useState } from "react";
import { useRouter } from "next/navigation";
import { AppHeader } from "@/components/layout/AppHeader";
import { AppFooter } from "@/components/layout/AppFooter";
import { AppBackgroundDecoration } from "@/components/layout/AppBackgroundDecoration";
import { VideoPlayer } from "@/components/results/VideoPlayer";
import { MomentTimeline } from "@/components/results/MomentTimeline";
import { MomentExplorerTab } from "@/components/results/MomentExplorerTab";
import { AllCandidatesTab } from "@/components/results/AllCandidatesTab";
import { TechnicalDetailsTab } from "@/components/results/TechnicalDetailsTab";
import { useAnalysis } from "@/context/AnalysisContext";

export default function ResultsPage() {
  const router = useRouter();
  const {
    video,
    query,
    activeCandidate,
    analysisResult,
    currentTime,
    isPlaying,
    setCurrentTime,
    setIsPlaying,
    selectCandidate,
    playMoment,
  } = useAnalysis();

  const [activeTab, setActiveTab] = useState<"explorer" | "all" | "technical">(
    "explorer"
  );

  const handleRefineQuery = () => {
    router.push("/analyze");
  };

  const handleSeek = (time: number) => {
    setCurrentTime(time);
  };

  return (
    <div className="relative min-h-screen flex flex-col bg-white text-[#111111] overflow-x-hidden">
      <AppBackgroundDecoration />
      <AppHeader />

      <main className="relative z-10 flex-1 max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-6 sm:py-10 flex flex-col gap-8 sm:gap-10">
        {/* Top Section: Video Player & Match Summary (Two Columns) */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
          {/* Left: Video Player */}
          <div className="lg:col-span-6 w-full">
            <VideoPlayer
              video={video}
              activeCandidate={activeCandidate}
              currentTime={currentTime}
              isPlaying={isPlaying}
              onTimeUpdate={setCurrentTime}
              onPlayPauseToggle={setIsPlaying}
            />
          </div>

          {/* Right: Analysis Result Card & Action CTAs */}
          <div className="lg:col-span-6 flex flex-col gap-5">
            {/* Header row with Analysis Result badge, Title, and Completed badge */}
            <div className="flex items-start justify-between gap-4">
              <div className="flex flex-col gap-1.5">
                <div className="inline-flex items-center gap-1.5 px-3 py-0.5 rounded-full text-[11px] font-bold bg-[#FFEBEF] text-[#FE2C55] w-fit shadow-2xs">
                  <svg
                    width="10"
                    height="10"
                    viewBox="0 0 24 24"
                    fill="currentColor"
                    aria-hidden="true"
                  >
                    <path d="M5 3.868v16.264c0 .858.94 1.383 1.666.936l13.011-8.132a1.096 1.096 0 0 0 0-1.872L6.666 2.932C5.94 2.485 5 3.01 5 3.868Z" />
                  </svg>
                  <span>ANALYSIS RESULT</span>
                </div>

                <h1 className="text-2xl sm:text-3xl font-black tracking-tight text-[#111111]">
                  Your matching moment
                </h1>
              </div>

              {/* Green Completed Status Badge */}
              <div className="inline-flex items-center gap-1.5 px-3.5 py-1 rounded-full text-xs font-bold bg-[#DCFCE7]/70 border border-[#86EFAC]/40 text-[#16A34A] shadow-2xs flex-shrink-0">
                <svg
                  width="13"
                  height="13"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="3"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  aria-hidden="true"
                >
                  <polyline points="20 6 9 17 4 12" />
                </svg>
                <span>Completed</span>
              </div>
            </div>

            {/* Top Match & Query Info Cards */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              {/* TOP MATCH Card */}
              <div className="bg-white rounded-2xl p-4 sm:p-5 border border-[#E5E7EB] shadow-xs flex flex-col justify-between">
                <div>
                  <span className="text-[11px] font-extrabold text-[#FE2C55] tracking-wider uppercase">
                    TOP MATCH
                  </span>
                  <div className="text-2xl sm:text-3xl font-black font-mono text-[#FE2C55] mt-1 tracking-tight">
                    {activeCandidate.startFormatted} – {activeCandidate.endFormatted}
                  </div>
                </div>
                <div className="text-xs text-[#6B7280] font-medium mt-2">
                  Predicted segment • {activeCandidate.durationSeconds.toFixed(1)} seconds
                </div>
              </div>

              {/* Query Card */}
              <div className="bg-white rounded-2xl p-4 sm:p-5 border border-[#E5E7EB] shadow-xs flex flex-col justify-between">
                <div className="flex items-center gap-1.5 text-xs font-bold text-[#111111]">
                  <svg
                    width="14"
                    height="14"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="2"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    aria-hidden="true"
                  >
                    <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
                  </svg>
                  <span>Query</span>
                </div>
                <p className="text-xs sm:text-sm text-[#4B5563] font-normal leading-relaxed mt-1 line-clamp-3">
                  {query}
                </p>
                <div className="text-[10px] text-[#9CA3AF] font-mono mt-2">
                  Video: {video.filename || "perfume_ad.mp4"}
                </div>
              </div>
            </div>

            {/* Action Buttons: Play Moment & Refine Query */}
            <div className="flex flex-wrap items-center gap-3 pt-1">
              <button
                type="button"
                onClick={() => playMoment(activeCandidate)}
                className="inline-flex items-center justify-center gap-2 px-6 py-3 text-sm font-bold text-white bg-[#000000] hover:bg-[#222222] rounded-full shadow-md hover:shadow-lg transition-all active:scale-[0.98] cursor-pointer"
              >
                <svg
                  width="14"
                  height="14"
                  viewBox="0 0 24 24"
                  fill="currentColor"
                  aria-hidden="true"
                >
                  <path d="M5 3.868v16.264c0 .858.94 1.383 1.666.936l13.011-8.132a1.096 1.096 0 0 0 0-1.872L6.666 2.932C5.94 2.485 5 3.01 5 3.868Z" />
                </svg>
                <span>Play Moment</span>
              </button>

              <button
                type="button"
                onClick={handleRefineQuery}
                className="inline-flex items-center justify-center gap-2 px-5 py-3 text-sm font-semibold text-[#111111] bg-white hover:bg-[#F8F9FA] border border-[#E5E7EB] hover:border-[#111111]/40 rounded-full shadow-2xs hover:shadow-xs transition-all active:scale-[0.98] cursor-pointer"
              >
                <svg
                  width="15"
                  height="15"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="2"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  aria-hidden="true"
                >
                  <line x1="4" x2="20" y1="21" y2="21" />
                  <line x1="4" x2="20" y1="3" y2="3" />
                  <line x1="4" x2="20" y1="12" y2="12" />
                  <circle cx="9" cy="21" r="2" />
                  <circle cx="15" cy="3" r="2" />
                  <circle cx="12" cy="12" r="2" />
                </svg>
                <span>Refine Query</span>
              </button>
            </div>
          </div>
        </div>

        {/* Middle Section: Moment Timeline */}
        <MomentTimeline
          durationSeconds={video.durationSeconds || 25.0}
          candidates={analysisResult.candidates}
          activeCandidate={activeCandidate}
          currentTime={currentTime}
          onSelectCandidate={selectCandidate}
          onSeek={handleSeek}
        />

        {/* Bottom Section: Tabs Navigation & Panels */}
        <div className="flex flex-col gap-6">
          {/* Tabs Navigation Pills */}
          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={() => setActiveTab("explorer")}
              className={`px-5 py-2 text-xs sm:text-sm font-bold rounded-full transition-all cursor-pointer ${
                activeTab === "explorer"
                  ? "bg-[#000000] text-white shadow-xs"
                  : "bg-white text-[#6B7280] hover:text-[#111111] border border-[#E5E7EB]"
              }`}
            >
              Moment Explorer
            </button>

            <button
              type="button"
              onClick={() => setActiveTab("all")}
              className={`px-5 py-2 text-xs sm:text-sm font-bold rounded-full transition-all cursor-pointer ${
                activeTab === "all"
                  ? "bg-[#000000] text-white shadow-xs"
                  : "bg-white text-[#6B7280] hover:text-[#111111] border border-[#E5E7EB]"
              }`}
            >
              All Candidates ({analysisResult.candidates.length})
            </button>

            <button
              type="button"
              onClick={() => setActiveTab("technical")}
              className={`px-5 py-2 text-xs sm:text-sm font-bold rounded-full transition-all cursor-pointer ${
                activeTab === "technical"
                  ? "bg-[#000000] text-white shadow-xs"
                  : "bg-white text-[#6B7280] hover:text-[#111111] border border-[#E5E7EB]"
              }`}
            >
              Technical Details
            </button>
          </div>

          {/* Active Tab Content */}
          <div className="animate-fade-in">
            {activeTab === "explorer" && (
              <MomentExplorerTab
                keyframes={analysisResult.keyframes}
                candidates={analysisResult.candidates}
                activeCandidate={activeCandidate}
                onSeekTo={handleSeek}
                onSelectCandidate={selectCandidate}
              />
            )}

            {activeTab === "all" && (
              <AllCandidatesTab
                candidates={analysisResult.candidates}
                activeCandidate={activeCandidate}
                onSelectCandidate={selectCandidate}
              />
            )}

            {activeTab === "technical" && (
              <TechnicalDetailsTab details={analysisResult.technicalDetails} />
            )}
          </div>
        </div>
      </main>

      <AppFooter />
    </div>
  );
}

