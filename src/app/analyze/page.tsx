"use client";

import React, { useState } from "react";
import { useRouter } from "next/navigation";
import { AppHeader } from "@/components/layout/AppHeader";
import { AppFooter } from "@/components/layout/AppFooter";
import { AppBackgroundDecoration } from "@/components/layout/AppBackgroundDecoration";
import { VideoDropzone } from "@/components/upload/VideoDropzone";
import { UploadedVideoCard } from "@/components/upload/UploadedVideoCard";
import { QueryInput } from "@/components/upload/QueryInput";
import { useAnalysis } from "@/context/AnalysisContext";

export default function AnalyzePage() {
  const router = useRouter();
  const {
    video,
    query,
    setQuery,
    selectedChipId,
    setSelectedChipId,
    uploadCustomVideo,
    loadSampleDemo,
    removeVideo,
  } = useAnalysis();

  const [validationError, setValidationError] = useState<string | null>(null);

  const hasVideo = Boolean(video && video.filename);
  const hasQuery = Boolean(query && query.trim().length > 0);
  const isValid = hasVideo && hasQuery;

  const handleSelectChip = (chipId: string, chipText: string) => {
    setSelectedChipId(chipId);
    setQuery(chipText);
  };

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!hasVideo) {
      setValidationError("Please select or upload a video before analyzing.");
      return;
    }
    if (!hasQuery) {
      setValidationError("Please describe the moment you want to find.");
      return;
    }
    setValidationError(null);
    router.push("/analyze/processing");
  };

  return (
    <div className="relative min-h-screen flex flex-col bg-white text-[#111111] overflow-x-hidden">
      <AppBackgroundDecoration />
      <AppHeader />

      <main className="relative z-10 flex-1 max-w-4xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-8 sm:py-12">
        {/* Header Section */}
        <div className="flex flex-col items-start mb-8 sm:mb-10">
          {/* Badge */}
          <div className="inline-flex items-center gap-1.5 px-3.5 py-1 rounded-full text-xs font-bold bg-[#FFEBEF] text-[#FE2C55] mb-4 shadow-2xs">
            <svg
              width="10"
              height="10"
              viewBox="0 0 24 24"
              fill="currentColor"
              aria-hidden="true"
            >
              <path d="M5 3.868v16.264c0 .858.94 1.383 1.666.936l13.011-8.132a1.096 1.096 0 0 0 0-1.872L6.666 2.932C5.94 2.485 5 3.01 5 3.868Z" />
            </svg>
            <span>Video Moment Intelligence</span>
          </div>

          {/* Heading */}
          <h1 className="text-3xl sm:text-5xl font-black tracking-tight text-[#111111] leading-tight mb-2">
            Analyze a <span className="text-[#FE2C55]">video</span>
          </h1>

          <p className="text-sm sm:text-base text-[#6B7280]">
            Upload an advertisement video and describe the moment to find.
          </p>
        </div>

        {/* Form Container */}
        <form onSubmit={handleSubmit} className="flex flex-col gap-6 sm:gap-8">
          {/* Upload Area: Dropzone + Uploaded Card */}
          <div className="flex flex-col gap-4">
            <VideoDropzone
              onVideoSelected={uploadCustomVideo}
              onUseSampleDemo={loadSampleDemo}
            />

            {hasVideo && (
              <div className="animate-fade-in">
                <UploadedVideoCard video={video} onRemove={removeVideo} />
              </div>
            )}
          </div>

          {/* Query Section */}
          <div className="pt-2">
            <QueryInput
              query={query}
              onChange={setQuery}
              selectedChipId={selectedChipId}
              onSelectChip={handleSelectChip}
            />
          </div>

          {/* Validation Error Message */}
          {validationError && (
            <div className="p-3.5 rounded-xl bg-[#FEE2E2] border border-[#FCA5A5] text-[#DC2626] text-xs sm:text-sm font-semibold flex items-center gap-2">
              <svg
                width="16"
                height="16"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
                aria-hidden="true"
              >
                <circle cx="12" cy="12" r="10" />
                <line x1="12" x2="12" y1="8" y2="12" />
                <line x1="12" x2="12.01" y1="16" y2="16" />
              </svg>
              <span>{validationError}</span>
            </div>
          )}

          {/* CTA Button */}
          <div className="flex justify-center pt-4 sm:pt-6">
            <button
              type="submit"
              disabled={!isValid}
              className="inline-flex items-center justify-center gap-2 px-10 py-4 text-base font-bold text-white bg-[#000000] hover:bg-[#222222] disabled:opacity-40 disabled:cursor-not-allowed rounded-full shadow-md hover:shadow-lg transition-all active:scale-[0.98] cursor-pointer min-w-[240px]"
            >
              <span>Find Moments</span>
              <span className="text-white/80">→</span>
            </button>
          </div>
        </form>
      </main>

      <AppFooter />
    </div>
  );
}
