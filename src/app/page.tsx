"use client";

import React from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { AppHeader } from "@/components/layout/AppHeader";
import { AppFooter } from "@/components/layout/AppFooter";
import { AppBackgroundDecoration } from "@/components/layout/AppBackgroundDecoration";
import { HeroVideoPreview } from "@/components/landing/HeroVideoPreview";
import { useAnalysis } from "@/context/AnalysisContext";

export default function LandingPage() {
  const router = useRouter();
  const { loadSampleDemo } = useAnalysis();

  const handleTrySampleDemo = () => {
    loadSampleDemo();
    router.push("/analyze");
  };

  return (
    <div className="relative min-h-screen flex flex-col bg-white text-[#111111] overflow-x-hidden">
      <AppBackgroundDecoration />
      <AppHeader />

      <main className="relative z-10 flex-1 flex flex-col justify-center max-w-7xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-10 md:py-16">
        {/* Two-Column Hero Section */}
        <div className="grid grid-cols-1 lg:grid-cols-12 gap-12 lg:gap-8 items-center mb-16 lg:mb-24">
          {/* Left Column: Headline and Call-To-Actions */}
          <div className="lg:col-span-6 flex flex-col items-start text-left max-w-xl">
            {/* Small Pink Badge */}
            <div className="inline-flex items-center gap-1.5 px-3.5 py-1 rounded-full text-xs font-bold bg-[#FFEBEF] text-[#FE2C55] mb-6 shadow-2xs">
              <svg
                width="12"
                height="12"
                viewBox="0 0 24 24"
                fill="currentColor"
                aria-hidden="true"
              >
                <path d="M5 3.868v16.264c0 .858.94 1.383 1.666.936l13.011-8.132a1.096 1.096 0 0 0 0-1.872L6.666 2.932C5.94 2.485 5 3.01 5 3.868Z" />
              </svg>
              <span>Video Moment Intelligence</span>
            </div>

            {/* Main Headline */}
            <h1 className="text-4xl sm:text-5xl lg:text-6xl font-black tracking-tight text-[#111111] leading-[1.08] mb-6">
              Find the moment <br />
              that <span className="text-[#FE2C55]">matters.</span>
            </h1>

            {/* Description */}
            <p className="text-base sm:text-lg text-[#6B7280] leading-relaxed mb-8 max-w-lg">
              Describe any moment in an advertisement video. LUMAE finds the
              corresponding time interval using Moment-DETR.
            </p>

            {/* Action Buttons */}
            <div className="flex flex-wrap items-center gap-4 w-full sm:w-auto">
              <Link
                href="/analyze"
                className="inline-flex items-center justify-center gap-2.5 px-7 py-3.5 text-sm sm:text-base font-bold text-white bg-[#000000] hover:bg-[#222222] rounded-full shadow-md hover:shadow-lg transition-all active:scale-[0.98] w-full sm:w-auto"
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
                <span>Analyze a Video</span>
                <span className="text-white/80">→</span>
              </Link>

              <button
                type="button"
                onClick={handleTrySampleDemo}
                className="inline-flex items-center justify-center gap-2 px-6 py-3.5 text-sm sm:text-base font-semibold text-[#111111] bg-white hover:bg-[#F8F9FA] border border-[#E5E7EB] hover:border-[#111111]/30 rounded-full shadow-2xs hover:shadow-xs transition-all active:scale-[0.98] w-full sm:w-auto cursor-pointer"
              >
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
                  <path d="m16 13 5.223 3.482a.5.5 0 0 0 .777-.416V7.934a.5.5 0 0 0-.777-.416L16 11" />
                  <rect width="14" height="12" x="2" y="6" rx="2" />
                </svg>
                <span>Try Sample Demo</span>
              </button>
            </div>
          </div>

          {/* Right Column: Hero Video Preview Card */}
          <div className="lg:col-span-6 flex justify-center lg:justify-end">
            <HeroVideoPreview />
          </div>
        </div>

        {/* Bottom Three Features */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6 sm:gap-8 pt-8 border-t border-[#E5E7EB]/80">
          {/* Feature 1 */}
          <div className="flex items-start gap-3.5 p-4 rounded-2xl hover:bg-[#F8F9FA]/60 transition-colors">
            <div className="w-10 h-10 rounded-xl bg-[#FFEBEF] flex items-center justify-center text-[#FE2C55] flex-shrink-0">
              <svg
                width="20"
                height="20"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
                aria-hidden="true"
              >
                <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2" />
              </svg>
            </div>
            <div>
              <h3 className="text-base font-bold text-[#111111] mb-1">
                Temporal Grounding
              </h3>
              <p className="text-sm text-[#6B7280]">
                Locate moments by language.
              </p>
            </div>
          </div>

          {/* Feature 2 */}
          <div className="flex items-start gap-3.5 p-4 rounded-2xl hover:bg-[#F8F9FA]/60 transition-colors">
            <div className="w-10 h-10 rounded-xl bg-[#FFEBEF] flex items-center justify-center text-[#FE2C55] flex-shrink-0">
              <svg
                width="20"
                height="20"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
                aria-hidden="true"
              >
                <polygon points="12 2 2 7 12 12 22 7 12 2" />
                <polyline points="2 17 12 22 22 17" />
                <polyline points="2 12 12 17 22 12" />
              </svg>
            </div>
            <div>
              <h3 className="text-base font-bold text-[#111111] mb-1">
                Ad Video Focus
              </h3>
              <p className="text-sm text-[#6B7280]">
                Built for advertisements.
              </p>
            </div>
          </div>

          {/* Feature 3 */}
          <div className="flex items-start gap-3.5 p-4 rounded-2xl hover:bg-[#F8F9FA]/60 transition-colors">
            <div className="w-10 h-10 rounded-xl bg-[#FFEBEF] flex items-center justify-center text-[#FE2C55] flex-shrink-0">
              <svg
                width="20"
                height="20"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
                aria-hidden="true"
              >
                <path d="M18 20V10" />
                <path d="M12 20V4" />
                <path d="M6 20v-6" />
              </svg>
            </div>
            <div>
              <h3 className="text-base font-bold text-[#111111] mb-1">
                Research-backed
              </h3>
              <p className="text-sm text-[#6B7280]">
                Based on Moment-DETR & CLIP.
              </p>
            </div>
          </div>
        </div>
      </main>

      <AppFooter />
    </div>
  );
}
