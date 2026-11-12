"use client";

import React from "react";
import Link from "next/link";
import { usePathname, useRouter } from "next/navigation";
import { BrandLogo } from "@/components/ui/BrandLogo";
import { useAnalysis } from "@/context/AnalysisContext";

export function AppHeader() {
  const pathname = usePathname();
  const router = useRouter();
  const { loadSampleDemo } = useAnalysis();

  const isAnalyzeActive =
    pathname.startsWith("/analyze") || pathname === "/";
  const isExamplesActive = pathname === "/examples";
  const isLibraryActive = pathname === "/library";
  const isResearchActive = pathname === "/research";

  const handleNewAnalysis = () => {
    loadSampleDemo();
    router.push("/analyze");
  };

  return (
    <header className="relative z-20 w-full border-b border-[#E5E7EB]/80 bg-white/90 backdrop-blur-md">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 h-18 flex items-center justify-between">
        {/* Left: Brand Logo */}
        <div className="flex-shrink-0">
          <BrandLogo size="md" />
        </div>

        {/* Center: Main Navigation Pill */}
        <nav
          className="hidden md:flex items-center gap-1 bg-[#F8F9FA] p-1.5 rounded-full border border-[#E5E7EB]/70 shadow-xs"
          aria-label="Main Navigation"
        >
          <Link
            href="/analyze"
            className={`px-5 py-1.5 text-sm font-semibold rounded-full transition-all duration-200 ${
              isAnalyzeActive
                ? "bg-[#000000] text-white shadow-xs"
                : "text-[#6B7280] hover:text-[#111111] hover:bg-white/80"
            }`}
          >
            Analyze
          </Link>

          <Link
            href="/examples"
            className={`px-5 py-1.5 text-sm font-semibold rounded-full transition-all duration-200 ${
              isExamplesActive
                ? "bg-[#000000] text-white shadow-xs"
                : "text-[#6B7280] hover:text-[#111111] hover:bg-white/80"
            }`}
          >
            Examples
          </Link>

          <Link
            href="/library"
            className={`px-5 py-1.5 text-sm font-semibold rounded-full transition-all duration-200 ${
              isLibraryActive
                ? "bg-[#000000] text-white shadow-xs"
                : "text-[#6B7280] hover:text-[#111111] hover:bg-white/80"
            }`}
          >
            Library
          </Link>

          <Link
            href="/research"
            className={`px-5 py-1.5 text-sm font-semibold rounded-full transition-all duration-200 ${
              isResearchActive
                ? "bg-[#000000] text-white shadow-xs"
                : "text-[#6B7280] hover:text-[#111111] hover:bg-white/80"
            }`}
          >
            Research
          </Link>
        </nav>

        {/* Right: Action or New Analysis */}
        <div className="flex items-center gap-3">
          {pathname.includes("/results") ? (
            <button
              onClick={handleNewAnalysis}
              className="inline-flex items-center gap-1.5 px-4 py-2 text-xs sm:text-sm font-semibold text-[#111111] bg-white border border-[#E5E7EB] hover:border-[#111111] rounded-full shadow-2xs hover:shadow-xs transition-all active:scale-[0.98]"
            >
              <svg
                width="14"
                height="14"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2.5"
                strokeLinecap="round"
                strokeLinejoin="round"
                aria-hidden="true"
              >
                <path d="M12 5v14M5 12h14" />
              </svg>
              <span>New analysis</span>
            </button>
          ) : (
            <Link
              href="/analyze"
              className="inline-flex items-center gap-1.5 px-4 py-2 text-xs sm:text-sm font-semibold text-white bg-[#000000] hover:bg-[#222222] rounded-full shadow-2xs hover:shadow-xs transition-all active:scale-[0.98]"
            >
              <span>Launch Demo</span>
              <svg
                width="12"
                height="12"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2.5"
                strokeLinecap="round"
                strokeLinejoin="round"
                aria-hidden="true"
              >
                <path d="M5 12h14M12 5l7 7-7 7" />
              </svg>
            </Link>
          )}
        </div>
      </div>

      {/* Mobile Navigation Row */}
      <div className="flex md:hidden items-center justify-around px-4 py-2 bg-[#F8F9FA] border-t border-[#E5E7EB]/60">
        <Link
          href="/analyze"
          className={`px-3 py-1 text-xs font-semibold rounded-full ${
            isAnalyzeActive ? "bg-[#000000] text-white" : "text-[#6B7280]"
          }`}
        >
          Analyze
        </Link>
        <Link
          href="/examples"
          className={`px-3 py-1 text-xs font-semibold rounded-full ${
            isExamplesActive ? "bg-[#000000] text-white" : "text-[#6B7280]"
          }`}
        >
          Examples
        </Link>
        <Link
          href="/library"
          className={`px-3 py-1 text-xs font-semibold rounded-full ${
            isLibraryActive ? "bg-[#000000] text-white" : "text-[#6B7280]"
          }`}
        >
          Library
        </Link>
        <Link
          href="/research"
          className={`px-3 py-1 text-xs font-semibold rounded-full ${
            isResearchActive ? "bg-[#000000] text-white" : "text-[#6B7280]"
          }`}
        >
          Research
        </Link>
      </div>
    </header>
  );
}
