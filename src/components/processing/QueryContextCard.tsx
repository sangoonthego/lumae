"use client";

import React from "react";

interface QueryContextCardProps {
  query: string;
}

export function QueryContextCard({ query }: QueryContextCardProps) {
  const displayQuery = query || "The person demonstrates how to spray and use the perfume.";

  return (
    <div className="w-full bg-white rounded-2xl p-4 sm:p-5 border border-[#E5E7EB] shadow-xs flex flex-col gap-2 transition-all">
      <div className="flex items-center gap-2">
        <div className="w-7 h-7 rounded-xl bg-[#FFEBEF] text-[#FE2C55] flex items-center justify-center flex-shrink-0">
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
            <circle cx="11" cy="11" r="8" />
            <line x1="21" y1="21" x2="16.65" y2="16.65" />
          </svg>
        </div>
        <span className="text-xs font-bold text-[#6B7280] uppercase tracking-wider">
          Search query
        </span>
      </div>

      <p className="text-xs sm:text-sm font-semibold text-[#111111] leading-relaxed break-words pl-0.5">
        &ldquo;{displayQuery}&rdquo;
      </p>
    </div>
  );
}

