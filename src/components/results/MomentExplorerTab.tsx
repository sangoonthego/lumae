"use client";

import React from "react";
import Image from "next/image";
import { KeyframeFrame, MomentCandidate } from "@/lib/mockData";

interface MomentExplorerTabProps {
  keyframes: KeyframeFrame[];
  candidates: MomentCandidate[];
  activeCandidate: MomentCandidate;
  onSeekTo: (seconds: number) => void;
  onSelectCandidate: (candidate: MomentCandidate, autoPlay?: boolean) => void;
}

export function MomentExplorerTab({
  keyframes,
  candidates,
  activeCandidate,
  onSeekTo,
  onSelectCandidate,
}: MomentExplorerTabProps) {
  const top3 = candidates.slice(0, 3);

  return (
    <div className="grid grid-cols-1 lg:grid-cols-12 gap-6 items-start">
      {/* Left: 3 Keyframe Cards (Before, Inside Top Match, After) */}
      <div className="lg:col-span-7 grid grid-cols-1 sm:grid-cols-3 gap-4">
        {keyframes.map((frame) => {
          const isInside = frame.type === "inside";

          return (
            <div
              key={frame.id}
              onClick={() => onSeekTo(frame.timestampSeconds)}
              className={`bg-white rounded-2xl p-2.5 transition-all duration-200 cursor-pointer group flex flex-col ${
                isInside
                  ? "border-2 border-[#FE2C55] shadow-sm hover:shadow-md ring-2 ring-[#FE2C55]/20"
                  : "border border-[#E5E7EB] hover:border-[#111111]/30 hover:shadow-xs"
              }`}
            >
              {/* Image container */}
              <div className="relative aspect-[4/3] w-full rounded-xl overflow-hidden bg-black/5 mb-2.5">
                <Image
                  src={frame.imageSrc}
                  alt={frame.label}
                  fill
                  sizes="(max-width: 640px) 100vw, 220px"
                  className="object-cover group-hover:scale-105 transition-transform duration-300"
                />
                {/* Timestamp badge */}
                <span className="absolute bottom-1.5 right-1.5 bg-black/75 backdrop-blur-xs text-white text-[10px] font-mono px-1.5 py-0.5 rounded-sm">
                  {frame.timestampFormatted}
                </span>
              </div>

              {/* Labels */}
              <div className="flex flex-col px-1 pb-1">
                <span
                  className={`text-xs font-bold leading-tight ${
                    isInside ? "text-[#FE2C55]" : "text-[#111111]"
                  }`}
                >
                  {frame.label}
                </span>
                <span className="text-[11px] font-mono text-[#6B7280] mt-0.5">
                  {frame.sublabel}
                </span>
              </div>
            </div>
          );
        })}
      </div>

      {/* Right: Candidate Moments (Top 3) */}
      <div className="lg:col-span-5 bg-white rounded-3xl p-5 sm:p-6 border border-[#E5E7EB] shadow-xs flex flex-col gap-3">
        {/* Header */}
        <div className="flex items-center gap-2 mb-1">
          <svg
            width="18"
            height="18"
            viewBox="0 0 24 24"
            fill="none"
            stroke="#FE2C55"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
            aria-hidden="true"
          >
            <polygon points="12 2 2 7 12 12 22 7 12 2" />
            <polyline points="2 17 12 22 22 17" />
            <polyline points="2 12 12 17 22 12" />
          </svg>
          <h3 className="text-sm sm:text-base font-bold text-[#111111]">
            Candidate Moments (Top 3)
          </h3>
        </div>

        {/* Top 3 List */}
        <div className="flex flex-col gap-2">
          {top3.map((cand) => {
            const isSelected = activeCandidate.id === cand.id;
            const isTop = cand.isTopMatch;

            return (
              <div
                key={cand.id}
                onClick={() => onSelectCandidate(cand, true)}
                className={`p-3 rounded-2xl border transition-all duration-200 cursor-pointer flex items-center justify-between gap-3 ${
                  isSelected || isTop
                    ? "bg-[#FFF0F5] border-[#FE2C55]/25 shadow-2xs"
                    : "bg-white hover:bg-[#F8F9FA] border-[#E5E7EB]"
                }`}
              >
                {/* Left: Rank & Interval */}
                <div className="flex items-center gap-3">
                  <span className="text-xs font-mono font-bold text-[#6B7280]">
                    {cand.label}
                  </span>
                  <span className="text-sm font-mono font-black text-[#111111]">
                    {cand.startFormatted} – {cand.endFormatted}
                  </span>
                </div>

                {/* Right: Tag */}
                <div>
                  {isTop ? (
                    <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-bold bg-[#FFEBEF] text-[#FE2C55] border border-[#FE2C55]/15">
                      Top match
                    </span>
                  ) : (
                    <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-semibold bg-[#F3F4F6] text-[#6B7280]">
                      Alternative
                    </span>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

