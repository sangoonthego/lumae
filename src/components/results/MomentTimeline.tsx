"use client";

import React, { useState } from "react";
import { MomentCandidate } from "@/lib/mockData";

interface MomentTimelineProps {
  durationSeconds: number;
  candidates: MomentCandidate[];
  activeCandidate: MomentCandidate;
  currentTime: number;
  onSelectCandidate: (candidate: MomentCandidate, autoPlay?: boolean) => void;
  onSeek: (time: number) => void;
}

export function MomentTimeline({
  durationSeconds = 25.0,
  candidates,
  activeCandidate,
  currentTime,
  onSelectCandidate,
  onSeek,
}: MomentTimelineProps) {
  const [hoveredCandidate, setHoveredCandidate] = useState<MomentCandidate | null>(null);

  const duration = durationSeconds > 0 ? durationSeconds : 25.0;
  const topMatch = candidates.find((c) => c.isTopMatch) || candidates[0];
  const otherCandidates = candidates.filter((c) => !c.isTopMatch).slice(0, 4);

  const timeTicks = [0, 5, 10, 15, 20, 25];

  const handleTrackClick = (e: React.MouseEvent<HTMLDivElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const clickX = e.clientX - rect.left;
    const pct = Math.max(0, Math.min(1, clickX / rect.width));
    onSeek(pct * duration);
  };

  return (
    <div className="w-full bg-white rounded-3xl p-5 sm:p-6 border border-[#E5E7EB] shadow-xs flex flex-col gap-4">
      {/* Header Row: Title, Range, Legend */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3">
        <div className="flex items-center gap-3">
          <h2 className="text-base sm:text-lg font-bold text-[#111111]">
            Moment Timeline
          </h2>
          <span className="text-xs font-mono font-medium text-[#6B7280]">
            00:00 – 00:{Math.floor(duration).toString().padStart(2, "0")}
          </span>
        </div>

        {/* Legend */}
        <div className="flex items-center gap-4 text-xs font-medium text-[#6B7280]">
          <div className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-xs bg-[#FE2C55]" />
            <span>Top match</span>
          </div>
          <div className="flex items-center gap-1.5">
            <span className="w-2.5 h-2.5 rounded-xs bg-[#CBD5E1]" />
            <span>Other candidates</span>
          </div>
        </div>
      </div>

      {/* Timeline Interactive Track */}
      <div className="relative pt-2 pb-6">
        <div
          onClick={handleTrackClick}
          className="relative w-full h-4 bg-[#E5E7EB] rounded-full cursor-pointer overflow-visible select-none transition-all group"
        >
          {/* Other candidates (gray pills) */}
          {otherCandidates.map((candidate) => {
            const isSelected = activeCandidate.id === candidate.id;
            const leftPct = (candidate.startSeconds / duration) * 100;
            const widthPct =
              ((candidate.endSeconds - candidate.startSeconds) / duration) * 100;

            return (
              <div
                key={candidate.id}
                onClick={(e) => {
                  e.stopPropagation();
                  onSelectCandidate(candidate, true);
                }}
                onMouseEnter={() => setHoveredCandidate(candidate)}
                onMouseLeave={() => setHoveredCandidate(null)}
                className={`absolute top-0 bottom-0 rounded-full cursor-pointer transition-all ${
                  isSelected
                    ? "bg-[#64748B] ring-2 ring-[#000000] z-20"
                    : "bg-[#CBD5E1] hover:bg-[#94A3B8] z-10"
                }`}
                style={{
                  left: `${leftPct}%`,
                  width: `${widthPct}%`,
                }}
              />
            );
          })}

          {/* Top-1 Match interval (Pink with circular handles) */}
          {topMatch && (
            <div
              onClick={(e) => {
                e.stopPropagation();
                onSelectCandidate(topMatch, true);
              }}
              onMouseEnter={() => setHoveredCandidate(topMatch)}
              onMouseLeave={() => setHoveredCandidate(null)}
              className={`absolute top-0 bottom-0 bg-[#FE2C55] rounded-full cursor-pointer z-30 transition-all ${
                activeCandidate.id === topMatch.id
                  ? "shadow-[0_0_12px_#FE2C55]"
                  : "opacity-90 hover:opacity-100"
              }`}
              style={{
                left: `${(topMatch.startSeconds / duration) * 100}%`,
                width: `${
                  ((topMatch.endSeconds - topMatch.startSeconds) / duration) * 100
                }%`,
              }}
            >
              {/* Left circular handle */}
              <span className="absolute -left-1.5 top-1/2 -translate-y-1/2 w-4 h-4 bg-white border-2 border-[#FE2C55] rounded-full shadow-xs" />
              {/* Right circular handle */}
              <span className="absolute -right-1.5 top-1/2 -translate-y-1/2 w-4 h-4 bg-white border-2 border-[#FE2C55] rounded-full shadow-xs" />
            </div>
          )}

          {/* Current Video Playhead */}
          <div
            className="absolute -top-1.5 -bottom-1.5 w-1 bg-[#111111] rounded-full z-40 pointer-events-none shadow-sm transition-all"
            style={{
              left: `${(currentTime / duration) * 100}%`,
            }}
          >
            <div className="absolute -top-1 -left-1 w-3 h-3 bg-[#111111] rounded-full" />
          </div>
        </div>

        {/* Hover Tooltip */}
        {hoveredCandidate && (
          <div
            className="absolute -top-8 px-2.5 py-1 rounded-md bg-[#111111] text-white text-[11px] font-mono font-semibold pointer-events-none transform -translate-x-1/2 z-50 whitespace-nowrap shadow-md"
            style={{
              left: `${
                ((hoveredCandidate.startSeconds + hoveredCandidate.endSeconds) /
                  (2 * duration)) *
                100
              }%`,
            }}
          >
            {hoveredCandidate.isTopMatch ? "Top match: " : "Candidate: "}
            {hoveredCandidate.startFormatted} – {hoveredCandidate.endFormatted}
          </div>
        )}

        {/* Time Tick Labels below track */}
        <div className="relative w-full flex justify-between mt-3 text-xs font-mono text-[#9CA3AF] select-none">
          {timeTicks.map((tick) => (
            <span key={tick} className="text-center">
              {tick}s
            </span>
          ))}
        </div>
      </div>
    </div>
  );
}

