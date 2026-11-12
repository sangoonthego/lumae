"use client";

import React from "react";
import Image from "next/image";
import { MockVideoData } from "@/lib/mockData";

interface UploadedVideoCardProps {
  video: MockVideoData;
  onRemove: () => void;
}

export function UploadedVideoCard({ video, onRemove }: UploadedVideoCardProps) {
  return (
    <div className="w-full bg-white rounded-2xl p-3 sm:p-4 border border-[#E5E7EB] shadow-xs flex items-center justify-between gap-4 transition-all">
      {/* Left: Thumbnail with duration badge */}
      <div className="flex items-center gap-4 min-w-0">
        <div className="relative w-28 sm:w-36 h-16 sm:h-20 rounded-xl overflow-hidden bg-black flex-shrink-0 border border-[#E5E7EB]/80">
          <Image
            src={video.thumbnailSrc || "/mock/perfume_ad_thumb.jpg"}
            alt={video.filename || "Video thumbnail"}
            fill
            sizes="(max-width: 640px) 112px, 144px"
            className="object-cover"
          />
          {/* Duration overlay badge */}
          <div className="absolute bottom-1.5 right-1.5 bg-black/75 backdrop-blur-xs text-white text-[10px] sm:text-xs font-semibold px-1.5 py-0.5 rounded-sm flex items-center gap-1">
            <svg
              width="8"
              height="8"
              viewBox="0 0 24 24"
              fill="currentColor"
              aria-hidden="true"
            >
              <path d="M5 3.868v16.264c0 .858.94 1.383 1.666.936l13.011-8.132a1.096 1.096 0 0 0 0-1.872L6.666 2.932C5.94 2.485 5 3.01 5 3.868Z" />
            </svg>
            <span>{video.durationFormatted}</span>
          </div>
        </div>

        {/* Video metadata */}
        <div className="min-w-0 flex flex-col justify-center">
          <div className="font-bold text-sm sm:text-base text-[#111111] truncate mb-1">
            {video.filename}
          </div>

          <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-xs text-[#6B7280]">
            {/* Duration */}
            <div className="flex items-center gap-1">
              <svg
                width="13"
                height="13"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
                aria-hidden="true"
              >
                <circle cx="12" cy="12" r="10" />
                <polyline points="12 6 12 12 16 14" />
              </svg>
              <span>{video.durationFormatted}</span>
            </div>

            {/* Resolution */}
            <div className="flex items-center gap-1">
              <svg
                width="13"
                height="13"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
                aria-hidden="true"
              >
                <rect width="20" height="14" x="2" y="3" rx="2" />
                <line x1="8" x2="16" y1="21" y2="21" />
                <line x1="12" x2="12" y1="17" y2="21" />
              </svg>
              <span>{video.resolution}</span>
            </div>

            {/* Size */}
            <div className="flex items-center gap-1">
              <svg
                width="13"
                height="13"
                viewBox="0 0 24 24"
                fill="none"
                stroke="currentColor"
                strokeWidth="2"
                strokeLinecap="round"
                strokeLinejoin="round"
                aria-hidden="true"
              >
                <ellipse cx="12" cy="5" rx="9" ry="3" />
                <path d="M3 5V19A9 3 0 0 0 21 19V5" />
                <path d="M3 12A9 3 0 0 0 21 12" />
              </svg>
              <span>{video.fileSizeFormatted}</span>
            </div>
          </div>
        </div>
      </div>

      {/* Right: Divider + Remove Button */}
      <div className="flex items-center gap-3 pl-2 sm:pl-4 border-l border-[#E5E7EB]">
        <button
          type="button"
          onClick={onRemove}
          className="w-9 h-9 sm:w-10 sm:h-10 rounded-xl bg-white hover:bg-[#FFEBEF] text-[#6B7280] hover:text-[#FE2C55] border border-[#E5E7EB] hover:border-[#FE2C55]/30 flex items-center justify-center transition-all cursor-pointer"
          aria-label="Remove uploaded video"
          title="Remove video"
        >
          <svg
            width="18"
            height="18"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            strokeLinejoin="round"
            aria-hidden="true"
          >
            <path d="M3 6h18" />
            <path d="M19 6v14c0 1-1 2-2 2H7c-1 0-2-1-2-2V6" />
            <path d="M8 6V4c0-1 1-2 2-2h4c1 0 2 1 2 2v2" />
            <line x1="10" x2="10" y1="11" y2="17" />
            <line x1="14" x2="14" y1="11" y2="17" />
          </svg>
        </button>
      </div>
    </div>
  );
}

