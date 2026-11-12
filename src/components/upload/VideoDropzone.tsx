"use client";

import React, { useRef, useState, DragEvent, ChangeEvent } from "react";

interface VideoDropzoneProps {
  onVideoSelected: (file: File) => void;
  onUseSampleDemo: () => void;
  disabled?: boolean;
}

export function VideoDropzone({
  onVideoSelected,
  onUseSampleDemo,
  disabled = false,
}: VideoDropzoneProps) {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [errorMessage, setErrorMessage] = useState<string | null>(null);

  const handleDragOver = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    if (!disabled) setIsDragging(true);
  };

  const handleDragLeave = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  const handleDrop = (e: DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    if (disabled) return;

    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      const file = e.dataTransfer.files[0];
      validateAndSelect(file);
    }
  };

  const handleFileInputChange = (e: ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const file = e.target.files[0];
      validateAndSelect(file);
    }
  };

  const validateAndSelect = (file: File) => {
    setErrorMessage(null);
    if (!file.type.includes("mp4") && !file.name.toLowerCase().endsWith(".mp4")) {
      setErrorMessage("Please upload an MP4 video file.");
      return;
    }
    onVideoSelected(file);
  };

  return (
    <div className="w-full">
      <input
        ref={fileInputRef}
        type="file"
        accept="video/mp4,video/*"
        className="hidden"
        onChange={handleFileInputChange}
        disabled={disabled}
      />

      <div
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => !disabled && fileInputRef.current?.click()}
        className={`w-full rounded-2xl border-2 border-dashed transition-all duration-200 cursor-pointer p-8 sm:p-10 flex flex-col items-center justify-center text-center select-none ${
          isDragging
            ? "border-[#FE2C55] bg-[#FFEBEF]/40 scale-[0.99]"
            : "border-[#E5E7EB] hover:border-[#FE2C55]/50 bg-white hover:bg-[#F8F9FA]"
        }`}
      >
        {/* Pink Cloud Upload Icon in Soft Pink Rounded Container */}
        <div className="w-16 h-16 rounded-2xl bg-[#FFEBEF] flex items-center justify-center text-[#FE2C55] mb-4 shadow-2xs group-hover:scale-105 transition-transform">
          <svg
            width="32"
            height="32"
            viewBox="0 0 24 24"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.8"
            strokeLinecap="round"
            strokeLinejoin="round"
            aria-hidden="true"
          >
            <path d="M4 14.899A7 7 0 1 1 15.71 8h1.79a4.5 4.5 0 0 1 2.5 8.242" />
            <path d="M12 12v9" />
            <path d="m16 16-4-4-4 4" />
          </svg>
        </div>

        <h3 className="text-base sm:text-lg font-bold text-[#111111] mb-1">
          Drop a video or click to upload
        </h3>

        <p className="text-xs sm:text-sm text-[#6B7280]">
          MP4 only. Up to 5 minutes (≤ 300 seconds).
        </p>

        {errorMessage && (
          <div className="mt-3 text-xs font-semibold text-[#DC2626] bg-[#FEE2E2] px-3 py-1.5 rounded-lg">
            {errorMessage}
          </div>
        )}
      </div>

      <div className="flex items-center justify-center gap-2 mt-3 text-xs text-[#6B7280]">
        <span>No video on hand?</span>
        <button
          type="button"
          onClick={onUseSampleDemo}
          className="text-[#FE2C55] font-semibold hover:underline cursor-pointer"
        >
          Use bundled perfume advertisement demo
        </button>
      </div>
    </div>
  );
}

