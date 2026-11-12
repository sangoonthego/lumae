"use client";

import React, { useRef, useState } from "react";
import { MockVideoData } from "@/lib/mockData";

interface ProcessingVideoPreviewProps {
  video: MockVideoData;
}

export function ProcessingVideoPreview({ video }: ProcessingVideoPreviewProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [isMuted, setIsMuted] = useState(false);

  const duration = video.durationSeconds || 25.0;

  const handleTogglePlay = () => {
    if (!videoRef.current) return;
    if (isPlaying) {
      videoRef.current.pause();
      setIsPlaying(false);
    } else {
      videoRef.current.play().then(() => setIsPlaying(true)).catch(() => {});
    }
  };

  const handleTimeUpdate = () => {
    if (videoRef.current) {
      setCurrentTime(videoRef.current.currentTime);
    }
  };

  const handleToggleMute = () => {
    if (!videoRef.current) return;
    videoRef.current.muted = !isMuted;
    setIsMuted(!isMuted);
  };

  const handleToggleFullscreen = () => {
    if (!videoRef.current) return;
    if (document.fullscreenElement) {
      document.exitFullscreen().catch(() => {});
    } else {
      videoRef.current.requestFullscreen().catch(() => {});
    }
  };

  const formatTime = (sec: number) => {
    const m = Math.floor(sec / 60);
    const s = Math.floor(sec % 60);
    return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
  };

  return (
    <div className="w-full flex flex-col gap-2.5">
      {/* Video Container */}
      <div className="relative w-full rounded-3xl overflow-hidden bg-[#0D1117] shadow-[0_15px_35px_rgba(0,0,0,0.12)] border border-[#E5E7EB] group aspect-[4/3] sm:aspect-[16/10] flex items-center justify-center">
        {/* Video / Poster */}
        <video
          ref={videoRef}
          src={video.videoSrc || "/mock/perfume_ad.mp4"}
          poster={video.posterSrc || "/mock/perfume_bottle.jpg"}
          onTimeUpdate={handleTimeUpdate}
          onEnded={() => setIsPlaying(false)}
          playsInline
          preload="metadata"
          className="w-full h-full object-cover cursor-pointer"
          onClick={handleTogglePlay}
        />

        {/* Centered Play Button when paused */}
        {!isPlaying && (
          <button
            type="button"
            onClick={handleTogglePlay}
            className="absolute inset-0 m-auto w-16 h-16 rounded-full bg-black/45 hover:bg-black/65 backdrop-blur-xs border border-white/40 flex items-center justify-center text-white transition-all transform hover:scale-105 active:scale-95 cursor-pointer shadow-lg z-10"
            aria-label="Play preview video"
          >
            <svg
              width="24"
              height="24"
              viewBox="0 0 24 24"
              fill="currentColor"
              className="ml-0.5"
              aria-hidden="true"
            >
              <path d="M5 3.868v16.264c0 .858.94 1.383 1.666.936l13.011-8.132a1.096 1.096 0 0 0 0-1.872L6.666 2.932C5.94 2.485 5 3.01 5 3.868Z" />
            </svg>
          </button>
        )}

        {/* Bottom Video Controls Overlay */}
        <div className="absolute bottom-0 inset-x-0 bg-gradient-to-t from-black/90 via-black/50 to-transparent p-3.5 sm:p-4 pt-8 text-white flex flex-col gap-2 z-10">
          <div className="flex items-center gap-3">
            {/* Play/Pause Button */}
            <button
              type="button"
              onClick={handleTogglePlay}
              className="text-white hover:text-[#25F4EE] transition-colors cursor-pointer"
              aria-label={isPlaying ? "Pause" : "Play"}
            >
              {isPlaying ? (
                <svg width="15" height="15" viewBox="0 0 24 24" fill="currentColor">
                  <rect x="6" y="4" width="4" height="16" />
                  <rect x="14" y="4" width="4" height="16" />
                </svg>
              ) : (
                <svg width="15" height="15" viewBox="0 0 24 24" fill="currentColor">
                  <polygon points="5 3 19 12 5 21 5 3" />
                </svg>
              )}
            </button>

            {/* Time display */}
            <span className="text-xs font-mono text-white/90">
              {formatTime(currentTime)} / {formatTime(duration)}
            </span>

            {/* Scrub / Progress bar */}
            <div
              className="flex-1 h-2 bg-white/30 rounded-full cursor-pointer relative overflow-visible flex items-center group/track"
              onClick={(e) => {
                const rect = e.currentTarget.getBoundingClientRect();
                const p = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
                if (videoRef.current) {
                  videoRef.current.currentTime = p * duration;
                  setCurrentTime(p * duration);
                }
              }}
            >
              <div
                className="h-full bg-white rounded-full relative"
                style={{ width: `${(currentTime / duration) * 100}%` }}
              >
                <span className="absolute -right-1.5 top-1/2 -translate-y-1/2 w-3.5 h-3.5 bg-[#FE2C55] rounded-full border-2 border-white shadow-xs" />
              </div>
            </div>

            {/* Volume Mute/Unmute */}
            <button
              type="button"
              onClick={handleToggleMute}
              className="text-white/80 hover:text-white transition-colors cursor-pointer"
              aria-label={isMuted ? "Unmute" : "Mute"}
            >
              {isMuted ? (
                <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5" />
                  <line x1="23" y1="9" x2="17" y2="15" />
                  <line x1="17" y1="9" x2="23" y2="15" />
                </svg>
              ) : (
                <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5" />
                  <path d="M15.54 8.46a5 5 0 0 1 0 7.07" />
                </svg>
              )}
            </button>

            {/* Fullscreen */}
            <button
              type="button"
              onClick={handleToggleFullscreen}
              className="text-white/80 hover:text-white transition-colors cursor-pointer"
              aria-label="Fullscreen"
            >
              <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <polyline points="15 3 21 3 21 9" />
                <polyline points="9 21 3 21 3 15" />
                <line x1="21" x2="14" y1="3" y2="10" />
                <line x1="3" x2="10" y1="21" y2="14" />
              </svg>
            </button>
          </div>
        </div>
      </div>

      {/* Video Metadata line */}
      <div className="text-xs font-mono text-[#6B7280] text-center px-1">
        Target video: <span className="font-semibold text-[#111111]">{video.filename || "perfume_ad.mp4"}</span> · {video.durationFormatted || "25.0s"} span
      </div>
    </div>
  );
}
