"use client";

import React, { useRef, useState } from "react";

export function HeroVideoPreview() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [isPlaying, setIsPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(5.4);
  const duration = 25.0;

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

  const handleSeek = (time: number) => {
    if (videoRef.current) {
      videoRef.current.currentTime = time;
      setCurrentTime(time);
    }
  };

  const formatTime = (sec: number) => {
    const m = Math.floor(sec / 60);
    const s = Math.floor(sec % 60);
    return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
  };

  return (
    <div className="relative w-full max-w-lg lg:max-w-xl mx-auto">
      {/* Main Video Card Container with slight tilt/elevation matching Reference 1 */}
      <div className="relative z-10 bg-white rounded-3xl p-3 sm:p-4 shadow-[0_20px_50px_rgba(0,0,0,0.12)] border border-[#E5E7EB] transform md:rotate-1 hover:rotate-0 transition-transform duration-500 overflow-hidden group">
        <div className="relative aspect-[4/3] sm:aspect-[16/10] w-full rounded-2xl overflow-hidden bg-black">
          {/* Video element */}
          <video
            ref={videoRef}
            src="/mock/perfume_ad.mp4"
            poster="/mock/perfume_bottle.jpg"
            onTimeUpdate={handleTimeUpdate}
            onEnded={() => setIsPlaying(false)}
            playsInline
            preload="metadata"
            className="w-full h-full object-cover cursor-pointer"
            onClick={handleTogglePlay}
          />

          {/* Centered Translucent Play Button (visible when paused) */}
          {!isPlaying && (
            <button
              onClick={handleTogglePlay}
              className="absolute inset-0 m-auto w-16 h-16 sm:w-20 sm:h-20 rounded-full bg-black/40 hover:bg-black/60 backdrop-blur-md border border-white/40 flex items-center justify-center text-white transition-all transform hover:scale-105 active:scale-95 shadow-lg group-hover:bg-black/50"
              aria-label="Play sample video"
            >
              <svg
                width="28"
                height="28"
                viewBox="0 0 24 24"
                fill="currentColor"
                className="ml-1"
                aria-hidden="true"
              >
                <path d="M5 3.868v16.264c0 .858.94 1.383 1.666.936l13.011-8.132a1.096 1.096 0 0 0 0-1.872L6.666 2.932C5.94 2.485 5 3.01 5 3.868Z" />
              </svg>
            </button>
          )}

          {/* Bottom video timeline bar overlay */}
          <div className="absolute bottom-0 inset-x-0 bg-gradient-to-t from-black/80 via-black/40 to-transparent p-3 sm:p-4 pt-8">
            <div className="flex items-center justify-between text-white/90 text-xs font-mono font-medium mb-1.5">
              <span>{formatTime(currentTime)} / 00:25</span>
              <span className="text-[#FE2C55] font-semibold text-[11px] bg-white/20 backdrop-blur-xs px-2 py-0.5 rounded-full">
                00:05.4 – 00:07.7
              </span>
            </div>

            {/* Timeline track */}
            <div
              className="relative w-full h-3 bg-white/30 rounded-full cursor-pointer overflow-visible flex items-center"
              onClick={(e) => {
                const rect = e.currentTarget.getBoundingClientRect();
                const pos = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
                handleSeek(pos * duration);
              }}
            >
              {/* Highlighted Top-1 Moment (5.4s to 7.7s of 25s = 21.6% to 30.8%) */}
              <div
                className="absolute top-0 bottom-0 bg-[#FE2C55] rounded-full shadow-[0_0_8px_#FE2C55]"
                style={{
                  left: `${(5.4 / duration) * 100}%`,
                  width: `${((7.7 - 5.4) / duration) * 100}%`,
                }}
              >
                {/* Circular handle left */}
                <span className="absolute -left-1 top-1/2 -translate-y-1/2 w-3.5 h-3.5 bg-white border-2 border-[#FE2C55] rounded-full shadow-sm" />
                {/* Circular handle right */}
                <span className="absolute -right-1 top-1/2 -translate-y-1/2 w-3.5 h-3.5 bg-white border-2 border-[#FE2C55] rounded-full shadow-sm" />
              </div>

              {/* Current playhead */}
              <div
                className="absolute top-1/2 -translate-y-1/2 w-2 h-4 bg-white rounded-xs shadow-md pointer-events-none"
                style={{
                  left: `${(currentTime / duration) * 100}%`,
                }}
              />
            </div>
          </div>
        </div>

        {/* Floating Predicted Moment Card (as in screenshot 1) */}
        <div className="absolute bottom-5 right-5 sm:bottom-6 sm:right-6 bg-white/95 backdrop-blur-md rounded-2xl p-3.5 sm:p-4 shadow-[0_12px_30px_rgba(0,0,0,0.18)] border border-[#E5E7EB] flex items-center gap-3 animate-fade-in z-20">
          <div className="w-9 h-9 rounded-xl bg-[#FFEBEF] flex items-center justify-center text-[#FE2C55] flex-shrink-0">
            <svg
              width="20"
              height="20"
              viewBox="0 0 24 24"
              fill="currentColor"
              aria-hidden="true"
            >
              <path d="M12 2l2.4 7.2L22 12l-7.6 2.8L12 22l-2.4-7.2L2 12l7.6-2.8z" />
            </svg>
          </div>
          <div>
            <div className="text-[11px] font-semibold text-[#6B7280] tracking-wide uppercase">
              Predicted moment
            </div>
            <div className="text-base sm:text-lg font-black text-[#FE2C55] tracking-tight">
              00:05.4 – 00:07.7
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
