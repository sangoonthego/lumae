"use client";

import React, { useRef, useEffect, useState, useCallback } from "react";
import { MockVideoData, MomentCandidate } from "@/lib/mockData";

interface VideoPlayerProps {
  video: MockVideoData;
  activeCandidate: MomentCandidate;
  currentTime: number;
  isPlaying: boolean;
  onTimeUpdate: (time: number) => void;
  onPlayPauseToggle: (playing: boolean) => void;
}

export function VideoPlayer({
  video,
  activeCandidate,
  currentTime,
  isPlaying,
  onTimeUpdate,
  onPlayPauseToggle,
}: VideoPlayerProps) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [duration, setDuration] = useState<number>(video.durationSeconds || 25.0);
  const [playbackIntervalOnly, setPlaybackIntervalOnly] = useState<boolean>(false);

  // Sync internal video element with external isPlaying prop
  useEffect(() => {
    if (!videoRef.current) return;
    if (isPlaying) {
      videoRef.current.play().catch(() => {});
    } else {
      videoRef.current.pause();
    }
  }, [isPlaying]);

  // Sync external currentTime changes (e.g. clicking keyframe or candidate)
  useEffect(() => {
    if (!videoRef.current) return;
    if (Math.abs(videoRef.current.currentTime - currentTime) > 0.3) {
      videoRef.current.currentTime = currentTime;
    }
  }, [currentTime]);

  const handleLoadedMetadata = () => {
    if (videoRef.current && videoRef.current.duration && !isNaN(videoRef.current.duration)) {
      setDuration(videoRef.current.duration);
    }
  };

  const handleTimeUpdateInternal = () => {
    if (!videoRef.current) return;
    const cur = videoRef.current.currentTime;
    onTimeUpdate(cur);

    // If currently playing a specific moment interval, pause once end is reached
    if (playbackIntervalOnly) {
      if (cur >= activeCandidate.endSeconds) {
        videoRef.current.pause();
        onPlayPauseToggle(false);
        setPlaybackIntervalOnly(false);
      }
    }
  };

  const handleTogglePlay = useCallback(() => {
    if (!videoRef.current) return;
    if (isPlaying) {
      videoRef.current.pause();
      onPlayPauseToggle(false);
      setPlaybackIntervalOnly(false);
    } else {
      videoRef.current.play().then(() => onPlayPauseToggle(true)).catch(() => {});
    }
  }, [isPlaying, onPlayPauseToggle]);

  const handleSeek = (time: number) => {
    if (videoRef.current) {
      videoRef.current.currentTime = time;
      onTimeUpdate(time);
    }
  };

  const formatTime = (sec: number) => {
    const m = Math.floor(sec / 60);
    const s = Math.floor(sec % 60);
    return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
  };

  // Moment interval percentage
  const intervalStartPct = (activeCandidate.startSeconds / duration) * 100;
  const intervalWidthPct =
    ((activeCandidate.endSeconds - activeCandidate.startSeconds) / duration) * 100;

  return (
    <div
      className="relative w-full rounded-3xl overflow-hidden bg-black shadow-[0_15px_35px_rgba(0,0,0,0.12)] border border-[#E5E7EB] group aspect-[16/10]"
    >
      <video
        ref={videoRef}
        src={video.videoSrc || "/mock/perfume_ad.mp4"}
        poster={video.posterSrc || "/mock/perfume_ad_spray.jpg"}
        onLoadedMetadata={handleLoadedMetadata}
        onTimeUpdate={handleTimeUpdateInternal}
        onEnded={() => onPlayPauseToggle(false)}
        playsInline
        preload="metadata"
        className="w-full h-full object-cover cursor-pointer"
        onClick={handleTogglePlay}
      />

      {/* Centered Translucent Play Button (visible when paused) */}
      {!isPlaying && (
        <button
          onClick={handleTogglePlay}
          className="absolute inset-0 m-auto w-16 h-16 rounded-full bg-black/45 hover:bg-black/65 backdrop-blur-xs border border-white/40 flex items-center justify-center text-white transition-all transform hover:scale-105 active:scale-95 cursor-pointer shadow-lg z-10"
          aria-label="Play video"
        >
          <svg
            width="26"
            height="26"
            viewBox="0 0 24 24"
            fill="currentColor"
            className="ml-0.5"
            aria-hidden="true"
          >
            <path d="M5 3.868v16.264c0 .858.94 1.383 1.666.936l13.011-8.132a1.096 1.096 0 0 0 0-1.872L6.666 2.932C5.94 2.485 5 3.01 5 3.868Z" />
          </svg>
        </button>
      )}

      {/* Bottom video controls matching Screenshot 4 */}
      <div className="absolute bottom-0 inset-x-0 bg-gradient-to-t from-black/90 via-black/50 to-transparent p-3.5 sm:p-4 pt-8 text-white flex flex-col gap-2 z-10">
        <div className="flex items-center gap-3">
          {/* Play/Pause */}
          <button
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

          {/* Timeline scrub track with highlighted moment segment */}
          <div
            className="flex-1 h-2 bg-white/30 rounded-full cursor-pointer relative overflow-visible flex items-center group/track"
            onClick={(e) => {
              const rect = e.currentTarget.getBoundingClientRect();
              const p = Math.max(0, Math.min(1, (e.clientX - rect.left) / rect.width));
              handleSeek(p * duration);
            }}
          >
            {/* Pink Highlight Interval */}
            <div
              className="absolute top-0 bottom-0 bg-[#FE2C55] rounded-full shadow-[0_0_8px_#FE2C55]"
              style={{
                left: `${intervalStartPct}%`,
                width: `${intervalWidthPct}%`,
              }}
            >
              {/* Left handle */}
              <span className="absolute -left-1 top-1/2 -translate-y-1/2 w-3 h-3 bg-white border-2 border-[#FE2C55] rounded-full shadow-xs" />
              {/* Right handle */}
              <span className="absolute -right-1 top-1/2 -translate-y-1/2 w-3 h-3 bg-white border-2 border-[#FE2C55] rounded-full shadow-xs" />
            </div>

            {/* Current Playhead */}
            <div
              className="absolute top-1/2 -translate-y-1/2 w-2 h-3.5 bg-white rounded-xs shadow-md pointer-events-none"
              style={{
                left: `${(currentTime / duration) * 100}%`,
              }}
            />
          </div>

          {/* Volume */}
          <button
            className="text-white/80 hover:text-white"
            aria-label="Volume"
            type="button"
          >
            <svg
              width="15"
              height="15"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5" />
              <path d="M15.54 8.46a5 5 0 0 1 0 7.07" />
            </svg>
          </button>

          {/* Fullscreen */}
          <button
            className="text-white/80 hover:text-white"
            aria-label="Fullscreen"
            type="button"
          >
            <svg
              width="15"
              height="15"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <polyline points="15 3 21 3 21 9" />
              <polyline points="9 21 3 21 3 15" />
              <line x1="21" x2="14" y1="3" y2="10" />
              <line x1="3" x2="10" y1="21" y2="14" />
            </svg>
          </button>
        </div>
      </div>
    </div>
  );
}
