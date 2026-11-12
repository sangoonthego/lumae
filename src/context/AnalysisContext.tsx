"use client";

import React, { createContext, useContext, useState, useEffect, useCallback, useMemo } from "react";
import {
  MockVideoData,
  MomentCandidate,
  MockAnalysisResult,
  DEFAULT_SAMPLE_VIDEO,
  DEFAULT_QUERY,
  MOCK_CANDIDATES,
  createMockAnalysisResult,
  formatSecondsToMinutes,
} from "@/lib/mockData";

interface AnalysisContextType {
  video: MockVideoData;
  customFile: File | null;
  videoUrl: string;
  query: string;
  selectedChipId: string | null;
  activeCandidate: MomentCandidate;
  analysisResult: MockAnalysisResult;
  currentTime: number;
  isPlaying: boolean;
  activeInterval: { start: number; end: number } | null;
  setQuery: (q: string) => void;
  setSelectedChipId: (id: string | null) => void;
  selectCandidate: (candidate: MomentCandidate, autoPlay?: boolean) => void;
  setCurrentTime: (time: number) => void;
  setIsPlaying: (playing: boolean) => void;
  playMoment: (candidate?: MomentCandidate) => void;
  stopPlayback: () => void;
  uploadCustomVideo: (file: File) => Promise<void>;
  loadSampleDemo: () => void;
  removeVideo: () => void;
  refineQuery: () => void;
  resetAnalysis: () => void;
}

const AnalysisContext = createContext<AnalysisContextType | null>(null);

export function AnalysisProvider({ children }: { children: React.ReactNode }) {
  const [video, setVideo] = useState<MockVideoData>(DEFAULT_SAMPLE_VIDEO);
  const [customFile, setCustomFile] = useState<File | null>(null);
  const [videoUrl, setVideoUrl] = useState<string>(DEFAULT_SAMPLE_VIDEO.videoSrc);
  const [query, setQuery] = useState<string>(DEFAULT_QUERY);
  const [selectedChipId, setSelectedChipId] = useState<string | null>("using-product");
  const [activeCandidate, setActiveCandidate] = useState<MomentCandidate>(MOCK_CANDIDATES[0]);
  const [currentTime, setCurrentTime] = useState<number>(0);
  const [isPlaying, setIsPlaying] = useState<boolean>(false);
  const [activeInterval, setActiveInterval] = useState<{ start: number; end: number } | null>({
    start: MOCK_CANDIDATES[0].startSeconds,
    end: MOCK_CANDIDATES[0].endSeconds,
  });

  const analysisResult = useMemo<MockAnalysisResult>(() => {
    return createMockAnalysisResult(query, video);
  }, [query, video]);

  // Clean up object URLs on change or unmount
  useEffect(() => {
    return () => {
      if (videoUrl.startsWith("blob:")) {
        URL.revokeObjectURL(videoUrl);
      }
    };
  }, [videoUrl]);

  const removeVideo = useCallback(() => {
    if (videoUrl.startsWith("blob:")) {
      URL.revokeObjectURL(videoUrl);
    }
    setCustomFile(null);
    setVideo({
      id: "no-video",
      filename: "",
      title: "No video selected",
      durationSeconds: 0,
      durationFormatted: "00:00",
      resolution: "",
      width: 0,
      height: 0,
      fileSizeBytes: 0,
      fileSizeFormatted: "",
      format: "",
      fps: 30,
      videoSrc: "",
      posterSrc: "",
      thumbnailSrc: "",
    });
    setVideoUrl("");
  }, [videoUrl]);

  const loadSampleDemo = useCallback(() => {
    if (videoUrl.startsWith("blob:")) {
      URL.revokeObjectURL(videoUrl);
    }
    setCustomFile(null);
    setVideo(DEFAULT_SAMPLE_VIDEO);
    setVideoUrl(DEFAULT_SAMPLE_VIDEO.videoSrc);
    setQuery(DEFAULT_QUERY);
    setSelectedChipId("using-product");
    setActiveCandidate(MOCK_CANDIDATES[0]);
    setActiveInterval({
      start: MOCK_CANDIDATES[0].startSeconds,
      end: MOCK_CANDIDATES[0].endSeconds,
    });
  }, [videoUrl]);

  const uploadCustomVideo = useCallback(
    async (file: File) => {
      if (videoUrl.startsWith("blob:")) {
        URL.revokeObjectURL(videoUrl);
      }
      const newBlobUrl = URL.createObjectURL(file);
      setCustomFile(file);
      setVideoUrl(newBlobUrl);

      // Attempt to load duration and dimensions from native HTML5 Video
      let durationSec = 25.0;
      let width = 1920;
      let height = 1080;

      try {
        const tempVideo = document.createElement("video");
        tempVideo.preload = "metadata";
        tempVideo.src = newBlobUrl;
        await new Promise<void>((resolve) => {
          tempVideo.onloadedmetadata = () => {
            if (tempVideo.duration && !isNaN(tempVideo.duration) && isFinite(tempVideo.duration)) {
              durationSec = Math.min(300, Math.max(1, tempVideo.duration));
            }
            if (tempVideo.videoWidth) width = tempVideo.videoWidth;
            if (tempVideo.videoHeight) height = tempVideo.videoHeight;
            resolve();
          };
          tempVideo.onerror = () => resolve();
          // Timeout fallback
          setTimeout(resolve, 1500);
        });
      } catch {
        // Fallback to default metadata
      }

      const sizeMB = (file.size / (1024 * 1024)).toFixed(1);

      setVideo({
        id: `custom-video-${Date.now()}`,
        filename: file.name,
        title: file.name.replace(/\.[^/.]+$/, ""),
        durationSeconds: durationSec,
        durationFormatted: formatSecondsToMinutes(durationSec),
        resolution: `${width} × ${height}`,
        width,
        height,
        fileSizeBytes: file.size,
        fileSizeFormatted: `${sizeMB} MB`,
        format: file.type ? file.type.replace("video/", "").toUpperCase() : "MP4",
        fps: 30,
        videoSrc: newBlobUrl,
        posterSrc: DEFAULT_SAMPLE_VIDEO.posterSrc,
        thumbnailSrc: DEFAULT_SAMPLE_VIDEO.thumbnailSrc,
      });
    },
    [videoUrl]
  );

  const selectCandidate = useCallback((candidate: MomentCandidate, autoPlay = false) => {
    setActiveCandidate(candidate);
    setActiveInterval({
      start: candidate.startSeconds,
      end: candidate.endSeconds,
    });
    setCurrentTime(candidate.startSeconds);
    if (autoPlay) {
      setIsPlaying(true);
    }
  }, []);

  const playMoment = useCallback(
    (candidate?: MomentCandidate) => {
      const target = candidate ?? activeCandidate;
      setActiveCandidate(target);
      setActiveInterval({
        start: target.startSeconds,
        end: target.endSeconds,
      });
      setCurrentTime(target.startSeconds);
      setIsPlaying(true);
    },
    [activeCandidate]
  );

  const stopPlayback = useCallback(() => {
    setIsPlaying(false);
  }, []);

  const refineQuery = useCallback(() => {
    // Keeps selected video, allows user to edit query on /analyze
  }, []);

  const resetAnalysis = useCallback(() => {
    loadSampleDemo();
  }, [loadSampleDemo]);

  return (
    <AnalysisContext.Provider
      value={{
        video,
        customFile,
        videoUrl,
        query,
        selectedChipId,
        activeCandidate,
        analysisResult,
        currentTime,
        isPlaying,
        activeInterval,
        setQuery,
        setSelectedChipId,
        selectCandidate,
        setCurrentTime,
        setIsPlaying,
        playMoment,
        stopPlayback,
        uploadCustomVideo,
        loadSampleDemo,
        removeVideo,
        refineQuery,
        resetAnalysis,
      }}
    >
      {children}
    </AnalysisContext.Provider>
  );
}

export function useAnalysis() {
  const context = useContext(AnalysisContext);
  if (!context) {
    throw new Error("useAnalysis must be used within an AnalysisProvider");
  }
  return context;
}
