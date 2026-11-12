export interface MomentCandidate {
  id: string;
  rank: number;
  label: string;
  isTopMatch: boolean;
  startSeconds: number;
  endSeconds: number;
  durationSeconds: number;
  startFormatted: string;
  endFormatted: string;
  rawScore?: number;
  tag: string;
  description: string;
}

export interface ProcessingStage {
  id: number;
  title: string;
  description?: string;
  activeDetails?: string;
  completedSummary?: string;
  details?: string;
  mockMeta?: { label: string; value: string; icon?: string }[];
  status: "pending" | "running" | "completed";
  durationMs: number;
}

export interface KeyframeFrame {
  id: string;
  type: "before" | "inside" | "after";
  label: string;
  sublabel: string;
  timestampSeconds: number;
  timestampFormatted: string;
  imageSrc: string;
  description: string;
}

export interface MockVideoData {
  id: string;
  filename: string;
  title: string;
  durationSeconds: number;
  durationFormatted: string;
  resolution: string;
  width: number;
  height: number;
  fileSizeBytes: number;
  fileSizeFormatted: string;
  format: string;
  fps: number;
  videoSrc: string;
  posterSrc: string;
  thumbnailSrc: string;
}

export interface TechnicalDetails {
  modelName: string;
  architecture: string;
  videoEncoder: string;
  temporalGroundingHead: string;
  task: string;
  sampleFps: string;
  proposalsCount: number;
  inferenceMode: string;
  contextWindow: string;
  lossFunction: string;
  frameworkVersion: string;
}

export interface MockAnalysisResult {
  analysisId: string;
  mode: "demo";
  query: string;
  video: MockVideoData;
  candidates: MomentCandidate[];
  topCandidate: MomentCandidate;
  keyframes: KeyframeFrame[];
  technicalDetails: TechnicalDetails;
  completedAt: string;
}

export const formatSecondsToTimestamp = (seconds: number): string => {
  const m = Math.floor(seconds / 60);
  const s = seconds % 60;
  const wholeSeconds = Math.floor(s);
  const tenths = Math.floor((s - wholeSeconds) * 10);
  return `${String(m).padStart(2, "0")}:${String(wholeSeconds).padStart(2, "0")}.${tenths}`;
};

export const formatSecondsToMinutes = (seconds: number): string => {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
};

export const DEFAULT_SAMPLE_VIDEO: MockVideoData = {
  id: "perfume-ad-01",
  filename: "perfume_ad.mp4",
  title: "LUMAE Eau De Parfum — Luxury Spring Launch Ad",
  durationSeconds: 25.0,
  durationFormatted: "00:25",
  resolution: "1920 × 1080",
  width: 1920,
  height: 1080,
  fileSizeBytes: 26004684, // 24.8 MB
  fileSizeFormatted: "24.8 MB",
  format: "MP4",
  fps: 30,
  videoSrc: "/mock/perfume_ad.mp4",
  posterSrc: "/mock/perfume_ad_spray.jpg",
  thumbnailSrc: "/mock/perfume_ad_thumb.jpg",
};

export const DEFAULT_QUERY =
  "The person demonstrates how to spray and use the perfume.";

export const QUERY_SUGGESTIONS = [
  { id: "product-close-up", label: "Product close-up", icon: "target", text: "Product close-up showing the bottle and brand logo." },
  { id: "using-product", label: "Using the product", icon: "check", text: "The person demonstrates how to spray and use the perfume." },
  { id: "opening-package", label: "Opening the package", icon: "box", text: "Unboxing the perfume bottle from its luxury packaging." },
  { id: "show-logo", label: "Show logo", icon: "sparkles", text: "Close-up hero display of the LUMAE brand logo." },
];

export const MOCK_CANDIDATES: MomentCandidate[] = [
  {
    id: "cand-1",
    rank: 1,
    label: "01",
    isTopMatch: true,
    startSeconds: 5.4,
    endSeconds: 7.7,
    durationSeconds: 2.3,
    startFormatted: "00:05.4",
    endFormatted: "00:07.7",
    rawScore: 0.942,
    tag: "Top match",
    description: "Actor holds perfume bottle, sprays fine fragrance mist toward face and neckline.",
  },
  {
    id: "cand-2",
    rank: 2,
    label: "02",
    isTopMatch: false,
    startSeconds: 12.1,
    endSeconds: 14.8,
    durationSeconds: 2.7,
    startFormatted: "00:12.1",
    endFormatted: "00:14.8",
    rawScore: 0.781,
    tag: "Alternative",
    description: "Secondary angle showing tactile bottle grip and fine mist dispersion.",
  },
  {
    id: "cand-3",
    rank: 3,
    label: "03",
    isTopMatch: false,
    startSeconds: 18.2,
    endSeconds: 20.6,
    durationSeconds: 2.4,
    startFormatted: "00:18.2",
    endFormatted: "00:20.6",
    rawScore: 0.635,
    tag: "Alternative",
    description: "Lifestyle cut demonstrating casual application before evening out.",
  },
  {
    id: "cand-4",
    rank: 4,
    label: "04",
    isTopMatch: false,
    startSeconds: 0.8,
    endSeconds: 3.2,
    durationSeconds: 2.4,
    startFormatted: "00:00.8",
    endFormatted: "00:03.2",
    rawScore: 0.492,
    tag: "Candidate",
    description: "Opening beauty hero shot with gentle camera push-in on glass flacon.",
  },
  {
    id: "cand-5",
    rank: 5,
    label: "05",
    isTopMatch: false,
    startSeconds: 3.8,
    endSeconds: 5.2,
    durationSeconds: 1.4,
    startFormatted: "00:03.8",
    endFormatted: "00:05.2",
    rawScore: 0.448,
    tag: "Candidate",
    description: "Hand reaches into frame, lifting bottle from marble pedestal.",
  },
  {
    id: "cand-6",
    rank: 6,
    label: "06",
    isTopMatch: false,
    startSeconds: 7.9,
    endSeconds: 10.1,
    durationSeconds: 2.2,
    startFormatted: "00:07.9",
    endFormatted: "00:10.1",
    rawScore: 0.412,
    tag: "Candidate",
    description: "Satisfied expression and subtle head tilt catching natural daylight.",
  },
  {
    id: "cand-7",
    rank: 7,
    label: "07",
    isTopMatch: false,
    startSeconds: 10.4,
    endSeconds: 11.9,
    durationSeconds: 1.5,
    startFormatted: "00:10.4",
    endFormatted: "00:11.9",
    rawScore: 0.384,
    tag: "Candidate",
    description: "Macro focus on gold nozzle atomizer with droplet reflection.",
  },
  {
    id: "cand-8",
    rank: 8,
    label: "08",
    isTopMatch: false,
    startSeconds: 15.0,
    endSeconds: 17.5,
    durationSeconds: 2.5,
    startFormatted: "00:15.0",
    endFormatted: "00:17.5",
    rawScore: 0.341,
    tag: "Candidate",
    description: "Soft floral background reveal framing the glass bottle geometry.",
  },
  {
    id: "cand-9",
    rank: 9,
    label: "09",
    isTopMatch: false,
    startSeconds: 21.0,
    endSeconds: 23.4,
    durationSeconds: 2.4,
    startFormatted: "00:21.0",
    endFormatted: "00:23.4",
    rawScore: 0.298,
    tag: "Candidate",
    description: "Closing hero packshot with embossed LUMAE branding.",
  },
  {
    id: "cand-10",
    rank: 10,
    label: "10",
    isTopMatch: false,
    startSeconds: 23.5,
    endSeconds: 25.0,
    durationSeconds: 1.5,
    startFormatted: "00:23.5",
    endFormatted: "00:25.0",
    rawScore: 0.255,
    tag: "Candidate",
    description: "Final call-to-action lockup with fragrance edition name.",
  },
];

export const MOCK_KEYFRAMES: KeyframeFrame[] = [
  {
    id: "kf-before",
    type: "before",
    label: "Before",
    sublabel: "00:04.0",
    timestampSeconds: 4.0,
    timestampFormatted: "00:04.0",
    imageSrc: "/mock/keyframe_before.jpg",
    description: "Subject holding perfume bottle, preparing to press nozzle.",
  },
  {
    id: "kf-inside",
    type: "inside",
    label: "Inside (Top match)",
    sublabel: "00:06.5",
    timestampSeconds: 6.5,
    timestampFormatted: "00:06.5",
    imageSrc: "/mock/keyframe_inside.jpg",
    description: "Active spray demonstration with visible mist particles.",
  },
  {
    id: "kf-after",
    type: "after",
    label: "After",
    sublabel: "00:08.0",
    timestampSeconds: 8.0,
    timestampFormatted: "00:08.0",
    imageSrc: "/mock/keyframe_after.jpg",
    description: "Product lowered, subject enjoying the fragrance scent.",
  },
];

export const INITIAL_PROCESSING_STAGES: ProcessingStage[] = [
  {
    id: 1,
    title: "1. Validating video",
    description: "Checking video container format, stream codecs, and aspect ratio.",
    activeDetails: "Inspecting MP4 container and metadata streams...",
    completedSummary: "MP4 · 00:25 · 1080 × 1920",
    mockMeta: [
      { label: "File format", value: "MP4", icon: "file" },
      { label: "Duration", value: "00:25", icon: "clock" },
      { label: "Resolution", value: "1080 × 1920", icon: "monitor" },
    ],
    status: "pending",
    durationMs: 1200,
  },
  {
    id: 2,
    title: "2. Extracting video frames",
    description: "Processing frames from the uploaded video.",
    activeDetails: "Frame 5 of 13 · Sampling at 0.5 FPS",
    completedSummary: "Sampled 13 frames (0.5 FPS)",
    details: "Sampled 13 frames (0.5 FPS)",
    status: "pending",
    durationMs: 1400,
  },
  {
    id: 3,
    title: "3. Encoding video & query",
    description: "Generating multimodal token representations with CLIP ViT-B/32.",
    activeDetails: "Using cached CLIP features (ViT-B/32)",
    completedSummary: "CLIP ViT-B/32 features encoded",
    details: "Using cached CLIP features (ViT-B/32)",
    status: "pending",
    durationMs: 1600,
  },
  {
    id: 4,
    title: "4. Locating matching moments",
    description: "Predicting candidate temporal intervals with Transformer cross-attention.",
    activeDetails: "Running Moment-DETR inference...",
    completedSummary: "Candidate moments localized",
    details: "Running Moment-DETR inference...",
    status: "pending",
    durationMs: 2200,
  },
  {
    id: 5,
    title: "5. Ranking candidate intervals",
    description: "Applying Hungarian matching to rank proposals by joint saliency score.",
    activeDetails: "Sorting Top-10 proposals by Hungarian loss...",
    completedSummary: "Top 10 candidates ranked",
    status: "pending",
    durationMs: 1200,
  },
  {
    id: 6,
    title: "6. Preparing your results",
    description: "Formatting timestamps, keyframes, and candidate breakdown for workspace view.",
    activeDetails: "Finalizing moment exploration results...",
    completedSummary: "Ready",
    status: "pending",
    durationMs: 1000,
  },
];

export const MOCK_TECHNICAL_DETAILS: TechnicalDetails = {
  modelName: "Moment-DETR",
  architecture: "Transformer Encoder-Decoder with Moment Queries",
  videoEncoder: "CLIP ViT-B/32 (Cached Features)",
  temporalGroundingHead: "Hungarian Matcher with L1 + GIoU Loss",
  task: "Natural-language temporal grounding in advertisement video",
  sampleFps: "0.5 FPS (13 sampled keyframes)",
  proposalsCount: 10,
  inferenceMode: "Deterministic Demo Simulation",
  contextWindow: "Full 25.0s video span",
  lossFunction: "Cross-Entropy Saliency + Interval GIoU",
  frameworkVersion: "LUMAE Engine v2.4-lite",
};

export const createMockAnalysisResult = (
  query = DEFAULT_QUERY,
  video = DEFAULT_SAMPLE_VIDEO
): MockAnalysisResult => {
  return {
    analysisId: `lumae-analysis-${Date.now().toString(36)}`,
    mode: "demo",
    query,
    video,
    candidates: MOCK_CANDIDATES,
    topCandidate: MOCK_CANDIDATES[0],
    keyframes: MOCK_KEYFRAMES,
    technicalDetails: MOCK_TECHNICAL_DETAILS,
    completedAt: new Date().toISOString(),
  };
};

