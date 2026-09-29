"use client";

import { ChangeEvent, DragEvent, useEffect, useRef, useState } from "react";
import type { MockScenario, QueryDefinition, QueryPrediction, TemporalWindow } from "@/lib/analysisTypes";
import { presets } from "@/lib/analysisTypes";
import { loadBundledSampleVideo } from "@/lib/lumaeApi";
import { saliencyPoints, selectPrediction, visibleTimelineWindows } from "@/lib/analysisView";
import { analysisStages, useVideoAnalysis } from "@/lib/useVideoAnalysis";

const poster = "/lumae-demo.png";
const sample = "/lumae-demo.mp4";
const initialMode = process.env.NEXT_PUBLIC_LUMAE_USE_MOCK === "true" ? "mock" : "real";
const stamp = (seconds: number) => `${Math.floor(seconds / 60)}:${String(Math.floor(seconds % 60)).padStart(2, "0")}`;
const exact = (seconds: number) => `${seconds.toFixed(2)}s`;
const label = (intent: string) => intent === "hook_candidate" ? "Hook" : intent === "demo_moment" ? "Demo" : intent === "benefit_moment" ? "Result" : "Custom";

function Mark({ size = 24 }: { size?: number }) {
  return <svg width={size} height={size} viewBox="0 0 30 32" aria-hidden="true"><path d="M3 3.5c0-2 2.2-3 3.9-2l20 13c1.5 1 1.5 3 0 4l-20 13C5.2 32.5 3 31.5 3 29.5V3.5Z" fill="#25f4ee"/><path d="m12 6 15 9c1.5 1 1.5 3 0 4l-15 9V6Z" fill="#fe2c55"/></svg>;
}

function I({ name, size = 18 }: { name: string; size?: number }) {
  const d: Record<string, string> = {
    upload: "M12 16V3m0 0L7 8m5-5 5 5M4 15v4a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-4",
    video: "M3 5h18v14H3zM10 9l5 3-5 3z",
    search: "M17 17l5 5M19 10.5a8.5 8.5 0 1 1-17 0 8.5 8.5 0 0 1 17 0Z",
    play: "m7 4 13 8-13 8V4Z", pause: "M8 4v16M16 4v16",
    volume: "M4 9h4l5-4v14l-5-4H4V9Zm12 0a4 4 0 0 1 0 6m2-9a8 8 0 0 1 0 12",
    expand: "M9 3H3v6m12-6h6v6M3 15v6h6m12-6v6h-6",
    spark: "m12 2 2 7 7 3-7 2-2 7-2-7-7-2 7-3 2-7",
    arrow: "M4 12h16m-6-6 6 6-6 6", close: "M5 5 19 19M19 5 5 19",
    info: "M12 16v-5m0-4h.01M21 12a9 9 0 1 1-18 0 9 9 0 0 1 18 0Z",
    download: "M12 3v12m0 0 5-5m-5 5-5-5M4 18v3h16v-3",
    chart: "M4 20h16M6 17v-4m6 4V8m6 9V4", bookmark: "M5 3h14v18l-7-5-7 5V3Z",
    chevron: "m6 9 6 6 6-6",
  };
  return <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d={d[name] ?? d.info}/></svg>;
}

async function readCodec(file: File): Promise<string> {
  const header = new TextDecoder("latin1").decode(await file.slice(0, 2_000_000).arrayBuffer());
  if (header.includes("avc1") || header.includes("avc3")) return "H.264";
  if (header.includes("hvc1") || header.includes("hev1")) return "HEVC";
  if (header.includes("av01")) return "AV1";
  if (header.includes("vp09")) return "VP9";
  return "Unknown";
}

function Saliency({ prediction, duration, cursor, raw, seek }: {
  prediction: QueryPrediction; duration: number; cursor: number; raw: boolean; seek: (time: number) => void;
}) {
  const scores = saliencyPoints(prediction);
  if (!scores.length) return <div className="panel-empty">No saliency scores returned for this query.</div>;
  const min = Math.min(...prediction.rawSaliencyScores);
  const max = Math.max(...prediction.rawSaliencyScores);
  const range = max - min;
  const x = (time: number) => 40 + Math.min(1, time / duration) * 540;
  const y = (normalized: number) => 145 - normalized * 108;
  const line = scores.map((point, index) => `${index ? "L" : "M"}${x(point.time)} ${y(point.normalized)}`).join(" ");
  const peak = scores.reduce((best, point) => point.raw > best.raw ? point : best, scores[0]);
  const top = prediction.windows[0];
  return <div className="chart"><svg viewBox="0 0 600 180" role="img" aria-label={`Saliency for ${prediction.query}`}>
    <defs><linearGradient id="saliency-gradient" x1="0" y1="0" x2="0" y2="1"><stop stopColor="#fe2c55" stopOpacity=".19"/><stop offset="1" stopColor="#fe2c55" stopOpacity="0"/></linearGradient></defs>
    {top && <rect x={x(top.start)} y="26" width={x(top.end) - x(top.start)} height="119" fill="#fe2c55" opacity=".07"/>}
    {[0, .5, 1].map(value => <g key={value}><line x1="40" x2="580" y1={y(value)} y2={y(value)} stroke="#e9edf2"/><text x="3" y={y(value) + 4} fontSize="11" fill="#667085">{raw ? (min + range * value).toFixed(1) : value.toFixed(1)}</text></g>)}
    {[0, .25, .5, .75, 1].map(value => <g key={value}><line x1={x(duration * value)} x2={x(duration * value)} y1="26" y2="145" stroke="#edf0f3"/><text x={x(duration * value)} y="165" textAnchor="middle" fontSize="11" fill="#667085">{stamp(duration * value)}</text></g>)}
    {scores.length > 1 && <><path d={`${line} L${x(scores.at(-1)?.time ?? 0)} 145 L40 145Z`} fill="url(#saliency-gradient)"/><path d={line} fill="none" stroke="#fe2c55" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round"/></>}
    <circle cx={x(peak.time)} cy={y(peak.normalized)} r="4.5" fill="#fe2c55" stroke="white" strokeWidth="2"/>
    <line x1={x(cursor)} x2={x(cursor)} y1="26" y2="145" stroke="#161c27" strokeDasharray="4 4"/>
    <rect x="40" y="26" width="540" height="119" fill="transparent" className="chart-hit" onClick={event => { const bounds = event.currentTarget.getBoundingClientRect(); seek((event.clientX - bounds.left) / bounds.width * duration); }}/>
    {scores.map((point, index) => <circle key={index} cx={x(point.time)} cy={y(point.normalized)} r="7" fill="transparent" onClick={() => seek(point.time)}><title>{`${stamp(point.time)} · Raw: ${point.raw.toFixed(4)} · Normalized: ${point.normalized.toFixed(3)}`}</title></circle>)}
  </svg><div className="chart-key"><span><i/> Saliency score</span><span>Peak at {stamp(peak.time)}</span></div></div>;
}

export default function Home() {
  const [mode, setMode] = useState<"mock" | "real">(initialMode);
  const [scenario, setScenario] = useState<MockScenario>("standard");
  const { modelInfo, modelError, analysis, error: analysisError, busy, stage, run, reset } = useVideoAnalysis(mode, scenario);
  const [selectedId, setSelectedId] = useState("demo");
  const [query, setQuery] = useState(presets[1].query);
  const [file, setFile] = useState<File | null>(null);
  const [videoUrl, setVideoUrl] = useState("");
  const [videoMeta, setVideoMeta] = useState({ duration: 0, width: 0, height: 0 });
  const [codec, setCodec] = useState("Unknown");
  const [playing, setPlaying] = useState(false);
  const [cursor, setCursor] = useState(0);
  const [muted, setMuted] = useState(false);
  const [volume, setVolume] = useState(1);
  const [raw, setRaw] = useState(false);
  const [localError, setLocalError] = useState("");
  const [dragging, setDragging] = useState(false);
  const [showAll, setShowAll] = useState(false);
  const [recentQueries, setRecentQueries] = useState<string[]>([]);
  const video = useRef<HTMLVideoElement>(null);
  const player = useRef<HTMLDivElement>(null);
  const input = useRef<HTMLInputElement>(null);
  const clipEnd = useRef<number | null>(null);
  const blobUrl = useRef<string | null>(null);
  const lastCursorUpdate = useRef(0);

  const duration = videoMeta.duration || 30;
  const maxDuration = modelInfo?.model.maxVideoDurationSeconds ?? 150;
  const predictions = analysis?.predictions ?? [];
  const selected = selectPrediction(analysis, selectedId);
  const top = selected?.windows[0];
  const matches = selected?.windows ?? [];
  const timeline = analysis ? visibleTimelineWindows(analysis) : [];
  const clips = predictions.filter(prediction => prediction.windows.length && (prediction.intent !== "hook_candidate" || analysis?.interpretation.hook.candidate));
  const isSample = videoUrl === sample;
  const displayError = localError || analysisError?.message || "";
  const showScenario = mode === "mock" && (process.env.NODE_ENV === "development" || process.env.NEXT_PUBLIC_LUMAE_USE_MOCK === "true");

  useEffect(() => () => { if (blobUrl.current) URL.revokeObjectURL(blobUrl.current); }, []);

  function seek(start: number, end?: number) {
    if (!video.current) return;
    video.current.currentTime = Math.max(0, Math.min(duration, start));
    setCursor(video.current.currentTime);
    clipEnd.current = end ?? null;
    if (end !== undefined) void video.current.play().catch(() => setLocalError("Video playback was blocked by the browser."));
  }

  function choose(next: File) {
    setLocalError("");
    if (next.type !== "video/mp4" && !next.name.toLowerCase().endsWith(".mp4")) { setLocalError("Unsupported format. Please upload an MP4 video."); return; }
    if (next.size > 2 * 1024 ** 3) { setLocalError("Video exceeds the 2 GB upload limit."); return; }
    if (blobUrl.current) URL.revokeObjectURL(blobUrl.current);
    blobUrl.current = URL.createObjectURL(next);
    video.current?.pause();
    setFile(next); setVideoUrl(blobUrl.current); setVideoMeta({ duration: 0, width: 0, height: 0 });
    setCodec("Inspecting…"); void readCodec(next).then(setCodec).catch(() => setCodec("Unknown"));
    setMode(initialMode); setScenario("standard"); setCursor(0); reset();
  }

  function onFile(event: ChangeEvent<HTMLInputElement>) { if (event.target.files?.[0]) choose(event.target.files[0]); event.target.value = ""; }
  function onDrop(event: DragEvent<HTMLDivElement>) { event.preventDefault(); setDragging(false); if (event.dataTransfer.files[0]) choose(event.dataTransfer.files[0]); }

  async function loadSample() {
    setLocalError("");
    try {
      const demoFile = await loadBundledSampleVideo();
      if (blobUrl.current) URL.revokeObjectURL(blobUrl.current);
      blobUrl.current = null;
      video.current?.pause();
      setFile(demoFile); setVideoUrl(sample); setVideoMeta({ duration: 30, width: 1280, height: 720 });
      setCodec("H.264"); setMode("mock"); setScenario("standard"); setSelectedId("demo");
      setQuery(presets[1].query); setCursor(0); reset();
    } catch { setLocalError("Bundled sample video is unavailable."); }
  }

  function clear() {
    video.current?.pause();
    if (blobUrl.current) URL.revokeObjectURL(blobUrl.current);
    blobUrl.current = null;
    setFile(null); setVideoUrl(""); setVideoMeta({ duration: 0, width: 0, height: 0 });
    setCodec("Unknown"); setCursor(0); setLocalError(""); setMode(initialMode); setScenario("standard"); reset();
  }

  async function runAnalysis() {
    setLocalError("");
    if (!file) { setLocalError("Add a short-form video to begin."); return; }
    if (!videoMeta.duration) { setLocalError("Wait for the video metadata to load, then retry."); return; }
    if (videoMeta.duration > maxDuration && scenario !== "video_too_long") { setLocalError(`Video duration exceeds the ${maxDuration} second model limit.`); return; }
    if (!query.trim()) { setLocalError("Enter a query before running analysis."); return; }
    const preset = presets.find(item => item.query === query.trim());
    const queries: QueryDefinition[] = preset ? presets : [...presets, { id: "custom", intent: "custom", query: query.trim() }];
    setSelectedId(preset?.id ?? "custom");
    const result = await run(file, { durationSeconds: videoMeta.duration, width: videoMeta.width, height: videoMeta.height, codec }, queries);
    if (result) {
      if (!preset) setRecentQueries(previous => [query.trim(), ...previous.filter(item => item !== query.trim())].slice(0, 3));
      if (!result.predictions.some(prediction => prediction.windows.length)) setLocalError("No temporal predictions were returned for these queries.");
    }
  }

  function selectQuery(definition: QueryDefinition) { setSelectedId(definition.id); setQuery(definition.query); }
  function selectWindow(prediction: QueryPrediction, window: TemporalWindow, play = false) { setSelectedId(prediction.queryId); setQuery(prediction.query); seek(window.start, play ? window.end : undefined); }
  function exportJson() {
    if (!analysis) return;
    const url = URL.createObjectURL(new Blob([JSON.stringify(analysis, null, 2)], { type: "application/json" }));
    const link = document.createElement("a"); link.href = url; link.download = "lumae-analysis-v1.json"; link.click(); URL.revokeObjectURL(url);
  }

  return <div className="app-shell">
    <header className="topbar">
      <button className="brand" onClick={() => void loadSample()}><Mark/><span>LUMAE</span></button>
      <nav className="topnav" aria-label="Main navigation"><button className="active" onClick={clear}>Home</button><button onClick={() => document.getElementById("query-composer")?.scrollIntoView()}>Analyze</button><button onClick={() => input.current?.click()}>Library</button><button onClick={() => document.getElementById("model-info")?.scrollIntoView()}>Models</button><button onClick={() => document.getElementById("model-info")?.scrollIntoView()}>Research</button></nav>
      <div className="topbar-right"><span className="tagline">VIDEO INTELLIGENCE<br/>FOR A BRIGHTER STORY</span><span className="avatar">JD</span><span className="user-name">Jordan Doe</span><I name="chevron" size={15}/></div>
    </header>
    <main className="workspace">
      <aside className="sidebar panel"><div><h2>Add Video</h2>
        <div className={`dropzone ${dragging ? "dragging" : ""}`} onDragOver={event => { event.preventDefault(); setDragging(true); }} onDragLeave={() => setDragging(false)} onDrop={onDrop} onClick={() => input.current?.click()} role="button" tabIndex={0} onKeyDown={event => { if (event.key === "Enter" || event.key === " ") input.current?.click(); }}><I name="upload" size={34}/><strong>Drag and drop video here</strong><span>MP4 · up to 2 GB · max {maxDuration}s</span></div>
        <input className="sr-only" ref={input} type="file" accept="video/mp4,.mp4" onChange={onFile} aria-label="Upload MP4 video"/>
        <button className="primary upload-button" onClick={() => input.current?.click()}><I name="upload"/> Upload Video</button>
        {file && <div className="file-meta"><strong>{file.name}</strong><span>{videoMeta.duration ? stamp(videoMeta.duration) : "Loading…"} · {videoMeta.width || "—"} × {videoMeta.height || "—"} · MP4</span><small>Codec: {codec}</small><small>{!videoMeta.duration ? "Validating video…" : videoMeta.duration <= maxDuration ? "Validated · ready for analysis" : "Too long for current model"}</small></div>}
        <div className="sidebar-links"><button className="selected" onClick={clear}><I name="play"/> New Analysis</button><button onClick={() => input.current?.click()}><I name="video"/> My Videos</button><button onClick={() => document.getElementById("query-composer")?.scrollIntoView()}><I name="bookmark"/> Saved Queries</button><button onClick={() => document.getElementById("model-info")?.scrollIntoView()}><I name="chart"/> Reports / Research</button></div>
      </div><div className="sidebar-bottom"><strong>Turn videos into insights.</strong><p>Find key moments, understand content, and discover what matters with Lumae.</p><div className="wave wave-cyan"/><div className="wave wave-pink"/><span>LUMAE<br/>v1.0.0</span></div></aside>

      <section className="center panel" aria-label="Video and highlighted clips">{videoUrl ? <>
        <div className="video-shell" ref={player}><video ref={video} src={videoUrl} poster={isSample ? poster : undefined} playsInline preload="metadata" onLoadedMetadata={event => { const element = event.currentTarget; setVideoMeta({ duration: element.duration, width: element.videoWidth, height: element.videoHeight }); if (element.duration > maxDuration) setLocalError(`Video duration exceeds the ${maxDuration} second model limit.`); }} onTimeUpdate={event => { const next = event.currentTarget.currentTime; if (performance.now() - lastCursorUpdate.current > 100 || next === 0) { setCursor(next); lastCursorUpdate.current = performance.now(); } if (clipEnd.current !== null && next >= clipEnd.current) { event.currentTarget.pause(); clipEnd.current = null; setCursor(next); } }} onPlay={() => setPlaying(true)} onPause={() => setPlaying(false)} onError={() => setLocalError("This video could not be played. Try another MP4 file.")}/>
          {isSample && <div className="video-title"><span>SMALL CHOICES</span><strong>Brighter<br/>everyday.</strong><i/></div>}
          <div className="video-controls"><div className="video-progress" onClick={event => { const bounds = event.currentTarget.getBoundingClientRect(); seek((event.clientX - bounds.left) / bounds.width * duration); }} role="slider" tabIndex={0} aria-label="Seek video" aria-valuemin={0} aria-valuemax={duration} aria-valuenow={cursor} onKeyDown={event => { if (event.key === "ArrowRight") seek(cursor + 2); if (event.key === "ArrowLeft") seek(cursor - 2); }}><span style={{ width: `${cursor / duration * 100}%` }}/></div><div className="control-row"><button onClick={() => { if (!video.current) return; clipEnd.current = null; if (video.current.paused) void video.current.play(); else video.current.pause(); }} aria-label={playing ? "Pause video" : "Play video"}><I name={playing ? "pause" : "play"} size={22}/></button><span>{stamp(cursor)} / {stamp(duration)}</span><div className="spacer"/><button onClick={() => { if (video.current) video.current.muted = !muted; setMuted(!muted); }} aria-label={muted ? "Unmute" : "Mute"}><I name="volume" size={21}/></button><input type="range" min="0" max="1" step="0.05" value={muted ? 0 : volume} onChange={event => { const next = Number(event.target.value); setVolume(next); setMuted(next === 0); if (video.current) { video.current.volume = next; video.current.muted = next === 0; } }} aria-label="Volume"/><button onClick={() => void player.current?.requestFullscreen()} aria-label="Fullscreen"><I name="expand" size={21}/></button></div></div>
        </div>
        {mode === "mock" && <div className="mode-line"><b><i/> Demo Data Mode</b><span>Illustrative predictions · no inference was run</span></div>}
        <div className="timeline-area"><div className="timeline-track" aria-label="Temporal predictions"><span className="timeline-base"/>{timeline.map(({ prediction, window }) => <button key={prediction.queryId} className={`timeline-window ${prediction.intent} ${selected?.queryId === prediction.queryId ? "focused" : ""}`} style={{ left: `${window.start / duration * 100}%`, width: `${(window.end - window.start) / duration * 100}%` }} onClick={() => selectWindow(prediction, window)} title={`${label(prediction.intent)} · ${prediction.query} · ${exact(window.start)}–${exact(window.end)} · model confidence ${window.confidence.toFixed(4)}`}>{label(prediction.intent)}</button>)}<span className="timeline-cursor" style={{ left: `${cursor / duration * 100}%` }}/></div><div className="time-labels">{[0,.25,.5,.75,1].map(value => <span key={value}>{stamp(duration * value)}</span>)}</div>{analysis?.interpretation.hook.status === "not_confidently_detected" && <p className="timeline-note">No confident Hook Candidate</p>}</div>
        <div className="section-heading clips-heading"><h2>Highlighted Clips from Query</h2><button className="text-action" onClick={() => setShowAll(!showAll)}>{showAll ? "Show fewer" : "View All Clips"} <I name="arrow" size={16}/></button></div>
        {analysis ? clips.length ? <div className="clip-grid">{(showAll ? clips : clips.slice(0,3)).map(prediction => { const window = prediction.windows[0]; return <article className={`clip-card ${selected?.queryId === prediction.queryId ? "active" : ""}`} key={prediction.queryId}><button className="clip-thumb" style={{ backgroundImage: isSample ? `url(${poster})` : "linear-gradient(135deg,#dce4eb,#8193a5)" }} onClick={() => selectWindow(prediction, window, true)} aria-label={`Play ${label(prediction.intent)} clip`}><span>{(window.end - window.start).toFixed(1)}s</span></button><div className="clip-body"><b className={prediction.intent}>{label(prediction.intent)}</b><span>{exact(window.start)} – {exact(window.end)}</span><small>Model confidence {window.confidence.toFixed(4)}</small><p>“{prediction.query}”</p><button className="clip-play" onClick={() => selectWindow(prediction, window, true)}><I name="play" size={15}/> Play Clip</button></div></article>; })}</div> : <div className="empty-clips"><strong>No temporal predictions returned.</strong><span>Try another query or video.</span></div> : <div className="empty-clips"><I name="spark" size={27}/><strong>Moments appear after analysis.</strong><span>Ask Lumae to find the parts of a video that matter.</span></div>}
      </> : <div className="empty-video"><div><I name="video" size={36}/></div><h1>Add a short-form video to begin.</h1><p>Ask Lumae to find product demos, results, hooks, or any moment you can describe.</p><button className="primary" onClick={() => input.current?.click()}><I name="upload"/> Upload video</button><button className="text-action" onClick={() => void loadSample()}>Explore sample demo <I name="arrow" size={16}/></button><div className="empty-timeline"><span/><span/><span/></div></div>}</section>

      <section className="right-column" aria-label="Analysis insights">
        {mode === "mock" && !videoUrl && <div className="mock-banner">Demo Data Mode · illustrative responses</div>}
        <div className="panel query-panel" id="query-composer"><div className="section-heading"><h2>Query Video</h2>{analysis && <button className="icon-button" onClick={exportJson} title="Export analysis JSON" aria-label="Export analysis JSON"><I name="download" size={17}/></button>}</div><div className="query-input"><I name="search" size={20}/><input value={query} onChange={event => { setQuery(event.target.value); setSelectedId("custom"); }} onKeyDown={event => { if (event.key === "Enter") void runAnalysis(); }} placeholder="Describe a moment you want to find" aria-label="Video query"/><button onClick={() => { setQuery(""); setSelectedId("custom"); }} aria-label="Clear query"><I name="close" size={14}/></button></div><div className="query-actions"><div className="query-chips">{presets.map(preset => <button key={preset.id} className={selectedId === preset.id ? "selected" : ""} onClick={() => selectQuery(preset)} title={preset.query}>{label(preset.intent)}</button>)}</div><button className="primary run-button" disabled={busy} onClick={() => void runAnalysis()}><I name="spark" size={17}/>{busy ? "Analyzing…" : "Run Analysis"}</button></div><p className="query-hint">{selected?.query ?? "Select an intent or write a custom query."}</p>{recentQueries.length > 0 && <div className="recent-queries"><span>Recent</span>{recentQueries.map(item => <button key={item} title={item} onClick={() => { setQuery(item); setSelectedId("custom"); }}>{item}</button>)}</div>}{showScenario && <div className="scenario-control"><label htmlFor="mock-scenario">Mock scenario</label><select id="mock-scenario" value={scenario} onChange={event => { setScenario(event.target.value as MockScenario); reset(); setLocalError(""); }}><option value="standard">Standard · no Hook</option><option value="confident_hook">Confident Hook</option><option value="no_predictions">No predictions</option><option value="video_too_long">VIDEO_TOO_LONG error</option><option value="malformed_response">Malformed response</option></select></div>}{displayError && <div className="error-message" role="alert">{analysisError?.code ? `${analysisError.code}: ` : ""}{displayError}</div>}{busy && <div className="progress-list" aria-live="polite">{analysisStages.map((item,index) => <span className={index < stage ? "done" : index === stage ? "current" : ""} key={item}>{index < stage ? "✓" : `${index+1}.`} {item}</span>)}</div>}</div>

        <div className="panel benchmark-panel"><div className="section-heading"><div><h2>{modelInfo?.model.stage === "baseline" || !modelInfo ? "Baseline Benchmark" : "Model Benchmark"}</h2><p>{modelInfo ? `${modelInfo.benchmark.dataset} · ${modelInfo.model.displayName ?? modelInfo.model.id}` : "Model-level benchmark · awaiting model metadata"}</p></div><span title="Dataset-level metrics, not accuracy for this video"><I name="info" size={17}/></span></div>{modelInfo ? <><div className="metrics-grid">{[["R1@0.5",modelInfo.benchmark.MR_R1_0_5,"Top prediction overlaps ground truth at temporal IoU ≥ 0.5"],["R1@0.7",modelInfo.benchmark.MR_R1_0_7,"Same measure with stricter localization"],["MR mAP",modelInfo.benchmark.MR_mAP,"Moment retrieval ranking performance"],["HL mAP",modelInfo.benchmark.HL_VeryGood_mAP,"VeryGood highlight ranking performance"],["HIT@1",modelInfo.benchmark.HL_VeryGood_HIT1,"Top ranked highlight is positive (VeryGood)"]].map(([key,value,help]) => <div className="metric-card" key={key as string} title={help as string}><span>{key}</span><strong>{Number(value).toFixed(2)}</strong><small>baseline</small></div>)}</div><details className="benchmark-details"><summary>Detailed benchmark <I name="chevron" size={15}/></summary><div className="detail-metrics"><span>Short mAP <b>{modelInfo.benchmark.MR_short_mAP.toFixed(2)}</b></span><span>Middle mAP <b>{modelInfo.benchmark.MR_middle_mAP.toFixed(2)}</b></span><span>Long mAP <b>{modelInfo.benchmark.MR_long_mAP.toFixed(2)}</b></span></div><p>MR mAP@0.5 {modelInfo.benchmark.MR_mAP_0_5.toFixed(2)} · MR mAP@0.75 {modelInfo.benchmark.MR_mAP_0_75.toFixed(2)}. Short temporal moments remain challenging. These are dataset metrics, not per-video scores.</p></details></> : <div className="panel-empty">{modelError?.message ?? "Loading model-level benchmark…"}</div>}</div>

        <div className="panel saliency-panel"><div className="section-heading"><div><h2>Saliency Score <span title="Relative prominence, not a probability"><I name="info" size={15}/></span></h2><p>Query-conditioned temporal prominence</p></div><div className="segmented"><button className={!raw ? "selected" : ""} onClick={() => setRaw(false)}>Normalized</button><button className={raw ? "selected" : ""} onClick={() => setRaw(true)}>Raw</button></div></div>{selected ? <><div className="saliency-tabs">{predictions.map(prediction => <button key={prediction.queryId} className={selected.queryId === prediction.queryId ? "selected" : ""} onClick={() => { setSelectedId(prediction.queryId); setQuery(prediction.query); }}>{label(prediction.intent)}</button>)}</div><Saliency prediction={selected} duration={duration} cursor={cursor} raw={raw} seek={seek}/><p className="chart-note">Normalized values are for visualization. Raw model saliency is preserved; neither is a probability.</p><details className="score-table"><summary>View raw score data</summary><div>{saliencyPoints(selected).map((point,index) => <span key={index}>{stamp(point.time)} <b>{point.raw.toFixed(4)}</b></span>)}</div></details></> : <div className="panel-empty">Saliency appears after analysis.</div>}</div>

        <div className="insight-pair"><div className="panel mini-panel"><div className="section-heading"><h2>Top Moment</h2><small>MODEL OUTPUT</small></div>{top && selected ? <><div className="top-moment"><button className="moment-thumb" style={{ backgroundImage: isSample ? `url(${poster})` : "linear-gradient(135deg,#dce4eb,#8193a5)" }} onClick={() => selectWindow(selected, top, true)} aria-label="Play top moment"><I name="play" size={18}/></button><div><strong>{exact(top.start)} – {exact(top.end)}</strong><span>{label(selected.intent)} · model confidence</span><b>{top.confidence.toFixed(4)}</b></div></div><p className="moment-query">“{selected.query}”</p></> : <div className="panel-empty">No predicted moment yet.</div>}</div><div className="panel mini-panel matches-panel"><div className="section-heading"><h2>Query Matches</h2><small>MODEL OUTPUT</small></div>{matches.length ? matches.map((window,index) => <button key={index} className="match-row" onClick={() => seek(window.start, window.end)}><span className="match-play"><I name="play" size={12}/></span><span>{exact(window.start)} – {exact(window.end)}</span><b>{window.confidence.toFixed(4)}</b></button>) : <div className="panel-empty">No matches returned.</div>}</div></div>

        <div className="panel interpretation-panel"><div className="section-heading"><div><h2>Lumae Interpretation</h2><p>Product layer · distinct from raw predictions</p></div><I name="spark" size={20}/></div>{analysis ? <div className="interpretation-grid"><div><span>Hook Candidate <small>{analysis.interpretation.hook.provenance}</small></span><strong>{analysis.interpretation.hook.candidate ? `${exact(analysis.interpretation.hook.candidate.start)} – ${exact(analysis.interpretation.hook.candidate.end)}` : "No confident Hook Candidate"}</strong></div><div><span>Demo Moment <small>{analysis.interpretation.demoMoment?.provenance ?? "MODEL_OUTPUT"}</small></span><strong>{analysis.interpretation.demoMoment ? `${exact(analysis.interpretation.demoMoment.start)} – ${exact(analysis.interpretation.demoMoment.end)}` : "Not detected"}</strong></div><div><span>Result / Benefit <small>{analysis.interpretation.resultMoment?.provenance ?? "MODEL_OUTPUT"}</small></span><strong>{analysis.interpretation.resultMoment ? `${exact(analysis.interpretation.resultMoment.start)} – ${exact(analysis.interpretation.resultMoment.end)}` : "Not detected"}</strong></div><div><span>Core Action Region <small>{analysis.interpretation.coreActionRegion?.provenance ?? "DERIVED"}</small></span><strong>{analysis.interpretation.coreActionRegion ? `${exact(analysis.interpretation.coreActionRegion.start)} – ${exact(analysis.interpretation.coreActionRegion.end)}` : "Not detected"}</strong></div></div> : <div className="panel-empty">Interpretation appears after analysis.</div>}<p className="interpretation-note">Hook Candidate is a Lumae heuristic, not a native Moment-DETR class. Core Action Region is derived from temporal overlap.</p></div>

        <div className="panel model-panel" id="model-info"><div className="section-heading"><h2>Model / Research</h2><small>{modelInfo?.model.status?.replaceAll("_", " ").toUpperCase() ?? "RESEARCH DEMO"}</small></div><div className="model-facts"><span>Model <b>{modelInfo?.model.displayName ?? "Awaiting model metadata"}</b></span><span>Dataset <b>{modelInfo?.benchmark.dataset ?? "—"}</b></span><span>Sampling <b>{modelInfo?.model.temporalResolutionSeconds ? `~${modelInfo.model.temporalResolutionSeconds} seconds` : "—"}</b></span><span>Max input <b>{modelInfo?.model.maxVideoDurationSeconds ? `${modelInfo.model.maxVideoDurationSeconds} seconds` : "—"}</b></span></div><p>The CLIP-only baseline is visual; API v1 does not guarantee audio or speech understanding.</p><div className="model-links"><a href={process.env.NEXT_PUBLIC_MOMENT_DETR_REPO_URL || "https://github.com/jayleicn/moment_detr"} target="_blank" rel="noreferrer">Moment-DETR ↗</a>{process.env.NEXT_PUBLIC_LUMAE_PROJECT_REPO_URL && <a href={process.env.NEXT_PUBLIC_LUMAE_PROJECT_REPO_URL} target="_blank" rel="noreferrer">Lumae project ↗</a>}{process.env.NEXT_PUBLIC_LUMAE_COLAB_URL && <a href={process.env.NEXT_PUBLIC_LUMAE_COLAB_URL} target="_blank" rel="noreferrer">Colab notebook ↗</a>}</div></div>
      </section>
    </main>
  </div>;
}
