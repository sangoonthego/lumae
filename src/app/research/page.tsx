"use client";

import React from "react";
import Link from "next/link";
import { AppHeader } from "@/components/layout/AppHeader";
import { AppFooter } from "@/components/layout/AppFooter";
import { AppBackgroundDecoration } from "@/components/layout/AppBackgroundDecoration";

export default function ResearchPage() {
  const highlights = [
    {
      title: "Moment-DETR Architecture",
      desc: "Direct temporal interval prediction using bipartite Hungarian matching, eliminating complex sliding-window heuristics and anchor post-processing.",
      badge: "Architecture",
    },
    {
      title: "CLIP Multimodal Embeddings",
      desc: "Zero-shot visual-semantic alignment via OpenAI CLIP (ViT-B/32). Frame-level visual tokens cross-attend with natural-language text representations.",
      badge: "Encoding",
    },
    {
      title: "Ad Video Temporal Grounding",
      desc: "Specialized temporal localization on high-density promotional videos, isolating key product hooks, usage demonstrations, and branded packshots.",
      badge: "Domain",
    },
    {
      title: "Joint Loss Formulation",
      desc: "Combining generalized Intersection over Union (GIoU), smooth L1 interval offset loss, and Cross-Entropy temporal saliency ranking.",
      badge: "Optimization",
    },
  ];

  return (
    <div className="relative min-h-screen flex flex-col bg-white text-[#111111] overflow-x-hidden">
      <AppBackgroundDecoration />
      <AppHeader />

      <main className="relative z-10 flex-1 max-w-5xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-10 sm:py-14 flex flex-col gap-10">
        <div className="flex flex-col items-start gap-2">
          <div className="inline-flex items-center gap-1.5 px-3 py-0.5 rounded-full text-xs font-bold bg-[#FFEBEF] text-[#FE2C55]">
            Research & Methodology
          </div>
          <h1 className="text-3xl sm:text-5xl font-black text-[#111111] tracking-tight">
            Moment-DETR & <span className="text-[#FE2C55]">CLIP</span> Grounding
          </h1>
          <p className="text-sm sm:text-base text-[#6B7280] max-w-2xl leading-relaxed">
            LUMAE transforms unstructured marketing queries into calibrated time intervals across high-velocity advertisement video content.
          </p>
        </div>

        {/* 4 Pillars Grid */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {highlights.map((item, idx) => (
            <div
              key={idx}
              className="bg-white rounded-3xl p-6 border border-[#E5E7EB] shadow-xs flex flex-col justify-between hover:shadow-sm transition-all"
            >
              <div>
                <span className="text-xs font-mono font-bold text-[#FE2C55] bg-[#FFEBEF] px-2.5 py-1 rounded-full mb-3 inline-block">
                  {item.badge}
                </span>
                <h3 className="text-lg font-bold text-[#111111] mb-2">
                  {item.title}
                </h3>
                <p className="text-xs sm:text-sm text-[#6B7280] leading-relaxed">
                  {item.desc}
                </p>
              </div>
            </div>
          ))}
        </div>

        {/* Technical Callout Card */}
        <div className="bg-[#F8F9FA] rounded-3xl p-6 sm:p-8 border border-[#E5E7EB] flex flex-col md:flex-row items-center justify-between gap-6">
          <div className="flex flex-col gap-1.5">
            <h3 className="text-lg font-black text-[#111111]">
              Try the Interactive Prototype
            </h3>
            <p className="text-xs sm:text-sm text-[#6B7280]">
              Experience simulated inference on the sample perfume advertisement video.
            </p>
          </div>
          <Link
            href="/analyze"
            className="px-6 py-3 rounded-full bg-[#000000] hover:bg-[#222222] text-white text-sm font-bold shadow-sm transition-all flex-shrink-0"
          >
            Launch Grounding Demo →
          </Link>
        </div>
      </main>

      <AppFooter />
    </div>
  );
}

