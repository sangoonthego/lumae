"use client";

import React from "react";
import Image from "next/image";
import Link from "next/link";
import { useRouter } from "next/navigation";
import { AppHeader } from "@/components/layout/AppHeader";
import { AppFooter } from "@/components/layout/AppFooter";
import { AppBackgroundDecoration } from "@/components/layout/AppBackgroundDecoration";
import { useAnalysis } from "@/context/AnalysisContext";

export default function LibraryPage() {
  const router = useRouter();
  const { loadSampleDemo } = useAnalysis();

  const handleSelectPerfume = () => {
    loadSampleDemo();
    router.push("/analyze");
  };

  const libraryItems = [
    {
      id: "lib-1",
      filename: "perfume_ad.mp4",
      title: "LUMAE Eau De Parfum Launch",
      duration: "00:25",
      resolution: "1920 × 1080",
      size: "24.8 MB",
      date: "Nov 2026",
      image: "/mock/perfume_ad_spray.jpg",
      status: "Grounding Ready",
    },
    {
      id: "lib-2",
      filename: "summer_collection_spot.mp4",
      title: "Summer Apparel Lookbook",
      duration: "00:15",
      resolution: "1080 × 1920",
      size: "18.2 MB",
      date: "Nov 2026",
      image: "/mock/perfume_bottle.jpg",
      status: "Features Cached",
    },
    {
      id: "lib-3",
      filename: "cosmetics_unboxing_teaser.mp4",
      title: "Velvet Glow Foundation Reveal",
      duration: "00:22",
      resolution: "1920 × 1080",
      size: "21.4 MB",
      date: "Nov 2026",
      image: "/mock/keyframe_before.jpg",
      status: "Features Cached",
    },
    {
      id: "lib-4",
      filename: "skincare_serum_dropper.mp4",
      title: "Hydra Bloom Facial Serum",
      duration: "00:30",
      resolution: "1080 × 1920",
      size: "28.6 MB",
      date: "Nov 2026",
      image: "/mock/keyframe_inside.jpg",
      status: "Grounding Ready",
    },
  ];

  return (
    <div className="relative min-h-screen flex flex-col bg-white text-[#111111] overflow-x-hidden">
      <AppBackgroundDecoration />
      <AppHeader />

      <main className="relative z-10 flex-1 max-w-6xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-10 sm:py-14 flex flex-col gap-8">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
          <div>
            <div className="inline-flex items-center gap-1.5 px-3 py-0.5 rounded-full text-xs font-bold bg-[#FFEBEF] text-[#FE2C55] mb-2">
              Mock Asset Catalog
            </div>
            <h1 className="text-3xl sm:text-4xl font-black text-[#111111]">
              Video <span className="text-[#FE2C55]">Library</span>
            </h1>
            <p className="text-xs sm:text-sm text-[#6B7280] mt-1">
              Indexed advertisement videos available for instant temporal moment search.
            </p>
          </div>

          <Link
            href="/analyze"
            className="inline-flex items-center gap-2 px-5 py-2.5 rounded-full bg-[#000000] text-white text-xs sm:text-sm font-bold hover:bg-[#222222] transition-all self-start sm:self-center"
          >
            <span>+ Upload New Video</span>
          </Link>
        </div>

        {/* Video Grid */}
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-6">
          {libraryItems.map((item) => (
            <div
              key={item.id}
              className="bg-white rounded-3xl p-3.5 border border-[#E5E7EB] shadow-xs flex flex-col justify-between hover:shadow-md transition-all group"
            >
              <div>
                <div className="relative aspect-[16/10] w-full rounded-2xl overflow-hidden bg-black/5 mb-3">
                  <Image
                    src={item.image}
                    alt={item.title}
                    fill
                    sizes="(max-width: 640px) 100vw, 240px"
                    className="object-cover group-hover:scale-105 transition-transform duration-300"
                  />
                  <span className="absolute bottom-2 right-2 bg-black/75 backdrop-blur-xs text-white text-[10px] font-mono px-1.5 py-0.5 rounded-sm">
                    {item.duration}
                  </span>
                </div>

                <div className="text-xs font-bold text-[#111111] truncate mb-0.5">
                  {item.title}
                </div>
                <div className="text-[11px] font-mono text-[#6B7280] truncate mb-2">
                  {item.filename}
                </div>

                <div className="flex flex-col gap-1 text-[11px] text-[#6B7280] bg-[#F8F9FA] p-2.5 rounded-xl border border-[#E5E7EB]/60 mb-3">
                  <div className="flex justify-between">
                    <span>Resolution:</span>
                    <span className="font-mono text-[#111111]">{item.resolution}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>File size:</span>
                    <span className="font-mono text-[#111111]">{item.size}</span>
                  </div>
                </div>
              </div>

              <button
                type="button"
                onClick={handleSelectPerfume}
                className="w-full py-2 rounded-full bg-[#F8F9FA] hover:bg-[#000000] text-[#111111] hover:text-white border border-[#E5E7EB] hover:border-[#000000] text-xs font-bold transition-all text-center cursor-pointer"
              >
                Analyze Video →
              </button>
            </div>
          ))}
        </div>
      </main>

      <AppFooter />
    </div>
  );
}
