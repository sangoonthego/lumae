"use client";

import React from "react";
import Image from "next/image";
import { useRouter } from "next/navigation";
import { AppHeader } from "@/components/layout/AppHeader";
import { AppFooter } from "@/components/layout/AppFooter";
import { AppBackgroundDecoration } from "@/components/layout/AppBackgroundDecoration";
import { useAnalysis } from "@/context/AnalysisContext";

export default function ExamplesPage() {
  const router = useRouter();
  const { loadSampleDemo } = useAnalysis();

  const handleLaunchPerfume = () => {
    loadSampleDemo();
    router.push("/analyze");
  };

  const handleInspectPerfumeResults = () => {
    loadSampleDemo();
    router.push("/analyze/results");
  };

  const examples = [
    {
      id: "perfume",
      title: "LUMAE Eau De Parfum — Luxury Launch",
      category: "Beauty & Fragrance",
      duration: "00:25",
      query: "The person demonstrates how to spray and use the perfume.",
      image: "/mock/perfume_ad_spray.jpg",
      predictedInterval: "00:05.4 – 00:07.7",
      isPrimary: true,
    },
    {
      id: "beverage",
      title: "Sparkling Citrus — Refreshment Campaign",
      category: "Food & Beverage",
      duration: "00:30",
      query: "The condensation droplets slide down the cold glass bottle.",
      image: "/mock/perfume_bottle.jpg",
      predictedInterval: "00:08.2 – 00:11.0",
      isPrimary: false,
    },
    {
      id: "smartwatch",
      title: "Chrono Pulse — Fitness Precision",
      category: "Consumer Electronics",
      duration: "00:20",
      query: "Runner checks heart rate and swipes the touchscreen display.",
      image: "/mock/perfume_ad_thumb.jpg",
      predictedInterval: "00:04.1 – 00:06.8",
      isPrimary: false,
    },
  ];

  return (
    <div className="relative min-h-screen flex flex-col bg-white text-[#111111] overflow-x-hidden">
      <AppBackgroundDecoration />
      <AppHeader />

      <main className="relative z-10 flex-1 max-w-6xl w-full mx-auto px-4 sm:px-6 lg:px-8 py-10 sm:py-14 flex flex-col gap-8">
        <div className="flex flex-col items-start gap-2">
          <div className="inline-flex items-center gap-1.5 px-3 py-0.5 rounded-full text-xs font-bold bg-[#FFEBEF] text-[#FE2C55]">
            Interactive Examples
          </div>
          <h1 className="text-3xl sm:text-4xl font-black text-[#111111]">
            Curated Advertisement <span className="text-[#FE2C55]">Demos</span>
          </h1>
          <p className="text-sm sm:text-base text-[#6B7280] max-w-xl">
            Explore pre-configured advertisement samples to experience natural-language temporal grounding in action.
          </p>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {examples.map((item) => (
            <div
              key={item.id}
              className="bg-white rounded-3xl p-4 border border-[#E5E7EB] shadow-xs flex flex-col justify-between hover:shadow-md transition-all group"
            >
              <div>
                <div className="relative aspect-[16/10] w-full rounded-2xl overflow-hidden bg-black/5 mb-4">
                  <Image
                    src={item.image}
                    alt={item.title}
                    fill
                    sizes="(max-width: 768px) 100vw, 340px"
                    className="object-cover group-hover:scale-105 transition-transform duration-300"
                  />
                  <span className="absolute top-2.5 left-2.5 bg-black/75 backdrop-blur-xs text-white text-[10px] font-semibold px-2 py-0.5 rounded-full">
                    {item.category}
                  </span>
                  <span className="absolute bottom-2.5 right-2.5 bg-black/75 backdrop-blur-xs text-white text-[10px] font-mono px-2 py-0.5 rounded-sm">
                    {item.duration}
                  </span>
                </div>

                <h3 className="font-bold text-base text-[#111111] mb-2">
                  {item.title}
                </h3>

                <div className="text-xs text-[#6B7280] bg-[#F8F9FA] p-3 rounded-xl border border-[#E5E7EB]/60 mb-3">
                  <span className="font-semibold text-[#111111] block mb-1">
                    Target Query:
                  </span>
                  &ldquo;{item.query}&rdquo;
                </div>

                <div className="flex items-center justify-between text-xs font-mono mb-4 px-1">
                  <span className="text-[#6B7280]">Predicted:</span>
                  <span className="font-bold text-[#FE2C55]">
                    {item.predictedInterval}
                  </span>
                </div>
              </div>

              <div className="flex items-center gap-2 pt-2 border-t border-[#E5E7EB]/70">
                <button
                  type="button"
                  onClick={handleLaunchPerfume}
                  className="flex-1 py-2.5 px-3 rounded-full bg-[#000000] hover:bg-[#222222] text-white text-xs font-bold transition-all text-center cursor-pointer"
                >
                  Analyze Video
                </button>
                <button
                  type="button"
                  onClick={handleInspectPerfumeResults}
                  className="py-2.5 px-3.5 rounded-full bg-white hover:bg-[#F8F9FA] text-[#111111] border border-[#E5E7EB] text-xs font-semibold transition-all text-center cursor-pointer"
                >
                  View Result
                </button>
              </div>
            </div>
          ))}
        </div>
      </main>

      <AppFooter />
    </div>
  );
}
