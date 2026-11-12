import React from "react";
import Link from "next/link";
import { BrandLogo } from "@/components/ui/BrandLogo";

export function AppFooter() {
  return (
    <footer className="relative z-10 w-full mt-auto border-t border-[#E5E7EB] bg-white/80 py-8">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 flex flex-col md:flex-row items-center justify-between gap-4">
        <div className="flex items-center gap-3">
          <BrandLogo size="sm" />
          <span className="text-xs text-[#6B7280]">
            Video Moment Intelligence • Advertisement Temporal Grounding
          </span>
        </div>

        <div className="flex flex-wrap items-center justify-center gap-6 text-xs text-[#6B7280]">
          <Link href="/research" className="hover:text-[#111111] transition-colors">
            Moment-DETR Paper
          </Link>
          <Link href="/research" className="hover:text-[#111111] transition-colors">
            CLIP ViT-B/32
          </Link>
          <Link href="/examples" className="hover:text-[#111111] transition-colors">
            Sample Demos
          </Link>
          <span className="text-[#16A34A] font-medium flex items-center gap-1.5">
            <span className="w-1.5 h-1.5 rounded-full bg-[#16A34A] inline-block" />
            Frontend Prototype • Deterministic Simulation
          </span>
        </div>
      </div>
    </footer>
  );
}

