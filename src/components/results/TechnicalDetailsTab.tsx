"use client";

import React from "react";
import { TechnicalDetails } from "@/lib/mockData";

interface TechnicalDetailsTabProps {
  details: TechnicalDetails;
}

export function TechnicalDetailsTab({ details }: TechnicalDetailsTabProps) {
  const specs = [
    { label: "Grounding Model", value: details.modelName },
    { label: "Visual Encoder", value: details.videoEncoder },
    { label: "Model Architecture", value: details.architecture },
    { label: "Temporal Grounding Head", value: details.temporalGroundingHead },
    { label: "Core Task", value: details.task },
    { label: "Frame Sampling Rate", value: details.sampleFps },
    { label: "Candidate Proposals", value: `${details.proposalsCount} proposals evaluated` },
    { label: "Inference Engine", value: details.inferenceMode },
    { label: "Context Span", value: details.contextWindow },
    { label: "Loss Formulation", value: details.lossFunction },
    { label: "Pipeline Release", value: details.frameworkVersion },
  ];

  return (
    <div className="w-full bg-white rounded-3xl p-5 sm:p-6 border border-[#E5E7EB] shadow-xs flex flex-col gap-4">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-2 pb-2 border-b border-[#E5E7EB]">
        <div>
          <h3 className="text-base font-bold text-[#111111]">
            Research & Model Specifications
          </h3>
          <p className="text-xs text-[#6B7280]">
            Grounding advertisement video moments with Transformer Moment Queries and multimodal CLIP embeddings.
          </p>
        </div>
        <span className="text-xs font-semibold text-[#FE2C55] bg-[#FFEBEF] px-3 py-1 rounded-full border border-[#FE2C55]/20">
          Research-backed
        </span>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3.5 pt-1">
        {specs.map((item, idx) => (
          <div
            key={idx}
            className="p-3.5 rounded-2xl bg-[#F8F9FA] border border-[#E5E7EB]/70 flex flex-col gap-1"
          >
            <span className="text-[11px] font-bold text-[#6B7280] uppercase tracking-wide">
              {item.label}
            </span>
            <span className="text-xs sm:text-sm font-semibold text-[#111111]">
              {item.value}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

