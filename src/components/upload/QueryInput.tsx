"use client";

import React, { ChangeEvent } from "react";
import { QUERY_SUGGESTIONS } from "@/lib/mockData";

interface QueryInputProps {
  query: string;
  onChange: (query: string) => void;
  selectedChipId: string | null;
  onSelectChip: (chipId: string, text: string) => void;
}

export function QueryInput({
  query,
  onChange,
  selectedChipId,
  onSelectChip,
}: QueryInputProps) {
  const maxLength = 256;

  const handleTextChange = (e: ChangeEvent<HTMLTextAreaElement>) => {
    const val = e.target.value.slice(0, maxLength);
    onChange(val);
  };

  return (
    <div className="w-full flex flex-col gap-3">
      <label
        htmlFor="query-input"
        className="text-base sm:text-lg font-bold text-[#111111]"
      >
        What moment are you looking for?
      </label>

      {/* Main Rounded Query Box */}
      <div className="relative w-full rounded-2xl border border-[#E5E7EB] bg-white shadow-2xs focus-within:border-[#FE2C55] focus-within:ring-2 focus-within:ring-[#FE2C55]/20 transition-all">
        <textarea
          id="query-input"
          rows={2}
          value={query}
          onChange={handleTextChange}
          placeholder="e.g., The person demonstrates how to spray and use the perfume."
          className="w-full px-4 sm:px-5 py-3.5 sm:py-4 text-sm sm:text-base text-[#111111] placeholder-[#9CA3AF] bg-transparent border-0 outline-none resize-none font-normal leading-relaxed"
        />

        {/* Character Count */}
        <div className="flex justify-end px-4 pb-2.5 text-xs text-[#9CA3AF] font-mono select-none">
          {query.length} / {maxLength}
        </div>
      </div>

      {/* Suggestion Chips */}
      <div className="flex flex-wrap items-center gap-2 pt-1">
        <span className="text-xs font-medium text-[#6B7280] mr-1">
          Example queries:
        </span>

        {QUERY_SUGGESTIONS.map((chip) => {
          const isSelected = selectedChipId === chip.id;
          return (
            <button
              key={chip.id}
              type="button"
              onClick={() => onSelectChip(chip.id, chip.text)}
              className={`inline-flex items-center gap-1.5 px-3.5 py-1.5 rounded-full text-xs font-semibold transition-all duration-150 cursor-pointer ${
                isSelected
                  ? "bg-[#000000] text-white shadow-xs"
                  : "bg-[#FFEBEF] text-[#FE2C55] hover:bg-[#FFD9E2]"
              }`}
            >
              {isSelected ? (
                <svg
                  width="12"
                  height="12"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="3"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  aria-hidden="true"
                >
                  <polyline points="20 6 9 17 4 12" />
                </svg>
              ) : (
                <span className="w-1.5 h-1.5 rounded-full bg-[#FE2C55]" />
              )}
              <span>{chip.label}</span>
            </button>
          );
        })}
      </div>
    </div>
  );
}

