import React from "react";
import Link from "next/link";

interface BrandLogoProps {
  className?: string;
  size?: "sm" | "md" | "lg";
}

export function BrandLogo({ className = "", size = "md" }: BrandLogoProps) {
  const sizeClasses = {
    sm: "text-lg gap-2",
    md: "text-2xl gap-2.5",
    lg: "text-3xl gap-3",
  };

  const slashDimensions = {
    sm: "w-1.5 h-4",
    md: "w-2 h-5.5",
    lg: "w-2.5 h-7",
  };

  return (
    <Link
      href="/"
      className={`inline-flex items-center font-black tracking-tight select-none group text-[#111111] hover:opacity-90 transition-opacity ${sizeClasses[size]} ${className}`}
      aria-label="LUMAE Home"
    >
      <span className="flex items-center gap-1">
        {/* TikTok Cyan Slash */}
        <span
          className={`inline-block ${slashDimensions[size]} bg-[#25F4EE] rounded-full transform -skew-x-12 shadow-[0_0_10px_rgba(37,244,238,0.4)]`}
          aria-hidden="true"
        />
        {/* TikTok Pink Slash */}
        <span
          className={`inline-block ${slashDimensions[size]} bg-[#FE2C55] rounded-full transform -skew-x-12 shadow-[0_0_10px_rgba(254,44,85,0.4)]`}
          aria-hidden="true"
        />
      </span>
      <span className="font-extrabold tracking-[-0.03em] uppercase">LUMAE</span>
    </Link>
  );
}

