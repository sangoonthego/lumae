import React from "react";

interface StatusBadgeProps {
  variant: "pink" | "green" | "gray" | "outline" | "active";
  children: React.ReactNode;
  icon?: React.ReactNode;
  className?: string;
}

export function StatusBadge({
  variant,
  children,
  icon,
  className = "",
}: StatusBadgeProps) {
  const variantStyles = {
    pink: "bg-[#FFEBEF] text-[#FE2C55] border border-[#FE2C55]/15",
    green: "bg-[#DCFCE7]/70 text-[#16A34A] border border-[#86EFAC]/40",
    gray: "bg-[#F3F4F6] text-[#6B7280] border border-[#E5E7EB]",
    outline: "bg-white text-[#111111] border border-[#E5E7EB]",
    active: "bg-[#000000] text-white",
  };

  return (
    <span
      className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-bold tracking-tight select-none ${variantStyles[variant]} ${className}`}
    >
      {icon && <span className="flex-shrink-0">{icon}</span>}
      <span>{children}</span>
    </span>
  );
}

