import React from "react";

interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: "primary" | "secondary" | "outline" | "ghost" | "pink";
  size?: "sm" | "md" | "lg";
  icon?: React.ReactNode;
  iconPosition?: "left" | "right";
  children: React.ReactNode;
}

export function Button({
  variant = "primary",
  size = "md",
  icon,
  iconPosition = "left",
  children,
  className = "",
  disabled,
  ...props
}: ButtonProps) {
  const baseStyles =
    "inline-flex items-center justify-center font-semibold rounded-full transition-all duration-200 select-none active:scale-[0.98] disabled:opacity-50 disabled:pointer-events-none cursor-pointer focus-visible:outline-2 focus-visible:outline-offset-2 focus-visible:outline-[#FE2C55]";

  const variantStyles = {
    primary:
      "bg-[#000000] text-white hover:bg-[#222222] shadow-sm hover:shadow-md",
    secondary:
      "bg-white text-[#111111] border border-[#E5E7EB] hover:bg-[#F8F9FA] hover:border-[#111111]/30 shadow-2xs",
    outline:
      "bg-transparent text-[#111111] border border-[#E5E7EB] hover:bg-[#F8F9FA] hover:border-[#111111]",
    ghost: "bg-transparent text-[#6B7280] hover:text-[#111111] hover:bg-[#F8F9FA]",
    pink: "bg-[#FE2C55] text-white hover:bg-[#E01E46] shadow-sm hover:shadow-md",
  };

  const sizeStyles = {
    sm: "text-xs px-3.5 py-1.5 gap-1.5",
    md: "text-sm px-5 py-2.5 gap-2",
    lg: "text-base px-7 py-3.5 gap-2.5",
  };

  return (
    <button
      className={`${baseStyles} ${variantStyles[variant]} ${sizeStyles[size]} ${className}`}
      disabled={disabled}
      {...props}
    >
      {icon && iconPosition === "left" && <span className="flex-shrink-0">{icon}</span>}
      <span>{children}</span>
      {icon && iconPosition === "right" && <span className="flex-shrink-0">{icon}</span>}
    </button>
  );
}

