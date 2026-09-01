import type { ReactNode } from "react";

import { cn } from "@/lib/utils";

type Tone = "neutral" | "success" | "warning" | "danger" | "info";

const TONE_CLASS: Record<Tone, string> = {
  neutral: "bg-slate-100 text-slate-600",
  success: "bg-emerald-100 text-emerald-700",
  warning: "bg-amber-100 text-amber-700",
  danger: "bg-rose-100 text-rose-700",
  info: "bg-brand-100 text-brand-700",
};

export function Badge({
  children,
  tone = "neutral",
  className,
}: {
  children: ReactNode;
  tone?: Tone;
  className?: string;
}) {
  return (
    <span
      className={cn(
        "inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium",
        TONE_CLASS[tone],
        className,
      )}
    >
      {children}
    </span>
  );
}

/** 分类标签 -> 语义色，benign 为绿色，其余按威胁程度取暖色。 */
export function labelTone(label: string): Tone {
  if (label === "benign") return "success";
  if (label === "unknown") return "neutral";
  if (label === "ddos" || label === "ransomware") return "danger";
  return "warning";
}

export function statusTone(status: string): Tone {
  switch (status) {
    case "succeeded":
      return "success";
    case "failed":
      return "danger";
    case "running":
      return "info";
    default:
      return "neutral";
  }
}
