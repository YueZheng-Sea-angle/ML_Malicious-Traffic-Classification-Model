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

/** 分类标签 -> 语义色（DataCon T1：firefox 为正常对照，其余为代理/隧道工具）。 */
export function labelTone(label: string): Tone {
  if (label === "firefox") return "success";
  if (label === "unknown") return "neutral";
  if (label === "openvpn-udp" || label === "openvpn-tls" || label === "wireguard-udp") {
    return "warning";
  }
  return "info";
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
