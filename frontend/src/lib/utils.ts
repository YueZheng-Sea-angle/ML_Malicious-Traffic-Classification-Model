import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function formatBytes(bytes: number) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(2)} MB`;
}

export function formatPercent(value: number) {
  return `${(value * 100).toFixed(1)}%`;
}

export function formatTime(value?: string | null) {
  if (!value) return "—";
  return new Date(value).toLocaleString("zh-CN", { hour12: false });
}
