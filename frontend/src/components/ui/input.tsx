import type { InputHTMLAttributes } from "react";

import { cn } from "@/lib/utils";

export interface InputProps extends InputHTMLAttributes<HTMLInputElement> {
  label?: string;
  hint?: string;
  error?: string;
}

export function Input({ className, label, hint, error, id, ...props }: InputProps) {
  const inputId = id ?? props.name;
  return (
    <div className="space-y-1.5">
      {label ? (
        <label htmlFor={inputId} className="block text-sm font-medium text-slate-700 dark:text-slate-300">
          {label}
        </label>
      ) : null}
      <input
        id={inputId}
        className={cn(
          "w-full rounded-lg border bg-white px-3 py-2 text-sm text-slate-900 shadow-sm transition-colors",
          "placeholder:text-slate-400 focus:border-brand-500 focus:outline-none focus:ring-2 focus:ring-brand-500/20",
          "dark:border-slate-600 dark:bg-slate-950 dark:text-slate-100 dark:placeholder:text-slate-500",
          error ? "border-rose-400" : "border-slate-300",
          className,
        )}
        {...props}
      />
      {error ? <p className="text-xs text-rose-600">{error}</p> : null}
      {!error && hint ? <p className="text-xs text-slate-400">{hint}</p> : null}
    </div>
  );
}
