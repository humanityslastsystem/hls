// Segmented toggle — the shared Ink filled-active control used across the viewer
// (Timeline group/color, Source Artifact/Folder, Rollouts Flat/By task).
// Framed 1.5px border + offset shadow; the active option is filled foreground.
import type { ComponentType } from "react";

import { cn } from "~/lib/utils";

export interface SegOption<T extends string> {
  value: T;
  label: string;
  icon?: ComponentType<{ className?: string }>;
  disabled?: boolean;
  /** Tooltip of the option, such as why it is disabled. */
  title?: string;
}

export function Segmented<T extends string>({
  value,
  onChange,
  options,
  label,
  className,
}: {
  value: T;
  onChange: (v: T) => void;
  options: SegOption<T>[];
  /** Optional small uppercase prefix label (e.g. "group", "color"). */
  label?: string;
  className?: string;
}) {
  return (
    <div className={cn("flex items-center gap-1.5", className)}>
      {label && (
        <span className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
          {label}
        </span>
      )}
      <div className="inline-flex border-[1.5px] border-foreground p-0.5 shadow-[var(--shadow-sm)]">
        {options.map((o) => {
          const Icon = o.icon;
          const active = value === o.value;
          return (
            <button
              key={o.value}
              type="button"
              disabled={o.disabled}
              title={o.title}
              onClick={() => onChange(o.value)}
              className={cn(
                "inline-flex items-center gap-1 px-2 py-1 font-mono text-[11px] transition-colors",
                "disabled:cursor-not-allowed disabled:opacity-40",
                active
                  ? "bg-foreground text-background"
                  : "text-muted-foreground hover:text-foreground",
              )}
            >
              {Icon && <Icon className="h-3.5 w-3.5" />}
              {o.label}
            </button>
          );
        })}
      </div>
    </div>
  );
}
