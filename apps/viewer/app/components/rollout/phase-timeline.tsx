// PhaseTimeline — compact horizontal Gantt: build → deploy → eval
// Ink styling: square bars, hairlines, ink/neutral fills, mono duration labels.
// Surfaces on the Overview tab. Tolerates missing timestamps (skips a phase).

import { buildFacts, valueOf } from "~/lib/build";
import { cn } from "~/lib/utils";
import type { EvalEpisode, RolloutBuild } from "~/lib/types";

// ---------------------------------------------------------------------------
// Duration formatting
// ---------------------------------------------------------------------------

function fmtDurSec(sec: number): string {
  if (sec < 60) return `${Math.round(sec)}s`;
  if (sec < 3600) return `${Math.floor(sec / 60)}m${Math.round(sec % 60)}s`;
  return `${Math.floor(sec / 3600)}h${Math.floor((sec % 3600) / 60)}m`;
}

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface Phase {
  label: string;
  start: number;
  end: number;
  /** Tailwind fill classes (bg + text) */
  fillCls: string;
  /** Tailwind bar classes for the bar background */
  barCls: string;
}

// ---------------------------------------------------------------------------
// Component
// ---------------------------------------------------------------------------

interface PhaseTimelineProps {
  build: RolloutBuild | null | undefined;
  evals: EvalEpisode[];
  className?: string;
}

export function PhaseTimeline({ build, evals, className }: PhaseTimelineProps) {
  // Collect phases
  const phases: Phase[] = [];

  // Build phase: the agent's part
  const span = build ? valueOf(buildFacts(build).span) : null;
  if (span != null) {
    phases.push({
      label: "Build",
      start: span.start,
      end: span.end,
      fillCls: "text-foreground",
      barCls: "bg-foreground/20 border border-border",
    });
  }

  // Terminal eval
  const terminal =
    evals.find((e) => e.eval_id === "terminal") ?? evals[evals.length - 1];

  // Deploy phase: from the end of the agent's part to eval.started_at
  if (span != null && terminal?.started_at != null && terminal.started_at > span.end) {
    phases.push({
      label: "Deploy",
      start: span.end,
      end: terminal.started_at,
      fillCls: "text-muted-foreground",
      barCls: "bg-muted border border-border/60",
    });
  }

  // Eval phase
  if (terminal?.started_at != null && terminal?.finished_at != null) {
    phases.push({
      label: "Eval",
      start: terminal.started_at,
      end: terminal.finished_at,
      fillCls: "text-foreground",
      barCls: "bg-foreground/10 border border-border",
    });
  }

  if (phases.length === 0) return null;

  // Calculate total span
  const tMin = Math.min(...phases.map((p) => p.start));
  const tMax = Math.max(...phases.map((p) => p.end));
  const totalSpan = tMax - tMin;
  if (totalSpan <= 0) return null;

  return (
    <div className={cn("space-y-2", className)}>
      <div className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
        Phase timeline
      </div>
      {/* Gantt bar */}
      <div className="relative h-6 border-[1.5px] border-border bg-muted/30 shadow-[var(--shadow-sm)]">
        {phases.map((p) => {
          const left = ((p.start - tMin) / totalSpan) * 100;
          const width = Math.max(((p.end - p.start) / totalSpan) * 100, 1);
          return (
            <div
              key={p.label}
              className={cn("absolute top-0 bottom-0", p.barCls)}
              style={{ left: `${left}%`, width: `${width}%` }}
              title={`${p.label}: ${fmtDurSec(p.end - p.start)}`}
            />
          );
        })}
      </div>
      {/* Legend row */}
      <div className="flex flex-wrap items-center gap-x-5 gap-y-1">
        {phases.map((p) => {
          const dur = p.end - p.start;
          return (
            <div key={p.label} className="flex items-center gap-1.5">
              <span
                className={cn("inline-block h-3 w-3 shrink-0", p.barCls)}
              />
              <span className={cn("font-mono text-[11px]", p.fillCls)}>
                {p.label}
              </span>
              <span className="font-mono text-[11px] text-muted-foreground tabular-nums">
                {fmtDurSec(dur)}
              </span>
            </div>
          );
        })}
        <span className="ml-auto font-mono text-[10px] text-muted-foreground tabular-nums">
          total {fmtDurSec(totalSpan)}
        </span>
      </div>
    </div>
  );
}
