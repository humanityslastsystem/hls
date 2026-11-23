// Compact one-line header for a single trajectory step. Shows step id, source,
// model, a message preview (first line, or "thinking…" for reasoning-only, or
// "⚙ toolNames" for tool-only steps), and — when timestamps exist — the step
// duration (+Δ) and time-since-start. Ink styling: mono numerals, hairlines.

import type { Step } from "~/lib/types";
import { cn } from "~/lib/utils";
import { getFirstLine } from "./content-renderer";

function formatMs(durationMs: number): string {
  const seconds = durationMs / 1000;
  if (seconds < 60) return `${seconds.toFixed(1)}s`;
  const minutes = Math.floor(seconds / 60);
  const rem = seconds % 60;
  return `${minutes}m ${rem.toFixed(0)}s`;
}

function formatStepDuration(
  prevTimestamp: string | null,
  currentTimestamp: string | null,
): string | null {
  if (!prevTimestamp || !currentTimestamp) return null;
  const durationMs =
    new Date(currentTimestamp).getTime() - new Date(prevTimestamp).getTime();
  if (durationMs < 0 || Number.isNaN(durationMs)) return null;
  return formatMs(durationMs);
}

const sourceColors: Record<string, string> = {
  system: "text-muted-foreground",
  user: "text-primary",
  agent: "text-foreground",
};

export function StepTrigger({
  step,
  prevTimestamp,
  startTimestamp,
}: {
  step: Step;
  prevTimestamp: string | null;
  startTimestamp: string | null;
}) {
  const stepDuration = formatStepDuration(prevTimestamp, step.timestamp);
  const sinceStart = formatStepDuration(startTimestamp, step.timestamp);
  const firstLine = getFirstLine(step.message);

  // Reasoning-only / tool-only steps carry no message text — surface a
  // meaningful fallback so the row never reads as empty.
  const toolNames = (step.tool_calls ?? []).map((t) => t.function_name).join(", ");
  const fallback = step.reasoning_content
    ? "thinking…"
    : toolNames
      ? `⚙ ${toolNames}`
      : null;

  return (
    <div className="flex min-w-0 flex-1 items-center gap-3 overflow-hidden">
      <div className="flex min-w-0 flex-1 items-center gap-2 overflow-hidden">
        <span className="shrink-0 font-mono text-xs tabular-nums text-muted-foreground">
          #{step.step_id}
        </span>
        <span
          className={cn(
            "shrink-0 font-mono text-xs font-medium",
            sourceColors[step.source] ?? "text-muted-foreground",
          )}
        >
          {step.source}
        </span>
        {step.model_name && (
          <span className="hidden shrink-0 font-mono text-xs text-muted-foreground sm:inline">
            {step.model_name}
          </span>
        )}
        <span className="min-w-0 truncate text-xs text-foreground/90">
          {firstLine || (
            <span className="italic text-muted-foreground">
              {fallback ?? "No message"}
            </span>
          )}
        </span>
      </div>
      {(sinceStart || stepDuration) && (
        <div className="flex shrink-0 items-center gap-3 font-mono text-[11px] tabular-nums text-muted-foreground">
          {stepDuration && <span>+{stepDuration}</span>}
          {sinceStart && <span className="text-muted-foreground/70">{sinceStart}</span>}
        </div>
      )}
    </div>
  );
}
