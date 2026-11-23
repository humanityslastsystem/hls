import { useQuery } from "@tanstack/react-query";
import { ChevronDown, ChevronRight } from "lucide-react";
import { useState } from "react";

import { fileUrl } from "~/lib/api";
import { buildFiles, buildOf, roleLogs } from "~/lib/build";
import type { RoleLog } from "~/lib/build";
import { summarizeSlo } from "~/lib/slo";
import type { EvalEpisode, RolloutRecord } from "~/lib/types";

// ---------------------------------------------------------------------------
// Format helpers
// ---------------------------------------------------------------------------

function fmtDuration(
  a: number | null | undefined,
  b: number | null | undefined,
): string {
  if (a == null || b == null) return "—";
  const s = b - a;
  if (s < 60) return `${Math.round(s)}s`;
  if (s < 3600) return `${Math.floor(s / 60)}m ${Math.round(s % 60)}s`;
  return `${Math.floor(s / 3600)}h ${Math.floor((s % 3600) / 60)}m`;
}

// ---------------------------------------------------------------------------
// CR-collapse: simulate terminal carriage-return overwrite behavior
// ---------------------------------------------------------------------------

function crCollapse(raw: string): string {
  return raw
    .split("\n")
    .map((line) => {
      const parts = line.split("\r");
      // find last non-empty part
      let result = "";
      for (const p of parts) {
        if (p.length > 0) result = p;
        else if (parts.length === 1) result = p;
      }
      return result;
    })
    .join("\n");
}

// ---------------------------------------------------------------------------
// Parse checkpoint string
// ---------------------------------------------------------------------------

function parseCheckpoint(checkpoint: string): {
  backend: string;
  artifactId: string;
} {
  const colonIdx = checkpoint.indexOf(":");
  if (colonIdx === -1) {
    return { backend: "unknown", artifactId: checkpoint };
  }
  return {
    backend: checkpoint.slice(0, colonIdx),
    artifactId: checkpoint.slice(colonIdx + 1),
  };
}

// ---------------------------------------------------------------------------
// Stat tile (same pattern as build-panel.tsx)
// ---------------------------------------------------------------------------

function StatTile({
  label,
  value,
}: {
  label: string;
  value: React.ReactNode;
}) {
  return (
    <div className="ink-block bg-card px-3 py-2">
      <div className="text-[10px] font-mono uppercase tracking-widest text-muted-foreground">
        {label}
      </div>
      <div className="mt-1 font-mono text-sm font-semibold tabular-nums leading-snug text-foreground">
        {value}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Section header
// ---------------------------------------------------------------------------

function SectionHeader({ children }: { children: React.ReactNode }) {
  return (
    <h3 className="font-mono text-[10px] font-semibold uppercase tracking-widest text-muted-foreground">
      {children}
    </h3>
  );
}

// ---------------------------------------------------------------------------
// Role section: image digest + optional logs
// ---------------------------------------------------------------------------

// One log of a role, fetched when it is opened.
function LogView({
  path,
  open,
  task,
  rolloutId,
}: {
  path: string;
  open: boolean;
  task: string;
  rolloutId: string;
}) {
  const { data, isLoading, isError } = useQuery({
    queryKey: ["container-log", task, rolloutId, path],
    queryFn: async () => {
      const r = await fetch(fileUrl(task, rolloutId, path));
      if (!r.ok) throw new Error(`fetch log: ${r.status}`);
      return r.text();
    },
    enabled: open,
    staleTime: Infinity,
  });

  if (!open) return null;
  return (
    <div className="border-t border-border">
      {isLoading && (
        <p className="px-3 py-2 font-mono text-xs text-muted-foreground animate-pulse">
          Loading log…
        </p>
      )}
      {isError && (
        <p className="px-3 py-2 font-mono text-xs text-muted-foreground">
          Log not available.
        </p>
      )}
      {data != null && (
        <pre className="overflow-auto max-h-80 font-mono text-xs px-3 py-2 text-foreground whitespace-pre-wrap leading-5">
          {crCollapse(data)}
        </pre>
      )}
    </div>
  );
}

function RoleSection({
  role,
  imageId,
  logs,
  noLogReason,
  task,
  rolloutId,
}: {
  role: string;
  imageId: string | null;
  /** The role's logs, a button each; none shows "log not captured". */
  logs: RoleLog[];
  /** Why there is no log, when the record says; shown as the marker's tooltip. */
  noLogReason?: string;
  task: string;
  rolloutId: string;
}) {
  // The label of the log shown, if any.
  const [openLog, setOpenLog] = useState<string | null>(null);

  const imageShort = imageId
    ? imageId.length > 60
      ? `…${imageId.slice(-56)}`
      : imageId
    : null;

  return (
    <div className="border border-border">
      {/* Role header + digest */}
      <div className="flex items-start gap-3 px-3 py-2">
        <span className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground w-20 shrink-0 pt-0.5">
          {role}
        </span>
        {imageShort ? (
          <span className="font-mono text-xs text-foreground break-all flex-1" title={imageId ?? undefined}>
            {imageShort}
          </span>
        ) : (
          <span className="font-mono text-xs text-muted-foreground/50 italic flex-1">
            no image digest
          </span>
        )}
        {logs.length > 0 ? (
          <div className="flex shrink-0 items-center gap-3">
            {logs.map((log) => (
              <button
                key={log.label}
                type="button"
                onClick={() => setOpenLog((o) => (o === log.label ? null : log.label))}
                title={log.path}
                className="font-mono text-[10px] text-muted-foreground hover:text-foreground transition-colors flex items-center gap-1"
              >
                {openLog === log.label ? (
                  <ChevronDown className="h-3 w-3" />
                ) : (
                  <ChevronRight className="h-3 w-3" />
                )}
                {log.label}
              </button>
            ))}
          </div>
        ) : (
          <span
            className="shrink-0 font-mono text-[10px] text-muted-foreground/50 italic"
            title={noLogReason}
          >
            log not captured
          </span>
        )}
      </div>

      {logs.map((log) => (
        <LogView
          key={log.path}
          path={log.path}
          open={openLog === log.label}
          task={task}
          rolloutId={rolloutId}
        />
      ))}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main panel
// ---------------------------------------------------------------------------

export function DeployPanel({ rec }: { rec: RolloutRecord }) {
  // Pick the "terminal" eval episode (the one with eval_id === "terminal",
  // or fall back to the last eval).
  const terminal: EvalEpisode | undefined =
    rec.evals.find((e) => e.eval_id === "terminal") ??
    rec.evals[rec.evals.length - 1];

  const checkpoint = terminal?.checkpoint ?? "";
  const { backend, artifactId } = parseCheckpoint(checkpoint);

  const checkpointDisplay =
    artifactId.length > 60 ? `${artifactId.slice(0, 57)}…` : artifactId;

  // SLO p99 — score.breakdown.slo is nested per tenant; see ~/lib/slo for the
  // exact diagnostics shape (StreamingLatencySLO's p99_ttft_ms vs NoSLO's {n}).
  const slo = summarizeSlo(terminal?.score);
  const sloLabel = slo?.label ?? "SLO p99";
  const p99Ms =
    slo?.p99Ms != null ? `${Math.round(slo.p99Ms)} ms` : "—";

  const imageDigests = rec.config.image_digests ?? {};

  const containerLogs = terminal?.container_logs ?? {};
  const hasBuild = "value" in buildOf(rec);
  const producerLog = buildFiles(rec).producerLog;

  return (
    <div className="w-full space-y-6 px-6 py-4">
      {/* ------------------------------------------------------------------ */}
      {/* Checkpoint section                                                   */}
      {/* ------------------------------------------------------------------ */}
      <section className="space-y-2">
        <SectionHeader>Checkpoint</SectionHeader>
        {terminal ? (
          <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
            <StatTile label="Backend" value={backend} />
            <div className="col-span-1 sm:col-span-3">
              <StatTile
                label="Checkpoint"
                value={
                  <span
                    className="truncate text-[11px]"
                    title={artifactId}
                  >
                    {checkpointDisplay || "—"}
                  </span>
                }
              />
            </div>
          </div>
        ) : (
          <div className="border border-border bg-muted/20 p-4">
            <p className="font-mono text-xs text-muted-foreground">
              No eval episodes recorded.
            </p>
          </div>
        )}
      </section>

      {/* ------------------------------------------------------------------ */}
      {/* Timing                                                               */}
      {/* ------------------------------------------------------------------ */}
      {terminal && (
        <section className="space-y-2">
          <SectionHeader>Timing</SectionHeader>
          <div className="grid grid-cols-2 gap-2 sm:grid-cols-4">
            <StatTile
              label="Eval duration"
              value={fmtDuration(terminal.started_at, terminal.finished_at)}
            />
            <StatTile label={sloLabel} value={p99Ms} />
          </div>
        </section>
      )}

      {/* ------------------------------------------------------------------ */}
      {/* Deployed roles                                                        */}
      {/* ------------------------------------------------------------------ */}
      <section className="space-y-2">
        <SectionHeader>Deployed roles</SectionHeader>
        {Object.keys(imageDigests).length === 0 && !hasBuild ? (
          <div className="border border-border bg-muted/20 p-4">
            <p className="font-mono text-xs text-muted-foreground">No image digests recorded.</p>
          </div>
        ) : (
          <div className="space-y-1">
            {/* Producer (build agent) row — its log in the transcript's copy of the agent's logs */}
            {hasBuild && (
              <RoleSection
                key="producer"
                role="producer"
                imageId={null}
                logs={"value" in producerLog ? [{ label: "log", path: producerLog.value }] : []}
                noLogReason={"absent" in producerLog ? producerLog.absent : undefined}
                task={rec.task_id}
                rolloutId={rec.rollout_id}
              />
            )}
            {/* Each image's role: its logs among the eval's role logs, <role>.out and <role>.err */}
            {Object.entries(imageDigests).map(([role, imageId]) => {
              const found = roleLogs(rec.instance, containerLogs, role);
              return (
                <RoleSection
                  key={role}
                  role={role}
                  imageId={imageId}
                  logs={"value" in found ? found.value : []}
                  noLogReason={"absent" in found ? found.absent : undefined}
                  task={rec.task_id}
                  rolloutId={rec.rollout_id}
                />
              );
            })}
          </div>
        )}
      </section>
    </div>
  );
}
