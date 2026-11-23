// Compare view — side-by-side comparison of selected rollouts.
// Rows = metrics, columns = one per rollout.
// Entry point: /compare?ids=<enc task/id>,<enc task/id>,...

import { useQueries } from "@tanstack/react-query";
import { ArrowLeft, AlertCircle } from "lucide-react";
import { Link, useSearchParams } from "react-router";

import { GateBadge, StatusBadge, BackendBadge } from "~/components/rollout/badges";
import { Fact } from "~/components/rollout/build-fact";
import { fetchRollout } from "~/lib/api";
import { recordFacts, valueOf } from "~/lib/build";
import { summarizeSlo } from "~/lib/slo";
import { cn } from "~/lib/utils";
import type { RolloutRecord } from "~/lib/types";

// ---------------------------------------------------------------------------
// Format helpers (no middot — commas/slashes/spaces only)
// ---------------------------------------------------------------------------

function fmtScore(v: number | null | undefined): string {
  if (v == null) return "—";
  return (v * 100).toFixed(1) + "%";
}

function fmtCost(v: number): string {
  if (v < 0.01) return `$${v.toFixed(4)}`;
  return `$${v.toFixed(2)}`;
}

function fmtMs(ms: number | null | undefined): string {
  if (ms == null) return "—";
  return `${Math.round(ms).toLocaleString()} ms`;
}

function fmtDuration(a: number, b: number): string {
  const s = b - a;
  if (s < 60) return `${Math.round(s)}s`;
  if (s < 3600) return `${Math.floor(s / 60)}m ${Math.round(s % 60)}s`;
  return `${Math.floor(s / 3600)}h ${Math.floor((s % 3600) / 60)}m`;
}

// ---------------------------------------------------------------------------
// Parse ?ids= param into [{task, id}] pairs.
// Each entry is "<task>/<rollout_id>" — split on LAST slash.
// ---------------------------------------------------------------------------

function parseIds(raw: string | null): Array<{ task: string; id: string }> {
  if (!raw) return [];
  return raw
    .split(",")
    .map((s) => decodeURIComponent(s.trim()))
    .filter(Boolean)
    .map((s) => {
      const lastSlash = s.lastIndexOf("/");
      if (lastSlash < 0) return null;
      return { task: s.slice(0, lastSlash), id: s.slice(lastSlash + 1) };
    })
    .filter((x): x is { task: string; id: string } => x !== null);
}

// ---------------------------------------------------------------------------
// Metric row definitions
// ---------------------------------------------------------------------------

interface MetricRow {
  key: string;
  label: string;
  /** Extract display value from a RolloutRecord */
  value: (rec: RolloutRecord) => React.ReactNode;
  /** Extract numeric value for best-value comparison (higher = better unless invert=true) */
  numeric?: (rec: RolloutRecord) => number | null;
  invert?: boolean; // lower is better
}

const METRIC_ROWS: MetricRow[] = [
  {
    key: "status",
    label: "Status",
    value: (rec) => <StatusBadge status={rec.status} />,
    // no numeric comparison for status
  },
  {
    key: "score",
    label: "Score",
    value: (rec) => {
      const terminal = terminalEval(rec);
      const v = typeof terminal?.score?.value === "number" ? terminal.score.value : null;
      return (
        <span className="font-mono tabular-nums text-xs">
          {fmtScore(v)}
        </span>
      );
    },
    numeric: (rec) => {
      const terminal = terminalEval(rec);
      const v = terminal?.score?.value;
      return typeof v === "number" ? v : null;
    },
  },
  {
    key: "gate",
    label: "Gate",
    value: (rec) => {
      const terminal = terminalEval(rec);
      const passed =
        (terminal?.verdict_summary?.gate_passed as boolean | null | undefined) ??
        (terminal?.score?.gate_passed as boolean | null | undefined) ??
        null;
      return <GateBadge passed={passed} />;
    },
    numeric: (rec) => {
      const terminal = terminalEval(rec);
      const passed =
        (terminal?.verdict_summary?.gate_passed as boolean | null | undefined) ??
        (terminal?.score?.gate_passed as boolean | null | undefined) ??
        null;
      return passed == null ? null : passed ? 1 : 0;
    },
  },
  {
    key: "p99",
    label: "SLO p99",
    value: (rec) => {
      const slo = terminalSlo(rec);
      if (slo && slo.metric === "none") {
        return (
          <span className="font-mono tabular-nums text-xs text-muted-foreground">
            no SLO{slo.n != null ? ` (n=${slo.n})` : ""}
          </span>
        );
      }
      return (
        <span className="font-mono tabular-nums text-xs">
          {slo ? fmtMs(slo.p99Ms) : fmtMs(null)}
          {slo?.metric === "ttft" ? " (TTFT)" : ""}
        </span>
      );
    },
    numeric: (rec) => terminalSlo(rec)?.p99Ms ?? null,
    invert: true, // lower is better
  },
  {
    key: "cost",
    label: "Build cost",
    value: (rec) => (
      <span className="font-mono tabular-nums text-xs">
        <Fact fact={recordFacts(rec).cost} format={fmtCost} />
      </span>
    ),
    numeric: (rec) => valueOf(recordFacts(rec).cost),
    invert: true,
  },
  {
    key: "model",
    label: "Model fingerprint",
    value: (rec) => {
      const fp = modelFingerprint(rec);
      return (
        <span className="font-mono text-xs truncate max-w-[140px] block" title={fp ?? undefined}>
          {fp ?? "—"}
        </span>
      );
    },
  },
  {
    key: "backend",
    label: "Backend",
    value: (rec) => {
      const substrate = rec.config?.substrate as Record<string, unknown> | undefined;
      const kind = substrate?.kind as string | null | undefined;
      return <BackendBadge backend={kind} />;
    },
  },
  {
    key: "duration",
    label: "Build duration",
    value: (rec) => (
      <span className="font-mono tabular-nums text-xs">
        <Fact fact={recordFacts(rec).span} format={(sp) => fmtDuration(sp.start, sp.end)} />
      </span>
    ),
    numeric: (rec) => {
      const span = valueOf(recordFacts(rec).span);
      return span == null ? null : span.end - span.start;
    },
    invert: true,
  },
];

// ---------------------------------------------------------------------------
// Data accessors
// ---------------------------------------------------------------------------

function terminalEval(rec: RolloutRecord) {
  const evals = rec.evals ?? [];
  return evals.find((e) => e.eval_id === "terminal") ?? evals[evals.length - 1] ?? null;
}

function terminalSlo(rec: RolloutRecord) {
  const terminal = terminalEval(rec);
  if (!terminal) return null;
  return summarizeSlo(terminal.score);
}

function modelFingerprint(rec: RolloutRecord): string | null {
  const spec = rec.config?.spec as Record<string, unknown> | undefined;
  const model = spec?.model as Record<string, unknown> | undefined;
  return (model?.fingerprint as string | null | undefined) ?? null;
}

// ---------------------------------------------------------------------------
// Best-value computation (per metric row across loaded rollouts)
// ---------------------------------------------------------------------------

function bestIndex(
  row: MetricRow,
  recs: (RolloutRecord | null)[],
): number | null {
  if (!row.numeric) return null;
  let best: number | null = null;
  let bestIdx: number | null = null;
  recs.forEach((rec, i) => {
    if (!rec) return;
    const v = row.numeric!(rec);
    if (v == null) return;
    if (best == null) {
      best = v;
      bestIdx = i;
    } else {
      const isBetter = row.invert ? v < best : v > best;
      if (isBetter) {
        best = v;
        bestIdx = i;
      }
    }
  });
  return bestIdx;
}

// ---------------------------------------------------------------------------
// Metric label cell
// ---------------------------------------------------------------------------

function MetricLabel({ label }: { label: string }) {
  return (
    <div className="flex items-center border-r border-border px-3 py-3 min-w-[130px]">
      <span className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground font-semibold whitespace-nowrap">
        {label}
      </span>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Column header
// ---------------------------------------------------------------------------

function ColumnHeader({
  task,
  id,
  rec,
  loading,
  error,
}: {
  task: string;
  id: string;
  rec?: RolloutRecord;
  loading: boolean;
  error?: Error | null;
}) {
  const shortId = id.length > 22 ? id.slice(0, 22) : id;
  return (
    <div className="flex flex-col gap-1 border-l border-border px-3 py-2 min-w-[180px] max-w-[240px]">
      <Link
        to={`/rollouts/${encodeURIComponent(task)}/${encodeURIComponent(id)}`}
        className="font-mono text-xs text-muted-foreground hover:text-foreground transition-colors truncate"
        title={`${task}/${id}`}
      >
        {task}
      </Link>
      <Link
        to={`/rollouts/${encodeURIComponent(task)}/${encodeURIComponent(id)}`}
        className="font-mono text-[11px] text-foreground hover:underline transition-colors tabular-nums truncate"
        title={id}
      >
        {shortId}
      </Link>
      {loading && (
        <span className="font-mono text-[10px] text-muted-foreground animate-pulse">
          loading…
        </span>
      )}
      {error && !loading && (
        <span className="font-mono text-[10px] text-destructive flex items-center gap-1">
          <AlertCircle className="h-3 w-3 shrink-0" />
          fetch error
        </span>
      )}
      {rec && !loading && !error && (
        <StatusBadge status={rec.status} />
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main compare page
// ---------------------------------------------------------------------------

export default function ComparePage() {
  const [searchParams] = useSearchParams();
  const pairs = parseIds(searchParams.get("ids"));

  // Fetch all rollouts in parallel
  const results = useQueries({
    queries: pairs.map(({ task, id }) => ({
      queryKey: ["rollout", task, id],
      queryFn: () => fetchRollout(task, id),
      retry: 1,
    })),
  });

  // Empty state
  if (pairs.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center py-20 gap-4">
        <p className="font-mono text-sm text-muted-foreground">
          No rollouts selected for comparison.
        </p>
        <Link
          to="/"
          className="inline-flex items-center gap-1.5 font-mono text-xs text-muted-foreground hover:text-foreground transition-colors"
        >
          <ArrowLeft className="h-3.5 w-3.5" />
          Back to rollouts
        </Link>
      </div>
    );
  }

  const recs = results.map((r) => r.data ?? null);

  return (
    <div className="flex flex-col min-h-0">
      {/* Page header */}
      <div className="flex items-center justify-between border-b border-border px-4 py-2 shrink-0">
        <div className="flex items-center gap-2">
          <Link
            to="/"
            className="inline-flex items-center gap-1 font-mono text-xs text-muted-foreground hover:text-foreground transition-colors"
          >
            <ArrowLeft className="h-3.5 w-3.5" />
            Rollouts
          </Link>
          <span className="text-muted-foreground text-xs">/</span>
          <span className="font-mono text-xs text-foreground font-medium">
            Compare ({pairs.length})
          </span>
        </div>
      </div>

      {/* Comparison table */}
      <div className="overflow-auto flex-1">
        <table className="border-collapse w-max min-w-full">
          {/* Column headers */}
          <thead>
            <tr>
              {/* Row label header (empty corner) */}
              <th className="sticky left-0 z-20 bg-background border-b border-r border-border px-3 py-2 min-w-[130px]" />
              {pairs.map(({ task, id }, i) => (
                <th
                  key={`${task}/${id}`}
                  className="border-b border-l border-border bg-background sticky top-0 z-10 align-top"
                >
                  <ColumnHeader
                    task={task}
                    id={id}
                    rec={recs[i] ?? undefined}
                    loading={results[i].isLoading}
                    error={results[i].error}
                  />
                </th>
              ))}
            </tr>
          </thead>

          {/* Metric rows */}
          <tbody>
            {METRIC_ROWS.map((row) => {
              const best = bestIndex(row, recs);
              return (
                <tr key={row.key} className="group">
                  {/* Metric label (sticky left) */}
                  <td className="sticky left-0 z-10 bg-background border-b border-border">
                    <MetricLabel label={row.label} />
                  </td>

                  {/* Value per rollout */}
                  {pairs.map(({ task, id }, i) => {
                    const rec = recs[i];
                    const result = results[i];
                    const isBest = best === i;

                    return (
                      <td
                        key={`${task}/${id}`}
                        className={cn(
                          "border-b border-l border-border px-3 py-3 min-w-[180px] max-w-[240px] align-middle",
                          isBest && "bg-accent/30",
                        )}
                      >
                        {result.isLoading ? (
                          <span className="font-mono text-xs text-muted-foreground animate-pulse">
                            …
                          </span>
                        ) : result.error ? (
                          <span className="font-mono text-xs text-destructive/70">—</span>
                        ) : rec ? (
                          <div className={cn(isBest && "font-semibold")}>
                            {row.value(rec)}
                          </div>
                        ) : (
                          <span className="font-mono text-xs text-muted-foreground">—</span>
                        )}
                      </td>
                    );
                  })}
                </tr>
              );
            })}
          </tbody>
        </table>

        {/* Legend */}
        <div className="px-4 py-3 border-t border-border">
          <p className="font-mono text-[10px] text-muted-foreground">
            Shaded cells indicate best value per row (highest score/gate, lowest p99/cost/duration).
          </p>
        </div>
      </div>
    </div>
  );
}
