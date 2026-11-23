import { useVirtualizer } from "@tanstack/react-virtual";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import Fuse from "fuse.js";

import { fetchTracePage, fetchTraceRow, fetchTraceSummary } from "~/lib/api";
import { evalProgress } from "~/lib/eval-progress";
import type { EvalProgress } from "~/lib/eval-progress";
import { cn } from "~/lib/utils";
import type { RolloutEvent } from "~/components/rollout/use-rollout-events";
import type { RolloutRecord, TraceSummary, VerifierSummary } from "~/lib/types";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

interface TraceRow {
  turn_index: number | null;
  request_id: string;
  session_id: string | null;
  tenant_id?: string | null;
  prompt: string | null;
  artifact_text: string;
  artifact_tokens: number | null;
  ground_truth: string | null;
  correct: boolean | null;
  token_match: boolean | null;
  worst_gap: number | null;
  n_tokens: number | null;
  first_above_cap_index: number | null;
  latency_ms: number | null;
  error: string | null;
  reference_text?: string | null;
  predicted?: string | null;
  oracle_predicted?: string | null;
  oracle_correct?: boolean | null;
  verifier_cells?: { name: string; passed: boolean | null; detail: string }[];
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const ALL_ROWS_LIMIT = 1_000_000; // fetch every row in one shot (server has no max)
const ROW_HEIGHT = 28; // px — compact single-line rows
const EXPANDED_HEIGHT = 200; // px — expanded row height estimate

// ---------------------------------------------------------------------------
// Client-side filter / search / sort
// ---------------------------------------------------------------------------

export interface SortState {
  col: string;
  dir: "asc" | "desc";
}

const FUSE_KEYS = [
  "request_id",
  "artifact_text",
  "predicted",
  "oracle_predicted",
  "ground_truth",
  "reference_text",
  "session_id",
];

// Filter by the `only` chip selection. Mirrors the old server-side `only` param.
function filterByOnly(rows: TraceRow[], only: OnlyFilter): TraceRow[] {
  if (!only) return rows;
  if (only === "incorrect") return rows.filter((r) => r.correct === false);
  if (only === "diverged") return rows.filter((r) => r.token_match === false);
  if (only === "failing")
    return rows.filter((r) => (r.verifier_cells ?? []).some((c) => c.passed === false));
  // Any other string is a session id.
  return rows.filter((r) => r.session_id === only);
}

// Fuzzy search across the text-ish columns.
// The caller is expected to pass a memoized Fuse instance keyed on allRows so
// that the index is not rebuilt on every sort/filter change.
function searchRows(fuse: Fuse<TraceRow> | null, rows: TraceRow[], query: string): TraceRow[] {
  const q = query.trim();
  if (!q) return rows;
  if (!fuse) return rows;
  return fuse.search(q).map((r) => r.item);
}

// Extract the comparable value for a sort column. Returns null/undefined for
// "missing" so those sort last regardless of direction.
function sortValue(row: TraceRow, col: string): unknown {
  if (col.startsWith("verifier:")) {
    const idx = Number(col.slice("verifier:".length));
    return row.verifier_cells?.[idx]?.passed ?? null;
  }
  return (row as unknown as Record<string, unknown>)[col];
}

function compareValues(a: unknown, b: unknown): number {
  const aMissing = a == null;
  const bMissing = b == null;
  // Nulls/undefined always sort LAST (in both directions — handled by caller
  // re-flipping; here we keep them after non-missing in the ascending frame).
  if (aMissing && bMissing) return 0;
  if (aMissing) return 1;
  if (bMissing) return -1;
  if (typeof a === "number" && typeof b === "number") return a - b;
  if (typeof a === "boolean" && typeof b === "boolean") {
    return a === b ? 0 : a ? 1 : -1;
  }
  const as = String(a).toLowerCase();
  const bs = String(b).toLowerCase();
  return as < bs ? -1 : as > bs ? 1 : 0;
}

// Stable sort. Nulls/undefined always sink to the bottom in both directions.
function sortRows(rows: TraceRow[], sortState: SortState | null): TraceRow[] {
  if (!sortState) return rows;
  const { col, dir } = sortState;
  const indexed = rows.map((row, i) => ({ row, i }));
  indexed.sort((x, y) => {
    const av = sortValue(x.row, col);
    const bv = sortValue(y.row, col);
    const aMissing = av == null;
    const bMissing = bv == null;
    // Missing values pinned to the end regardless of direction.
    if (aMissing || bMissing) {
      if (aMissing && bMissing) return x.i - y.i;
      return aMissing ? 1 : -1;
    }
    const cmp = compareValues(av, bv);
    if (cmp !== 0) return dir === "asc" ? cmp : -cmp;
    return x.i - y.i; // stable
  });
  return indexed.map((e) => e.row);
}

// A clickable header cell that drives the shared sort state.
function SortableHeaderCell({
  col,
  label,
  className,
  sortState,
  onSort,
  title,
  children,
}: {
  col: string;
  label: React.ReactNode;
  className?: string;
  sortState: SortState | null;
  onSort: (col: string) => void;
  title?: string;
  children?: React.ReactNode;
}) {
  const active = sortState?.col === col;
  const arrow = active ? (sortState?.dir === "asc" ? " ▲" : " ▼") : "";
  return (
    <div
      className={cn("cursor-pointer select-none hover:text-foreground", className)}
      onClick={() => onSort(col)}
      title={title}
    >
      {label}
      {children}
      {arrow}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Format helpers
// ---------------------------------------------------------------------------

function fmtMs(ms: number | null | undefined): string {
  if (ms == null) return "—";
  return `${Math.round(ms)}ms`;
}

function truncate(s: string | null | undefined, n: number): string {
  if (!s) return "—";
  return s.length > n ? `${s.slice(0, n)}…` : s;
}

function fmtVerifierRollup(v: VerifierSummary): string {
  if (v.kind === "token_level") {
    if (v.flip_rate != null && v.rate_ratio != null) {
      const flipPct = (v.flip_rate * 100).toFixed(2);
      const curve = v.curve_fired_levels;
      const curveText =
        curve == null
          ? ""
          : curve.length === 0
            ? ", curve clean"
            : `, curve fired @ ${curve.map((c) => c.toFixed(2)).join(", ")} nats`;
      return `flip ${flipPct}% (${v.rate_ratio.toFixed(2)}x of ceiling)${curveText}`;
    }
    return "";
  }
  if (v.kind === "task_relative") {
    const art = v.artifact_value != null ? `${(v.artifact_value * 100).toFixed(1)}%` : "?";
    const ref = v.reference_value != null ? `${(v.reference_value * 100).toFixed(1)}%` : "?";
    return `${art} / Oracle ${ref}`;
  }
  if (v.kind === "task_absolute") {
    const art = v.artifact_value != null ? `${(v.artifact_value * 100).toFixed(1)}%` : "?";
    const min = v.min_value != null ? `threshold ${(v.min_value * 100).toFixed(1)}%` : "";
    return min ? `${art} / ${min}` : art;
  }
  return "";
}

// ---------------------------------------------------------------------------
// Rollup row
// ---------------------------------------------------------------------------

function RollupRow({ summary, loaded, total }: { summary: TraceSummary; loaded: number; total: number }) {
  const kind = summary.verifier_kind ?? null;
  const v0 = summary.verifiers?.[0] ?? null;
  // Narrow the union once: the token_level card below reads TokenLevelSummary
  // fields, which `kind` (the trace-level verifier_kind string) cannot narrow.
  const tokenV0 = v0 && v0.kind === "token_level" ? v0 : null;

  return (
    <div className="flex flex-wrap items-center gap-x-6 gap-y-1 ink-block bg-card px-4 py-2 font-mono text-xs">
      <span className="tabular-nums font-semibold">
        {summary.total.toLocaleString()} requests
      </span>

      {/* token_level: pooled flip rate ceiling (rate_multiplier x q_max) plus a
          flip-count curve (count of gaps >= c must stay under its calibrated
          ceiling at each level c); this is a trace-level aggregate gate, not a
          per-token verdict. */}
      {kind === "token_level" && tokenV0 && (
        <>
          {tokenV0.flip_rate != null && tokenV0.rate_ratio != null && (
            <span className="flex items-center gap-2">
              <span className="text-muted-foreground">flip rate</span>
              <span className="tabular-nums text-foreground font-semibold">
                {(tokenV0.flip_rate * 100).toFixed(2)}%
              </span>
              <span className="text-muted-foreground text-[10px]">
                ({tokenV0.rate_ratio.toFixed(2)}x of ceiling)
              </span>
              {tokenV0.curve_fired_levels != null && (
                <span className="text-muted-foreground text-[10px]">
                  {tokenV0.curve_fired_levels.length === 0
                    ? "curve clean"
                    : `curve fired @ ${tokenV0.curve_fired_levels.map((c) => c.toFixed(2)).join(", ")} nats`}
                </span>
              )}
              <span
                title="Two-threshold calibrated gate: passes when the pooled flip rate stays under its calibrated ceiling (rate_multiplier x q_max) and the flip-count curve stays under its calibrated ceiling at every gap level c. This is a trace-level aggregate, not a per-token verdict."
                className={cn(
                  "inline-flex items-center px-1.5 py-0 text-[10px] font-mono font-semibold border",
                  tokenV0.gate_passed
                    ? "bg-ok/15 text-ok border-ok/30"
                    : "bg-destructive/15 text-destructive border-destructive/30",
                )}
              >
                {tokenV0.gate_passed ? "PASS" : "FAIL"}
              </span>
            </span>
          )}

        </>
      )}

      {/* task_relative: Artifact vs Oracle gate — make it the visual focus */}
      {kind === "task_relative" && summary.artifact_value != null && summary.reference_value != null && (
        <span className="flex items-center gap-2">
          <span className="text-muted-foreground">Artifact</span>
          <span className="tabular-nums text-foreground font-semibold">
            {(summary.artifact_value * 100).toFixed(1)}%
          </span>
          {summary.tol != null && summary.tol !== 0 && (
            <span className="text-muted-foreground text-[10px]">
              +&thinsp;{(summary.tol * 100).toFixed(1)}%
            </span>
          )}
          <span className="text-muted-foreground">&ge;</span>
          <span className="text-muted-foreground">Oracle</span>
          <span className="tabular-nums text-foreground font-semibold">
            {(summary.reference_value * 100).toFixed(1)}%
          </span>
          <span className={cn(
            "inline-flex items-center px-1.5 py-0 text-[10px] font-mono font-semibold border",
            summary.gate_passed
              ? "bg-ok/15 text-ok border-ok/30"
              : "bg-destructive/15 text-destructive border-destructive/30"
          )}>
            {summary.gate_passed ? "PASS" : "FAIL"}
          </span>
        </span>
      )}

      {/* task_absolute: score vs threshold */}
      {kind === "task_absolute" && summary.artifact_value != null && (
        <span className="flex items-center gap-2">
          <span className="text-muted-foreground">Artifact</span>
          <span className="tabular-nums text-foreground font-semibold">
            {(summary.artifact_value * 100).toFixed(1)}%
          </span>
          {summary.min_value != null && (
            <>
              <span className="text-muted-foreground">threshold &ge;</span>
              <span className="tabular-nums text-foreground font-semibold">
                {(summary.min_value * 100).toFixed(1)}%
              </span>
            </>
          )}
          <span className={cn(
            "inline-flex items-center px-1.5 py-0 text-[10px] font-mono font-semibold border",
            summary.gate_passed
              ? "bg-ok/15 text-ok border-ok/30"
              : "bg-destructive/15 text-destructive border-destructive/30"
          )}>
            {summary.gate_passed ? "PASS" : "FAIL"}
          </span>
        </span>
      )}

      {/* p99 always shown */}
      {summary.p99_ms != null && (
        <span className="text-muted-foreground">
          p99{" "}
          <span className="tabular-nums text-foreground font-semibold">
            {Math.round(summary.p99_ms).toLocaleString()} ms
          </span>
        </span>
      )}

      <span className="ml-auto font-mono text-[10px] text-muted-foreground tabular-nums">
        showing {loaded.toLocaleString()} of {total.toLocaleString()}
      </span>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Multi-verifier rollup strip
// ---------------------------------------------------------------------------

function MultiVerifierRollupStrip({
  summary,
  loaded,
  total,
}: {
  summary: TraceSummary;
  loaded: number;
  total: number;
}) {
  const verifiers = summary.verifiers ?? [];
  const pillBase =
    "inline-flex items-center px-1.5 py-0 text-[10px] font-mono font-semibold border";

  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-1 ink-block bg-card px-4 py-2 font-mono text-xs">
      {/* Overall gate */}
      <span className="flex items-center gap-1.5">
        <span className="text-muted-foreground font-semibold">GATE</span>
        <span
          className={cn(
            pillBase,
            summary.gate_passed
              ? "bg-ok/15 text-ok border-ok/30"
              : "bg-destructive/15 text-destructive border-destructive/30",
          )}
        >
          {summary.gate_passed ? "PASS" : "FAIL"}
        </span>
      </span>

      {/* Per-verifier chips */}
      {verifiers.map((v) => {
        const rollup = fmtVerifierRollup(v);
        return (
          <span key={v.name} className="flex items-center gap-1.5">
            <span className="text-foreground font-semibold">{v.name}</span>
            <span
              className={cn(
                pillBase,
                v.gate_passed
                  ? "bg-ok/15 text-ok border-ok/30"
                  : "bg-destructive/15 text-destructive border-destructive/30",
              )}
            >
              {v.gate_passed ? "PASS" : "FAIL"}
            </span>
            {rollup && (
              <span className="text-muted-foreground text-[10px]">{rollup}</span>
            )}
          </span>
        );
      })}

      {/* p99 */}
      {summary.p99_ms != null && (
        <span className="text-muted-foreground">
          p99{" "}
          <span className="tabular-nums text-foreground font-semibold">
            {Math.round(summary.p99_ms).toLocaleString()} ms
          </span>
        </span>
      )}

      <span className="ml-auto font-mono text-[10px] text-muted-foreground tabular-nums">
        showing {loaded.toLocaleString()} of {total.toLocaleString()}
      </span>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Lazy-loaded full prompt + artifact for an expanded inline row.
//
// The dense list ships LIGHT rows (no prompt, truncated artifact_text). When a
// row is expanded we fetch the full cached row and render the untruncated text.
// Shows the light values instantly, then swaps to full when the query resolves.
// ---------------------------------------------------------------------------

function ExpandedFullText({
  row,
  task,
  id,
  evalId,
}: {
  row: TraceRow;
  task: string;
  id: string;
  evalId: string;
}) {
  const { data: full } = useQuery({
    queryKey: ["trace-row", task, id, evalId, row.request_id],
    queryFn: () => fetchTraceRow(task, id, evalId, row.request_id),
    enabled: true,
  });

  return (
    <>
      {(full?.prompt ?? row.prompt) && (
        <div>
          <div className="text-[9px] uppercase tracking-widest text-muted-foreground mb-0.5">Prompt</div>
          <pre className="whitespace-pre-wrap text-foreground text-[10px] leading-[1.6] max-h-32 overflow-auto">
            {full?.prompt ?? row.prompt}
          </pre>
        </div>
      )}
      {(full?.artifact_text ?? row.artifact_text) && (
        <div>
          <div className="text-[9px] uppercase tracking-widest text-muted-foreground mb-0.5">Artifact</div>
          <pre className="whitespace-pre-wrap text-foreground text-[10px] leading-[1.6] max-h-24 overflow-auto">
            {full?.artifact_text ?? row.artifact_text}
          </pre>
        </div>
      )}
    </>
  );
}

// ---------------------------------------------------------------------------
// Multi-verifier matrix table
// ---------------------------------------------------------------------------

type MultiOnlyFilter = "all" | "failing";

function MultiVerifierTable({
  task,
  id,
  evalId,
  verifiers,
  onlyFilter,
  query,
  sortState,
  onSort,
  onLoadedCount,
  onTotalCount,
}: {
  task: string;
  id: string;
  evalId: string;
  verifiers: VerifierSummary[];
  onlyFilter: MultiOnlyFilter;
  query: string;
  sortState: SortState | null;
  onSort: (col: string) => void;
  onLoadedCount: (n: number) => void;
  onTotalCount: (n: number) => void;
}) {
  const [expandedRows, setExpandedRows] = useState<Set<number>>(new Set());
  const parentRef = useRef<HTMLDivElement>(null);

  // Fetch ALL rows once, keyed on (task, id, evalId).
  const { data, isLoading } = useQuery({
    queryKey: ["trace-all", task, id, evalId],
    queryFn: () =>
      fetchTracePage(task, id, evalId, { offset: 0, limit: ALL_ROWS_LIMIT }),
  });

  const allRows = useMemo(() => (data?.rows as TraceRow[] | undefined) ?? [], [data]);

  // Build Fuse index only when allRows changes, not on every sort/filter change.
  const fuse = useMemo(
    () =>
      allRows.length > 0
        ? new Fuse(allRows, { keys: FUSE_KEYS, threshold: 0.3, ignoreLocation: true })
        : null,
    [allRows],
  );

  // Map the multi-verifier chip onto the shared OnlyFilter.
  const only: OnlyFilter = onlyFilter === "failing" ? "failing" : null;

  // Compute the display list entirely client-side.
  const displayRows = useMemo(
    () => sortRows(searchRows(fuse, filterByOnly(allRows, only), query), sortState),
    [fuse, allRows, only, query, sortState],
  );

  // Reset expansion when the visible set changes (allRows included so a
  // refetch / live seal doesn't leave stale index-based expansion state).
  useEffect(() => {
    setExpandedRows(new Set());
  }, [allRows, only, query, sortState]);

  useEffect(() => {
    onLoadedCount(displayRows.length);
  }, [displayRows.length, onLoadedCount]);

  useEffect(() => {
    onTotalCount(allRows.length);
  }, [allRows.length, onTotalCount]);

  const virtualizer = useVirtualizer({
    count: displayRows.length,
    getScrollElement: () => parentRef.current,
    estimateSize: (index) => (expandedRows.has(index) ? EXPANDED_HEIGHT : ROW_HEIGHT),
    overscan: 10,
  });

  const virtualItems = virtualizer.getVirtualItems();

  const toggleRow = useCallback(
    (index: number) => {
      setExpandedRows((prev) => {
        const next = new Set(prev);
        if (next.has(index)) next.delete(index);
        else next.add(index);
        return next;
      });
      requestAnimationFrame(() => virtualizer.measure());
    },
    [virtualizer],
  );

  // Each per-tenant verifier column is labelled by the tenant it grades (recovered
  // from the rows it actually scored), not the repeated verifier class name.
  const verifierTenants = useMemo(
    () =>
      verifiers.map((_v, vi) => {
        const r = displayRows.find(
          (row) => row.verifier_cells?.[vi]?.passed != null && row.tenant_id,
        );
        return r?.tenant_id ?? null;
      }),
    [verifiers, displayRows],
  );

  // Column template: Request, Artifact, Oracle, Expected, [narrow per-tenant verifier
  // cols], Latency. Text columns get the width; each verifier column holds a single
  // check (= artifact correct) under its tenant header.
  const cols = [
    "1.3fr",
    "2fr",
    "2fr",
    "1.1fr",
    ...verifiers.map(() => "64px"),
    "64px",
  ].join(" ");

  const pillBase =
    "inline-flex items-center px-1 py-0 text-[9px] font-mono font-medium leading-tight";

  if (isLoading) {
    return (
      <div className="border border-border bg-muted/20 p-4">
        <p className="font-mono text-xs text-muted-foreground animate-pulse">Loading…</p>
      </div>
    );
  }

  if (displayRows.length === 0) {
    return (
      <div className="border border-border bg-muted/20 p-4">
        <p className="font-mono text-xs text-muted-foreground">No matching turns.</p>
      </div>
    );
  }

  return (
    <div className="flex flex-col border border-border">
      {/* Sticky header */}
      <div
        className="sticky top-0 z-10 grid whitespace-nowrap border-b-2 border-foreground bg-card font-mono text-[9px] uppercase tracking-widest text-muted-foreground"
        style={{ gridTemplateColumns: cols }}
      >
        <SortableHeaderCell col="request_id" label="Request" className="px-3 py-1.5" sortState={sortState} onSort={onSort} />
        <SortableHeaderCell col="artifact_text" label="Artifact" className="px-2 py-1.5" sortState={sortState} onSort={onSort} />
        <SortableHeaderCell col="reference_text" label="Oracle" className="px-2 py-1.5" sortState={sortState} onSort={onSort} />
        <SortableHeaderCell col="ground_truth" label="Expected" className="px-2 py-1.5" sortState={sortState} onSort={onSort} />
        {verifiers.map((v, vi) => (
          <SortableHeaderCell
            key={`${v.name}-${vi}`}
            col={`verifier:${vi}`}
            label={verifierTenants[vi] ?? v.name}
            className="px-2 py-1.5 text-center whitespace-nowrap normal-case tracking-normal text-[10px] text-foreground"
            sortState={sortState}
            onSort={onSort}
            title={verifierTenants[vi] ? `${verifierTenants[vi]} -- ${v.name} (check = artifact correct on this item)` : v.name}
          />
        ))}
        <SortableHeaderCell col="latency_ms" label="Latency" className="px-2 py-1.5 text-right" sortState={sortState} onSort={onSort} />
      </div>

      <div ref={parentRef} className="overflow-auto" style={{ maxHeight: "calc(100vh - 280px)" }}>
        <div style={{ height: `${virtualizer.getTotalSize()}px`, position: "relative" }}>
          {virtualItems.map((vitem) => {
            const row = displayRows[vitem.index];

            return (
              <div
                key={vitem.key}
                ref={virtualizer.measureElement}
                data-index={vitem.index}
                style={{
                  position: "absolute",
                  top: 0,
                  left: 0,
                  right: 0,
                  transform: `translateY(${vitem.start}px)`,
                }}
              >
                {row ? (
                  <div>
                    <button
                      type="button"
                      onClick={() => toggleRow(vitem.index)}
                      className={cn(
                        "grid w-full items-center border-b border-border/40 font-mono text-[11px] transition-colors hover:bg-accent/50 text-left",
                        expandedRows.has(vitem.index) && "bg-accent/30",
                      )}
                      style={{ gridTemplateColumns: cols, minHeight: `${ROW_HEIGHT}px` }}
                      data-testid="turn-row"
                    >
                      <div className="truncate px-3 py-1 text-foreground font-semibold">
                        {truncate(row.request_id, 30)}
                      </div>
                      <div className="truncate px-2 py-1 text-foreground">
                        {row.artifact_text
                          ? truncate(row.artifact_text, 30)
                          : <span className="text-muted-foreground italic text-[10px]">empty</span>}
                      </div>
                      <div className="truncate px-2 py-1 text-muted-foreground">
                        {row.reference_text ? truncate(row.reference_text, 30) : "—"}
                      </div>
                      <div className="truncate px-2 py-1 text-foreground">
                        {row.ground_truth != null ? truncate(String(row.ground_truth), 18) : "—"}
                      </div>
                      {/* Per-tenant verifier cells -- check = artifact correct on this item */}
                      {verifiers.map((v, vi) => {
                        const cell = row.verifier_cells?.[vi];
                        if (!cell || cell.passed == null) {
                          return (
                            <div key={`${v.name}-${vi}`} className="px-2 py-1 text-center text-muted-foreground">
                              —
                            </div>
                          );
                        }
                        return (
                          <div
                            key={`${v.name}-${vi}`}
                            className="px-2 py-1 text-center"
                            title={cell.detail || undefined}
                          >
                            <span
                              className={cn(
                                pillBase,
                                cell.passed
                                  ? "bg-ok/15 text-ok"
                                  : "bg-destructive/15 text-destructive",
                              )}
                            >
                              {cell.passed ? "✓" : "✗"}
                            </span>
                          </div>
                        );
                      })}
                      <div className="px-2 py-1 text-right tabular-nums text-muted-foreground text-[10px]">
                        {fmtMs(row.latency_ms)}
                      </div>
                    </button>

                    {expandedRows.has(vitem.index) && (
                      <div className="border-b border-border bg-muted/10 px-3 py-2 font-mono text-[11px] space-y-1.5">
                        <ExpandedFullText row={row} task={task} id={id} evalId={evalId} />
                        {/* Per-verifier detail */}
                        {(row.verifier_cells ?? []).map((cell, vi) => (
                          <div key={`${verifiers[vi]?.name ?? "v"}-${vi}`}>
                            <div className="text-[9px] uppercase tracking-widest text-muted-foreground mb-0.5">
                              {verifiers[vi]?.name ?? `Verifier ${vi}`}
                            </div>
                            <span
                              className={cn(
                                "text-[10px] font-mono",
                                cell.passed ? "text-ok" : "text-destructive",
                              )}
                            >
                              {cell.passed ? "✓" : "✗"} {cell.detail}
                            </span>
                          </div>
                        ))}
                        {row.error && (
                          <div className="text-destructive text-[10px]">{row.error}</div>
                        )}
                      </div>
                    )}
                  </div>
                ) : (
                  <div
                    className="border-b border-border/40 flex items-center px-3"
                    style={{ height: `${ROW_HEIGHT}px` }}
                  >
                    <span className="font-mono text-[10px] text-muted-foreground animate-pulse">
                      Loading…
                    </span>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

function MultiVerifierFilterChips({
  only,
  onChange,
  loaded,
  total,
  searchInput,
  onSearchInput,
}: {
  only: MultiOnlyFilter;
  onChange: (v: MultiOnlyFilter) => void;
  loaded: number;
  total: number;
  searchInput: string;
  onSearchInput: (v: string) => void;
}) {
  const pill =
    "inline-flex items-center h-7 px-3 text-[11px] font-mono border-[1.5px] border-border cursor-pointer transition-colors select-none";

  function Chip({ value, label }: { value: MultiOnlyFilter; label: string }) {
    const active = only === value;
    return (
      <button
        type="button"
        onClick={() => onChange(value)}
        className={cn(
          pill,
          active
            ? "bg-foreground text-background border-foreground"
            : "bg-card text-muted-foreground hover:text-foreground hover:border-foreground/40",
        )}
      >
        {label}
      </button>
    );
  }

  return (
    <div className="flex flex-wrap items-center gap-2">
      <Chip value="all" label="All" />
      <Chip value="failing" label="Failing any" />
      <input
        type="text"
        value={searchInput}
        onChange={(e) => onSearchInput(e.target.value)}
        placeholder="Search…"
        className="h-7 w-48 px-2 text-xs font-mono border-[1.5px] border-border bg-background placeholder:text-muted-foreground focus:outline-none focus:border-foreground focus:shadow-[var(--shadow-sm)]"
      />
      <span className="ml-auto font-mono text-[10px] text-muted-foreground tabular-nums">
        showing {loaded.toLocaleString()} of {total.toLocaleString()}
      </span>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Filter chips
// ---------------------------------------------------------------------------

type OnlyFilter = string | null;

function FilterChips({
  only,
  onChange,
  sessionIds,
  showSessionChips,
  verifierKind,
  searchInput,
  onSearchInput,
}: {
  only: OnlyFilter;
  onChange: (v: OnlyFilter) => void;
  sessionIds: string[];
  showSessionChips: boolean;
  verifierKind?: string | null;
  searchInput: string;
  onSearchInput: (v: string) => void;
}) {
  const pill =
    "inline-flex items-center h-7 px-3 text-[11px] font-mono border-[1.5px] border-border cursor-pointer transition-colors select-none";

  function Chip({ value, label }: { value: OnlyFilter; label: string }) {
    const active = only === value;
    return (
      <button
        type="button"
        onClick={() => onChange(value)}
        className={cn(
          pill,
          active
            ? "bg-foreground text-background border-foreground"
            : "bg-card text-muted-foreground hover:text-foreground hover:border-foreground/40",
        )}
      >
        {label}
      </button>
    );
  }

  const showDiverged = verifierKind === "token_level";

  return (
    <div className="flex flex-wrap items-center gap-2">
      <Chip value={null} label="All" />
      <Chip value="incorrect" label="Incorrect" />
      {showDiverged && <Chip value="diverged" label="Diverged" />}
      {showSessionChips &&
        sessionIds.map((sid) => (
          <Chip key={sid} value={sid} label={sid} />
        ))}
      <input
        type="text"
        value={searchInput}
        onChange={(e) => onSearchInput(e.target.value)}
        placeholder="Search…"
        className="h-7 w-48 px-2 text-xs font-mono border-[1.5px] border-border bg-background placeholder:text-muted-foreground focus:outline-none focus:border-foreground focus:shadow-[var(--shadow-sm)]"
      />
    </div>
  );
}

// ---------------------------------------------------------------------------
// Table header
// ---------------------------------------------------------------------------

function TableHeader({
  showSession,
  showReference,
  verifierKind,
  sortState,
  onSort,
}: {
  showSession: boolean;
  showReference: boolean;
  verifierKind?: string | null;
  sortState: SortState | null;
  onSort: (col: string) => void;
}) {
  const kind = verifierKind ?? null;

  if (kind === "task_relative") {
    const baseCols = showSession ? ["2fr", "0.8fr"] : ["2fr"];
    const cols = [...baseCols, "1.5fr", "1.5fr", "1.5fr", "70px", "70px", "70px"].join(" ");
    return (
      <div
        className="sticky top-0 z-10 grid whitespace-nowrap border-b-2 border-foreground bg-card font-mono text-[9px] uppercase tracking-widest text-muted-foreground"
        style={{ gridTemplateColumns: cols }}
      >
        <SortableHeaderCell col="request_id" label="Request" className="px-3 py-1.5" sortState={sortState} onSort={onSort} />
        {showSession && <SortableHeaderCell col="session_id" label="Session" className="px-2 py-1.5" sortState={sortState} onSort={onSort} />}
        <SortableHeaderCell col="predicted" label="Artifact" className="px-2 py-1.5" sortState={sortState} onSort={onSort} />
        <SortableHeaderCell col="oracle_predicted" label="Oracle" className="px-2 py-1.5" sortState={sortState} onSort={onSort} />
        <SortableHeaderCell col="ground_truth" label="Expected" className="px-2 py-1.5" sortState={sortState} onSort={onSort} />
        <SortableHeaderCell col="correct" label="Artifact" className="px-2 py-1.5" sortState={sortState} onSort={onSort} />
        <SortableHeaderCell col="oracle_correct" label="Oracle" className="px-2 py-1.5" sortState={sortState} onSort={onSort} />
        <SortableHeaderCell col="latency_ms" label="Latency" className="px-2 py-1.5 text-right" sortState={sortState} onSort={onSort} />
      </div>
    );
  }

  if (kind === "task_absolute") {
    const baseCols = showSession ? ["2fr", "0.8fr"] : ["2fr"];
    const cols = [...baseCols, "1.5fr", "1.5fr", "80px", "70px"].join(" ");
    return (
      <div
        className="sticky top-0 z-10 grid whitespace-nowrap border-b-2 border-foreground bg-card font-mono text-[9px] uppercase tracking-widest text-muted-foreground"
        style={{ gridTemplateColumns: cols }}
      >
        <SortableHeaderCell col="request_id" label="Request" className="px-3 py-1.5" sortState={sortState} onSort={onSort} />
        {showSession && <SortableHeaderCell col="session_id" label="Session" className="px-2 py-1.5" sortState={sortState} onSort={onSort} />}
        <SortableHeaderCell col="predicted" label="Artifact" className="px-2 py-1.5" sortState={sortState} onSort={onSort} />
        <SortableHeaderCell col="ground_truth" label="Expected" className="px-2 py-1.5" sortState={sortState} onSort={onSort} />
        <SortableHeaderCell col="correct" label="Correct" className="px-2 py-1.5" sortState={sortState} onSort={onSort} />
        <SortableHeaderCell col="latency_ms" label="Latency" className="px-2 py-1.5 text-right" sortState={sortState} onSort={onSort} />
      </div>
    );
  }

  // token_level / generic — existing layout
  const cols = (() => {
    const base = showSession ? ["2fr", "0.8fr"] : ["2fr"];
    base.push("1.5fr"); // Artifact
    if (showReference) base.push("1.5fr"); // Reference (task-level only)
    base.push("1.5fr", "110px", "70px"); // Expected, Oracle/Greedy-match, Latency
    return base.join(" ");
  })();

  return (
    <div
      className="sticky top-0 z-10 grid whitespace-nowrap border-b-2 border-foreground bg-card font-mono text-[9px] uppercase tracking-widest text-muted-foreground"
      style={{ gridTemplateColumns: cols }}
    >
      <SortableHeaderCell col="request_id" label="Request" className="px-3 py-1.5" sortState={sortState} onSort={onSort} />
      {showSession && <SortableHeaderCell col="session_id" label="Session" className="px-2 py-1.5" sortState={sortState} onSort={onSort} />}
      <SortableHeaderCell col="artifact_text" label="Artifact" className="px-2 py-1.5" sortState={sortState} onSort={onSort} />
      {showReference && (
        <SortableHeaderCell col="reference_text" label="Reference" className="px-2 py-1.5" sortState={sortState} onSort={onSort}>
          <span
            title="Reference model's decoded answer (task-level verifiers only)."
            className="ml-1 cursor-help opacity-50 hover:opacity-80"
          >
            ?
          </span>
        </SortableHeaderCell>
      )}
      <SortableHeaderCell col="ground_truth" label="Expected" className="px-2 py-1.5" sortState={sortState} onSort={onSort} />
      <SortableHeaderCell col="token_match" label="Token match" className="px-2 py-1.5" sortState={sortState} onSort={onSort}>
        <span
          title="Per-request teacher-forced token gap vs the oracle. This column mirrors one trace-level verdict: the two-threshold calibrated gate (pooled flip rate + worst gap vs calibrated ceilings) decides pass/fail for the whole trace, and every row shows that same outcome."
          className="ml-1 cursor-help opacity-50 hover:opacity-80"
        >
          ?
        </span>
      </SortableHeaderCell>
      <SortableHeaderCell col="latency_ms" label="Latency" className="px-2 py-1.5 text-right" sortState={sortState} onSort={onSort} />
    </div>
  );
}

// ---------------------------------------------------------------------------
// Dense table row (collapsed)
// ---------------------------------------------------------------------------

function TurnRow({
  row,
  onClick,
  isExpanded,
  showSession,
  showReference,
  verifierKind,
  task,
  id,
  evalId,
}: {
  row: TraceRow;
  onClick: () => void;
  isExpanded: boolean;
  showSession: boolean;
  showReference: boolean;
  verifierKind?: string | null;
  task: string;
  id: string;
  evalId: string;
}) {
  const pillBase = "inline-flex items-center px-1 py-0 text-[9px] font-mono font-medium leading-tight";
  const kind = verifierKind ?? null;

  // When expanded, lazy-load the FULL row (untruncated prompt/artifact/reference).
  // The collapsed cells keep using the light `row`; the expanded sections render
  // the truncated text instantly, then swap to full when this resolves.
  const { data: full } = useQuery({
    queryKey: ["trace-row", task, id, evalId, row.request_id],
    queryFn: () => fetchTraceRow(task, id, evalId, row.request_id),
    enabled: isExpanded,
  });

  const cols = (() => {
    if (kind === "task_relative") {
      const base = showSession ? ["2fr", "0.8fr"] : ["2fr"];
      return [...base, "1.5fr", "1.5fr", "1.5fr", "70px", "70px", "70px"].join(" ");
    }
    if (kind === "task_absolute") {
      const base = showSession ? ["2fr", "0.8fr"] : ["2fr"];
      return [...base, "1.5fr", "1.5fr", "80px", "70px"].join(" ");
    }
    // token_level / generic
    const base = showSession ? ["2fr", "0.8fr"] : ["2fr"];
    base.push("1.5fr"); // Artifact
    if (showReference) base.push("1.5fr"); // Reference
    base.push("1.5fr", "110px", "70px"); // Expected, Oracle/Greedy-match, Latency
    return base.join(" ");
  })();

  // Shared session cell
  const sessionCell = showSession && (
    <div className="truncate px-2 py-1 text-muted-foreground text-[10px]">
      {row.session_id ? truncate(row.session_id, 12) : "—"}
    </div>
  );

  if (kind === "task_relative") {
    return (
      <div>
        <button
          type="button"
          onClick={onClick}
          className={cn(
            "grid w-full items-center border-b border-border/40 font-mono text-[11px] transition-colors hover:bg-accent/50 text-left",
            isExpanded && "bg-accent/30",
          )}
          style={{ gridTemplateColumns: cols, minHeight: `${ROW_HEIGHT}px` }}
          data-testid="turn-row"
        >
          <div className="truncate px-3 py-1 text-foreground font-semibold">
            {truncate(row.request_id, 30)}
          </div>
          {sessionCell}
          {/* Artifact col: predicted answer */}
          <div className="truncate px-2 py-1 text-foreground">
            {row.predicted != null ? truncate(row.predicted, 20) : (row.artifact_text ? truncate(row.artifact_text, 20) : <span className="text-muted-foreground italic text-[10px]">empty</span>)}
          </div>
          {/* Oracle col: oracle_predicted */}
          <div className="truncate px-2 py-1 text-muted-foreground">
            {row.oracle_predicted != null ? truncate(row.oracle_predicted, 20) : "—"}
          </div>
          {/* Expected col: ground_truth */}
          <div className="truncate px-2 py-1 text-foreground">
            {row.ground_truth != null ? truncate(String(row.ground_truth), 20) : "—"}
          </div>
          {/* Artifact correct pill */}
          <div className="px-2 py-1">
            {row.correct === true && <span className={cn(pillBase, "bg-ok/15 text-ok")}>ok</span>}
            {row.correct === false && <span className={cn(pillBase, "bg-destructive/15 text-destructive")}>err</span>}
            {row.correct == null && <span className="text-muted-foreground">—</span>}
          </div>
          {/* Oracle correct pill */}
          <div className="px-2 py-1">
            {row.oracle_correct === true && <span className={cn(pillBase, "bg-ok/15 text-ok")}>ok</span>}
            {row.oracle_correct === false && <span className={cn(pillBase, "bg-destructive/15 text-destructive")}>err</span>}
            {row.oracle_correct == null && <span className="text-muted-foreground">—</span>}
          </div>
          <div className="px-2 py-1 text-right tabular-nums text-muted-foreground text-[10px]">
            {fmtMs(row.latency_ms)}
          </div>
        </button>

        {isExpanded && (
          <div className="border-b border-border bg-muted/10 px-3 py-2 font-mono text-[11px] space-y-1.5">
            {(full?.prompt ?? row.prompt) && (
              <div>
                <div className="text-[9px] uppercase tracking-widest text-muted-foreground mb-0.5">Prompt</div>
                <pre className="whitespace-pre-wrap text-foreground text-[10px] leading-[1.6] max-h-32 overflow-auto">
                  {full?.prompt ?? row.prompt}
                </pre>
              </div>
            )}
            {(row.predicted ?? full?.artifact_text ?? row.artifact_text) && (
              <div>
                <div className="text-[9px] uppercase tracking-widest text-muted-foreground mb-0.5">Artifact</div>
                <pre className="whitespace-pre-wrap text-foreground text-[10px] leading-[1.6] max-h-24 overflow-auto">
                  {row.predicted ?? full?.artifact_text ?? row.artifact_text}
                </pre>
              </div>
            )}
            {(full?.reference_text ?? row.reference_text) != null && (
              <div>
                <div className="text-[9px] uppercase tracking-widest text-muted-foreground mb-0.5">Oracle raw text</div>
                <pre className="whitespace-pre-wrap text-muted-foreground text-[10px] leading-[1.6] max-h-24 overflow-auto">
                  {full?.reference_text ?? row.reference_text}
                </pre>
              </div>
            )}
            {row.error && (
              <div className="text-destructive text-[10px]">{row.error}</div>
            )}
          </div>
        )}
      </div>
    );
  }

  if (kind === "task_absolute") {
    return (
      <div>
        <button
          type="button"
          onClick={onClick}
          className={cn(
            "grid w-full items-center border-b border-border/40 font-mono text-[11px] transition-colors hover:bg-accent/50 text-left",
            isExpanded && "bg-accent/30",
          )}
          style={{ gridTemplateColumns: cols, minHeight: `${ROW_HEIGHT}px` }}
          data-testid="turn-row"
        >
          <div className="truncate px-3 py-1 text-foreground font-semibold">
            {truncate(row.request_id, 30)}
          </div>
          {sessionCell}
          {/* Artifact col: predicted or artifact_text */}
          <div className="truncate px-2 py-1 text-foreground">
            {row.predicted != null ? truncate(row.predicted, 20) : (row.artifact_text ? truncate(row.artifact_text, 20) : <span className="text-muted-foreground italic text-[10px]">empty</span>)}
          </div>
          {/* Expected col: ground_truth */}
          <div className="truncate px-2 py-1 text-foreground">
            {row.ground_truth != null ? truncate(String(row.ground_truth), 20) : "—"}
          </div>
          {/* Correct pill */}
          <div className="px-2 py-1">
            {row.correct === true && <span className={cn(pillBase, "bg-ok/15 text-ok")}>correct</span>}
            {row.correct === false && <span className={cn(pillBase, "bg-destructive/15 text-destructive")}>incorrect</span>}
            {row.correct == null && <span className="text-muted-foreground">—</span>}
          </div>
          <div className="px-2 py-1 text-right tabular-nums text-muted-foreground text-[10px]">
            {fmtMs(row.latency_ms)}
          </div>
        </button>

        {isExpanded && (
          <div className="border-b border-border bg-muted/10 px-3 py-2 font-mono text-[11px] space-y-1.5">
            {(full?.prompt ?? row.prompt) && (
              <div>
                <div className="text-[9px] uppercase tracking-widest text-muted-foreground mb-0.5">Prompt</div>
                <pre className="whitespace-pre-wrap text-foreground text-[10px] leading-[1.6] max-h-32 overflow-auto">
                  {full?.prompt ?? row.prompt}
                </pre>
              </div>
            )}
            {(row.predicted ?? full?.artifact_text ?? row.artifact_text) && (
              <div>
                <div className="text-[9px] uppercase tracking-widest text-muted-foreground mb-0.5">Artifact</div>
                <pre className="whitespace-pre-wrap text-foreground text-[10px] leading-[1.6] max-h-24 overflow-auto">
                  {row.predicted ?? full?.artifact_text ?? row.artifact_text}
                </pre>
              </div>
            )}
            {row.error && (
              <div className="text-destructive text-[10px]">{row.error}</div>
            )}
            {row.artifact_tokens != null && (
              <div className="text-[10px] text-muted-foreground">output tokens: {row.artifact_tokens}</div>
            )}
          </div>
        )}
      </div>
    );
  }

  // token_level / generic — EXACTLY as before
  const oracleBadge = (() => {
    if (row.token_match === true) {
      return (
        <span className={cn(pillBase, "bg-ok/15 text-ok")} title="Trace-level calibrated gate passed (mirrored on every row)">
          token-match
        </span>
      );
    }
    if (row.token_match === false) {
      const atIdx = row.first_above_cap_index != null ? ` @${row.first_above_cap_index}` : "";
      return (
        <span
          className={cn(pillBase, "bg-amber-500/15 text-amber-600 dark:text-amber-400")}
          title={`Trace-level calibrated gate failed (mirrored on every row); worst gap${atIdx}`}
        >
          diverges{atIdx}
        </span>
      );
    }
    return <span className="text-muted-foreground">—</span>;
  })();

  return (
    <div>
      <button
        type="button"
        onClick={onClick}
        className={cn(
          "grid w-full items-center border-b border-border/40 font-mono text-[11px] transition-colors hover:bg-accent/50 text-left",
          isExpanded && "bg-accent/30",
        )}
        style={{
          gridTemplateColumns: cols,
          minHeight: `${ROW_HEIGHT}px`,
        }}
        data-testid="turn-row"
      >
        <div className="truncate px-3 py-1 text-foreground font-semibold">
          {truncate(row.request_id, 30)}
        </div>
        {showSession && (
          <div className="truncate px-2 py-1 text-muted-foreground text-[10px]">
            {row.session_id ? truncate(row.session_id, 12) : "—"}
          </div>
        )}
        <div className="truncate px-2 py-1 text-foreground">
          {row.artifact_text ? truncate(row.artifact_text, 20) : <span className="text-muted-foreground italic text-[10px]">empty</span>}
        </div>
        {showReference && (
          <div className="truncate px-2 py-1 text-muted-foreground">
            {row.reference_text != null ? truncate(row.reference_text, 20) : "—"}
          </div>
        )}
        <div className="truncate px-2 py-1 text-foreground">
          {row.ground_truth != null ? truncate(String(row.ground_truth), 20) : "—"}
        </div>
        <div className="px-2 py-1">{oracleBadge}</div>
        <div className="px-2 py-1 text-right tabular-nums text-muted-foreground text-[10px]">
          {fmtMs(row.latency_ms)}
        </div>
      </button>

      {isExpanded && (
        <div className="border-b border-border bg-muted/10 px-3 py-2 font-mono text-[11px] space-y-1.5">
          {(full?.prompt ?? row.prompt) && (
            <div>
              <div className="text-[9px] uppercase tracking-widest text-muted-foreground mb-0.5">Prompt</div>
              <pre className="whitespace-pre-wrap text-foreground text-[10px] leading-[1.6] max-h-32 overflow-auto">
                {full?.prompt ?? row.prompt}
              </pre>
            </div>
          )}
          {(full?.artifact_text ?? row.artifact_text) && (
            <div>
              <div className="text-[9px] uppercase tracking-widest text-muted-foreground mb-0.5">Artifact</div>
              <pre className="whitespace-pre-wrap text-foreground text-[10px] leading-[1.6] max-h-24 overflow-auto">
                {full?.artifact_text ?? row.artifact_text}
              </pre>
            </div>
          )}
          {(full?.reference_text ?? row.reference_text) != null ? (
            <div>
              <div className="text-[9px] uppercase tracking-widest text-muted-foreground mb-0.5">Reference</div>
              <pre className="whitespace-pre-wrap text-muted-foreground text-[10px] leading-[1.6] max-h-24 overflow-auto">
                {full?.reference_text ?? row.reference_text}
              </pre>
            </div>
          ) : null}
          {(row.n_tokens != null || row.worst_gap != null || row.first_above_cap_index != null) && (
            <div className="flex flex-wrap gap-x-4 text-[10px] text-muted-foreground">
              <span className="text-[9px] uppercase tracking-widest mr-1">Token-parity detail:</span>
              {row.n_tokens != null && <span>n_tokens: {row.n_tokens}</span>}
              {row.worst_gap != null && <span>worst_gap: {row.worst_gap.toFixed(4)}</span>}
              {row.first_above_cap_index != null && <span>first_above_cap_index: {row.first_above_cap_index}</span>}
            </div>
          )}
          {row.error && (
            <div className="text-destructive text-[10px]">{row.error}</div>
          )}
          {row.artifact_tokens != null && (
            <div className="text-[10px] text-muted-foreground">output tokens: {row.artifact_tokens}</div>
          )}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Virtualized turn list with infinite paging
// ---------------------------------------------------------------------------

function VirtualTurnList({
  task,
  id,
  evalId,
  only,
  query,
  sortState,
  onSort,
  onSessionIds,
  onLoadedCount,
  onTotalCount,
  showSession,
  verifierKind,
}: {
  task: string;
  id: string;
  evalId: string;
  only: OnlyFilter;
  query: string;
  sortState: SortState | null;
  onSort: (col: string) => void;
  onSessionIds: (ids: string[]) => void;
  onLoadedCount: (n: number) => void;
  onTotalCount: (n: number) => void;
  showSession: boolean;
  verifierKind?: string | null;
}) {
  const [expandedRows, setExpandedRows] = useState<Set<number>>(new Set());
  const parentRef = useRef<HTMLDivElement>(null);

  // Fetch ALL rows once, keyed on (task, id, evalId). Filter/search/sort happen
  // entirely client-side, so changing those never refetches.
  const { data, isLoading } = useQuery({
    queryKey: ["trace-all", task, id, evalId],
    queryFn: () =>
      fetchTracePage(task, id, evalId, { offset: 0, limit: ALL_ROWS_LIMIT }),
  });

  const allRows = useMemo(() => (data?.rows as TraceRow[] | undefined) ?? [], [data]);

  // Build Fuse index only when allRows changes, not on every sort/filter change.
  const fuse = useMemo(
    () =>
      allRows.length > 0
        ? new Fuse(allRows, { keys: FUSE_KEYS, threshold: 0.3, ignoreLocation: true })
        : null,
    [allRows],
  );

  // Derive session ids + reference presence from the full set.
  const sessionIds = useMemo(() => {
    const s = new Set<string>();
    for (const r of allRows) if (r.session_id) s.add(r.session_id);
    return Array.from(s).sort();
  }, [allRows]);
  const hasReference = useMemo(
    () => allRows.some((r) => r.reference_text != null),
    [allRows],
  );

  useEffect(() => {
    if (sessionIds.length > 0) onSessionIds(sessionIds);
  }, [sessionIds, onSessionIds]);

  // Compute the display list entirely client-side.
  const displayRows = useMemo(
    () => sortRows(searchRows(fuse, filterByOnly(allRows, only), query), sortState),
    [fuse, allRows, only, query, sortState],
  );

  // Reset expansion when the visible set changes (allRows included so a
  // refetch / live seal doesn't leave stale index-based expansion state).
  useEffect(() => {
    setExpandedRows(new Set());
  }, [allRows, only, query, sortState]);

  useEffect(() => {
    onLoadedCount(displayRows.length);
  }, [displayRows.length, onLoadedCount]);

  useEffect(() => {
    onTotalCount(allRows.length);
  }, [allRows.length, onTotalCount]);

  const virtualizer = useVirtualizer({
    count: displayRows.length,
    getScrollElement: () => parentRef.current,
    estimateSize: (index) => {
      return expandedRows.has(index) ? EXPANDED_HEIGHT : ROW_HEIGHT;
    },
    overscan: 10,
  });

  const virtualItems = virtualizer.getVirtualItems();

  const toggleRow = useCallback(
    (index: number) => {
      setExpandedRows((prev) => {
        const next = new Set(prev);
        if (next.has(index)) {
          next.delete(index);
        } else {
          next.add(index);
        }
        return next;
      });
      requestAnimationFrame(() => {
        virtualizer.measure();
      });
    },
    [virtualizer],
  );

  // Not yet received first response
  if (isLoading) {
    return (
      <div className="border border-border bg-muted/20 p-4">
        <p className="font-mono text-xs text-muted-foreground animate-pulse">
          Loading…
        </p>
      </div>
    );
  }

  if (displayRows.length === 0) {
    return (
      <div className="border border-border bg-muted/20 p-4">
        <p className="font-mono text-xs text-muted-foreground">
          No matching turns.
        </p>
      </div>
    );
  }

  return (
    <div className="flex flex-col border border-border">
      <TableHeader showSession={showSession} showReference={hasReference} verifierKind={verifierKind} sortState={sortState} onSort={onSort} />
      <div
        ref={parentRef}
        className="overflow-auto"
        style={{ maxHeight: "calc(100vh - 280px)" }}
      >
        <div
          style={{
            height: `${virtualizer.getTotalSize()}px`,
            position: "relative",
          }}
        >
          {virtualItems.map((vitem) => {
            const row = displayRows[vitem.index];

            return (
              <div
                key={vitem.key}
                ref={virtualizer.measureElement}
                data-index={vitem.index}
                style={{
                  position: "absolute",
                  top: 0,
                  left: 0,
                  right: 0,
                  transform: `translateY(${vitem.start}px)`,
                }}
              >
                {row ? (
                  <TurnRow
                    row={row}
                    onClick={() => toggleRow(vitem.index)}
                    isExpanded={expandedRows.has(vitem.index)}
                    showSession={showSession}
                    showReference={hasReference}
                    verifierKind={verifierKind}
                    task={task}
                    id={id}
                    evalId={evalId}
                  />
                ) : (
                  <div
                    className="border-b border-border/40 flex items-center px-3"
                    style={{ height: `${ROW_HEIGHT}px` }}
                  >
                    <span className="font-mono text-[10px] text-muted-foreground animate-pulse">
                      Loading…
                    </span>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Live eval progress banner
// ---------------------------------------------------------------------------

// What an eval's progress, read from the events of its drive stage's log, says.
function LiveEvalBanner({ progress }: { progress: EvalProgress }) {
  if (progress.phase === "none") return null;

  const starting = progress.phase === "starting";
  const step =
    progress.phase === "driven"
      ? `Drive ended${progress.outcome != null ? ` (${progress.outcome})` : ""}; verification and scoring follow.`
      : starting
        ? "Starting the artifact."
        : "Driving the artifact.";

  return (
    <div className="ink-block bg-muted/10 px-4 py-3 space-y-1">
      <div className="flex items-center gap-2">
        <span className="h-1.5 w-1.5 bg-primary animate-pulse shrink-0" />
        <span className="font-mono text-xs text-primary font-medium">
          {starting ? "Eval starting" : "Eval in progress"}
        </span>
      </div>
      <p className="font-mono text-[11px] text-muted-foreground">{step}</p>
      <p className="font-mono text-[11px] text-muted-foreground">
        Full trace will load when the eval seals.
      </p>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Generic / unknown verifier neutral view
// ---------------------------------------------------------------------------

function GenericNeutralView({
  task,
  id,
  evalId,
}: {
  task: string;
  id: string;
  evalId: string;
}) {
  const [expandedRows, setExpandedRows] = useState<Set<number>>(new Set());
  const parentRef = useRef<HTMLDivElement>(null);

  // Fetch ALL rows once, keyed on (task, id, evalId).
  const { data, isLoading } = useQuery({
    queryKey: ["trace-all", task, id, evalId],
    queryFn: () =>
      fetchTracePage(task, id, evalId, { offset: 0, limit: ALL_ROWS_LIMIT }),
  });

  const displayRows = useMemo(
    () => (data?.rows as TraceRow[] | undefined) ?? [],
    [data],
  );

  const virtualizer = useVirtualizer({
    count: displayRows.length,
    getScrollElement: () => parentRef.current,
    estimateSize: (index) => (expandedRows.has(index) ? EXPANDED_HEIGHT : ROW_HEIGHT),
    overscan: 10,
  });

  const virtualItems = virtualizer.getVirtualItems();

  const toggleRow = useCallback(
    (index: number) => {
      setExpandedRows((prev) => {
        const next = new Set(prev);
        if (next.has(index)) next.delete(index);
        else next.add(index);
        return next;
      });
      requestAnimationFrame(() => virtualizer.measure());
    },
    [virtualizer],
  );

  const cols = ["2fr", "1.5fr", "1.5fr", "70px"].join(" ");

  if (isLoading) {
    return (
      <div className="border border-border bg-muted/20 p-4">
        <p className="font-mono text-xs text-muted-foreground animate-pulse">Loading…</p>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-2">
      <div className="border border-border bg-muted/10 px-4 py-2 font-mono text-xs text-muted-foreground">
        Specialized eval view not implemented for this verifier.
      </div>
      <div className="flex flex-col border border-border">
        {/* Header */}
        <div
          className="sticky top-0 z-10 grid whitespace-nowrap border-b-2 border-foreground bg-card font-mono text-[9px] uppercase tracking-widest text-muted-foreground"
          style={{ gridTemplateColumns: cols }}
        >
          <div className="px-3 py-1.5">Request</div>
          <div className="px-2 py-1.5">Artifact</div>
          <div className="px-2 py-1.5">Expected</div>
          <div className="px-2 py-1.5 text-right">Latency</div>
        </div>
        <div ref={parentRef} className="overflow-auto" style={{ maxHeight: "calc(100vh - 280px)" }}>
          <div style={{ height: `${virtualizer.getTotalSize()}px`, position: "relative" }}>
            {virtualItems.map((vitem) => {
              const row = displayRows[vitem.index];
              return (
                <div
                  key={vitem.key}
                  ref={virtualizer.measureElement}
                  data-index={vitem.index}
                  style={{ position: "absolute", top: 0, left: 0, right: 0, transform: `translateY(${vitem.start}px)` }}
                >
                  {row ? (
                    <div>
                      <button
                        type="button"
                        onClick={() => toggleRow(vitem.index)}
                        className={cn(
                          "grid w-full items-center border-b border-border/40 font-mono text-[11px] transition-colors hover:bg-accent/50 text-left",
                          expandedRows.has(vitem.index) && "bg-accent/30",
                        )}
                        style={{ gridTemplateColumns: cols, minHeight: `${ROW_HEIGHT}px` }}
                        data-testid="turn-row"
                      >
                        <div className="truncate px-3 py-1 text-foreground font-semibold">
                          {truncate(row.request_id, 30)}
                        </div>
                        <div className="truncate px-2 py-1 text-foreground">
                          {row.artifact_text ? truncate(row.artifact_text, 20) : <span className="text-muted-foreground italic text-[10px]">empty</span>}
                        </div>
                        <div className="truncate px-2 py-1 text-foreground">
                          {row.ground_truth != null ? truncate(String(row.ground_truth), 20) : "—"}
                        </div>
                        <div className="px-2 py-1 text-right tabular-nums text-muted-foreground text-[10px]">
                          {fmtMs(row.latency_ms)}
                        </div>
                      </button>
                      {expandedRows.has(vitem.index) && (
                        <div className="border-b border-border bg-muted/10 px-3 py-2 font-mono text-[11px] space-y-1.5">
                          <ExpandedFullText row={row} task={task} id={id} evalId={evalId} />
                          {row.error && <div className="text-destructive text-[10px]">{row.error}</div>}
                        </div>
                      )}
                    </div>
                  ) : (
                    <div className="border-b border-border/40 flex items-center px-3" style={{ height: `${ROW_HEIGHT}px` }}>
                      <span className="font-mono text-[10px] text-muted-foreground animate-pulse">Loading…</span>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main panel
// ---------------------------------------------------------------------------

export function EvalPanel({
  rec,
  only: onlyProp,
  onOnlyChange,
  liveEvents = [],
}: {
  rec: RolloutRecord;
  only?: OnlyFilter;
  onOnlyChange?: (v: OnlyFilter) => void;
  liveEvents?: RolloutEvent[];
}) {
  const task = rec.task_id;
  const id = rec.rollout_id;

  const terminal =
    rec.evals.find((e) => e.eval_id === "terminal") ??
    rec.evals[rec.evals.length - 1];
  const evalId = terminal?.eval_id ?? "terminal";

  const [localOnly, setLocalOnly] = useState<OnlyFilter>(null);
  const only: OnlyFilter = onlyProp !== undefined ? onlyProp : localOnly;

  const [multiOnly, setMultiOnly] = useState<MultiOnlyFilter>("all");
  const [multiLoadedCount, setMultiLoadedCount] = useState(0);

  // Fuzzy-search box: `searchInput` is the live input; `query` is debounced and
  // is what the tables actually filter on.
  const [searchInput, setSearchInput] = useState("");
  const [query, setQuery] = useState("");
  useEffect(() => {
    const t = setTimeout(() => setQuery(searchInput), 150);
    return () => clearTimeout(t);
  }, [searchInput]);

  // Sort state, shared with whichever table renders. Clicking a header toggles.
  const [sortState, setSortState] = useState<SortState | null>(null);
  const handleSort = useCallback((col: string) => {
    setSortState((prev) => {
      if (prev?.col === col) {
        return { col, dir: prev.dir === "asc" ? "desc" : "asc" };
      }
      return { col, dir: "asc" };
    });
  }, []);

  // Total row counts (= allRows.length) reported up from the active table.
  const [totalCount, setTotalCount] = useState(0);
  const [multiTotalCount, setMultiTotalCount] = useState(0);

  // Session IDs derived from the full row set by the active table.
  const [sessionIds, setSessionIds] = useState<string[]>([]);
  const [loadedCount, setLoadedCount] = useState(0);

  const { data: summary, isLoading: summaryLoading } = useQuery({
    queryKey: ["trace-summary", task, id, evalId],
    queryFn: () => fetchTraceSummary(task, id, evalId),
    enabled: !!terminal,
  });

  const handleSessionIds = useCallback((ids: string[]) => {
    setSessionIds((prev) => {
      const merged = [...new Set([...prev, ...ids])].sort();
      if (merged.length === prev.length && merged.every((v, i) => v === prev[i])) {
        return prev;
      }
      return merged;
    });
    // Check for multi-turn: if any session now has more than 1 occurrence
    // We infer from the accumulated rows — if we see a session_id appearing more
    // than once across all seen IDs, it's multi-turn. But since onSessionIds only
    // sends unique IDs per page, we need to count differently.
    // Simpler heuristic: if #sessions < #total rows loaded, then some sessions
    // have multiple turns. We let the parent count rows vs sessions.
  }, []);

  const handleFilterChange = useCallback(
    (v: OnlyFilter) => {
      if (onOnlyChange) {
        onOnlyChange(v);
      } else {
        setLocalOnly(v);
      }
    },
    [onOnlyChange],
  );

  const handleLoadedCount = useCallback(
    (count: number) => {
      setLoadedCount(count);
    },
    [],
  );

  // Derived at render time — avoids the async race where handleLoadedCount
  // fires before sessionIds has updated from a prior async setState.
  const isMultiTurn = loadedCount > 0 && sessionIds.length > 0 && loadedCount > sessionIds.length;

  const evalInProgress = terminal != null && terminal.finished_at == null;
  const progress = useMemo(() => evalProgress(liveEvents), [liveEvents]);

  if (!terminal) {
    return (
      <div className="w-full px-6 py-4 space-y-3">
        {progress.phase !== "none" ? (
          <LiveEvalBanner progress={progress} />
        ) : (
          <div className="border border-border bg-muted/20 p-4">
            <p className="font-mono text-xs text-muted-foreground">
              No eval episodes found for this rollout.
            </p>
          </div>
        )}
      </div>
    );
  }

  const verifierKind = summary?.verifier_kind ?? null;
  const multi = (summary?.verifiers?.length ?? 0) > 1;

  return (
    <div className="flex flex-col w-full px-6 py-4 gap-3" style={{ minHeight: "calc(100vh - 140px)" }}>
      {evalInProgress && <LiveEvalBanner progress={progress} />}
      {summaryLoading ? (
        <div className="ink-block bg-card px-4 py-2">
          <span className="font-mono text-xs text-muted-foreground animate-pulse">
            Loading summary…
          </span>
        </div>
      ) : summary && multi ? (
        <>
          <MultiVerifierRollupStrip
            summary={summary}
            loaded={multiLoadedCount}
            total={multiTotalCount}
          />
          <MultiVerifierFilterChips
            only={multiOnly}
            onChange={setMultiOnly}
            loaded={multiLoadedCount}
            total={multiTotalCount}
            searchInput={searchInput}
            onSearchInput={setSearchInput}
          />
          <MultiVerifierTable
            key={evalId}
            task={task}
            id={id}
            evalId={evalId}
            verifiers={summary.verifiers ?? []}
            onlyFilter={multiOnly}
            query={query}
            sortState={sortState}
            onSort={handleSort}
            onLoadedCount={setMultiLoadedCount}
            onTotalCount={setMultiTotalCount}
          />
        </>
      ) : summary ? (
        <RollupRow summary={summary} loaded={loadedCount} total={totalCount} />
      ) : null}

      {!multi && (
        <FilterChips
          only={only}
          onChange={handleFilterChange}
          sessionIds={sessionIds}
          showSessionChips={isMultiTurn}
          verifierKind={verifierKind}
          searchInput={searchInput}
          onSearchInput={setSearchInput}
        />
      )}

      {!multi && summary != null && (verifierKind == null || verifierKind === "generic") && (
        <GenericNeutralView
          key={evalId}
          task={task}
          id={id}
          evalId={evalId}
        />
      )}

      {!multi && summary != null && verifierKind != null && verifierKind !== "generic" && (
        <VirtualTurnList
          key={evalId}
          task={task}
          id={id}
          evalId={evalId}
          only={only}
          query={query}
          sortState={sortState}
          onSort={handleSort}
          onSessionIds={handleSessionIds}
          onLoadedCount={handleLoadedCount}
          onTotalCount={setTotalCount}
          showSession={isMultiTurn}
          verifierKind={verifierKind}
        />
      )}

      {!multi && summary?.total === 0 && (
        <div className="border border-border bg-muted/20 p-4">
          <p className="font-mono text-xs text-muted-foreground">
            No trace events found for eval {evalId}.
          </p>
        </div>
      )}
    </div>
  );
}
