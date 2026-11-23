import { Command } from "cmdk";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import Fuse from "fuse.js";
import {
  createColumnHelper,
  flexRender,
  getCoreRowModel,
  getSortedRowModel,
  useReactTable,
  type SortingState,
} from "@tanstack/react-table";
import { useVirtualizer } from "@tanstack/react-virtual";
import {
  ArrowDown,
  ArrowUp,
  ArrowUpDown,
  ChevronDown,
  GitCompare,
  Layers,
  Search,
  Table2,
  X,
} from "lucide-react";
import { parseAsString, parseAsStringEnum, useQueryState } from "nuqs";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useNavigate } from "react-router";

import { BackendBadge, GateBadge, StatusBadge } from "~/components/rollout/badges";
import { Segmented } from "~/components/rollout/segmented";
import { useRolloutsStream } from "~/components/rollout/use-rollouts-stream";
import { Button } from "~/components/ui/button";
import { fetchAllRollouts, fetchRollout } from "~/lib/api";
import { useKeyboardTableNavigation } from "~/lib/hooks";
import { cn } from "~/lib/utils";
import type { RolloutSummary } from "~/lib/types";

// ---------------------------------------------------------------------------
// Format helpers
// ---------------------------------------------------------------------------

const rtf = new Intl.RelativeTimeFormat("en", { numeric: "auto" });

function formatRelative(ts: number | null | undefined): string {
  if (ts == null) return "—";
  const ms = ts * 1000;
  const diff = (ms - Date.now()) / 1000; // seconds, negative = past
  const abs = Math.abs(diff);
  if (abs < 60) return rtf.format(Math.round(diff), "second");
  if (abs < 3600) return rtf.format(Math.round(diff / 60), "minute");
  if (abs < 86400) return rtf.format(Math.round(diff / 3600), "hour");
  return rtf.format(Math.round(diff / 86400), "day");
}

function formatScore(score: number | null | undefined): string {
  if (score == null) return "—";
  return score.toFixed(3);
}

function formatCost(cost: number | null | undefined): string {
  if (cost == null) return "—";
  if (cost < 0.01) return `$${cost.toFixed(4)}`;
  return `$${cost.toFixed(2)}`;
}

function durationSec(
  a: number | null | undefined,
  b: number | null | undefined,
): number | null {
  if (a == null || b == null) return null;
  return b - a;
}

function formatDuration(
  a: number | null | undefined,
  b: number | null | undefined,
): string {
  const s = durationSec(a, b);
  if (s == null) return "—";
  if (s < 60) return `${Math.round(s)}s`;
  if (s < 3600) return `${Math.floor(s / 60)}m${Math.round(s % 60)}s`;
  return `${Math.floor(s / 3600)}h${Math.floor((s % 3600) / 60)}m`;
}

// ---------------------------------------------------------------------------
// Score pill (local, not shared — amber accepted per review)
// ---------------------------------------------------------------------------

function ScorePill({ score }: { score: number | null | undefined }) {
  if (score == null) return <span className="text-muted-foreground tabular-nums">—</span>;
  const cls =
    score >= 0.8
      ? "text-ok"
      : score >= 0.5
        ? "text-amber-600 dark:text-amber-400"
        : "text-destructive";
  return (
    <span className={cn("tabular-nums font-mono text-xs font-medium", cls)}>
      {formatScore(score)}
    </span>
  );
}

// ---------------------------------------------------------------------------
// Phase chip + progress bar (live, running rows only)
// ---------------------------------------------------------------------------

/**
 * Parse "N/M" phase fraction from the stream delta's `phase` field.
 * Returns [n, m] or null if not parseable.
 */
function parsePhase(phase: string | null | undefined): [number, number] | null {
  if (!phase) return null;
  const m = phase.match(/(\d+)\/(\d+)/);
  if (!m) return null;
  const n = parseInt(m[1], 10);
  const total = parseInt(m[2], 10);
  if (!isFinite(n) || !isFinite(total) || total <= 0) return null;
  return [n, total];
}

function PhaseChip({ phase }: { phase: string | null | undefined }) {
  if (!phase) return null;
  const fraction = parsePhase(phase);
  const pct = fraction ? (fraction[0] / fraction[1]) * 100 : null;

  return (
    <div className="flex flex-col gap-0.5 min-w-0">
      {/* chip */}
      <span className="inline-flex items-center gap-1 px-1.5 py-0.5 text-xs font-mono font-medium bg-primary/10 text-primary">
        <span className="h-1.5 w-1.5 bg-current animate-pulse shrink-0" />
        <span className="truncate">{phase}</span>
      </span>
      {/* thin progress bar */}
      {pct != null && (
        <div className="h-[3px] w-full bg-primary/15 relative overflow-hidden">
          <div
            className="absolute left-0 top-0 bottom-0 bg-primary/60 transition-all duration-300"
            style={{ width: `${Math.min(100, pct)}%` }}
          />
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Shared row-cell renderers — used by both the flat table's column defs and
// the grouped view's hand-rolled <tr>s so the two don't drift on the pieces
// that are identical between them. Pieces that already differ between the
// two views (checkbox/cursor markup tied to keyboard nav, the Model cell's
// title/muted-dash styling) are intentionally left per-caller — see
// TaskGroupSection / RolloutTable for details.
// ---------------------------------------------------------------------------

/** `"org/model-name"` -> `"model-name"`, matching each view's shortening rule. */
function shortenModel(m: string): string {
  return m.split("/").pop() ?? m;
}

function StatusCell({
  status,
  phase,
}: {
  status: RolloutSummary["status"];
  phase: RolloutSummary["phase"];
}) {
  const isInProgress = status === "running" || status === "stalled";
  return (
    <div className="flex flex-col gap-0.5">
      <StatusBadge status={status} />
      {isInProgress && phase && <PhaseChip phase={phase} />}
    </div>
  );
}

function GateCell({ passed }: { passed: RolloutSummary["gate_passed"] }) {
  return (
    <div className="flex justify-center">
      <GateBadge passed={passed} />
    </div>
  );
}

function ScoreCell({ score }: { score: RolloutSummary["score"] }) {
  return (
    <div className="text-right">
      <ScorePill score={score} />
    </div>
  );
}

/** Shared guard so a checkbox click inside a row doesn't also trigger the row's onClick. */
function handleRowClick(
  e: React.MouseEvent,
  row: RolloutSummary,
  onRowClick: (r: RolloutSummary) => void,
) {
  if ((e.target as HTMLElement).closest("input[type=checkbox]")) return;
  onRowClick(row);
}

// ---------------------------------------------------------------------------
// Select component (minimal, no external dep beyond Tailwind)
// ---------------------------------------------------------------------------

interface SelectFilterProps {
  label: string;
  options: string[];
  value: string;
  onValueChange: (v: string) => void;
}

function SelectFilter({ label, options, value, onValueChange }: SelectFilterProps) {
  return (
    <div className="relative">
      <select
        value={value}
        onChange={(e) => onValueChange(e.target.value)}
        className={cn(
          "h-7 pl-2.5 pr-7 text-xs font-mono font-medium border-[1.5px] border-border bg-background appearance-none cursor-pointer",
          "focus:outline-none focus:border-foreground focus:shadow-[var(--shadow-sm)]",
          value ? "text-foreground" : "text-muted-foreground",
        )}
      >
        <option value="">{label}</option>
        {options.map((o) => (
          <option key={o} value={o}>
            {o}
          </option>
        ))}
      </select>
      <ChevronDown className="pointer-events-none absolute right-2 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-muted-foreground" />
    </div>
  );
}

// ---------------------------------------------------------------------------
// Column definitions
// ---------------------------------------------------------------------------

const col = createColumnHelper<RolloutSummary>();

function SortIcon({ sorted }: { sorted: false | "asc" | "desc" }) {
  if (sorted === "asc") return <ArrowUp className="h-3 w-3" />;
  if (sorted === "desc") return <ArrowDown className="h-3 w-3" />;
  return <ArrowUpDown className="h-3 w-3 opacity-40" />;
}

function SortableHeader({
  column,
  children,
  className,
}: {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  column: any;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <button
      type="button"
      onClick={column.getToggleSortingHandler()}
      className={cn(
        "flex items-center gap-1 font-mono text-[10px] uppercase tracking-widest text-muted-foreground hover:text-foreground transition-colors",
        className,
      )}
    >
      {children}
      <SortIcon sorted={column.getIsSorted()} />
    </button>
  );
}

const columns = [
  col.accessor("task_id", {
    header: ({ column }) => <SortableHeader column={column}>Task</SortableHeader>,
    cell: ({ row }) => (
      <span className="font-mono text-xs font-medium">{row.original.task_id}</span>
    ),
    size: 220,
  }),
  col.accessor("rollout_id", {
    header: ({ column }) => <SortableHeader column={column}>Rollout ID</SortableHeader>,
    cell: ({ row }) => (
      <span className="font-mono text-[11px] text-muted-foreground">
        {row.original.rollout_id}
      </span>
    ),
    size: 180,
  }),
  col.accessor("status", {
    header: ({ column }) => <SortableHeader column={column}>Status</SortableHeader>,
    cell: ({ row }) => (
      <StatusCell status={row.original.status} phase={row.original.phase} />
    ),
    size: 150,
  }),
  col.accessor("score", {
    header: ({ column }) => (
      <div className="text-right w-full">
        <SortableHeader column={column}>Score</SortableHeader>
      </div>
    ),
    cell: ({ row }) => <ScoreCell score={row.original.score} />,
    size: 72,
    sortUndefined: "last",
  }),
  col.accessor("gate_passed", {
    header: ({ column }) => (
      <div className="text-center w-full">
        <SortableHeader column={column}>Gate</SortableHeader>
      </div>
    ),
    cell: ({ row }) => <GateCell passed={row.original.gate_passed} />,
    size: 72,
    sortUndefined: "last",
  }),
  col.accessor("backend", {
    header: ({ column }) => <SortableHeader column={column}>Backend</SortableHeader>,
    cell: ({ row }) => <BackendBadge backend={row.original.backend} />,
    size: 100,
  }),
  col.accessor("accelerator", {
    header: ({ column }) => <SortableHeader column={column}>Accel</SortableHeader>,
    cell: ({ row }) => (
      <span className="font-mono text-xs text-muted-foreground">
        {row.original.accelerator ?? "—"}
      </span>
    ),
    size: 80,
  }),
  col.accessor("model_fingerprint", {
    header: ({ column }) => <SortableHeader column={column}>Model</SortableHeader>,
    cell: ({ row }) => {
      const m = row.original.model_fingerprint;
      if (!m) return <span className="text-muted-foreground">—</span>;
      return (
        <span className="font-mono text-[11px]" title={m}>
          {shortenModel(m)}
        </span>
      );
    },
    size: 140,
  }),
  col.accessor("created_at", {
    header: ({ column }) => <SortableHeader column={column}>Launched</SortableHeader>,
    cell: ({ row }) => (
      <span className="text-xs text-muted-foreground tabular-nums whitespace-nowrap">
        {formatRelative(row.original.created_at)}
      </span>
    ),
    size: 110,
    sortUndefined: "last",
  }),
  col.display({
    id: "duration",
    header: () => (
      <div className="text-right font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
        Duration
      </div>
    ),
    cell: ({ row }) => (
      <div className="text-right tabular-nums text-xs text-muted-foreground">
        {formatDuration(row.original.created_at, row.original.finished_at)}
      </div>
    ),
    size: 80,
  }),
  col.accessor("cost_usd", {
    header: ({ column }) => (
      <div className="text-right w-full">
        <SortableHeader column={column}>Cost</SortableHeader>
      </div>
    ),
    cell: ({ row }) => (
      <div className="text-right tabular-nums text-xs text-muted-foreground">
        {formatCost(row.original.cost_usd)}
      </div>
    ),
    size: 80,
    sortUndefined: "last",
  }),
];

// ---------------------------------------------------------------------------
// Empty state
// ---------------------------------------------------------------------------

function EmptyState({ hasFilters }: { hasFilters: boolean }) {
  return (
    <div className="flex flex-col items-center justify-center py-20 text-center gap-4">
      <div className="w-10 h-10 border-[1.5px] border-border shadow-[var(--shadow-sm)] flex items-center justify-center text-muted-foreground">
        <Search className="h-5 w-5" />
      </div>
      <div>
        <p className="font-mono text-sm font-semibold text-foreground">
          {hasFilters ? "No rollouts match those filters" : "No rollouts yet"}
        </p>
        {!hasFilters && (
          <p className="text-xs text-muted-foreground mt-1">
            Launch one with{" "}
            <code className="font-mono bg-muted px-1 py-0.5">hls run &lt;task-module&gt; --arm &lt;arm&gt;</code>.
          </p>
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Virtualized flat table
// ---------------------------------------------------------------------------

function RolloutTable({
  rows,
  onRowClick,
  onRowHover,
  selectedIds,
  onToggleSelect,
  isLoading,
  hasFilters,
  sorting,
  onSortingChange,
}: {
  rows: RolloutSummary[];
  onRowClick: (r: RolloutSummary) => void;
  onRowHover: (r: RolloutSummary) => void;
  selectedIds: Set<string>;
  onToggleSelect: (id: string) => void;
  isLoading: boolean;
  hasFilters: boolean;
  sorting: SortingState;
  onSortingChange: (s: SortingState) => void;
}) {
  const table = useReactTable({
    data: rows,
    columns,
    getCoreRowModel: getCoreRowModel(),
    getSortedRowModel: getSortedRowModel(),
    onSortingChange: (updater) => {
      const next =
        typeof updater === "function" ? updater(sorting) : updater;
      onSortingChange(next);
    },
    state: { sorting },
    getRowId: (r) => `${r.task_id}/${r.rollout_id}`,
  });

  const { rows: tableRows } = table.getRowModel();

  const scrollRef = useRef<HTMLDivElement>(null);

  const virtualizer = useVirtualizer({
    count: tableRows.length,
    getScrollElement: () => scrollRef.current,
    estimateSize: () => 36,
    overscan: 10,
  });

  // Keyboard navigation
  const { cursorIndex } = useKeyboardTableNavigation({
    rows: tableRows.map((r) => r.original),
    onNavigate: onRowClick,
  });

  // Scroll cursor into view when it moves
  useEffect(() => {
    if (cursorIndex >= 0) {
      virtualizer.scrollToIndex(cursorIndex, { align: "auto" });
    }
  }, [cursorIndex, virtualizer]);

  const virtualRows = virtualizer.getVirtualItems();
  const totalSize = virtualizer.getTotalSize();
  const paddingTop = virtualRows.length > 0 ? (virtualRows[0]?.start ?? 0) : 0;
  const paddingBottom =
    virtualRows.length > 0
      ? totalSize - (virtualRows[virtualRows.length - 1]?.end ?? 0)
      : 0;

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-16">
        <span className="text-xs text-muted-foreground font-mono animate-pulse">
          Loading rollouts…
        </span>
      </div>
    );
  }

  if (tableRows.length === 0) {
    return <EmptyState hasFilters={hasFilters} />;
  }

  return (
    <div
      ref={scrollRef}
      className="overflow-auto border border-border"
      style={{ maxHeight: "calc(100vh - 240px)" }}
    >
      <table className="w-full border-collapse text-sm" style={{ tableLayout: "fixed" }}>
        <colgroup>
          <col style={{ width: 32 }} />
          {table.getFlatHeaders().map((header) => (
            <col key={header.id} style={{ width: header.getSize() }} />
          ))}
        </colgroup>
        <thead className="sticky top-0 z-10 bg-muted">
          {table.getHeaderGroups().map((hg) => (
            <tr key={hg.id} className="border-b-2 border-foreground">
              <th className="w-8 px-2 py-2" />
              {hg.headers.map((header) => (
                <th
                  key={header.id}
                  className="px-3 py-2 text-left align-middle"
                  style={{ width: header.getSize() }}
                >
                  {header.isPlaceholder
                    ? null
                    : flexRender(header.column.columnDef.header, header.getContext())}
                </th>
              ))}
            </tr>
          ))}
        </thead>
        <tbody>
          {paddingTop > 0 && (
            <tr>
              <td style={{ height: `${paddingTop}px` }} colSpan={columns.length + 1} />
            </tr>
          )}
          {virtualRows.map((vr) => {
            const row = tableRows[vr.index];
            if (!row) return null;
            const id = `${row.original.task_id}/${row.original.rollout_id}`;
            const isSelected = selectedIds.has(id);
            const isCursor = vr.index === cursorIndex;
            return (
              <tr
                key={row.id}
                data-index={vr.index}
                ref={virtualizer.measureElement}
                className={cn(
                  "border-b border-border/60 cursor-pointer transition-colors",
                  "hover:bg-accent/60",
                  isSelected && "bg-accent",
                  isCursor && "ring-inset ring-1 ring-ring bg-accent/40",
                )}
                onClick={(e) => handleRowClick(e, row.original, onRowClick)}
                onMouseEnter={() => onRowHover(row.original)}
              >
                {/* Ink keyboard-cursor indicator: left bar, square */}
                <td className="w-8 px-2 py-2 align-middle relative">
                  {isCursor && (
                    <span
                      aria-hidden
                      className="absolute left-0 top-0 bottom-0 w-[3px] bg-ring"
                    />
                  )}
                  <input
                    type="checkbox"
                    checked={isSelected}
                    onChange={() => onToggleSelect(id)}
                    onClick={(e) => e.stopPropagation()}
                    className="h-3.5 w-3.5 cursor-pointer border-border"
                    aria-label="Select rollout"
                  />
                </td>
                {row.getVisibleCells().map((cell) => (
                  <td
                    key={cell.id}
                    className="px-3 py-2 align-middle truncate"
                    style={{ maxWidth: cell.column.getSize() }}
                  >
                    {flexRender(cell.column.columnDef.cell, cell.getContext())}
                  </td>
                ))}
              </tr>
            );
          })}
          {paddingBottom > 0 && (
            <tr>
              <td
                style={{ height: `${paddingBottom}px` }}
                colSpan={columns.length + 1}
              />
            </tr>
          )}
        </tbody>
      </table>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Grouped view
// ---------------------------------------------------------------------------

function TaskGroupSection({
  taskId,
  rows,
  onRowClick,
  selectedIds,
  onToggleSelect,
}: {
  taskId: string;
  rows: RolloutSummary[];
  onRowClick: (r: RolloutSummary) => void;
  selectedIds: Set<string>;
  onToggleSelect: (id: string) => void;
}) {
  const scores = rows.map((r) => r.score).filter((s): s is number => s != null);
  const best = scores.length ? Math.max(...scores) : null;
  const passed = rows.filter((r) => r.gate_passed === true).length;
  const gated = rows.filter((r) => r.gate_passed != null).length;

  return (
    <div className="mb-8">
      <div className="flex items-baseline gap-3 px-1 pb-2 mb-2 border-b border-border">
        <h2 className="font-mono text-sm font-semibold">{taskId}</h2>
        <span className="text-xs text-muted-foreground font-mono">
          {rows.length} rollout{rows.length === 1 ? "" : "s"}
          {best != null && <>, best {best.toFixed(3)}</>}
          {gated > 0 && <>, {passed}/{gated} gate</>}
        </span>
      </div>
      <div className="border border-border">
        <table className="w-full border-collapse text-sm">
          <thead>
            <tr className="border-b border-border bg-muted">
              <th className="w-8 px-2 py-2" />
              <th className="px-3 py-2 text-left">
                <span className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
                  Rollout ID
                </span>
              </th>
              <th className="px-3 py-2 text-left">
                <span className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
                  Status
                </span>
              </th>
              <th className="px-3 py-2 text-right">
                <span className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
                  Score
                </span>
              </th>
              <th className="px-3 py-2 text-center">
                <span className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
                  Gate
                </span>
              </th>
              <th className="px-3 py-2 text-left">
                <span className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
                  Backend
                </span>
              </th>
              <th className="px-3 py-2 text-left">
                <span className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
                  Model
                </span>
              </th>
              <th className="px-3 py-2 text-left">
                <span className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
                  Launched
                </span>
              </th>
              <th className="px-3 py-2 text-right">
                <span className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
                  Cost
                </span>
              </th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r) => {
              const id = `${r.task_id}/${r.rollout_id}`;
              const isSelected = selectedIds.has(id);
              return (
                <tr
                  key={r.rollout_id}
                  className={cn(
                    "border-b border-border/60 cursor-pointer transition-colors hover:bg-accent/60",
                    isSelected && "bg-accent",
                  )}
                  onClick={(e) => handleRowClick(e, r, onRowClick)}
                >
                  <td className="w-8 px-2 py-2 align-middle">
                    <input
                      type="checkbox"
                      checked={isSelected}
                      onChange={() => onToggleSelect(id)}
                      onClick={(e) => e.stopPropagation()}
                      className="h-3.5 w-3.5 cursor-pointer"
                    />
                  </td>
                  <td className="px-3 py-2 font-mono text-[11px] text-muted-foreground truncate max-w-[160px]">
                    {r.rollout_id}
                  </td>
                  <td className="px-3 py-2">
                    <StatusCell status={r.status} phase={r.phase} />
                  </td>
                  <td className="px-3 py-2 text-right">
                    <ScoreCell score={r.score} />
                  </td>
                  <td className="px-3 py-2">
                    <GateCell passed={r.gate_passed} />
                  </td>
                  <td className="px-3 py-2">
                    <BackendBadge backend={r.backend} />
                  </td>
                  <td className="px-3 py-2 font-mono text-[11px] truncate max-w-[120px]">
                    {r.model_fingerprint ? shortenModel(r.model_fingerprint) : "—"}
                  </td>
                  <td className="px-3 py-2 text-xs text-muted-foreground tabular-nums whitespace-nowrap">
                    {formatRelative(r.created_at)}
                  </td>
                  <td className="px-3 py-2 text-right text-xs text-muted-foreground tabular-nums">
                    {formatCost(r.cost_usd)}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Command palette (cmdk)
// ---------------------------------------------------------------------------

interface CommandPaletteProps {
  open: boolean;
  onClose: () => void;
  rows: RolloutSummary[];
  onSelect: (r: RolloutSummary) => void;
}

function CommandPalette({ open, onClose, rows, onSelect }: CommandPaletteProps) {
  if (!open) return null;

  return (
    <div
      className="fixed inset-0 z-50 flex items-start justify-center pt-[20vh]"
      onClick={onClose}
    >
      <div className="absolute inset-0 bg-background/60 backdrop-blur-sm" />
      <div
        className="relative z-10 w-full max-w-lg border border-border bg-background shadow-lg"
        onClick={(e) => e.stopPropagation()}
      >
        <Command className="[&_[cmdk-input-wrapper]]:border-b [&_[cmdk-input-wrapper]]:border-border">
          <div className="flex items-center px-3 border-b border-border">
            <Search className="h-4 w-4 shrink-0 text-muted-foreground mr-2" />
            <Command.Input
              autoFocus
              placeholder="Jump to rollout…"
              className="flex-1 h-11 bg-transparent text-sm font-mono placeholder:text-muted-foreground outline-none"
            />
            <button
              type="button"
              onClick={onClose}
              className="text-muted-foreground hover:text-foreground"
            >
              <X className="h-4 w-4" />
            </button>
          </div>
          <Command.List className="max-h-64 overflow-y-auto">
            <Command.Empty className="py-6 text-center text-xs text-muted-foreground font-mono">
              No rollouts found.
            </Command.Empty>
            {rows.slice(0, 100).map((r) => (
              <Command.Item
                key={`${r.task_id}/${r.rollout_id}`}
                value={`${r.task_id} ${r.rollout_id} ${r.model_fingerprint ?? ""} ${r.backend ?? ""}`}
                onSelect={() => {
                  onSelect(r);
                  onClose();
                }}
                className="flex items-center gap-3 px-3 py-2 cursor-pointer text-sm data-[selected=true]:bg-accent data-[selected=true]:text-accent-foreground"
              >
                <span className="font-mono text-xs font-medium shrink-0">
                  {r.task_id}
                </span>
                <span className="font-mono text-[11px] text-muted-foreground truncate">
                  {r.rollout_id}
                </span>
                <StatusBadge status={r.status} />
              </Command.Item>
            ))}
          </Command.List>
        </Command>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main page
// ---------------------------------------------------------------------------

type ViewMode = "flat" | "grouped";

// Sort column IDs that map to SortingState
type SortDir = "asc" | "desc";

export default function Rollouts() {
  const navigate = useNavigate();
  const queryClient = useQueryClient();

  // Live stream deltas — used to inject `phase` + live status into running rows.
  // `connected` gates the fallback poll below: while the SSE stream is healthy,
  // its deltas + create-triggered invalidateQueries (below) keep the list fresh,
  // so the periodic full-list refetch is redundant. If the stream drops (onerror
  // closes SSE in use-rollouts-stream), polling resumes until it reconnects.
  const { deltas: streamDeltas, connected: streamConnected } = useRolloutsStream();

  const { data, isPending, error } = useQuery({
    queryKey: ["rollouts"],
    queryFn: fetchAllRollouts,
    refetchInterval: streamConnected ? false : 5_000,
  });

  // URL state — filters
  const [q, setQ] = useQueryState("q", parseAsString.withDefault(""));
  const [statusF, setStatusF] = useQueryState("status", parseAsString.withDefault(""));
  const [taskF, setTaskF] = useQueryState("task", parseAsString.withDefault(""));
  const [backendF, setBackendF] = useQueryState("backend", parseAsString.withDefault(""));
  const [gateF, setGateF] = useQueryState("gate", parseAsString.withDefault(""));
  const [view, setView] = useQueryState(
    "view",
    parseAsStringEnum<ViewMode>(["flat", "grouped"]).withDefault("flat"),
  );

  // URL state — sort (deferred in Task 7, wired here for deep-linkable sorting)
  const [sortCol, setSortCol] = useQueryState("sort", parseAsString.withDefault("created_at"));
  const [sortDir, setSortDir] = useQueryState(
    "dir",
    parseAsStringEnum<SortDir>(["asc", "desc"]).withDefault("desc"),
  );

  // Derived SortingState for react-table
  const sorting = useMemo<SortingState>(
    () => [{ id: sortCol, desc: sortDir === "desc" }],
    [sortCol, sortDir],
  );

  const handleSortingChange = useCallback(
    (next: SortingState) => {
      if (next.length === 0) {
        // User toggled off sorting — reset to default
        setSortCol(null);
        setSortDir(null);
      } else {
        const s = next[0];
        if (s) {
          setSortCol(s.id === "created_at" ? null : s.id);
          setSortDir(s.desc ? null : "asc"); // "desc" is the default so null = desc
        }
      }
    },
    [setSortCol, setSortDir],
  );

  // Local state
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [paletteOpen, setPaletteOpen] = useState(false);

  const searchRef = useRef<HTMLInputElement>(null);

  // Merge stream deltas over fetched rows so running rows show live phase.
  const all = useMemo(() => {
    const base = data ?? [];
    if (Object.keys(streamDeltas).length === 0) return base;
    return base.map((r) => {
      const key = `${r.task_id}/${r.rollout_id}`;
      const delta = streamDeltas[key];
      if (!delta) return r;
      // Only merge live-presentation fields the stream drives; never clobber a
      // non-null fetched value with a null/stale streamed one.
      return {
        ...r,
        phase: (delta.phase as RolloutSummary["phase"]) ?? r.phase,
        status: (delta.status as RolloutSummary["status"]) ?? r.status,
      } as RolloutSummary;
    });
  }, [data, streamDeltas]);

  // When the stream reports a key not in base (newly-created rollout), trigger
  // an immediate list refetch rather than waiting for the 5 s interval.
  useEffect(() => {
    const base = data ?? [];
    if (Object.keys(streamDeltas).length === 0) return;
    const baseKeys = new Set(base.map((r) => `${r.task_id}/${r.rollout_id}`));
    const hasUnknown = Object.keys(streamDeltas).some((k) => !baseKeys.has(k));
    if (hasUnknown) {
      queryClient.invalidateQueries({ queryKey: ["rollouts"] });
    }
  }, [streamDeltas, data, queryClient]);

  // Facet options
  const taskOptions = useMemo(() => {
    const s = new Set<string>();
    for (const r of all) s.add(r.task_id);
    return [...s].sort();
  }, [all]);

  const statusOptions = useMemo(() => {
    const s = new Set<string>();
    for (const r of all) if (r.status) s.add(r.status);
    return [...s].sort();
  }, [all]);

  const backendOptions = useMemo(() => {
    const s = new Set<string>();
    for (const r of all) if (r.backend) s.add(r.backend);
    return [...s].sort();
  }, [all]);

  // Fuse instance for text search — rebuilt when `all` changes
  const fuse = useMemo(
    () =>
      new Fuse(all, {
        keys: ["task_id", "rollout_id", "model_fingerprint", "backend", "accelerator", "agent_implementer_model"],
        threshold: 0.3,
        ignoreLocation: true,
      }),
    [all],
  );

  const filtered = useMemo(() => {
    const needle = q.trim();
    // Text-filter first via Fuse (or pass-through if empty)
    const textFiltered = needle ? fuse.search(needle).map((r) => r.item) : all;
    // Then apply facet filters on the text-filtered subset
    return textFiltered.filter((r) => {
      if (statusF && r.status !== statusF) return false;
      if (backendF && r.backend !== backendF) return false;
      if (taskF && r.task_id !== taskF) return false;
      if (gateF) {
        const tag =
          r.gate_passed === true ? "pass" : r.gate_passed === false ? "fail" : "n/a";
        if (tag !== gateF) return false;
      }
      return true;
    });
  }, [all, fuse, q, statusF, backendF, taskF, gateF]);

  const grouped = useMemo(() => {
    const m = new Map<string, RolloutSummary[]>();
    for (const r of filtered) {
      const arr = m.get(r.task_id);
      if (arr) arr.push(r);
      else m.set(r.task_id, [r]);
    }
    return [...m.entries()].sort((a, b) => a[0].localeCompare(b[0]));
  }, [filtered]);

  const hasFilters =
    !!q || !!statusF || !!backendF || !!taskF || !!gateF;

  const clearFilters = useCallback(() => {
    setQ(null);
    setStatusF(null);
    setBackendF(null);
    setTaskF(null);
    setGateF(null);
  }, [setQ, setStatusF, setBackendF, setTaskF, setGateF]);

  const onRowClick = useCallback(
    (r: RolloutSummary) =>
      navigate(
        `/rollouts/${encodeURIComponent(r.task_id)}/${encodeURIComponent(r.rollout_id)}`,
      ),
    [navigate],
  );

  // Hover prefetch: pre-warm the rollout detail cache before the user clicks
  const onRowHover = useCallback(
    (r: RolloutSummary) => {
      void queryClient.prefetchQuery({
        queryKey: ["rollout", r.task_id, r.rollout_id],
        queryFn: () => fetchRollout(r.task_id, r.rollout_id),
        staleTime: 30_000,
      });
    },
    [queryClient],
  );

  const toggleSelect = useCallback((id: string) => {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  }, []);

  // Global keyboard shortcuts
  useEffect(() => {
    function handler(e: KeyboardEvent) {
      if ((e.metaKey || e.ctrlKey) && e.key === "k") {
        e.preventDefault();
        setPaletteOpen((o) => !o);
      }
      if (e.key === "Escape") {
        if (paletteOpen) {
          setPaletteOpen(false);
        } else {
          searchRef.current?.blur();
        }
      }
      if (
        e.key === "/" &&
        document.activeElement?.tagName !== "INPUT" &&
        document.activeElement?.tagName !== "TEXTAREA" &&
        document.activeElement?.tagName !== "SELECT"
      ) {
        e.preventDefault();
        searchRef.current?.focus();
      }
    }
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [paletteOpen]);

  const nTasks = useMemo(
    () => new Set(all.map((r) => r.task_id)).size,
    [all],
  );

  const subtitleText = isPending
    ? "Loading…"
    : error
      ? "Failed to load rollouts."
      : `${filtered.length}${filtered.length !== all.length ? ` of ${all.length}` : ""} rollout${all.length === 1 ? "" : "s"} across ${nTasks} task${nTasks === 1 ? "" : "s"}`;

  return (
    <>
      <CommandPalette
        open={paletteOpen}
        onClose={() => setPaletteOpen(false)}
        rows={filtered}
        onSelect={onRowClick}
      />

      <div className="mx-auto w-full max-w-7xl px-6 py-8">
        {/* Page header */}
        <div className="mb-5 flex flex-wrap items-end justify-between gap-3">
          <div>
            <h1 className="font-mono text-sm font-bold uppercase tracking-widest">
              Rollouts
            </h1>
            <p className="text-xs text-muted-foreground mt-0.5 font-mono">
              {subtitleText}
            </p>
          </div>
          <div className="flex items-center gap-2">
            {selectedIds.size > 0 && (
              <Button
                variant="outline"
                size="sm"
                className="h-7 gap-1.5 text-xs"
                onClick={() => {
                  const ids = [...selectedIds];
                  navigate(`/compare?ids=${ids.map(encodeURIComponent).join(",")}`);
                }}
              >
                <GitCompare className="h-3.5 w-3.5" />
                Compare {selectedIds.size}
              </Button>
            )}
            <Segmented
              value={view}
              onChange={(v) => void setView(v)}
              options={[
                { value: "flat", label: "Flat", icon: Table2 },
                { value: "grouped", label: "By task", icon: Layers },
              ]}
            />
          </div>
        </div>

        {/* Filter bar */}
        <div className="mb-4 flex flex-wrap items-center gap-2">
          <div className="relative min-w-[200px] max-w-xs flex-1">
            <Search className="pointer-events-none absolute left-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-muted-foreground" />
            <input
              ref={searchRef}
              type="text"
              placeholder="Search task, id, model… (/ to focus)"
              value={q}
              onChange={(e) => setQ(e.target.value || null)}
              className={cn(
                "h-7 w-full pl-8 pr-8 text-xs font-mono border-[1.5px] border-border bg-background",
                "placeholder:text-muted-foreground focus:outline-none focus:border-foreground focus:shadow-[var(--shadow-sm)]",
              )}
            />
            {q ? (
              <button
                type="button"
                onClick={() => setQ(null)}
                className="absolute right-2.5 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground"
              >
                <X className="h-3.5 w-3.5" />
              </button>
            ) : (
              <div className="pointer-events-none absolute right-2 top-1/2 -translate-y-1/2 flex items-center gap-0.5">
                <kbd className="font-mono text-[10px] text-muted-foreground border-[1.5px] border-border bg-muted shadow-[var(--shadow-xs)] px-1">
                  ⌘K
                </kbd>
              </div>
            )}
          </div>

          <SelectFilter
            label="All tasks"
            options={taskOptions}
            value={taskF}
            onValueChange={(v) => setTaskF(v || null)}
          />
          <SelectFilter
            label="All status"
            options={statusOptions}
            value={statusF}
            onValueChange={(v) => setStatusF(v || null)}
          />
          <SelectFilter
            label="Any gate"
            options={["pass", "fail", "n/a"]}
            value={gateF}
            onValueChange={(v) => setGateF(v || null)}
          />
          <SelectFilter
            label="All backends"
            options={backendOptions}
            value={backendF}
            onValueChange={(v) => setBackendF(v || null)}
          />

          {hasFilters && (
            <Button
              variant="ghost"
              size="sm"
              className="h-7 text-xs text-muted-foreground gap-1"
              onClick={clearFilters}
            >
              <X className="h-3 w-3" /> Clear
            </Button>
          )}
        </div>

        {/* Error */}
        {error && (
          <div className="border border-destructive/40 bg-destructive/10 p-4 text-sm text-destructive font-mono">
            Failed to load rollouts.
          </div>
        )}

        {/* Content */}
        {!error &&
          (view === "grouped" ? (
            grouped.length === 0 && !isPending ? (
              <EmptyState hasFilters={hasFilters} />
            ) : (
              <div>
                {grouped.map(([taskId, rows]) => (
                  <TaskGroupSection
                    key={taskId}
                    taskId={taskId}
                    rows={rows}
                    onRowClick={onRowClick}
                    selectedIds={selectedIds}
                    onToggleSelect={toggleSelect}
                  />
                ))}
              </div>
            )
          ) : (
            <RolloutTable
              rows={filtered}
              onRowClick={onRowClick}
              onRowHover={onRowHover}
              selectedIds={selectedIds}
              onToggleSelect={toggleSelect}
              isLoading={isPending}
              hasFilters={hasFilters}
              sorting={sorting}
              onSortingChange={handleSortingChange}
            />
          ))}
      </div>
    </>
  );
}
