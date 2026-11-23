import { useQuery, useQueryClient } from "@tanstack/react-query";
import { parseAsString, parseAsStringEnum, useQueryState } from "nuqs";
import { useCallback, useEffect } from "react";
import { Link, useNavigate, useParams } from "react-router";

import { StatusBadge } from "~/components/rollout/badges";
import { BuildPanel } from "~/components/rollout/build-panel";
import { DeployPanel } from "~/components/rollout/deploy-panel";
import { EvalPanel } from "~/components/rollout/eval-panel";
import { OverviewPanel } from "~/components/rollout/overview-panel";
import { PromptPanel } from "~/components/rollout/prompt-panel";
import { SourcePanel } from "~/components/rollout/source-panel";
import { TimelinePanel } from "~/components/rollout/timeline-panel";
import { useRolloutEvents, type RolloutEvent } from "~/components/rollout/use-rollout-events";
import { fetchRollout, fetchTraceSummary, recordUrl } from "~/lib/api";
import { useGotoList, useNumberKeyTabs } from "~/lib/hooks";
import { cn } from "~/lib/utils";
import type { RolloutRecord } from "~/lib/types";

// ---------------------------------------------------------------------------
// Tab definition
// ---------------------------------------------------------------------------

type Tab =
  | "overview"
  | "prompt"
  | "build"
  | "source"
  | "deploy"
  | "eval"
  | "timeline";

interface TabDef {
  key: Tab;
  label: string;
  count?: number | null;
}

const ALL_TABS: TabDef[] = [
  { key: "overview", label: "Overview" },
  { key: "prompt", label: "Prompt" },
  { key: "build", label: "Build" },
  { key: "source", label: "Source" },
  { key: "deploy", label: "Deploy" },
  { key: "eval", label: "Eval" },
  { key: "timeline", label: "Timeline" },
];

// ---------------------------------------------------------------------------
// Live-follow effect: on terminal SSE event, refetch the sealed record
// ---------------------------------------------------------------------------

function useLiveFollow(task: string, id: string, running: boolean) {
  const queryClient = useQueryClient();
  const { events, done } = useRolloutEvents(task, id, running);

  useEffect(() => {
    if (done) {
      void queryClient.invalidateQueries({ queryKey: ["rollout", task, id] });
    }
  }, [done, task, id, queryClient]);

  return { events, done };
}

// ---------------------------------------------------------------------------
// Loading skeleton
// ---------------------------------------------------------------------------

function LoadingState() {
  return (
    <div className="flex items-center justify-center py-16">
      <span className="font-mono text-xs text-muted-foreground animate-pulse">
        Loading rollout…
      </span>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Tab bar
// ---------------------------------------------------------------------------

function TabBar({
  tabs,
  active,
  onSelect,
}: {
  tabs: TabDef[];
  active: Tab;
  onSelect: (t: Tab) => void;
}) {
  return (
    <nav className="flex w-full shrink-0 items-stretch justify-center gap-0 overflow-x-auto overflow-y-hidden border-b border-border">
      {tabs.map((t) => (
        <button
          key={t.key}
          type="button"
          onClick={() => onSelect(t.key)}
          className={cn(
            "relative inline-flex items-center gap-1.5 whitespace-nowrap px-3 py-2 font-mono text-xs transition-colors focus-visible:outline-none",
            active === t.key
              ? "-mb-px border-b-[3px] border-foreground bg-accent text-foreground"
              : "text-muted-foreground hover:text-foreground",
          )}
        >
          <span>{t.label}</span>
          {t.count != null && (
            <span className="font-mono text-[10px] tabular-nums text-muted-foreground">
              {t.count}
            </span>
          )}
        </button>
      ))}
    </nav>
  );
}

// ---------------------------------------------------------------------------
// Page header
// ---------------------------------------------------------------------------

function RolloutHeader({
  rec,
  task,
  id,
  running,
  liveDone,
}: {
  rec: RolloutRecord;
  task: string;
  id: string;
  running: boolean;
  liveDone: boolean;
}) {
  const shortId = id.slice(0, 20);
  return (
    <div className="flex items-center justify-between gap-3 border-b border-border px-4 py-2">
      <div className="flex items-center gap-2 min-w-0">
        {/* Breadcrumb */}
        <Link
          to="/"
          className="font-mono text-xs text-muted-foreground hover:text-foreground transition-colors shrink-0"
        >
          Rollouts
        </Link>
        <span className="text-muted-foreground text-xs">/</span>
        <span className="font-mono text-xs font-medium text-foreground truncate">
          {task}
        </span>
        <span className="text-muted-foreground text-xs">/</span>
        <span className="font-mono text-xs text-muted-foreground truncate">
          {shortId}
        </span>
        <StatusBadge status={rec.status} />
        {running && !liveDone && (
          <span className="h-2 w-2 bg-primary animate-pulse shrink-0" />
        )}
      </div>
      <div className="flex items-center gap-3 shrink-0">
        <a
          href={recordUrl(task, id)}
          target="_blank"
          rel="noreferrer"
          className="font-mono text-xs text-muted-foreground hover:text-foreground transition-colors"
        >
          record ↗
        </a>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main route
// ---------------------------------------------------------------------------

export default function Rollout() {
  const { task = "", id = "" } = useParams();
  const navigate = useNavigate();

  const [tab, setTab] = useQueryState(
    "tab",
    parseAsStringEnum<Tab>([
      "overview",
      "prompt",
      "build",
      "source",
      "deploy",
      "eval",
      "timeline",
    ]).withDefault("overview"),
  );

  // Eval panel filter — lifted to URL so it's deep-linkable
  const [evalOnly, setEvalOnly] = useQueryState(
    "only",
    parseAsString.withDefault(""),
  );

  const { data: rec } = useQuery<RolloutRecord>({
    queryKey: ["rollout", task, id],
    queryFn: () => fetchRollout(task, id),
    refetchInterval: (query) => {
      const s = query.state.data?.status;
      return s === "running" || s === "stalled" ? 5000 : false;
    },
  });

  // Terminal eval id, for the Eval-tab request count. Shares the cache key with
  // EvalPanel's own trace-summary query so this is not a duplicate fetch.
  const evals = rec?.evals ?? [];
  const terminalEval =
    evals.find((e) => e.eval_id === "terminal") ?? evals[evals.length - 1];
  const terminalEvalId = terminalEval?.eval_id ?? "terminal";
  const { data: traceSummary } = useQuery({
    queryKey: ["trace-summary", task, id, terminalEvalId],
    queryFn: () => fetchTraceSummary(task, id, terminalEvalId),
    enabled: !!terminalEval,
  });

  const running = rec?.status === "running";
  const { events: liveEvents, done: liveDone } = useLiveFollow(task, id, running);

  // Navigate back to list via g l or Escape (when no dialog is open)
  const goList = useCallback(() => navigate("/"), [navigate]);
  useGotoList({ onActivate: goList });

  useEffect(() => {
    function handler(e: KeyboardEvent) {
      const tag = (e.target as HTMLElement)?.tagName;
      if (tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT") return;
      if (e.key === "Escape") {
        navigate("/");
      }
    }
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [navigate]);

  // Eval's tab label counts the trace's requests. Build and Source carry none: the build record
  // keeps no step count and lists no source files.
  const tabs: TabDef[] = ALL_TABS.map((t) => {
    if (t.key === "eval") return { ...t, count: traceSummary?.total ?? null };
    return t;
  });

  // Number keys 1–N switch tabs (hooks must be called unconditionally)
  useNumberKeyTabs({
    keys: tabs.map((t) => t.key),
    onSelect: (key) => void setTab(key),
    enabled: !!rec,
  });

  if (!rec) return <LoadingState />;

  const activeTab = tab ?? "overview";

  return (
    <div className="flex flex-col" style={{ minHeight: "calc(100vh - 48px)" }}>
      <RolloutHeader
        rec={rec}
        task={task}
        id={id}
        running={running}
        liveDone={liveDone}
      />

      <TabBar tabs={tabs} active={activeTab} onSelect={setTab} />

      <div className="flex-1 overflow-auto">
        {activeTab === "overview" && <OverviewPanel rec={rec} />}
        {activeTab === "prompt" && <PromptPanel rec={rec} />}
        {activeTab === "build" && <BuildPanel rec={rec} liveEvents={liveEvents} />}
        {activeTab === "source" && <SourcePanel rec={rec} />}
        {activeTab === "deploy" && <DeployPanel rec={rec} />}
        {activeTab === "eval" && (
          <EvalPanel
            rec={rec}
            only={evalOnly || null}
            onOnlyChange={(v) => void setEvalOnly(v ?? null)}
            liveEvents={liveEvents}
          />
        )}
        {activeTab === "timeline" && <TimelinePanel rec={rec} />}
      </div>
    </div>
  );
}
