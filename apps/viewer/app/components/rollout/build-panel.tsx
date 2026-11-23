import * as Collapsible from "@radix-ui/react-collapsible";
import { useQuery } from "@tanstack/react-query";
import Fuse from "fuse.js";
import { ChevronRight } from "lucide-react";
import { parseAsInteger, useQueryState } from "nuqs";
import { useMemo, useRef, useState } from "react";

import { StepContent } from "~/components/trajectory/step-content";
import { StepTrigger } from "~/components/trajectory/step-trigger";
import { fetchTrajectory, fileUrl } from "~/lib/api";
import { cn } from "~/lib/utils";
import type { RolloutEvent } from "~/components/rollout/use-rollout-events";
import { Fact } from "~/components/rollout/build-fact";
import { buildFacts, buildFiles, buildOf, buildProgress, valueOf } from "~/lib/build";
import type { BuildFact, BuildProgress } from "~/lib/build";
import type {
  ContentPart,
  ObservationContent,
  RolloutBuild,
  RolloutRecord,
  Step,
  Trajectory,
} from "~/lib/types";

// ---------------------------------------------------------------------------
// Format helpers
// ---------------------------------------------------------------------------

function fmtDuration(a: number, b: number): string {
  const s = b - a;
  if (s < 60) return `${Math.round(s)}s`;
  if (s < 3600) return `${Math.floor(s / 60)}m ${Math.round(s % 60)}s`;
  return `${Math.floor(s / 3600)}h ${Math.floor((s % 3600) / 60)}m`;
}

function fmtCost(v: number): string {
  if (v < 0.01) return `$${v.toFixed(4)}`;
  return `$${v.toFixed(2)}`;
}

function fmtNum(v: number): string {
  return v.toLocaleString();
}

// ---------------------------------------------------------------------------
// Meta bar — a single dense hairline strip above the accordion
// ---------------------------------------------------------------------------

function MetaItem({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <span className="flex items-baseline gap-1.5">
      <span className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
        {label}
      </span>
      <span className="font-mono text-xs tabular-nums text-foreground">{value}</span>
    </span>
  );
}

export function BuildMetaBar({ build }: { build: RolloutBuild }) {
  const facts = buildFacts(build);
  return (
    <div className="flex flex-wrap items-baseline gap-x-6 gap-y-1 ink-block bg-card px-3 py-2">
      <MetaItem
        label="image"
        value={
          <Fact
            fact={facts.image}
            format={(img) => (
              <span className="truncate" title={`${img.key}\nsha256 ${img.sha256}`}>
                {img.key.length > 40 ? `…${img.key.slice(-37)}` : img.key}
              </span>
            )}
          />
        }
      />
      <MetaItem label="steps" value={<Fact fact={facts.steps} format={fmtNum} />} />
      <MetaItem
        label="tokens"
        value={
          <>
            <Fact fact={facts.inputTokens} format={fmtNum} /> /{" "}
            <Fact fact={facts.outputTokens} format={fmtNum} />
          </>
        }
      />
      <MetaItem label="cost" value={<Fact fact={facts.cost} format={fmtCost} />} />
      <MetaItem
        label="duration"
        value={<Fact fact={facts.span} format={(s) => fmtDuration(s.start, s.end)} />}
      />
    </div>
  );
}

// ---------------------------------------------------------------------------
// Step search helpers
// ---------------------------------------------------------------------------

function extractObservationText(content: ObservationContent): string {
  if (!content) return "";
  if (typeof content === "string") return content;
  return content
    .map((p: ContentPart) => (p.type === "text" ? (p.text ?? "") : ""))
    .join(" ");
}

function stepToText(step: Step): string {
  const parts: string[] = [];

  // message
  if (typeof step.message === "string") {
    parts.push(step.message);
  } else if (Array.isArray(step.message)) {
    for (const p of step.message) {
      if (p.type === "text" && p.text) parts.push(p.text);
    }
  }

  // reasoning
  if (step.reasoning_content) parts.push(step.reasoning_content);

  // tool calls
  if (step.tool_calls) {
    for (const tc of step.tool_calls) {
      parts.push(tc.function_name);
      try {
        parts.push(JSON.stringify(tc.arguments));
      } catch {
        // ignore
      }
    }
  }

  // observation
  if (step.observation?.results) {
    for (const r of step.observation.results) {
      parts.push(extractObservationText(r.content));
    }
  }

  return parts.join(" ");
}

// ---------------------------------------------------------------------------
// One collapsible step row
// ---------------------------------------------------------------------------

function StepRow({
  step,
  prevTimestamp,
  startTimestamp,
  imageBase,
  defaultOpen = false,
}: {
  step: Step;
  prevTimestamp: string | null;
  startTimestamp: string | null;
  imageBase: BuildFact<string>;
  defaultOpen?: boolean;
}) {
  return (
    <Collapsible.Root
      defaultOpen={defaultOpen}
      className="group border-b border-border last:border-b-0"
    >
      <Collapsible.Trigger
        className={cn(
          "flex w-full items-center gap-2 px-3 py-1.5 text-left transition-colors",
          "hover:bg-accent/50 focus-visible:outline-none focus-visible:bg-accent/50",
        )}
      >
        <ChevronRight className="h-3.5 w-3.5 shrink-0 text-muted-foreground transition-transform group-data-[state=open]:rotate-90" />
        <StepTrigger
          step={step}
          prevTimestamp={prevTimestamp}
          startTimestamp={startTimestamp}
        />
      </Collapsible.Trigger>
      <Collapsible.Content className="border-t border-border bg-muted/10">
        <StepContent step={step} imageBase={imageBase} />
      </Collapsible.Content>
    </Collapsible.Root>
  );
}

// ---------------------------------------------------------------------------
// Trajectory accordion
// ---------------------------------------------------------------------------

// How often a running instance's page asks again for the trajectory of a build that has finished.
const TRAJECTORY_POLL_MS = 5000;

// The trajectory endpoint says whether there is one: the build record does not. It is asked again
// when the rollout stops running, and *missing* shows while it has none. The stage publishes the
// trajectory with its other outputs just after it logs that the build finished, hours before the
// instance ends, so a running instance is asked again until it has one.
function TrajectoryList({
  rec,
  built,
  missing,
}: {
  rec: RolloutRecord;
  built: boolean;
  missing: React.ReactNode;
}) {
  const task = rec.task_id;
  const id = rec.rollout_id;
  const running = rec.status === "running";

  // Deep-link: ?expand=<step_id> opens that step on load.
  const [expand] = useQueryState("expand", parseAsInteger);

  const { data, isLoading, error } = useQuery({
    queryKey: ["trajectory", task, id, running],
    queryFn: () => fetchTrajectory(task, id),
    retry: false,
    refetchInterval: (query) =>
      running && built && query.state.data == null ? TRAJECTORY_POLL_MS : false,
  });

  // The trajectory's image paths are relative to the directory of its file.
  const dir = buildFiles(rec).transcriptDir;
  const imageBase: BuildFact<string> = "value" in dir ? { value: fileUrl(task, id, dir.value) } : dir;

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-10">
        <span className="animate-pulse font-mono text-xs text-muted-foreground">
          Loading trajectory…
        </span>
      </div>
    );
  }

  if (error || !data) return <>{missing}</>;

  const traj = data as Trajectory;
  const steps = traj.steps ?? [];

  if (steps.length === 0) {
    return (
      <div className="border border-border bg-muted/20 p-4">
        <p className="font-mono text-xs text-muted-foreground">
          Trajectory has no steps.
        </p>
      </div>
    );
  }

  const startTs = steps[0]?.timestamp ?? null;

  return (
    <TrajectoryListInner
      steps={steps}
      startTs={startTs}
      imageBase={imageBase}
      expand={expand}
    />
  );
}

function TrajectoryListInner({
  steps,
  startTs,
  imageBase,
  expand,
}: {
  steps: Step[];
  startTs: string | null;
  imageBase: BuildFact<string>;
  expand: number | null;
}) {
  // Search state
  const [searchInput, setSearchInput] = useState("");
  const [searchQuery, setSearchQuery] = useState("");
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  // Build flat searchable objects once per steps array
  const searchable = useMemo(
    () => steps.map((step) => ({ step, text: stepToText(step) })),
    [steps],
  );

  // Build Fuse instance
  const fuse = useMemo(
    () => new Fuse(searchable, { keys: ["text"], threshold: 0.3, ignoreLocation: true }),
    [searchable],
  );

  // Order filtered results by Fuse relevance (default) or by turn number.
  const [orderByTurn, setOrderByTurn] = useState(false);

  // Filtered + ordered steps
  const visibleSteps = useMemo(() => {
    const result = searchQuery.trim()
      ? fuse.search(searchQuery.trim()).map((r) => r.item.step)
      : steps;
    if (!orderByTurn) return result;
    const order = new Map(steps.map((s, i) => [s, i]));
    return [...result].sort((a, b) => (order.get(a) ?? 0) - (order.get(b) ?? 0));
  }, [steps, fuse, searchQuery, orderByTurn]);

  function handleSearchChange(e: React.ChangeEvent<HTMLInputElement>) {
    const val = e.target.value;
    setSearchInput(val);
    if (debounceRef.current) clearTimeout(debounceRef.current);
    debounceRef.current = setTimeout(() => setSearchQuery(val), 150);
  }

  return (
    <div className="ink-panel bg-card">
      {/* Search bar */}
      <div className="flex items-center gap-3 px-3 py-2 border-b border-border">
        <input
          type="text"
          value={searchInput}
          onChange={handleSearchChange}
          placeholder="Search steps…"
          className="h-7 w-56 px-2 text-xs font-mono border-[1.5px] border-border bg-background placeholder:text-muted-foreground focus:outline-none focus:border-foreground focus:shadow-[var(--shadow-sm)]"
        />
        {searchInput.trim() && (
          <div className="flex h-7 items-center border-[1.5px] border-border">
            <button
              type="button"
              onClick={() => setOrderByTurn(false)}
              className={cn(
                "h-full px-2.5 text-[11px] font-mono transition-colors",
                !orderByTurn
                  ? "bg-foreground text-background"
                  : "text-muted-foreground hover:text-foreground",
              )}
            >
              Relevance
            </button>
            <button
              type="button"
              onClick={() => setOrderByTurn(true)}
              className={cn(
                "h-full border-l-[1.5px] border-border px-2.5 text-[11px] font-mono transition-colors",
                orderByTurn
                  ? "bg-foreground text-background"
                  : "text-muted-foreground hover:text-foreground",
              )}
            >
              Turn #
            </button>
          </div>
        )}
        <span className="font-mono text-[10px] text-muted-foreground tabular-nums">
          showing {visibleSteps.length} of {steps.length} steps
        </span>
      </div>
      {visibleSteps.map((step, i) => {
        const origIndex = steps.indexOf(step);
        return (
          <StepRow
            key={step.step_id ?? i}
            step={step}
            prevTimestamp={origIndex > 0 ? steps[origIndex - 1]?.timestamp ?? null : null}
            startTimestamp={startTs}
            imageBase={imageBase}
            defaultOpen={expand != null && step.step_id === expand}
          />
        );
      })}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Live build banner (running, no trajectory yet)
// ---------------------------------------------------------------------------

// What a running build's progress, read from the events of its stage's log, says.
function LiveBuildBanner({ progress }: { progress: BuildProgress }) {
  if (progress.phase === "none") return null;

  if (progress.phase === "finished") {
    return (
      <div className="border border-border bg-muted/10 px-4 py-3 font-mono text-xs text-muted-foreground">
        Build complete — publishing its outputs
        {progress.costUsd != null && (
          <span className="ml-2">
            cost{" "}
            <span className="font-semibold text-foreground">
              ${progress.costUsd.toFixed(4)}
            </span>
          </span>
        )}
      </div>
    );
  }

  return (
    <div className="animate-pulse border border-border/60 bg-muted/10 px-4 py-3 font-mono text-xs text-muted-foreground">
      {progress.phase === "starting"
        ? "Build starting — the sandbox is being set up; the agent has not started."
        : "Build in progress — trajectory will appear when complete."}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main panel
// ---------------------------------------------------------------------------

export function BuildPanel({
  rec,
  liveEvents = [],
}: {
  rec: RolloutRecord;
  liveEvents?: RolloutEvent[];
}) {
  const built = buildOf(rec);
  const build = valueOf(built);
  const running = rec.status === "running";
  const progress = useMemo(() => buildProgress(liveEvents), [liveEvents]);
  // Without a trajectory: the live banner while running, else the empty notice for a rollout
  // that built.
  const missing = running ? (
    <LiveBuildBanner progress={progress} />
  ) : build ? (
    <div className="border border-border bg-muted/20 p-4">
      <p className="font-mono text-xs text-muted-foreground">
        No build trajectory was captured.
      </p>
    </div>
  ) : null;

  return (
    <div className="w-full space-y-3 px-6 py-4">
      {"value" in built ? (
        <BuildMetaBar build={built.value} />
      ) : !running ? (
        <div className="border border-border bg-muted/20 p-4">
          <p className="font-mono text-xs text-muted-foreground">
            No build shown: {built.absent}.
          </p>
        </div>
      ) : null}

      <TrajectoryList rec={rec} built={progress.phase === "finished"} missing={missing} />
    </div>
  );
}
