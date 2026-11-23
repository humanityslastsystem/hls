import * as Popover from "@radix-ui/react-popover";
import { useQuery } from "@tanstack/react-query";
import { HelpCircle } from "lucide-react";
import { Fragment, useEffect, useRef, useState } from "react";

import { Segmented } from "~/components/rollout/segmented";
import {
  createTimeline,
  type TimelineController,
} from "~/components/rollout/timeline-canvas";
import { fetchTimeline } from "~/lib/api";
import type { RolloutRecord } from "~/lib/types";
import { cn } from "~/lib/utils";

type Group = "tenant" | "flat";
type ColorBy = "slo" | "tenant" | "lat";

export function TimelinePanel({ rec }: { rec: RolloutRecord }) {
  const task = rec.task_id;
  const id = rec.rollout_id;
  const terminal =
    rec.evals.find((e) => e.eval_id === "terminal") ??
    rec.evals[rec.evals.length - 1];
  const evalId = terminal?.eval_id ?? "terminal";

  const { data, isLoading, error } = useQuery({
    queryKey: ["timeline", task, id, evalId],
    queryFn: () => fetchTimeline(task, id, evalId),
    enabled: !!terminal,
  });

  const [group, setGroup] = useState<Group>("tenant");
  const [color, setColor] = useState<ColorBy>("slo");
  const [tenantOn, setTenantOn] = useState<boolean[]>([]);
  const [readout, setReadout] = useState("");

  const hostRef = useRef<HTMLDivElement | null>(null);
  const ctrlRef = useRef<TimelineController | null>(null);

  // initialise tenant toggles once data arrives
  useEffect(() => {
    if (data) setTenantOn(data.tenants.map(() => true));
  }, [data]);

  // (re)create the canvas controller when data or options change
  useEffect(() => {
    const host = hostRef.current;
    if (!host || !data || tenantOn.length !== data.tenants.length) return;
    ctrlRef.current?.destroy();
    ctrlRef.current = createTimeline(host, data, {
      group,
      color,
      tenantOn,
      onReadout: setReadout,
    });
    return () => {
      ctrlRef.current?.destroy();
      ctrlRef.current = null;
    };
  }, [data, group, color, tenantOn]);

  if (!terminal) return <Empty text="No eval to show a timeline for." />;
  if (isLoading) return <Empty text="Loading timeline..." />;
  if (error) return <Empty text="Failed to load timeline." />;
  if (!data || data.requests.length === 0)
    return <Empty text="No requests in this eval." />;

  return (
    <div
      className="flex flex-col gap-2 p-3"
      style={{ height: "calc(100vh - 130px)" }}
    >
      <div className="flex flex-wrap items-center gap-2">
        <Segmented
          label="group"
          value={group}
          onChange={setGroup}
          options={[
            { value: "tenant", label: "By tenant" },
            { value: "flat", label: "Flat" },
          ]}
        />
        <Segmented
          label="color"
          value={color}
          onChange={setColor}
          options={[
            { value: "slo", label: "SLO" },
            { value: "tenant", label: "Tenant" },
            { value: "lat", label: "Latency" },
          ]}
        />
        <div className="flex flex-wrap items-center gap-1.5">
          {data.tenants.map((t, i) => (
            <button
              key={t.id}
              type="button"
              onClick={() =>
                setTenantOn((p) => p.map((v, j) => (j === i ? !v : v)))
              }
              className={cn(
                "inline-flex h-7 items-center gap-1.5 border-[1.5px] border-[var(--block-border)] px-2.5 font-mono text-[11px]",
                tenantOn[i]
                  ? "bg-card text-foreground"
                  : "text-muted-foreground line-through opacity-50",
              )}
            >
              {t.id}
            </button>
          ))}
        </div>
        <span className="ml-auto font-mono text-[10px] text-muted-foreground">
          {readout}
        </span>
        <ControlsPopover />
      </div>
      <Legend />
      <div
        ref={hostRef}
        className="relative min-h-0 flex-1 border-[1.5px] border-[var(--block-border)] bg-card"
      />
    </div>
  );
}

function Empty({ text }: { text: string }) {
  return (
    <div className="flex h-full items-center justify-center font-mono text-xs text-muted-foreground">
      {text}
    </div>
  );
}

// ---- Legend: a visual key for the request-bar anatomy + SLO colors ----
function Legend() {
  return (
    <div className="flex flex-wrap items-center gap-x-6 gap-y-2 font-mono text-[10px] text-muted-foreground">
      <div className="flex items-center gap-2">
        <span className="uppercase tracking-widest">request</span>
        <span className="flex h-3.5 items-stretch border border-[var(--block-border)]">
          <span className="w-3 bg-foreground/25" title="scheduler slack" />
          <span
            className="w-5"
            style={{ backgroundColor: "var(--ok)", opacity: 0.4 }}
            title="waiting for first token"
          />
          <span className="w-8" style={{ backgroundColor: "var(--ok)" }} title="decoding" />
        </span>
        <span className="text-foreground/80">slack</span>
        <span aria-hidden>&rarr;</span>
        <span className="text-foreground/80">TTFT</span>
        <span aria-hidden>&rarr;</span>
        <span className="text-foreground/80">decode</span>
      </div>
      <div className="flex items-center gap-3">
        <span className="uppercase tracking-widest">SLO</span>
        <LegendDot color="var(--ok)" label="met" />
        <LegendDot color="var(--destructive)" label="violated" />
        <LegendDot color="var(--foreground)" label="error" />
      </div>
    </div>
  );
}

function LegendDot({ color, label }: { color: string; label: string }) {
  return (
    <span className="flex items-center gap-1.5">
      <span
        className="h-2.5 w-2.5 border border-black/10"
        style={{ backgroundColor: color }}
      />
      <span className="text-foreground/80">{label}</span>
    </span>
  );
}

// ---- Controls reference, tucked behind a popover ----
function Kbd({ children }: { children: React.ReactNode }) {
  return (
    <kbd className="border border-border bg-muted px-1 py-px text-[10px] leading-none">
      {children}
    </kbd>
  );
}

function ControlsSection({
  title,
  rows,
}: {
  title: string;
  rows: [React.ReactNode, string][];
}) {
  return (
    <div className="mb-2.5 last:mb-0">
      <div className="mb-1 text-[9px] uppercase tracking-widest text-muted-foreground">
        {title}
      </div>
      <div className="grid grid-cols-[auto_1fr] items-center gap-x-4 gap-y-1.5">
        {rows.map(([k, v], i) => (
          <Fragment key={i}>
            <span className="flex items-center gap-1 whitespace-nowrap">{k}</span>
            <span className="text-muted-foreground">{v}</span>
          </Fragment>
        ))}
      </div>
    </div>
  );
}

function ControlsPopover() {
  return (
    <Popover.Root>
      <Popover.Trigger asChild>
        <button
          type="button"
          className="inline-flex items-center gap-1 border-[1.5px] border-border px-2 py-1 font-mono text-[10px] text-muted-foreground transition-colors hover:border-foreground/40 hover:text-foreground"
        >
          <HelpCircle className="h-3 w-3" /> controls
        </button>
      </Popover.Trigger>
      <Popover.Portal>
        <Popover.Content
          align="end"
          sideOffset={6}
          className="z-50 border-[1.5px] border-foreground bg-card p-3 font-mono text-[11px] shadow-[var(--shadow-md)]"
        >
          <ControlsSection
            title="plot"
            rows={[
              [<Kbd>scroll</Kbd>, "scroll lanes"],
              [
                <>
                  <Kbd>&#8679;</Kbd> scroll
                </>,
                "pan time",
              ],
              [
                <>
                  <Kbd>&#8984;/Ctrl</Kbd> scroll
                </>,
                "zoom time",
              ],
              [<Kbd>drag</Kbd>, "pan"],
            ]}
          />
          <ControlsSection
            title="top strip"
            rows={[
              [<Kbd>scroll</Kbd>, "zoom time"],
              [<Kbd>drag</Kbd>, "pick window"],
            ]}
          />
          <ControlsSection
            title="right strip"
            rows={[
              [
                <>
                  <Kbd>scroll</Kbd> / <Kbd>drag</Kbd>
                </>,
                "zoom lanes",
              ],
              [<Kbd>click</Kbd>, "jump to lanes"],
            ]}
          />
          <Popover.Arrow className="fill-foreground" />
        </Popover.Content>
      </Popover.Portal>
    </Popover.Root>
  );
}
