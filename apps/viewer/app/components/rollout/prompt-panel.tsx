import { useQuery } from "@tanstack/react-query";
import type { Element, Parent, Root, Text } from "hast";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { visit } from "unist-util-visit";
import { fetchConfig, fileUrl } from "~/lib/api";
import { ABSENT, buildFiles, valueOf } from "~/lib/build";
import type { BriefProvenance, RolloutProvenance, RolloutRecord } from "~/lib/types";
import { cn } from "~/lib/utils";

// ---------------------------------------------------------------------------
// Block → display heading + spec fields map
// ---------------------------------------------------------------------------

const BLOCK_META: Record<
  string,
  { heading: string; fields: { label: string; path: string }[] }
> = {
  goal: {
    heading: "Goal",
    fields: [{ label: "Model fingerprint", path: "spec.model.fingerprint" }],
  },
  model: {
    heading: "Model",
    fields: [
      { label: "Fingerprint", path: "spec.model.fingerprint" },
      { label: "Base checkpoint", path: "spec.model.base_checkpoint" },
      { label: "Chat format", path: "spec.model.chat_format" },
    ],
  },
  hardware: {
    heading: "Hardware Envelope",
    fields: [
      { label: "Accelerator count", path: "spec.hardware.accelerator.count" },
      { label: "Accelerator kind", path: "spec.hardware.accelerator.kind" },
      {
        label: "Memory / device",
        path: "spec.hardware.accelerator.memory_gb_per_device",
      },
    ],
  },
  workload: {
    heading: "Workload",
    fields: [
      { label: "Arrival", path: "spec.workload.arrival.kind" },
      { label: "Output cap", path: "spec.workload.output_length_cap.max_t" },
      { label: "SLO target", path: "spec.workload.slo.target_ms" },
    ],
  },
  contract: {
    heading: "Network Contract",
    fields: [{ label: "Protocol", path: "contract" }],
  },
  build_instructions: {
    heading: "Build Instructions",
    fields: [{ label: "Server path", path: "server_path" }],
  },
  submission: {
    heading: "Submission",
    fields: [{ label: "Mode", path: "artifact.submission_mode" }],
  },
  // Rendered only when the producer sets a soft_budget (which is also what registers
  // the `get_budget` MCP tool). Without an entry here the block fell through to the
  // raw block name and rendered a lower-case "budget" heading.
  budget: {
    heading: "Your budget",
    fields: [],
  },
  // Header-LESS blocks (heading=""): render just the body via BriefProse, with no <h2>
  // and no provenance card -- like the verbatim text they carry. `uploads` is authored
  // by the task INCLUDING its own "## " heading, and `directives` is a task's standing
  // trailing text (e.g. a /goal for the ralph loop) -- neither wants a harness heading.
  uploads: {
    heading: "",
    fields: [],
  },
  directives: {
    heading: "",
    fields: [],
  },
};

// ---------------------------------------------------------------------------
// Spec path resolver
// ---------------------------------------------------------------------------

function resolveSpecPath(
  path: string,
  config: RolloutProvenance,
): string | null {
  const spec = config.spec as Record<string, unknown>;
  if (path === "server_path") return "/workspace/server.py";
  if (path === "contract") return "HTTP request/response";
  if (path.startsWith("spec.")) {
    let parts = path.slice(5).split(".");
    // Single-tenant tasks carry workload/data_source at the top level; multi-
    // tenant tasks nest them per tenant. When the top-level field is absent, fall
    // back to the first tenant so the brief's provenance still resolves.
    const head = parts[0];
    const tenants = (spec.tenants ?? []) as unknown[];
    if ((head === "workload" || head === "data_source") && spec[head] == null && tenants.length > 0) {
      parts = ["tenants", "0", ...parts];
    }
    let cur: unknown = spec;
    for (const p of parts) {
      if (cur == null || typeof cur !== "object") return null;
      cur = (cur as Record<string, unknown>)[p];
    }
    return cur != null ? String(cur) : null;
  }
  if (path.startsWith("artifact.")) {
    const parts = path.slice(9).split(".");
    let cur: unknown = config.artifact as Record<string, unknown>;
    for (const p of parts) {
      if (cur == null || typeof cur !== "object") return null;
      cur = (cur as Record<string, unknown>)[p];
    }
    return cur != null ? String(cur) : null;
  }
  return null;
}

// ---------------------------------------------------------------------------
// Rehype plugin: highlight override values in rendered markdown
// ---------------------------------------------------------------------------

function rehypeHighlightOverrides(values: string[]) {
  return () => (tree: unknown) => {
    if (values.length === 0) return;
    visit(
      tree as Root,
      "text",
      (node: Text, index: number | undefined, parent: Parent | undefined) => {
        if (!parent || index == null) return;
        const str = node.value;
        for (const val of values) {
          if (!val) continue;
          const idx = str.indexOf(val);
          if (idx !== -1) {
            const before = str.slice(0, idx);
            const after = str.slice(idx + val.length);
            const span: Element = {
              type: "element",
              tagName: "span",
              properties: {
                className: [
                  "bg-amber-50",
                  "dark:bg-amber-950/30",
                  "border-b",
                  "border-amber-400",
                  "dark:border-amber-500",
                  "px-0.5",
                  "cursor-help",
                ],
                title: "Value override",
                "data-override": val,
              },
              children: [{ type: "text", value: val }],
            };
            const replacement: (Text | Element)[] = [];
            if (before) replacement.push({ type: "text", value: before });
            replacement.push(span);
            if (after) replacement.push({ type: "text", value: after });
            parent.children.splice(index, 1, ...replacement);
            return; // only highlight first occurrence per text node
          }
        }
      },
    );
  };
}

// ---------------------------------------------------------------------------
// Provenance hover card
// ---------------------------------------------------------------------------

interface SectionProv {
  block: string;
  source: "base" | "task";
  rendered: string;
}

function ProvenanceCard({
  sec,
  config,
}: {
  sec: SectionProv;
  config: RolloutProvenance;
}) {
  const meta = BLOCK_META[sec.block];
  if (!meta) return null;

  const isTask = sec.source === "task";
  const tmplLabel = isTask
    ? `task template — block ${sec.block} (overrides default)`
    : `default brief — block ${sec.block}`;

  const fields = meta.fields
    .map((f) => ({
      ...f,
      value: resolveSpecPath(f.path, config),
    }))
    .filter((f) => f.value != null);

  const overrides = config.rollout_overrides ?? {};

  return (
    <span className="group relative inline-block">
      <span
        className="font-mono text-[9px] border border-muted-foreground/40 text-muted-foreground/60 inline-flex items-center justify-center w-[13px] h-[13px] ml-2 align-middle cursor-help select-none group-hover:text-muted-foreground transition-colors"
        style={{ verticalAlign: "2px" }}
        aria-label="Provenance"
      >
        i
      </span>
      {/* Popover card — fully reset typography */}
      <span
        className="absolute left-0 top-[calc(100%+6px)] z-50 hidden group-hover:block w-[300px] bg-popover border border-foreground shadow-[3px_3px_0_var(--color-border)] text-left"
        style={{
          fontSize: "11px",
          fontWeight: 400,
          lineHeight: "1.45",
          fontFamily: "var(--font-sans)",
          fontStyle: "normal",
          letterSpacing: 0,
          textTransform: "none",
          color: "var(--color-foreground)",
        }}
      >
        {/* Card header */}
        <span
          className="flex items-center gap-2 px-[11px] py-[7px] border-b border-border"
          style={{
            fontSize: "12px",
            lineHeight: "1.3",
            fontWeight: 600,
          }}
        >
          {meta.heading}
          <span
            className={cn(
              "ml-auto font-mono px-[5px] py-[1px]",
              isTask
                ? "bg-foreground text-background"
                : "border border-muted-foreground text-muted-foreground",
            )}
            style={{
              fontSize: "8.5px",
              fontWeight: 700,
              textTransform: "uppercase",
              letterSpacing: "0.3px",
              lineHeight: "1.5",
            }}
          >
            {sec.source}
          </span>
        </span>

        {/* Template row */}
        <span
          className="block px-[11px] py-[6px] border-b border-border/60"
          style={{ lineHeight: "1.45" }}
        >
          <span
            className="block"
            style={{
              fontSize: "8.5px",
              textTransform: "uppercase",
              letterSpacing: "0.5px",
              color: "var(--color-muted-foreground)",
              marginBottom: "3px",
              lineHeight: "1.3",
            }}
          >
            Template
          </span>
          <span
            className="font-mono block"
            style={{ fontSize: "11px", lineHeight: "1.5" }}
          >
            {tmplLabel}
          </span>
        </span>

        {/* Field rows */}
        {fields.length > 0 && (
          <span
            className="block px-[11px] py-[6px]"
            style={{ lineHeight: "1.45" }}
          >
            <span
              className="block"
              style={{
                fontSize: "8.5px",
                textTransform: "uppercase",
                letterSpacing: "0.5px",
                color: "var(--color-muted-foreground)",
                marginBottom: "3px",
                lineHeight: "1.3",
              }}
            >
              Interpolated values
            </span>
            {fields.map((f) => {
              const isOverridden = Object.entries(overrides).some(
                ([, v]) => String(v) === f.value,
              );
              return (
                <span
                  key={f.path}
                  className={cn(
                    "block py-[3px]",
                    isOverridden &&
                      "bg-amber-50 dark:bg-amber-950/30 mx-[-11px] px-[11px] mt-[3px]",
                  )}
                >
                  <span
                    className="font-mono block"
                    style={{
                      fontSize: "11px",
                      lineHeight: "1.4",
                      color: "var(--color-foreground)",
                    }}
                  >
                    {f.value}
                  </span>
                  <span
                    className="font-mono block"
                    style={{
                      fontSize: "10px",
                      lineHeight: "1.35",
                      color: "var(--color-muted-foreground)",
                      marginTop: "1px",
                    }}
                  >
                    from {f.path}
                  </span>
                  {isOverridden && (
                    <span
                      className="font-mono block text-amber-700 dark:text-amber-400"
                      style={{
                        fontSize: "10px",
                        lineHeight: "1.35",
                        marginTop: "1px",
                      }}
                    >
                      overridden via rollout axis
                    </span>
                  )}
                </span>
              );
            })}
          </span>
        )}
      </span>
    </span>
  );
}

// ---------------------------------------------------------------------------
// Markdown prose (shared by structured sections and the header-less directives block)
// ---------------------------------------------------------------------------

function BriefProse({
  text,
  overrides = [],
}: {
  text: string;
  overrides?: string[];
}) {
  return (
    <div className="brief-prose">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        rehypePlugins={[rehypeHighlightOverrides(overrides)]}
        components={{
          p: ({ children }) => (
            <p className="text-sm leading-[1.66] text-foreground/85 my-1">
              {children}
            </p>
          ),
          ul: ({ children }) => (
            <ul className="my-1 pl-5 list-disc space-y-0.5">{children}</ul>
          ),
          li: ({ children }) => (
            <li className="text-sm leading-[1.66] text-foreground/85">
              {children}
            </li>
          ),
          code: ({ children, className }) => {
            const isBlock = className?.startsWith("language-");
            if (isBlock) return <code className={className}>{children}</code>;
            return (
              <code className="font-mono text-[12.5px] bg-muted px-1 py-0.5 text-foreground">
                {children}
              </code>
            );
          },
          pre: ({ children }) => (
            <pre className="font-mono text-xs bg-muted px-3 py-2 overflow-x-auto my-2 text-foreground border border-border">
              {children}
            </pre>
          ),
          h2: ({ children }) => (
            <h2 className="text-[15px] font-semibold mt-4 mb-1.5 text-foreground">
              {children}
            </h2>
          ),
          h3: ({ children }) => (
            <h3 className="text-sm font-semibold mt-3 mb-1 text-foreground">
              {children}
            </h3>
          ),
          strong: ({ children }) => <strong>{children}</strong>,
        }}
      >
        {text}
      </ReactMarkdown>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Section renderer
// ---------------------------------------------------------------------------

function BriefSection({
  sec,
  config,
}: {
  sec: SectionProv;
  config: RolloutProvenance;
}) {
  const meta = BLOCK_META[sec.block];
  const heading = meta?.heading ?? sec.block;
  const isTask = sec.source === "task";
  const overrides = config.rollout_overrides ?? {};

  if (!sec.rendered.trim()) return null;

  // Header-less block (heading=""): render just the body, no <h2>, no provenance
  // card or task badge -- the directives are emitted verbatim like trailing prose.
  if (meta && meta.heading === "") {
    return (
      <div className="py-1.5 pb-3">
        <BriefProse
          text={sec.rendered}
          overrides={Object.values(overrides).map(String).filter(Boolean)}
        />
      </div>
    );
  }

  return (
    <div
      className={cn(
        "py-1.5 pb-3",
        isTask && "border-l-2 border-foreground pl-[18px]",
      )}
    >
      <h2 className="text-[15px] font-semibold mb-1.5 flex items-center gap-2 text-foreground leading-snug relative">
        {heading}
        <ProvenanceCard sec={sec} config={config} />
        {isTask && (
          <span
            className="font-mono bg-foreground text-background px-[6px] py-[1px] leading-none"
            style={{
              fontSize: "9px",
              fontWeight: 700,
              textTransform: "uppercase",
              letterSpacing: "0.3px",
            }}
          >
            task
          </span>
        )}
      </h2>
      <BriefProse
        text={sec.rendered}
        overrides={Object.values(overrides).map(String).filter(Boolean)}
      />
    </div>
  );
}

// ---------------------------------------------------------------------------
// Brief header bar
// ---------------------------------------------------------------------------

function BriefBar() {
  return (
    <div className="flex items-baseline gap-3 pb-3 mb-1.5 border-b border-border">
      <h1 className="text-[15px] font-bold m-0">Build brief</h1>
      <span className="font-mono text-[11.5px] text-muted-foreground">
        rendered
      </span>
      <div className="ml-auto flex items-center gap-3 text-[10.5px] text-muted-foreground">
        <span className="flex items-center gap-1.5">
          <span className="w-[9px] h-[9px] shrink-0 bg-foreground" />
          task override
        </span>
        <span className="flex items-center gap-1.5">
          <span className="w-[9px] h-[9px] shrink-0 border-[1.5px] border-amber-500" />
          value override
        </span>
        <span>hover the i for provenance</span>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// The brief's provenance: the file the build's brief_provenance output names
// ---------------------------------------------------------------------------

// The brief's provenance at *path*, a path below the rollout's directory; null when the file
// cannot be fetched or is not JSON.
async function fetchBriefProvenance(
  task: string,
  id: string,
  path: string,
): Promise<BriefProvenance | null> {
  try {
    const r = await fetch(fileUrl(task, id, path));
    if (!r.ok) return null;
    return JSON.parse(await r.text()) as BriefProvenance;
  } catch {
    return null;
  }
}

// ---------------------------------------------------------------------------
// Main panel
// ---------------------------------------------------------------------------

export function PromptPanel({ rec }: { rec: RolloutRecord }) {
  const task = rec.task_id;
  const id = rec.rollout_id;

  const { data: config } = useQuery({
    queryKey: ["rollout-config", task, id],
    queryFn: () => fetchConfig(task, id),
  });

  const file = buildFiles(rec).briefProvenance;
  const path = valueOf(file);
  const { data: prov, isLoading: provLoading } = useQuery({
    queryKey: ["brief-provenance", task, id, path],
    queryFn: () => fetchBriefProvenance(task, id, path as string),
    enabled: path != null,
  });

  if (provLoading) {
    return (
      <div className="mx-auto max-w-5xl px-6 py-8">
        <span className="font-mono text-xs text-muted-foreground animate-pulse">
          Loading brief…
        </span>
      </div>
    );
  }

  if (!prov || prov.sections.length === 0) {
    return (
      <div className="mx-auto max-w-5xl px-6 py-8">
        <p className="text-sm text-muted-foreground font-mono">
          No build brief was captured for this rollout.
        </p>
        {!prov && (
          <p className="mt-1 text-xs italic text-muted-foreground font-mono">
            absent: {"absent" in file ? file.absent : ABSENT.briefUnreadable}
          </p>
        )}
      </div>
    );
  }

  const docTitle = task.split("/")[0] ?? task;

  const fallbackConfig: RolloutProvenance = {
    task_id: task,
    spec: {},
    image_digests: {},
    rollout_overrides: {},
    artifact: {},
  };

  return (
    <div className="mx-auto max-w-5xl px-6 pt-6 pb-12">
      <BriefBar />
      <div>
        {/* Document title */}
        <div className="pb-3 pt-1.5">
          <h1 className="text-[21px] font-bold tracking-tight text-foreground mt-1.5 mb-0.5">
            HLS Build Brief
          </h1>
          <div className="font-mono text-[12.5px] text-muted-foreground">
            {docTitle}
          </div>
        </div>
        {/* Sections (including the trailing header-less `directives` block). */}
        {prov.sections.map((sec) => (
          <BriefSection
            key={sec.block}
            sec={sec}
            config={config ?? fallbackConfig}
          />
        ))}
      </div>
    </div>
  );
}
