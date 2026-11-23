import { Check, ChevronDown, ChevronRight, Copy } from "lucide-react";
import { useState } from "react";

import { BackendBadge, GateBadge, StatusBadge } from "~/components/rollout/badges";
import { Fact } from "~/components/rollout/build-fact";
import { PhaseTimeline } from "~/components/rollout/phase-timeline";
import { buildEgress, buildOf, builderModel, recordFacts, submissionMode, valueOf } from "~/lib/build";
import { summarizeSlo } from "~/lib/slo";
import { cn } from "~/lib/utils";
import type { RolloutRecord } from "~/lib/types";

// ---------------------------------------------------------------------------
// Format helpers
// ---------------------------------------------------------------------------

function fmtScore(v: number | null | undefined): string {
  if (v == null) return "—";
  return (v * 100).toFixed(1) + "%";
}

function fmtCost(v: number): string {
  if (v < 0.01) return `$${v.toFixed(4)}`;
  return `$${v.toFixed(2)}`;
}

function fmtTokens(v: number): string {
  if (v >= 1_000_000) return `${(v / 1_000_000).toFixed(1)}M`;
  if (v >= 1_000) return `${(v / 1_000).toFixed(1)}k`;
  return String(v);
}

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

function fmtMs(ms: number | null | undefined): string {
  if (ms == null) return "—";
  return `${Math.round(ms).toLocaleString()} ms`;
}

function fmtLocal(ts: number | null | undefined): string {
  if (ts == null) return "—";
  return new Date(ts * 1000).toLocaleString([], {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function scoreTone(v: number | null | undefined): "default" | "good" | "warn" | "bad" {
  if (v == null) return "default";
  if (v >= 0.8) return "good";
  if (v >= 0.5) return "warn";
  return "bad";
}

function headroomTone(
  p99: number | null | undefined,
  target: number | null | undefined,
): "default" | "good" | "warn" | "bad" {
  if (!p99 || !target) return "default";
  const ratio = target / p99;
  if (ratio >= 2) return "good";
  if (ratio >= 1) return "warn";
  return "bad";
}

// A dataset split size is either a flat number (older specs) or a Count object
// `{value, kind}` (current specs). Extract the numeric size from either shape.
function splitSize(v: unknown): string | number | null {
  if (v == null) return null;
  if (typeof v === "number" || typeof v === "string") return v;
  if (typeof v === "object" && "value" in (v as Record<string, unknown>)) {
    const inner = (v as Record<string, unknown>).value;
    if (typeof inner === "number" || typeof inner === "string") return inner;
  }
  return null;
}

// ---------------------------------------------------------------------------
// Copy button
// ---------------------------------------------------------------------------

function CopyBtn({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);
  function copy(e: React.MouseEvent) {
    e.stopPropagation();
    void navigator.clipboard.writeText(text).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1200);
    });
  }
  return (
    <button
      type="button"
      onClick={copy}
      className="shrink-0 p-0.5 text-muted-foreground/60 transition-colors hover:text-foreground"
      aria-label="Copy"
      title="Copy"
    >
      {copied ? <Check className="h-3 w-3 text-ok" /> : <Copy className="h-3 w-3" />}
    </button>
  );
}

// ---------------------------------------------------------------------------
// Stat tile / Card / Field
// ---------------------------------------------------------------------------

function Stat({
  label,
  value,
  sub,
  tone = "default",
}: {
  label: string;
  value: React.ReactNode;
  sub?: React.ReactNode;
  tone?: "default" | "good" | "warn" | "bad";
}) {
  const valueCls = {
    default: "text-foreground",
    good: "text-ok",
    warn: "text-amber-600 dark:text-amber-400",
    bad: "text-destructive",
  }[tone];
  return (
    <div className="ink-block bg-card px-3 py-2">
      <div className="font-mono text-[10px] uppercase tracking-widest text-muted-foreground">
        {label}
      </div>
      <div className={cn("mt-1 text-base font-semibold tabular-nums leading-snug", valueCls)}>
        {value}
      </div>
      {sub && (
        <div className="mt-0.5 font-mono text-[10px] text-muted-foreground">{sub}</div>
      )}
    </div>
  );
}

function Card({
  title,
  children,
  className,
}: {
  title?: string;
  children: React.ReactNode;
  className?: string;
}) {
  return (
    <section className={cn("ink-panel bg-card", className)}>
      {title && (
        <div className="border-b border-border px-3 py-2 font-mono text-[10px] font-semibold uppercase tracking-widest text-muted-foreground">
          {title}
        </div>
      )}
      <div className="p-3">{children}</div>
    </section>
  );
}

function Field({
  label,
  children,
  copy,
}: {
  label: string;
  children: React.ReactNode;
  copy?: string;
}) {
  return (
    <div className="flex items-baseline justify-between gap-4 py-1 text-sm">
      <span className="shrink-0 font-mono text-xs text-muted-foreground">{label}</span>
      <span className="flex min-w-0 items-center gap-1 text-right">
        <span className="truncate font-mono text-xs">{children}</span>
        {copy && <CopyBtn text={copy} />}
      </span>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Egress posture (network seal) of the deployed artifact: spec.network null => air-gap.
// Mirrors hls.egress.policy NetworkPolicy.domain_posture: an allow-list (mode "allow")
// with no entries permits nothing => air_gap; a deny-list (mode "deny") with no entries
// denies nothing => open; any non-empty list => filter.
// ---------------------------------------------------------------------------

type EgressPosture = "open" | "air_gap" | "filter";

function runtimeEgress(
  policy: Record<string, any> | null | undefined,
): { posture: EgressPosture; domains: string[] } {
  if (policy == null) return { posture: "air_gap", domains: [] };
  const dom = (policy.domains ?? {}) as { mode?: string; entries?: string[] };
  const entries = dom.entries ?? [];
  const posture: EgressPosture =
    entries.length > 0 ? "filter" : (dom.mode ?? "allow") === "allow" ? "air_gap" : "open";
  return { posture, domains: entries };
}

const EGRESS_LABEL: Record<EgressPosture, string> = {
  air_gap: "air-gap",
  open: "open",
  filter: "filter",
};

function EgressBadge({ posture }: { posture: EgressPosture }) {
  const tone =
    posture === "air_gap"
      ? "text-ok"
      : posture === "open"
        ? "text-amber-600 dark:text-amber-400"
        : "text-sky-600 dark:text-sky-400";
  return (
    <span className={cn("font-mono text-xs font-semibold uppercase tracking-wide", tone)}>
      {EGRESS_LABEL[posture]}
    </span>
  );
}

// ---------------------------------------------------------------------------
// Tenants. Per-tenant workload/data_source/verifiers live under
// spec.tenants[]; single-tenant tasks carry them at the top level. This mirrors
// TaskSpec.effective_tenants(): return the explicit list, else synthesize one
// "t0" tenant from the top-level fields so the panel always renders tenants.
// ---------------------------------------------------------------------------

function effectiveTenants(spec: Record<string, any>): Array<Record<string, any>> {
  const ts = (spec.tenants ?? []) as Array<Record<string, any>>;
  if (ts.length > 0) return ts;
  return [
    {
      id: "t0",
      workload: spec.workload ?? null,
      data_source: spec.data_source ?? null,
      verifiers: spec.verifiers ?? null,
    },
  ];
}

function fmtArrival(a: Record<string, any> | null | undefined): string {
  if (!a?.kind) return "—";
  if (a.kind === "Poisson" && a.rate_per_sec != null) return `Poisson, ${a.rate_per_sec}/s`;
  if (a.kind === "Gamma" && a.shape != null) return `Gamma, shape ${a.shape}, scale ${a.scale}`;
  return a.kind;
}

function fmtSlo(s: Record<string, any> | null | undefined): string {
  if (!s?.kind) return "—";
  if (s.target_ms != null) return `p99 ${fmtMs(s.target_ms)}`; // P99Latency
  const parts: string[] = [];
  if (s.ttft_ms != null) parts.push(`TTFT ${fmtMs(s.ttft_ms)}`); // P99StreamingSLO
  if (s.tpot_ms != null) parts.push(`TPOT ${fmtMs(s.tpot_ms)}`);
  return parts.length > 0 ? parts.join(", ") : s.kind;
}

// ---------------------------------------------------------------------------
// Error / invalid-artifact banner
// ---------------------------------------------------------------------------

function ErrorBanner({
  error,
}: {
  error: { type: string; message: string; traceback: string };
}) {
  const [open, setOpen] = useState(false);
  return (
    <div className="border-[1.5px] border-destructive/70 bg-destructive/10 shadow-[var(--shadow-md)] p-3">
      <div className="font-mono text-sm font-semibold text-destructive">
        {error.type}: {error.message}
      </div>
      {error.traceback && (
        <div className="mt-2">
          <button
            type="button"
            onClick={() => setOpen((o) => !o)}
            className="flex items-center gap-1 font-mono text-xs text-destructive/70 transition-colors hover:text-destructive"
          >
            {open ? <ChevronDown className="h-3 w-3" /> : <ChevronRight className="h-3 w-3" />}
            {open ? "Hide" : "Show"} traceback
          </button>
          {open && (
            <pre className="mt-2 max-h-64 overflow-auto whitespace-pre-wrap border border-destructive/20 bg-destructive/5 p-2 font-mono text-xs text-destructive/80">
              {error.traceback}
            </pre>
          )}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main panel — reads rec.config (the embedded provenance), the build record and the
// arm and budget the record's conditions name. Its substrate reads are tolerant of
// both the nested shape and the older flat one (backend / *_timeout_sec at the config
// root).
// ---------------------------------------------------------------------------

export function OverviewPanel({ rec }: { rec: RolloutRecord }) {
  const cfg = (rec.config ?? {}) as Record<string, any>;
  const spec = (cfg.spec ?? {}) as Record<string, any>;

  const model = (spec.model ?? {}) as Record<string, any>;
  const hardware = (spec.hardware ?? {}) as Record<string, any>;
  const accel = (hardware.accelerator ?? {}) as Record<string, any>;
  const contract = (spec.contract ?? {}) as Record<string, any>;
  const proto = (contract.protocol ?? {}) as Record<string, any>;
  // Per-tenant workload/data_source/verifiers live under spec.tenants[];
  // single-tenant tasks carry them at the top level. effectiveTenants() unifies
  // both, so the panel always renders tenants. The oracle reference is shared, so
  // search both the top-level verifiers and every tenant's verifiers.
  const tenants = effectiveTenants(spec);
  const oracleRef =
    [
      ...((spec.verifiers ?? []) as any[]),
      ...tenants.flatMap((t) => (t.verifiers ?? []) as any[]),
    ]
      .map((v) => v?.reference)
      .find((r) => r) ?? null;

  // Substrate facts. Backend config is namespaced under substrate.{docker,modal,
  // apptainer}; fall back to the flat keys (modal_region/host_devices) for a
  // record that holds them. Timeouts + backend stay backend-agnostic at the top level.
  const sub = (cfg.substrate ?? {}) as Record<string, any>;
  const backend = sub.backend ?? cfg.backend ?? null;
  const modalRegion = sub.modal?.region ?? sub.modal_region ?? cfg.modal_region ?? null;
  const dockerDevices = sub.docker?.host_devices ?? sub.host_devices ?? null;
  const appt = (sub.apptainer ?? {}) as Record<string, any>;
  const apptSlurm = (appt.slurm ?? {}) as Record<string, any>;
  const apptPartition = apptSlurm.artifact?.partition ?? apptSlurm.default?.partition ?? null;
  const requestTimeout = sub.request_timeout_sec ?? cfg.request_timeout_sec ?? null;
  const startupTimeout = sub.startup_timeout_sec ?? cfg.startup_timeout_sec ?? null;

  // The build's arm and budget, as the record's conditions name them.
  const implementer = builderModel(rec);
  const budget = rec.conditions?.budget ?? null;

  // Egress seal posture of the deployed artifact at runtime. See runtimeEgress.
  const runtime = runtimeEgress(spec.network);

  const imageDigests = (cfg.image_digests ?? {}) as Record<string, string>;

  // Terminal eval / scoring.
  const evals = rec.evals ?? [];
  const terminal = evals.find((e) => e.eval_id === "terminal") ?? evals[evals.length - 1];
  const score = (terminal?.score ?? {}) as Record<string, any>;
  const scoreVal = typeof score.value === "number" ? score.value : null;
  const slo = summarizeSlo(score);
  const gatePassed =
    (terminal?.verdict_summary?.gate_passed as boolean | null | undefined) ??
    (score.gate_passed as boolean | null | undefined);
  const headroom = slo?.p99Ms && slo?.targetMs ? slo.targetMs / slo.p99Ms : null;

  // Per-request verdict diagnostics. Two shapes: the per-verifier nesting
  // (`{ "0:GreedyMatch": { req: {...} } }`) for valid runs, and a FLAT
  // `{ message, returncode, log }` for invalid_artifact runs.
  const diagRoot = (terminal?.verdict_summary?.diagnostics ?? {}) as Record<string, any>;
  const invalidDiag = diagRoot as { message?: unknown; returncode?: unknown; log?: unknown };
  const isInvalidDiag =
    rec.status === "invalid_artifact" ||
    ("message" in diagRoot && typeof diagRoot.message !== "object");
  const verifierDiags = isInvalidDiag ? {} : diagRoot;
  // A grading's rebuild fails before any eval exists, so there are no verdict
  // diagnostics to read -- the record's own error is the whole account there.
  const invalidMessage =
    typeof invalidDiag.message === "string" ? invalidDiag.message : rec.error?.message;
  const invalidLog =
    typeof invalidDiag.log === "string" && invalidDiag.log ? invalidDiag.log : rec.error?.log;

  const overrides = (cfg.rollout_overrides ?? {}) as Record<string, unknown>;
  // The build's facts, each absent with the reason when the record has no build to read.
  const built = buildOf(rec);
  const build = recordFacts(rec);

  return (
    <div className="w-full space-y-4 px-6 py-4">
      {rec.status === "error" && rec.error && <ErrorBanner error={rec.error} />}

      {rec.status === "invalid_artifact" && (
        <div className="border-[1.5px] border-amber-500/70 bg-amber-500/10 shadow-[var(--shadow-md)] p-3">
          <div className="font-mono text-sm font-semibold text-amber-600 dark:text-amber-400">
            Invalid artifact — submission scored 0
          </div>
          {invalidMessage && (
            <div className="mt-1 text-sm text-amber-700 dark:text-amber-300">
              {invalidMessage}
            </div>
          )}
          {invalidLog && (
            <pre className="mt-2 max-h-64 overflow-auto whitespace-pre-wrap border border-amber-500/20 bg-amber-500/5 p-2 font-mono text-xs text-muted-foreground">
              {invalidLog}
            </pre>
          )}
        </div>
      )}

      {/* Stat strip */}
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-6">
        <Stat label="Status" value={<StatusBadge status={rec.status} />} />
        <Stat
          label="Score"
          value={fmtScore(scoreVal)}
          sub={score.sol != null ? `SoL ${score.sol} ${score.unit ?? ""}` : undefined}
          tone={scoreTone(scoreVal)}
        />
        <Stat label="Gate" value={<GateBadge passed={gatePassed} />} />
        <Stat
          label={slo?.label ?? "p99 latency"}
          value={fmtMs(slo?.p99Ms)}
          sub={
            headroom != null
              ? headroom >= 1
                ? `${headroom >= 10 ? Math.round(headroom) : headroom.toFixed(1)}x under target`
                : `${(1 / headroom).toFixed(1)}x over target`
              : slo?.n != null
                ? `n=${slo.n}`
                : undefined
          }
          tone={headroomTone(slo?.p99Ms, slo?.targetMs)}
        />
        <Stat
          label="Cost"
          value={<Fact fact={build.cost} format={fmtCost} />}
          sub={
            "value" in built ? (
              <>
                <Fact fact={build.inputTokens} format={fmtTokens} /> in /{" "}
                <Fact fact={build.outputTokens} format={fmtTokens} /> out
              </>
            ) : undefined
          }
        />
        <Stat
          label="Duration"
          value={fmtDuration(rec.created_at, rec.finished_at)}
          sub={fmtLocal(rec.created_at)}
        />
      </div>

      {/* Phase timeline */}
      <PhaseTimeline build={valueOf(built)} evals={rec.evals} />

      {/* Two-column dense config */}
      <div className="grid gap-4 lg:grid-cols-2">
        {/* LEFT — run configuration, split into sub-section cards */}
        <div className="space-y-3">
          {/* Run */}
          <Card title="Run">
            <Field label="Task" copy={rec.task_id}>{rec.task_id}</Field>
            <Field label="Rollout" copy={rec.rollout_id}>{rec.rollout_id}</Field>
            <Field label="Backend"><BackendBadge backend={backend} /></Field>
            <Field label="Implementer">
              <Fact fact={implementer} inline format={String} />
            </Field>
            {budget && <Field label="Max turns">{budget.max_turns ?? "none"}</Field>}
          </Card>

          {/* Model */}
          <Card title="Model">
            <Field label="Model">{model.base_checkpoint ?? "—"}</Field>
            {model.fingerprint && <Field label="Fingerprint" copy={model.fingerprint}>{model.fingerprint}</Field>}
            {model.chat_format && <Field label="Chat format">{model.chat_format}</Field>}
            <Field label="Oracle model">{oracleRef?.base_checkpoint ?? model.base_checkpoint ?? "—"}</Field>
            <Field label="Oracle decode">{oracleRef?.decode?.kind ?? "—"}</Field>
          </Card>

          {/* Hardware -- accelerator + backend-specific placement. */}
          <Card title="Hardware">
            <Field label="Accelerator">{accel.kind ?? "—"}</Field>
            <Field label="Accelerator count">{accel.count ?? "—"}</Field>
            {backend === "modal" && modalRegion && (
              <Field label="Modal region">{modalRegion}</Field>
            )}
            {backend === "docker" && dockerDevices != null && (
              <Field label="GPU devices">
                {Array.isArray(dockerDevices) ? dockerDevices.join(", ") : String(dockerDevices)}
              </Field>
            )}
            {backend === "apptainer" && (
              <>
                {apptPartition && <Field label="Slurm partition">{apptPartition}</Field>}
                {appt.account && <Field label="Slurm account">{appt.account}</Field>}
                {appt.scratch_dir && (
                  <Field label="Scratch dir" copy={appt.scratch_dir}>{appt.scratch_dir}</Field>
                )}
              </>
            )}
          </Card>

          {/* Network -- egress seal: build sandbox vs deployed artifact */}
          <Card title="Network">
            <Field label="Build egress">
              <Fact fact={buildEgress(rec)} inline format={String} />
            </Field>
            <Field label="Runtime egress">
              <EgressBadge posture={runtime.posture} />
              {runtime.posture === "filter" && runtime.domains.length > 0 && (
                <span className="ml-1 text-muted-foreground">{runtime.domains.join(", ")}</span>
              )}
            </Field>
            {backend === "apptainer" && appt.sealing_mode && (
              // The apptainer build sandbox's egress is sealed by a rootless
              // transparent gateway -- "enforced" seals it, "open" leaves it direct.
              <Field label="Build seal">
                <span
                  className={cn(
                    "font-mono text-xs font-semibold uppercase tracking-wide",
                    appt.sealing_mode === "enforced"
                      ? "text-ok"
                      : "text-amber-600 dark:text-amber-400",
                  )}
                >
                  {appt.sealing_mode}
                </span>
              </Field>
            )}
          </Card>

          {/* Workload -- per-tenant arrival + SLO; timeouts are run-level. */}
          <Card title={tenants.length > 1 ? `Workload (${tenants.length} tenants)` : "Workload"}>
            {tenants.length > 1 ? (
              tenants.map((t) => (
                <div key={t.id} className="border-b border-border/40 py-1 last:border-0">
                  <div className="font-mono text-[11px] font-semibold">{t.id}</div>
                  <Field label="Arrival">{fmtArrival(t.workload?.arrival)}</Field>
                  <Field label="SLO">{fmtSlo(t.workload?.slo)}</Field>
                </div>
              ))
            ) : (
              <>
                <Field label="Arrival">{fmtArrival(tenants[0]?.workload?.arrival)}</Field>
                <Field label="SLO">{fmtSlo(tenants[0]?.workload?.slo)}</Field>
                <Field label="Output cap (max tokens)">
                  {tenants[0]?.workload?.output_length_cap?.max_t ?? "uncapped"}
                </Field>
              </>
            )}
            <Field label="Request timeout">
              {requestTimeout != null ? `${requestTimeout}s` : "—"}
            </Field>
            <Field label="Startup timeout">
              {startupTimeout != null ? `${startupTimeout}s` : "—"}
            </Field>
          </Card>

          {/* Contract */}
          <Card title="Contract">
            <Field label="Wire protocol">{proto.kind ?? "—"}</Field>
            <Field label="Transport">{contract.transport ?? "—"}</Field>
            <Field label="Return token IDs">
              {proto.return_token_ids === true
                ? "yes"
                : proto.return_token_ids === false
                  ? "no"
                  : "—"}
            </Field>
            <Field label="Submission">
              <Fact fact={submissionMode(rec)} inline format={String} />
            </Field>
          </Card>

          {/* Dataset -- per-tenant data source + split. */}
          <Card title="Dataset">
            {tenants.length > 1 ? (
              tenants.map((t) => (
                <div key={t.id} className="border-b border-border/40 py-1 last:border-0">
                  <div className="font-mono text-[11px] font-semibold">{t.id}</div>
                  <Field label="Dataset">{t.data_source?.dataset?.path ?? "—"}</Field>
                  <Field label="Dev / test">
                    {`${splitSize(t.data_source?.split?.dev) ?? "full"} / ${splitSize(t.data_source?.split?.test) ?? "full"}`}
                  </Field>
                </div>
              ))
            ) : (
              <>
                <Field label="Dataset">{tenants[0]?.data_source?.dataset?.path ?? "—"}</Field>
                <Field label="Dev size">
                  {splitSize(tenants[0]?.data_source?.split?.dev) != null
                    ? String(splitSize(tenants[0]?.data_source?.split?.dev))
                    : "full"}
                </Field>
                <Field label="Test size">
                  {splitSize(tenants[0]?.data_source?.split?.test) != null
                    ? String(splitSize(tenants[0]?.data_source?.split?.test))
                    : "full"}
                </Field>
              </>
            )}
            <Field label="Seed">{spec.seed ?? "—"}</Field>
          </Card>

          {/* Provenance */}
          <Card title="Provenance">
            {Object.keys(imageDigests).length > 0 && (
              <Field
                label="Image digests"
                copy={Object.entries(imageDigests).map(([r, d]) => `${r}:${d}`).join("\n")}
              >
                {Object.entries(imageDigests)
                  .map(([r, d]) => `${r}:${String(d).slice(0, 12)}`)
                  .join(", ")}
              </Field>
            )}
            <Field
              label="hls / harbor"
              copy={`hls ${cfg.hls_commit ?? "?"} / harbor ${cfg.harbor_commit ?? "?"}`}
            >
              {cfg.hls_commit ?? "?"} / {cfg.harbor_commit ?? "?"}
            </Field>
          </Card>
        </div>

        {/* RIGHT — overrides + build + verdict diagnostics */}
        <div className="space-y-4">
          <Card title="Overrides">
            {Object.keys(overrides).length === 0 ? (
              <p className="font-mono text-xs text-muted-foreground">
                No rollout-axis overrides (task defaults).
              </p>
            ) : (
              <div className="space-y-1">
                {Object.entries(overrides).map(([k, v]) => (
                  <div
                    key={k}
                    className="flex items-baseline justify-between gap-3 font-mono text-xs"
                  >
                    <span className="text-muted-foreground">{k}</span>
                    <span className="break-all text-right">{String(v)}</span>
                  </div>
                ))}
              </div>
            )}
          </Card>

          {"absent" in built ? (
            <Card title="Build">
              <Field label="Build">
                <Fact fact={{ absent: built.absent }} inline format={String} />
              </Field>
            </Card>
          ) : (
            <Card title="Build">
              <Field
                label="Artifact image"
                copy={"value" in build.image ? build.image.value.key : undefined}
              >
                <Fact
                  fact={build.image}
                  inline
                  format={(img) => <span title={`sha256 ${img.sha256}`}>{img.key}</span>}
                />
              </Field>
              <Field label="Steps">
                <Fact fact={build.steps} inline format={String} />
              </Field>
              <Field label="Tokens (in/out)">
                <Fact fact={build.inputTokens} inline format={fmtTokens} /> /{" "}
                <Fact fact={build.outputTokens} inline format={fmtTokens} />
              </Field>
              <Field label="Cost">
                <Fact fact={build.cost} inline format={fmtCost} />
              </Field>
              <Field label="Build time">
                <Fact fact={build.span} inline format={(sp) => fmtDuration(sp.start, sp.end)} />
              </Field>
              <Field label="Source commit">
                <Fact fact={build.commit} inline format={(c) => c.slice(0, 10)} />
              </Field>
              <Field label="Source files">
                <Fact fact={build.sourceFiles} inline format={String} />
              </Field>
            </Card>
          )}

          {Object.keys(verifierDiags).length > 0 && (
            <Card title="Verdict diagnostics">
              <div className="space-y-3">
                {Object.entries(verifierDiags).map(([verifierName, perRequest]) => (
                  <div key={verifierName}>
                    <div className="mb-1 font-mono text-xs font-medium">{verifierName}</div>
                    <div className="space-y-1">
                      {typeof perRequest === "object" && perRequest != null ? (
                        (() => {
                          const entries = Object.entries(perRequest as Record<string, any>);
                          const CAP = 25;
                          return (
                            <>
                              {entries.slice(0, CAP).map(([reqId, diag]) => (
                                <DiagRow key={reqId} reqId={reqId} diag={diag} />
                              ))}
                              {entries.length > CAP && (
                                <div className="pt-0.5 font-mono text-[11px] text-muted-foreground/70">
                                  + {(entries.length - CAP).toLocaleString()} more — see the Eval tab
                                </div>
                              )}
                            </>
                          );
                        })()
                      ) : (
                        <div className="font-mono text-xs text-muted-foreground">
                          {String(perRequest)}
                        </div>
                      )}
                    </div>
                  </div>
                ))}
              </div>
              {slo && (
                <div className="mt-3 border-t border-border pt-3">
                  <div className="mb-1 font-mono text-xs font-medium">
                    SLO (latency){slo.tenantId ? ` — ${slo.tenantId}` : ""}
                  </div>
                  <div className="grid grid-cols-3 gap-x-6 gap-y-1 font-mono text-xs">
                    {slo.metric === "none" ? (
                      <DiagStat label="requests" value={slo.n ?? "—"} />
                    ) : (
                      <>
                        <DiagStat label={slo.label} value={fmtMs(slo.p99Ms)} />
                        <DiagStat
                          label={slo.metric === "legacy" ? "target" : "TTFT target"}
                          value={fmtMs(slo.targetMs)}
                        />
                        {slo.tpotP99Ms != null && (
                          <DiagStat label="p99 TPOT" value={fmtMs(slo.tpotP99Ms)} />
                        )}
                        {slo.tpotTargetMs != null && (
                          <DiagStat label="TPOT target" value={fmtMs(slo.tpotTargetMs)} />
                        )}
                        <DiagStat label="requests" value={slo.n ?? "—"} />
                      </>
                    )}
                  </div>
                </div>
              )}
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}

// One verdict diagnostics request row: renders the readable fields
// (worst_gap / n_tokens / first_above_cap_index) instead of [object Object].
function DiagRow({ reqId, diag }: { reqId: string; diag: unknown }) {
  if (typeof diag !== "object" || diag == null) {
    return (
      <div className="flex items-baseline justify-between gap-2 font-mono text-xs">
        <span className="truncate text-muted-foreground">{reqId}</span>
        <span className="tabular-nums">{String(diag)}</span>
      </div>
    );
  }
  const d = diag as Record<string, unknown>;
  const parts: string[] = [];
  if (d.worst_gap != null) parts.push(`gap ${Number(d.worst_gap)}`);
  if (d.n_tokens != null) parts.push(`${d.n_tokens} tok`);
  if (d.first_above_cap_index != null) parts.push(`first >cap @${d.first_above_cap_index}`);
  const summary = parts.length
    ? parts.join(", ")
    : Object.entries(d)
        .map(([k, v]) => `${k}=${v == null ? "—" : String(v)}`)
        .join(", ");
  return (
    <div className="flex items-baseline justify-between gap-2 font-mono text-xs">
      <span className="truncate text-muted-foreground" title={reqId}>{reqId}</span>
      <span className="shrink-0 tabular-nums">{summary}</span>
    </div>
  );
}

function DiagStat({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="flex justify-between gap-2">
      <span className="text-muted-foreground">{label}</span>
      <span className="tabular-nums">{value}</span>
    </div>
  );
}
