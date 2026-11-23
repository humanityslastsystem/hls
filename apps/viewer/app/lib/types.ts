// TypeScript types for HLS viewer API — mirrors src/hls/obs/record.py + src/hls/viewer/models.py

// BuildUsage: what harbor read of the agent's spend (hls.stages.agent_build.Usage)
export interface BuildUsage {
  n_input_tokens: number | null;
  n_cache_tokens: number | null;
  n_output_tokens: number | null;
  cost_usd: number | null; // harbor's estimate
}

// PackageRecord: package.json as the package stage writes it (hls.package.result.PackageResult)
export interface PackageRecord {
  task: string;
  status: "built" | "invalid_artifact";
  bundle_sha256: string | null;
  commit: string | null; // the commit of main that was built
  image: string | null; // the submission image's store key; set exactly when built
  sif_sha256: string | null; // set exactly when built
  wheels: string[];
  reason: string | null; // set exactly when invalid
  returncode: number | null;
  log_tail: string | null;
}

// RolloutBuild: a rollout's build as the stage pipeline records it (hls.obs.record.RolloutBuild).
// The views read it through lib/build.ts.
export interface RolloutBuild {
  build: Record<string, unknown>; // build.json as the agent_build stage wrote it
  outputs: Record<string, string>; // each agent_build output's store key, by name
  package: PackageRecord | null; // null when the package stage did not publish
  started_at: number | null;
  ended_at: number | null;
  usage: BuildUsage | null;
  upload_sha256: string | null;
}

// RolloutConditions: how the rollout ran (hls.obs.conditions.HarnessConditions), the fields the
// views read.
export interface RolloutConditions {
  // the arm whose agent built the submission, by its model and its harness ("claude-code", "codex")
  arm: { record: { model: string; harness: string } } | null;
  budget: { max_turns: number | null } | null; // the budgets that build ran under
}

// EvalEpisode
export interface EvalEpisode {
  eval_id: string;
  triggered_by: string;
  checkpoint: string;
  started_at: number;
  finished_at?: number | null;
  trace_path: string;
  verdict_summary: Record<string, any>;
  score: Record<string, any>;
  container_logs: Record<string, string>;
}

// RolloutProvenance
export interface RolloutProvenance {
  task_id: string;
  spec: Record<string, any>;
  image_digests: Record<string, string>;
  rollout_overrides: Record<string, any>;
  // Only in published records of the old commands; the stage pipeline records no substrate.
  substrate?: Record<string, any>;
  artifact: Record<string, any>;
  hls_commit?: string | null;
  harbor_commit?: string | null;
}

// RolloutRecord
// Every status the viewer serves a record with (tests/viewer/test_store_layout.py holds it to
// them): a sealed record's own (hls.obs.record.RolloutStatus), from "running" to
// "invalid_artifact", a scored-zero run whose submission was invalid or did not build ("built" is a
// build-only rollout whose submission built); a published run the old command never sealed
// ("unsealed", the scanner's UNSEALED); how a store instance that ended without a record ended (its
// plan's outcome, the scanner's INSTANCE_ENDINGS); and a store instance that has not ended and no
// orchestrator holds ("stalled", the scanner's STALLED).
export type RolloutStatus =
  | "running"
  | "ok"
  | "built"
  | "error"
  | "invalid_artifact"
  | "unsealed"
  | "succeeded"
  | "failed"
  | "void"
  | "cancelled"
  | "stalled";

export interface RolloutRecord {
  rollout_id: string;
  task_id: string;
  created_at: number;
  finished_at?: number | null;
  status: RolloutStatus;
  config: RolloutProvenance;
  // null for a published record, whose build is in the old recorder's shape, and for a store record
  // whose build is none or not a RolloutBuild (scanner)
  build?: RolloutBuild | null;
  // Only in a store record, which the scanner adds: the field locations at which its build fails to
  // validate as a RolloutBuild (`build.usage`), never a value; null when it validates or is none.
  build_invalid?: string[] | null;
  conditions?: RolloutConditions | null; // absent from published records
  evals: EvalEpisode[];
  // Only in a store instance's record, which the scanner adds: the instance's own store key. The
  // keys of its build's outputs lie below it, and the files endpoint takes them without it. A record
  // without it is a published one.
  instance?: string;
  // Only in published records of the old commands, as the only account of a failure; log is
  // a build failure's installer output.
  error?: {
    type: string;
    message: string;
    traceback: string;
    log?: string | null;
  } | null;
}

// RolloutSummary (list page) — mirrors hls.viewer.models.RolloutSummary
// created_at / finished_at are Unix timestamps (float seconds) from the Python side
export interface RolloutSummary {
  task_id: string;
  rollout_id: string;
  created_at?: number | null;
  finished_at?: number | null;
  status?: string | null;
  score?: number | null;
  gate_passed?: boolean | null;
  cost_usd?: number | null;
  image_digests?: Record<string, string>;
  agent_implementer_model?: string | null;
  model_fingerprint?: string | null;
  accelerator?: string | null;
  backend?: string | null;
  phase?: string | null;
  // allow extra enrichment fields
  [key: string]: unknown;
}

// ---- ATIF trajectory (build agent) ----

// Multimodal content (ATIF v1.6): a message is either a plain string or an
// array of typed parts. Observation content additionally allows null.
export interface ImageSource {
  media_type: string;
  path: string;
}

export interface ContentPart {
  type: "text" | "image";
  text?: string;
  source?: ImageSource;
}

export type MessageContent = string | ContentPart[];
export type ObservationContent = string | ContentPart[] | null;

export interface ToolCall {
  tool_call_id: string;
  function_name: string;
  arguments: Record<string, unknown>;
}

export interface ObservationResult {
  source_call_id: string | null;
  content: ObservationContent;
}

export interface Observation {
  results: ObservationResult[];
}

export interface StepMetrics {
  prompt_tokens: number | null;
  completion_tokens: number | null;
  cached_tokens: number | null;
  cost_usd?: number | null;
  extra?: Record<string, unknown>;
}

export interface Step {
  step_id: number;
  timestamp: string | null;
  source: "system" | "user" | "agent" | string;
  model_name: string | null;
  message: MessageContent;
  reasoning_content: string | null;
  tool_calls: ToolCall[] | null;
  observation: Observation | null;
  metrics: StepMetrics | null;
  extra?: Record<string, unknown>;
}

export interface TrajectoryAgent {
  name: string;
  version: string;
  model_name: string | null;
}

export interface FinalMetrics {
  total_prompt_tokens?: number | null;
  total_completion_tokens?: number | null;
  total_cached_tokens?: number | null;
  total_cost_usd?: number | null;
  total_steps?: number | null;
}

export interface Trajectory {
  schema_version: string;
  session_id: string;
  agent?: TrajectoryAgent | null;
  steps: Step[];
  final_metrics?: FinalMetrics | null;
}

// FileInfo
export interface FileInfo {
  path: string;
  name: string;
  is_dir: boolean;
  size?: number | null;
}

// PaginatedResponse
export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  page: number;
  page_size: number;
  total_pages: number;
}

// TraceSummary
export interface TraceSummary {
  total: number;
  token_match?: number | null;
  token_match_total?: number | null;
  correct?: number | null;
  graded?: number | null;
  p99_ms?: number | null;
  verifier_kind?: string | null;    // "token_level" | "task_relative" | "task_absolute" | "generic"
  gate_passed?: boolean | null;
  artifact_value?: number | null;
  reference_value?: number | null;
  min_value?: number | null;
  tol?: number | null;
  verifiers?: VerifierSummary[];
}

// VerifierSummary — one per verifier in a multi-verifier eval. A discriminated
// union on `kind`: each verifier kind carries only the diagnostics it emits
// (mirrors the per-kind models in hls/viewer/models.py).
interface VerifierSummaryBase {
  name: string; // full class name, e.g. "TokenFaithfulnessVerifier"
  gate_passed?: boolean | null;
}

export interface TokenLevelSummary extends VerifierSummaryBase {
  kind: "token_level"; // TokenFaithfulnessVerifier two-threshold diagnostics
  flip_rate?: number | null;   // fraction of teacher-forced tokens with gap > 0
  rate_ratio?: number | null;  // flip_rate / (rate_multiplier * calibration.q_max)
  // Gap levels (nats) whose flip-count ceiling the trace exceeded. [] = curve
  // checked and clean, null = the record carries no curve rows.
  curve_fired_levels?: number[] | null;
  n_tokens?: number | null;    // total teacher-forced tokens pooled across the trace
}

export interface TaskRelativeSummary extends VerifierSummaryBase {
  kind: "task_relative"; // ReferenceRelativeVerifier artifact-vs-reference diagnostics
  artifact_value?: number | null;
  reference_value?: number | null;
  tol?: number | null;
}

export interface TaskAbsoluteSummary extends VerifierSummaryBase {
  kind: "task_absolute"; // MetricVerifier absolute-threshold diagnostics
  artifact_value?: number | null;
  min_value?: number | null;
}

export interface GenericVerifierSummary extends VerifierSummaryBase {
  kind: "generic"; // no dedicated summary shape: identity and outcome only
}

export type VerifierSummary =
  | TokenLevelSummary
  | TaskRelativeSummary
  | TaskAbsoluteSummary
  | GenericVerifierSummary;

// TraceRow — per-turn row returned by trace_page
export interface TraceRow {
  turn_index: number | null;
  request_id: string;
  session_id: string | null;
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
  predicted?: string | null;          // artifact metric answer
  oracle_predicted?: string | null;   // oracle metric answer
  oracle_correct?: boolean | null;    // oracle's answer == expected
  verifier_cells?: { name: string; passed: boolean | null; detail: string }[];
}

// TracePage
export interface TracePage {
  offset: number;
  limit: number;
  total: number;
  rows: TraceRow[];
}

// BriefProvenance
export interface BriefProvenance {
  sections: {
    block: string;
    source: "base" | "task";
    rendered: string;
  }[];
}

export interface TimelineSLO {
  kind: string; // "streaming" | "latency"
  ttft_ms?: number | null;
  tpot_ms?: number | null;
  target_ms?: number | null;
}

export interface TimelineTenant {
  id: string;
  name?: string | null;
  req_n: number;
  slo?: TimelineSLO | null;
}

export interface TimelineRequest {
  ti: number; // index into TimelinePayload.tenants
  sid: string | null; // session id
  turn: number;
  send: number; // seconds from t0
  recv: number | null;
  plan: number | null;
  ttft: number | null;
  // Last streamed packet's timestamp (seconds from t0) -- the same instant the
  // server's StreamingLatencySLO._meets uses for the end of its TPOT window.
  // Distinct from `recv`, which is stamped after the SSE stream's trailing
  // `data: [DONE]` and so runs strictly >= the last packet. Null when no
  // per-packet timing was captured (e.g. return_token_ids=False) or on older
  // trace data predating this field; callers should fall back to `recv`.
  tlast?: number | null;
  ntok: number | null;
  err: boolean;
}

export interface TimelinePayload {
  t0: number;
  duration: number;
  tenants: TimelineTenant[];
  requests: TimelineRequest[];
}
