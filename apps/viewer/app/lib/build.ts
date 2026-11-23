// What the rollout views show of a rollout's build, read from its record as the stage pipeline
// writes it (RolloutBuild, hls.obs.record, and the arm its conditions name). A fact the record
// does not hold is absent, with the reason, and never filled in from another field.
//
// A published record, which the old recorder wrote, names no instance (the scanner names the
// instance of every store record it serves). Its build, the producer that names the arm that built
// it, and its evals' role logs are in the old recorder's shape, which the views do not read: each
// fact of them is absent for that reason.
//
// The files of a build are named by the store keys of the record's outputs. The viewer's files
// endpoint serves a store instance's file by its path below the instance's directory, which is the
// key without the instance's own key (the scanner adds that to the record as `instance`).

import type { RolloutBuild, RolloutRecord } from "./types";

// The names below are the stage pipeline's: tests/viewer/test_store_layout.py holds each to the
// writer's.

/** The directory of the transcript output that holds the copy of the agent's logs, beside its
 * INDEX.json and harbor's trajectory (hls.agent.transcript.COPY_DIR). */
export const TRANSCRIPT_LOGS = "logs";

/** What follows the harness's name in the name of the file harbor's agent writes its stream to in
 * the agent's logs (claude-code.txt, codex.txt). */
export const AGENT_LOG_SUFFIX = ".txt";

/** What the name of each event of the agent_build stage's build begins with. */
export const BUILD_EVENT_PREFIX = "build.";

/** The event the stage logs when the agent's clock starts. */
export const BUILD_STARTED = "build.started";

/** The event the stage logs when the build is done and the agent's logs are copied. */
export const BUILD_FINISHED = "build.finished";

/** The field of that event which holds the build's cost in USD. */
export const BUILD_COST_FIELD = "cost_usd";

/** One fact of a build record: its value, or the reason the record holds none. */
export type BuildFact<T> = { value: T } | { absent: string };

/** The submission image the package stage built. */
export interface BuiltImage {
  /** Its store key. */
  key: string;
  /** The sha256 of its file. */
  sha256: string;
}

/** When the agent's clock started and when its part ended (POSIX seconds, the runner's). */
export interface BuildSpan {
  start: number;
  end: number;
}

/** Every build fact the views show. */
export interface BuildFacts {
  /** The submission image, from the package record. */
  image: BuildFact<BuiltImage>;
  /** The agent's step count, which the record does not keep. */
  steps: BuildFact<number>;
  /** Input tokens, from harbor's usage. */
  inputTokens: BuildFact<number>;
  /** Output tokens, from harbor's usage. */
  outputTokens: BuildFact<number>;
  /** Harbor's cost estimate in USD, from its usage. */
  cost: BuildFact<number>;
  /** The agent's part, from the build's start and end. */
  span: BuildFact<BuildSpan>;
  /** The commit of main the package built, from the package record. */
  commit: BuildFact<string>;
  /** The submission's file count and size, which the record does not keep. */
  sourceFiles: BuildFact<number>;
}

/** Why each fact can be absent. */
export const ABSENT = {
  noPackage: "the package stage published no package.json",
  notBuilt: "the submission did not build, so the package stage made no image",
  steps: "the stage pipeline's build record keeps no step count",
  noUsage: "harbor read no usage from the agent's logs",
  noInputTokens: "harbor read no input token count",
  noOutputTokens: "harbor read no output token count",
  noCost: "harbor made no cost estimate",
  noSpan: "build.json records no start or end time",
  noCommit: "the source bundle has no main branch, or the build left no bundle",
  sourceFiles: "the stage pipeline's build record lists no source files: the source is a git bundle",
  noConditions: "the record holds no harness conditions, so it names no arm",
  noArm: "no agent built the artifact",
  noSubmissionMode: "the stage pipeline records no submission mode",
  noBuildEgress: "the rollout record does not state the build sandbox's egress policy",
  noBuild: "the record holds no build record",
  publishedBuild: "a published record's build is in a shape this viewer does not read",
  publishedProducer:
    "a published record names any arm that built it, its submission mode and its build's egress in its " +
    "producer, which this viewer does not read",
  publishedLogs: "a published record's evals name their role logs in a shape this viewer does not read",
  outsideInstance: "the output's key lies outside the rollout's instance",
  noSubmission: "the agent made no submission, so the build left no source bundle",
  notListed: "the rollout's files do not list it: it is gone, or a finalize scan withheld it",
  briefUnreadable: "the brief's provenance could not be read from the file the build recorded",
} as const;

/** Why a build record names no output *name*. */
export function absentOutput(name: string): string {
  return `the build recorded no ${name} output`;
}

/** Why a store record shows no build when its build does not validate as the stage pipeline's:
 * *where*, the field locations at which it fails, as the scanner names them (never a value). */
export function invalidBuild(where: readonly string[]): string {
  return `the record's build does not validate as the stage pipeline's build record at ${where.join(", ")}`;
}

/** Why the eval's record names no log of role *role*. */
export function absentRoleLog(role: string): string {
  return `the eval's stages left no log of a role named ${role}`;
}

function fact<T>(value: T | null | undefined, absent: string): BuildFact<T> {
  return value == null ? { absent } : { value };
}

/** Whether *rec* is a published record: the scanner names the instance of every store record it
 * serves, and of no other. */
function isPublished(rec: RolloutRecord): boolean {
  return rec.instance == null;
}

/** *rec*'s build as the stage pipeline records it; absent, with the reason, for a published record,
 * for a store record whose build the scanner found not to validate, and for one that holds none. */
export function buildOf(rec: RolloutRecord): BuildFact<RolloutBuild> {
  if (isPublished(rec)) return { absent: ABSENT.publishedBuild };
  if (rec.build != null) return { value: rec.build };
  return { absent: rec.build_invalid != null ? invalidBuild(rec.build_invalid) : ABSENT.noBuild };
}

/** The facts the views show of *rec*'s build: every one absent for the reason *rec* has no build
 * when it has none. */
export function recordFacts(rec: RolloutRecord): BuildFacts {
  const build = buildOf(rec);
  if ("value" in build) return buildFacts(build.value);
  const none = { absent: build.absent };
  return { image: none, steps: none, inputTokens: none, outputTokens: none, cost: none, span: none, commit: none,
           sourceFiles: none };
}

/** The facts the views show of *build*. */
export function buildFacts(build: RolloutBuild): BuildFacts {
  const { package: pkg, usage } = build;
  let image: BuildFact<BuiltImage>;
  if (pkg == null) image = { absent: ABSENT.noPackage };
  else if (pkg.image == null || pkg.sif_sha256 == null) image = { absent: ABSENT.notBuilt };
  else image = { value: { key: pkg.image, sha256: pkg.sif_sha256 } };
  const span: BuildFact<BuildSpan> =
    build.started_at == null || build.ended_at == null
      ? { absent: ABSENT.noSpan }
      : { value: { start: build.started_at, end: build.ended_at } };
  return {
    image,
    steps: { absent: ABSENT.steps },
    inputTokens:
      usage == null ? { absent: ABSENT.noUsage } : fact(usage.n_input_tokens, ABSENT.noInputTokens),
    outputTokens:
      usage == null ? { absent: ABSENT.noUsage } : fact(usage.n_output_tokens, ABSENT.noOutputTokens),
    cost: usage == null ? { absent: ABSENT.noUsage } : fact(usage.cost_usd, ABSENT.noCost),
    span,
    commit: pkg == null ? { absent: ABSENT.noPackage } : fact(pkg.commit, ABSENT.noCommit),
    sourceFiles: { absent: ABSENT.sourceFiles },
  };
}

/** The record of the arm whose agent built *rec*'s submission, as its conditions name it. */
function builderArm(rec: RolloutRecord): BuildFact<{ model: string; harness: string }> {
  if (isPublished(rec)) return { absent: ABSENT.publishedProducer };
  if (rec.conditions == null) return { absent: ABSENT.noConditions };
  return rec.conditions.arm == null ? { absent: ABSENT.noArm } : { value: rec.conditions.arm.record };
}

/** The model of the arm whose agent built *rec*'s submission, as its conditions name it. */
export function builderModel(rec: RolloutRecord): BuildFact<string> {
  const arm = builderArm(rec);
  return "value" in arm ? { value: arm.value.model } : arm;
}

/** How *rec*'s submission was handed in, which no record the views read states. */
export function submissionMode(rec: RolloutRecord): BuildFact<string> {
  return { absent: isPublished(rec) ? ABSENT.publishedProducer : ABSENT.noSubmissionMode };
}

/** The egress policy of *rec*'s build sandbox, which no record the views read states. */
export function buildEgress(rec: RolloutRecord): BuildFact<string> {
  return { absent: isPublished(rec) ? ABSENT.publishedProducer : ABSENT.noBuildEgress };
}

/** *f*'s value; null when it is absent. */
export function valueOf<T>(f: BuildFact<T>): T | null {
  return "value" in f ? f.value : null;
}

/** Where the files of a build lie, each as its path below the rollout's directory, which the
 * files endpoint takes. */
export interface BuildFiles {
  /** The brief's provenance, which the Prompt tab shows. */
  briefProvenance: BuildFact<string>;
  /** The directory of the transcript output, which holds harbor's trajectory: the trajectory's
   * image paths are relative to it. */
  transcriptDir: BuildFact<string>;
  /** The build agent's log, in the transcript's copy of the agent's logs: the file named for the
   * arm's harness. */
  producerLog: BuildFact<string>;
  /** The git bundle of the agent's submission. */
  sourceBundle: BuildFact<string>;
}

/** The path below the rollout's directory of store key *key*, which the files endpoint takes: the
 * key without the key of the rollout's *instance*. */
function instancePath(instance: string, key: string): BuildFact<string> {
  const prefix = `${instance}/`;
  return key.startsWith(prefix) ? { value: key.slice(prefix.length) } : { absent: ABSENT.outsideInstance };
}

/** The path below the rollout's directory of the output *name* of *rec*'s build. */
function outputPath(rec: RolloutRecord, name: string): BuildFact<string> {
  if (rec.instance == null) return { absent: ABSENT.publishedBuild };
  const build = buildOf(rec);
  if (!("value" in build)) return build;
  const key = build.value.outputs[name];
  if (key == null) {
    const noSubmission = name === "source_bundle" && build.value.outputs.no_submission != null;
    return { absent: noSubmission ? ABSENT.noSubmission : absentOutput(name) };
  }
  return instancePath(rec.instance, key);
}

/** One log of a role, to read from the rollout's files. */
export interface RoleLog {
  /** Which stream: the extension of the file, `out` or `err`. */
  label: string;
  /** Its path below the rollout's directory. */
  path: string;
}

/** The logs of role *role* among *containerLogs*, the role logs an eval's episode names (the file
 * name of each, `<role>.out` or `<role>.err`, to its store key), as paths below the rollout's
 * *instance*, stdout first. Absent for a published record, which names no instance, and when the
 * eval's stages left none of the role's. */
export function roleLogs(
  instance: string | undefined,
  containerLogs: Record<string, string>,
  role: string,
): BuildFact<RoleLog[]> {
  if (instance == null) return { absent: ABSENT.publishedLogs };
  const named = Object.entries(containerLogs).filter(([name]) => name.startsWith(`${role}.`));
  if (named.length === 0) return { absent: absentRoleLog(role) };
  const logs: RoleLog[] = [];
  let why = "";
  for (const [name, key] of named.sort(([a], [b]) => (a.endsWith(".out") ? 0 : 1) - (b.endsWith(".out") ? 0 : 1))) {
    const path = instancePath(instance, key);
    if ("value" in path) logs.push({ label: name.slice(role.length + 1), path: path.value });
    else why = path.absent;
  }
  return logs.length > 0 ? { value: logs } : { absent: why };
}

/** The files of *rec*'s build. */
export function buildFiles(rec: RolloutRecord): BuildFiles {
  const index = outputPath(rec, "transcript");
  const transcriptDir: BuildFact<string> =
    "value" in index ? { value: index.value.slice(0, index.value.lastIndexOf("/")) } : index;
  const arm = builderArm(rec);
  let producerLog: BuildFact<string>;
  if (!("value" in transcriptDir)) producerLog = transcriptDir;
  else if (!("value" in arm)) producerLog = arm;
  else {
    const file = `${arm.value.harness}${AGENT_LOG_SUFFIX}`;
    producerLog = { value: `${transcriptDir.value}/${TRANSCRIPT_LOGS}/${file}` };
  }
  return {
    briefProvenance: outputPath(rec, "brief_provenance"),
    transcriptDir,
    producerLog,
    sourceBundle: outputPath(rec, "source_bundle"),
  };
}

/** How far a running build has got, from the events of the agent_build stage's log. */
export interface BuildProgress {
  /** "none" before the stage logs a build event; "starting" from its first, while the sandbox is
   * set up and probed, until the agent's clock starts; "running" while the agent works;
   * "finished" once the stage has copied the agent's logs. */
  phase: "none" | "starting" | "running" | "finished";
  /** The build's cost in USD, from the finished event; null when it carries none. */
  costUsd: number | null;
}

/** The progress of the build whose events are *events*, the page's events of the instance's logs:
 * those of the newest attempt of the stage (each event names its log as `source`), since a retried
 * attempt builds again. */
export function buildProgress(events: readonly { type: string; data: Record<string, unknown> }[]): BuildProgress {
  const built = events.filter((ev) => ev.type.startsWith(BUILD_EVENT_PREFIX));
  const newest = built.length > 0 ? built[built.length - 1].data.source : undefined;
  let phase: BuildProgress["phase"] = "none";
  for (const ev of built) {
    if (ev.data.source !== newest) continue;
    if (ev.type === BUILD_FINISHED) {
      const cost = ev.data[BUILD_COST_FIELD];
      return { phase: "finished", costUsd: typeof cost === "number" ? cost : null };
    }
    phase = ev.type === BUILD_STARTED ? "running" : phase === "none" ? "starting" : phase;
  }
  return { phase, costUsd: null };
}
