// The build facts read from a stage pipeline rollout's build record (RolloutBuild). The fixture is
// the rollout.json finalize writes for a rollout with a build; each absent case changes it the way
// finalize records that case.

import { describe, expect, it } from "vitest";

import {
  ABSENT, absentOutput, absentRoleLog, buildEgress, buildFacts, buildFiles, builderModel, buildOf, buildProgress,
  invalidBuild, recordFacts, roleLogs, submissionMode, valueOf,
} from "./build";
import grading from "./fixtures/stage-grading.json";
import fixture from "./fixtures/stage-rollout.json";
import type { PackageRecord, RolloutBuild, RolloutRecord } from "./types";

const BUILD = fixture.build as RolloutBuild;
const PACKAGE = BUILD.package as PackageRecord;

describe("a build record with every fact", () => {
  const facts = buildFacts(BUILD);

  it("reads the image from the package record", () => {
    expect(facts.image).toEqual({ value: { key: PACKAGE.image, sha256: PACKAGE.sif_sha256 } });
  });

  it("reads tokens and cost from harbor's usage", () => {
    expect(facts.inputTokens).toEqual({ value: 1000 });
    expect(facts.outputTokens).toEqual({ value: 50 });
    expect(facts.cost).toEqual({ value: 12.5 });
  });

  it("reads the span from the build's start and end", () => {
    expect(facts.span).toEqual({ value: { start: 300, end: 400 } });
  });

  it("reads the commit from the package record", () => {
    expect(facts.commit).toEqual({ value: "f".repeat(40) });
  });

  it("shows steps and source files absent: the record keeps neither", () => {
    expect(facts.steps).toEqual({ absent: ABSENT.steps });
    expect(facts.sourceFiles).toEqual({ absent: ABSENT.sourceFiles });
  });
});

describe("a build record missing a fact", () => {
  it("says the package stage published nothing when there is no package record", () => {
    const facts = buildFacts({ ...BUILD, package: null });
    expect(facts.image).toEqual({ absent: ABSENT.noPackage });
    expect(facts.commit).toEqual({ absent: ABSENT.noPackage });
  });

  it("says the submission did not build when the package is invalid", () => {
    const invalid: PackageRecord = { ...PACKAGE, status: "invalid_artifact", image: null, sif_sha256: null,
                                     reason: "pyproject.toml is missing" };
    expect(buildFacts({ ...BUILD, package: invalid }).image).toEqual({ absent: ABSENT.notBuilt });
  });

  it("says the bundle had no main when the package names no commit", () => {
    expect(buildFacts({ ...BUILD, package: { ...PACKAGE, commit: null } }).commit)
      .toEqual({ absent: ABSENT.noCommit });
  });

  it("says harbor read no usage when the build records none", () => {
    const facts = buildFacts({ ...BUILD, usage: null });
    expect([facts.inputTokens, facts.outputTokens, facts.cost]).toEqual(
      [{ absent: ABSENT.noUsage }, { absent: ABSENT.noUsage }, { absent: ABSENT.noUsage }]);
  });

  it("names each usage field harbor left empty", () => {
    const usage = { n_input_tokens: null, n_cache_tokens: null, n_output_tokens: null, cost_usd: null };
    const facts = buildFacts({ ...BUILD, usage });
    expect([facts.inputTokens, facts.outputTokens, facts.cost]).toEqual(
      [{ absent: ABSENT.noInputTokens }, { absent: ABSENT.noOutputTokens }, { absent: ABSENT.noCost }]);
  });

  it("keeps a zero cost and zero tokens as values", () => {
    const usage = { n_input_tokens: 0, n_cache_tokens: null, n_output_tokens: 0, cost_usd: 0 };
    const facts = buildFacts({ ...BUILD, usage });
    expect([facts.inputTokens, facts.outputTokens, facts.cost])
      .toEqual([{ value: 0 }, { value: 0 }, { value: 0 }]);
  });

  it("has no span without both a start and an end", () => {
    expect(buildFacts({ ...BUILD, started_at: null }).span).toEqual({ absent: ABSENT.noSpan });
    expect(buildFacts({ ...BUILD, ended_at: null }).span).toEqual({ absent: ABSENT.noSpan });
  });
});

describe("valueOf", () => {
  it("is a fact's value, or null when it is absent", () => {
    expect(valueOf({ value: 3 })).toBe(3);
    expect(valueOf({ absent: "why" })).toBeNull();
  });
});

// The files of a build, which the record names by store key (RolloutBuild.outputs) and the files
// endpoint serves by their path below the instance's directory. The scanner adds the instance's
// own key to a store instance's record.

const INSTANCE = `runs/${fixture.task_id}/${fixture.rollout_id}`;
const OUTPUTS = "stages/agent_build/attempts/1/outputs";
const REC = { ...fixture, evals: [], instance: INSTANCE } as RolloutRecord;
// A published record as the scanner serves it: it names no instance, and its build and the arm
// that built it are in the old recorder's shape, which the scanner serves as none.
const { instance: _instance, ...unnamed } = REC;
const PUBLISHED = { ...unnamed, build: null, conditions: null } as RolloutRecord;

/** *REC* with its build's outputs changed by *change*. */
function withOutputs(change: (outputs: Record<string, string>) => void): RolloutRecord {
  const outputs = { ...BUILD.outputs };
  change(outputs);
  return { ...REC, build: { ...BUILD, outputs } };
}

describe("the files of a build in the store's layout", () => {
  it("lie in the instance the record names: the fixture's output keys all start with it", () => {
    for (const key of Object.values(BUILD.outputs)) expect(key.startsWith(`${INSTANCE}/`)).toBe(true);
  });

  const files = buildFiles(REC);

  it("names the brief's provenance by its output key below the instance", () => {
    expect(files.briefProvenance).toEqual({ value: `${OUTPUTS}/brief_provenance.json` });
  });

  it("names the transcript's directory, which holds the trajectory its image paths are relative to", () => {
    expect(files.transcriptDir).toEqual({ value: `${OUTPUTS}/transcript` });
  });

  it("names the agent's log in the transcript's copy of the agent's logs, by the arm's harness", () => {
    expect(files.producerLog).toEqual({ value: `${OUTPUTS}/transcript/logs/claude-code.txt` });
    const arm = fixture.conditions.arm;
    const codex = { ...REC, conditions: { ...REC.conditions!, arm: { ...arm, record: { ...arm.record, harness: "codex" } } } };
    expect(buildFiles(codex as RolloutRecord).producerLog).toEqual({ value: `${OUTPUTS}/transcript/logs/codex.txt` });
  });

  it("names the source bundle the agent committed", () => {
    expect(files.sourceBundle).toEqual({ value: `${OUTPUTS}/source.bundle` });
  });

  it("reads no old path: nothing it names is below build/", () => {
    for (const f of Object.values(files)) expect(valueOf(f)).not.toMatch(/(^|\/)build\//);
  });
});

describe("a build whose files the record does not name", () => {
  it("says the record holds no build, for a record that ran none", () => {
    const files = buildFiles({ ...REC, build: null });
    for (const f of Object.values(files)) expect(f).toEqual({ absent: ABSENT.noBuild });
  });

  it("says a published record's build is in a shape the viewer does not read", () => {
    for (const f of Object.values(buildFiles(PUBLISHED))) expect(f).toEqual({ absent: ABSENT.publishedBuild });
  });

  it("names each output the build did not record", () => {
    const files = buildFiles(withOutputs((o) => { delete o.brief_provenance; delete o.transcript; delete o.source_bundle; }));
    expect(files.briefProvenance).toEqual({ absent: absentOutput("brief_provenance") });
    expect(files.transcriptDir).toEqual({ absent: absentOutput("transcript") });
    expect(files.producerLog).toEqual({ absent: absentOutput("transcript") });
    expect(files.sourceBundle).toEqual({ absent: absentOutput("source_bundle") });
  });

  it("says the agent made no submission when the build left a no_submission record in place of a bundle", () => {
    const files = buildFiles(withOutputs((o) => { delete o.source_bundle; o.no_submission = `${INSTANCE}/${OUTPUTS}/no_submission.json`; }));
    expect(files.sourceBundle).toEqual({ absent: ABSENT.noSubmission });
  });

  it("reads no key that lies outside the instance", () => {
    const files = buildFiles(withOutputs((o) => { o.brief_provenance = "runs/fin/other/stages/agent_build/attempts/1/outputs/brief_provenance.json"; }));
    expect(files.briefProvenance).toEqual({ absent: ABSENT.outsideInstance });
  });

  it("names no producer log for a record that names no arm", () => {
    expect(buildFiles({ ...REC, conditions: null }).producerLog).toEqual({ absent: ABSENT.noConditions });
    const noArm = { ...REC, conditions: { ...REC.conditions!, arm: null } };
    expect(buildFiles(noArm).producerLog).toEqual({ absent: ABSENT.noArm });
  });
});

// What a running build's events say of its progress. The agent_build stage logs every event of its
// build as "build.<name>" in the log of its attempt; the first is its process ceiling, before the
// sandbox starts, and `build.started` is the start of the agent's clock.

/** A page event of a stage's log. */
function ev(type: string, data: Record<string, unknown> = {}) {
  return { type, ts: 1, data: { source: "stage agent_build attempt 1", ...data } };
}

describe("a build's progress, from the events its stage logs", () => {
  it("is none before the stage logs a build event, whatever other events there are", () => {
    expect(buildProgress([])).toEqual({ phase: "none", costUsd: null });
    expect(buildProgress([ev("instance_created"), ev("role_started", { role: "extract" })]).phase).toBe("none");
  });

  it("is starting from the stage's first build event until the agent's clock starts", () => {
    expect(buildProgress([ev("build.process_ceiling")]).phase).toBe("starting");
    expect(buildProgress([ev("build.process_ceiling"), ev("build.seal_probe"), ev("build.preflight")]).phase)
      .toBe("starting");
  });

  it("is running once the agent's clock starts", () => {
    expect(buildProgress([ev("build.preflight"), ev("build.started"), ev("build.heartbeat")]).phase).toBe("running");
  });

  it("is finished at the stage's finished event, with the cost it carries", () => {
    const events = [ev("build.started"), ev("build.finished", { cost_usd: 2.5, n_input_tokens: 10, n_output_tokens: 5 })];
    expect(buildProgress(events)).toEqual({ phase: "finished", costUsd: 2.5 });
  });

  it("is the newest attempt's when the stage was retried: an attempt that failed does not show as running", () => {
    const first = (type: string) => ({ type, ts: 1, data: { source: "stage agent_build attempt 1" } });
    const second = (type: string) => ({ type, ts: 2, data: { source: "stage agent_build attempt 2" } });
    expect(buildProgress([first("build.process_ceiling"), first("build.started"), second("build.process_ceiling")]).phase)
      .toBe("starting");
    expect(buildProgress([first("build.started"), second("build.process_ceiling"), second("build.started")]).phase)
      .toBe("running");
  });

  it("takes no cost from a finished event that carries none", () => {
    expect(buildProgress([ev("build.started"), ev("build.finished", { cost_usd: null })]))
      .toEqual({ phase: "finished", costUsd: null });
  });
});

// The role logs of an eval, which the grading's episode names by file name (<role>.out, <role>.err)
// and store key. The fixture is the grading.json finalize writes for the fixture rollout.

const LOGS = grading.episode.container_logs as Record<string, string>;
const DRIVE_LOGS = "stages/deploy_drive/attempts/1/logs";

describe("the role logs of an eval in the store's layout", () => {
  it("lie in the instance the record names: the fixture's keys all start with it", () => {
    for (const key of Object.values(LOGS)) expect(key.startsWith(`${INSTANCE}/`)).toBe(true);
  });

  it("are a role's out and err files, by their keys below the instance", () => {
    expect(roleLogs(INSTANCE, LOGS, "artifact")).toEqual({ value: [
      { label: "out", path: `${DRIVE_LOGS}/artifact.out` }, { label: "err", path: `${DRIVE_LOGS}/artifact.err` }] });
  });

  it("are only the files the role wrote, from the stage that ran it", () => {
    expect(roleLogs(INSTANCE, LOGS, "verify")).toEqual({ value: [
      { label: "err", path: "stages/verify/attempts/1/logs/verify.err" }] });
  });

  it("say the eval's stages left none for a role that wrote none", () => {
    expect(roleLogs(INSTANCE, LOGS, "sa-base")).toEqual({ absent: absentRoleLog("sa-base") });
  });

  it("match a role by its whole name: a role whose name starts another's has not its logs", () => {
    expect(roleLogs(INSTANCE, LOGS, "art")).toEqual({ absent: absentRoleLog("art") });
  });

  it("are not read for a published record, whose eval names its logs in the old recorder's shape", () => {
    // A published episode names each role's log by the role's name and its path in the old layout.
    const old = { artifact: "evals/terminal/artifact.log", driver: "evals/terminal/driver.log" };
    expect(roleLogs(undefined, old, "artifact")).toEqual({ absent: ABSENT.publishedLogs });
    expect(roleLogs(undefined, LOGS, "artifact")).toEqual({ absent: ABSENT.publishedLogs });
  });

  it("read no key that lies outside the instance", () => {
    const outside = { "driver.out": "runs/fin/other/stages/deploy_drive/attempts/1/logs/driver.out" };
    expect(roleLogs(INSTANCE, outside, "driver")).toEqual({ absent: ABSENT.outsideInstance });
  });
});

// What a published record shows of its build and of the agent that built it: none of it is read, and
// every fact is absent for that reason, never for one that is false of the record.

describe("a published record's build and builder", () => {
  it("has its build absent: the scanner serves none, and the record's is in the old recorder's shape", () => {
    expect(buildOf(PUBLISHED)).toEqual({ absent: ABSENT.publishedBuild });
    for (const f of Object.values(recordFacts(PUBLISHED))) expect(f).toEqual({ absent: ABSENT.publishedBuild });
  });

  it("names no implementer, submission mode or build egress: the old recorder's producer is not read", () => {
    expect(builderModel(PUBLISHED)).toEqual({ absent: ABSENT.publishedProducer });
    expect(submissionMode(PUBLISHED)).toEqual({ absent: ABSENT.publishedProducer });
    expect(buildEgress(PUBLISHED)).toEqual({ absent: ABSENT.publishedProducer });
  });
});

describe("a store record's build and builder", () => {
  it("is the build the record holds, with its facts", () => {
    expect(buildOf(REC)).toEqual({ value: BUILD });
    expect(recordFacts(REC)).toEqual(buildFacts(BUILD));
  });

  it("says the record holds no build record when it has none", () => {
    const none = { ...REC, build: null } as RolloutRecord;
    expect(buildOf(none)).toEqual({ absent: ABSENT.noBuild });
    for (const f of Object.values(recordFacts(none))) expect(f).toEqual({ absent: ABSENT.noBuild });
  });

  it("says where the record's build fails to validate when the scanner served it as none for that", () => {
    const where = ["build.ended_at", "build.usage"];
    const invalid = { ...REC, build: null, build_invalid: where } as RolloutRecord;
    expect(buildOf(invalid)).toEqual({ absent: invalidBuild(where) });
    expect(invalidBuild(where)).toContain("build.ended_at, build.usage");
    expect(buildFiles(invalid).sourceBundle).toEqual({ absent: invalidBuild(where) });
  });

  it("names the implementer from the record's conditions, and says why it names none", () => {
    expect(builderModel(REC)).toEqual({ value: fixture.conditions.arm.record.model });
    expect(builderModel({ ...REC, conditions: null })).toEqual({ absent: ABSENT.noConditions });
    expect(builderModel({ ...REC, conditions: { ...REC.conditions!, arm: null } })).toEqual({ absent: ABSENT.noArm });
  });

  it("says the stage pipeline records no submission mode and no build egress", () => {
    expect(submissionMode(REC)).toEqual({ absent: ABSENT.noSubmissionMode });
    expect(buildEgress(REC)).toEqual({ absent: ABSENT.noBuildEgress });
  });
});
