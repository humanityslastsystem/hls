// The build views on a rollout the stage pipeline sealed. The fixture is the rollout.json finalize
// writes for a rollout with a build (tests/finalize/test_rollout_records.py pins it to finalize's);
// every build field a view shows is rendered from its RolloutBuild, or shown absent with the reason.

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { NuqsTestingAdapter } from "nuqs/adapters/testing";
import { renderToStaticMarkup } from "react-dom/server";
import { MemoryRouter } from "react-router";
import { describe, expect, it } from "vitest";

import { BuildMetaBar, BuildPanel } from "~/components/rollout/build-panel";
import type { RolloutEvent } from "~/components/rollout/use-rollout-events";
import { OverviewPanel } from "~/components/rollout/overview-panel";
import { PhaseTimeline } from "~/components/rollout/phase-timeline";
import { SubmissionModeBar } from "~/components/rollout/source-panel";
import { ABSENT, invalidBuild } from "~/lib/build";
import fixture from "~/lib/fixtures/stage-rollout.json";
import type { EvalEpisode, RolloutBuild, RolloutRecord } from "~/lib/types";
import ComparePage from "~/routes/compare";

// A store instance's record carries the instance's own key, which the scanner adds. Its build's
// output keys lie below it, and the files endpoint serves each by the path below.
const INSTANCE = `runs/${fixture.task_id}/${fixture.rollout_id}`;
const OUTPUTS = "stages/agent_build/attempts/1/outputs";
const FILES = `/api/rollouts/${fixture.task_id}/${fixture.rollout_id}/files`;
const rec = { ...fixture, evals: [], instance: INSTANCE } as RolloutRecord;
const store = rec;
const BUILD = rec.build as RolloutBuild;
const IMAGE = fixture.build.package.image;
const SHA = fixture.build.package.sif_sha256;
const COMMIT = fixture.build.package.commit.slice(0, 10);
// A published record as the scanner serves it: it names no instance, and its build and the arm that
// built it are in the old recorder's shape, which the scanner serves as none.
const { instance: _instance, ...unnamed } = rec;
const published = { ...unnamed, build: null, conditions: null } as RolloutRecord;

/** *text* as React writes it into markup, in an element or an attribute. */
function esc(text: string): string {
  return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;")
    .replace(/'/g, "&#x27;");
}

/** The terminal eval of a rollout, run from *start* to *end*. */
function evalAt(start: number, end: number): EvalEpisode {
  return { eval_id: "terminal", triggered_by: "harness", checkpoint: "c", started_at: start, finished_at: end,
           trace_path: "", verdict_summary: {}, score: {}, container_logs: {} };
}

/** The marker of an absent fact, its reason in its tooltip, and inline after it when *inline*. */
function absent(reason: string, inline = false): string {
  return `title="${esc(reason)}">${inline ? esc(`absent: ${reason}`) : "absent"}</span>`;
}

/** The text of the overview field labelled *label* in *html*. */
function field(html: string, label: string): string {
  const cell = `>${label}</span><span[^>]*><span class="truncate[^"]*">(.*?)</span></span>`;
  const found = html.match(new RegExp(cell));
  return (found?.[1] ?? "").replace(/<[^>]*>/g, "").replace(/&#x27;/g, "'");
}

describe("the overview's build fields", () => {
  const html = renderToStaticMarkup(<OverviewPanel rec={rec} />);

  it("reads cost and tokens from the build's usage", () => {
    // The cost stat and the build card's cost.
    expect(html.split("$12.50").length - 1).toBe(2);
    expect(html).toContain("1.0k in / 50 out");
    expect(html).toContain(">1.0k / 50<");
  });

  it("reads the image from the package record and the build time from the build's span", () => {
    expect(html).toContain(`title="${esc(`sha256 ${SHA}`)}">${IMAGE}</span>`);
    expect(html).toContain("1m 40s");
  });

  it("reads the source commit from the package record", () => {
    expect(html).toContain(COMMIT);
  });

  it("shows steps and source files absent with the reason", () => {
    expect(html).toContain(absent(ABSENT.steps, true));
    expect(html).toContain(absent(ABSENT.sourceFiles, true));
  });

  it("shows a build-only rollout's status and its build", () => {
    const built = renderToStaticMarkup(<OverviewPanel rec={{ ...rec, status: "built" }} />);
    expect(built).toContain(">built</span>");
    expect(built.split("$12.50").length - 1).toBe(2);
  });

  it("shows the cost and the build absent for a published record, whose build the viewer does not read", () => {
    const old = renderToStaticMarkup(<OverviewPanel rec={published} />);
    expect(old).toContain(absent(ABSENT.publishedBuild));
    expect(field(old, "Build")).toBe(`absent: ${ABSENT.publishedBuild}`);
    expect(old).not.toContain("$12.50");
  });

  it("shows the cost and the build absent for a store record that holds no build", () => {
    const none = renderToStaticMarkup(<OverviewPanel rec={{ ...rec, build: null }} />);
    expect(none).toContain(absent(ABSENT.noBuild));
    expect(field(none, "Build")).toBe(`absent: ${ABSENT.noBuild}`);
  });

  it("shows each fact a build without usage or package lacks absent with the reason", () => {
    const bare = renderToStaticMarkup(
      <OverviewPanel rec={{ ...rec, build: { ...BUILD, usage: null, package: null, ended_at: null } }} />);
    expect(bare).toContain(absent(ABSENT.noUsage));
    expect(bare).toContain(absent(ABSENT.noUsage, true));
    expect(bare).toContain(absent(ABSENT.noPackage, true));
    expect(bare).toContain(absent(ABSENT.noSpan, true));
    expect(bare).not.toContain("$12.50");
  });
});

describe("the overview's build arm and budget", () => {
  const html = renderToStaticMarkup(<OverviewPanel rec={rec} />);

  it("names the model of the arm that built the submission, from the record's conditions", () => {
    // The arm's name is not its model, so the field shows which of the two it reads.
    const arm = { ...fixture.conditions.arm, record: { ...fixture.conditions.arm.record, name: "kimi-max" } };
    const named = renderToStaticMarkup(
      <OverviewPanel rec={{ ...rec, conditions: { ...rec.conditions!, arm } }} />);
    expect(field(named, "Implementer")).toBe("k3");
  });

  it("reads the turn cap from the build's budget", () => {
    expect(field(html, "Max turns")).toBe("none");
  });

  it("shows absent what no stage record carries: the submission mode and the build's egress", () => {
    expect(html).toContain(absent(ABSENT.noSubmissionMode, true));
    expect(html).toContain(absent(ABSENT.noBuildEgress, true));
  });

  it("says why it names no arm", () => {
    const noArm = renderToStaticMarkup(
      <OverviewPanel rec={{ ...rec, conditions: { ...rec.conditions!, arm: null, budget: null } }} />);
    expect(noArm).toContain(absent(ABSENT.noArm, true));
    expect(noArm).not.toContain(">Max turns<");
    const bare = renderToStaticMarkup(<OverviewPanel rec={{ ...rec, conditions: null }} />);
    expect(bare).toContain(absent(ABSENT.noConditions, true));
  });

  it("reads nothing of a published record's producer, and says so for each fact it holds", () => {
    const producer = { model_name: "old-model", max_turns: 7, artifact_mode: "git_repo", network: { domains: null } };
    const old = renderToStaticMarkup(<OverviewPanel rec={{ ...published,
      config: { ...published.config, artifact: { producer } } }} />);
    expect(field(old, "Implementer")).toBe(`absent: ${ABSENT.publishedProducer}`);
    expect(old).not.toContain(">Max turns<");
    expect(field(old, "Submission")).toBe(`absent: ${ABSENT.publishedProducer}`);
    expect(field(old, "Build egress")).toBe(`absent: ${ABSENT.publishedProducer}`);
    expect(old).not.toContain("old-model");
  });
});

describe("the overview's runtime egress", () => {
  /** The runtime egress the overview shows for a task whose runtime network policy is *network*. */
  function runtimeEgress(network: Record<string, unknown> | null): string {
    const html = renderToStaticMarkup(
      <OverviewPanel rec={{ ...rec, config: { ...rec.config, spec: { ...rec.config.spec, network } } }} />);
    return field(html, "Runtime egress");
  }

  it("shows a task without a network policy as the air gap, which the drive stage seals it in", () => {
    expect(rec.config.spec.network).toBeNull();
    expect(runtimeEgress(null)).toBe("air-gap");
  });

  it("reads a policy's posture from its domain list", () => {
    expect(runtimeEgress({ domains: { mode: "allow", entries: [] } })).toBe("air-gap");
    expect(runtimeEgress({ domains: { mode: "deny", entries: [] } })).toBe("open");
    expect(runtimeEgress({ domains: { mode: "allow", entries: ["api.example.org"] } }))
      .toBe("filterapi.example.org");
    expect(runtimeEgress({ domains: { mode: "deny", entries: ["tracker.example.org"] } }))
      .toBe("filtertracker.example.org");
  });
});

describe("the phase timeline", () => {
  it("draws the build from the build's span", () => {
    const html = renderToStaticMarkup(<PhaseTimeline build={rec.build} evals={[]} />);
    expect(html).toContain("Build: 1m40s");
  });

  it("draws the deploy from the end of the build to the terminal eval's start", () => {
    // The build ran 300 s to 400 s; the eval 460 s to 560 s.
    const html = renderToStaticMarkup(<PhaseTimeline build={rec.build} evals={[evalAt(460, 560)]} />);
    expect(html).toContain("Deploy: 1m0s");
    expect(html).toContain("Eval: 1m40s");
  });
});

describe("the compare page's build rows", () => {
  // A second rollout whose build cost less and took half as long, so it is the best of both rows.
  const cheaper = { ...rec, rollout_id: `${rec.rollout_id}-b`,
                    build: { ...BUILD, ended_at: 350, usage: { ...BUILD.usage!, cost_usd: 3 } } };
  const client = new QueryClient();
  for (const r of [rec, cheaper]) client.setQueryData(["rollout", r.task_id, r.rollout_id], r);
  const ids = [rec, cheaper].map((r) => encodeURIComponent(`${r.task_id}/${r.rollout_id}`)).join(",");
  const html = renderToStaticMarkup(
    <QueryClientProvider client={client}>
      <MemoryRouter initialEntries={[`/compare?ids=${ids}`]}>
        <ComparePage />
      </MemoryRouter>
    </QueryClientProvider>,
  );

  /** The value cells of the row labelled *label*: whether each is the row's best, and its text. */
  function row(label: string): { best: boolean; text: string }[] {
    const tr = html.split("<tr").find((r) => r.includes(`>${label}</span>`)) ?? "";
    return [...tr.matchAll(/<td class="([^"]*)">(.*?)<\/td>/g)].slice(1)
      .map((m) => ({ best: m[1].includes("bg-accent/30"), text: m[2].replace(/<[^>]*>/g, "") }));
  }

  it("reads each build's cost from its usage and marks the cheaper best", () => {
    expect(row("Build cost")).toEqual([{ best: false, text: "$12.50" }, { best: true, text: "$3.00" }]);
  });

  it("reads each build's duration from its span and marks the shorter best", () => {
    expect(row("Build duration")).toEqual([{ best: false, text: "1m 40s" }, { best: true, text: "50s" }]);
  });

  it("shows a published rollout's cost and duration absent, with the reason", () => {
    const old = { ...published, rollout_id: `${rec.rollout_id}-old` } as RolloutRecord;
    const oldClient = new QueryClient();
    for (const r of [rec, old]) oldClient.setQueryData(["rollout", r.task_id, r.rollout_id], r);
    const pair = [rec, old].map((r) => encodeURIComponent(`${r.task_id}/${r.rollout_id}`)).join(",");
    const page = renderToStaticMarkup(
      <QueryClientProvider client={oldClient}>
        <MemoryRouter initialEntries={[`/compare?ids=${pair}`]}>
          <ComparePage />
        </MemoryRouter>
      </QueryClientProvider>,
    );
    for (const label of ["Build cost", "Build duration"]) {
      const tr = page.split("<tr").find((r) => r.includes(`>${label}</span>`)) ?? "";
      expect(tr).toContain(absent(ABSENT.publishedBuild));
    }
  });
});

describe("the build tab's meta bar", () => {
  const html = renderToStaticMarkup(<BuildMetaBar build={BUILD} />);

  it("shows the image by its store key, with its digest", () => {
    expect(html).toContain(`title="${esc(`${IMAGE}\nsha256 ${SHA}`)}">…${IMAGE.slice(-37)}</span>`);
  });

  it("reads tokens, cost and duration from the record", () => {
    expect(html).toContain(`>${(1000).toLocaleString()} / 50<`);
    expect(html).toContain("$12.50");
    expect(html).toContain("1m 40s");
  });

  it("shows steps absent with the reason", () => {
    expect(html).toContain(absent(ABSENT.steps));
  });
});

describe("the build tab's trajectory", () => {
  /** The key of *r*'s trajectory query. */
  function trajectoryKey(r: RolloutRecord): unknown[] {
    return ["trajectory", r.task_id, r.rollout_id, r.status === "running"];
  }

  /** The build tab of *r*, the trajectory endpoint having answered *trajectory* (null for none) to
   * *client*. */
  function buildTab(r: RolloutRecord, trajectory: unknown, events: RolloutEvent[] = [],
                    client = new QueryClient(), search = ""): string {
    client.setQueryData(trajectoryKey(r), trajectory);
    return renderToStaticMarkup(
      <QueryClientProvider client={client}>
        <NuqsTestingAdapter searchParams={search}>
          <BuildPanel rec={r} liveEvents={events} />
        </NuqsTestingAdapter>
      </QueryClientProvider>,
    );
  }
  const step = { step_id: 1, timestamp: null, source: "agent", model_name: null, message: "hello",
                 reasoning_content: null, tool_calls: null, observation: null, metrics: null };
  // A running instance's record, as the scanner serves it before finalize writes the record: no
  // build record yet.
  const live = { ...store, status: "running", build: null, conditions: null } as RolloutRecord;
  /** An event of the agent_build stage's log, as the page's event stream sends it. */
  function stageEvent(type: string, data: Record<string, unknown> = {}): RolloutEvent {
    return { type, ts: 1, data: { source: "stage agent_build attempt 1", ...data } };
  }

  it("shows the trajectory the endpoint serves: the build record has no flag for it", () => {
    const html = buildTab(rec, { schema_version: "1", session_id: "s", steps: [step] });
    expect(html).toContain("showing 1 of 1 steps");
    expect(html).toContain("$12.50");
  });

  it("says none was captured when the endpoint has none", () => {
    expect(buildTab(rec, null)).toContain("No build trajectory was captured.");
  });

  it("shows the live banner while the rollout runs and the endpoint has none", () => {
    const started = [{ type: "build.started", ts: 1, data: {} }];
    const html = buildTab({ ...rec, status: "running" }, null, started);
    expect(html).toContain("Build in progress");
    expect(html).not.toContain("No build trajectory was captured.");
  });

  it("shows the live banner from the stage's first build event, before the agent's clock starts", () => {
    const html = buildTab(live, null, [stageEvent("build.process_ceiling")]);
    expect(html).toContain("Build starting");
    expect(html).not.toContain("Build in progress");
  });

  it("shows no banner for a running instance whose stage has logged no build event", () => {
    const early = [stageEvent("role_started", { role: "extract" }), stageEvent("image_extracted")];
    const html = buildTab(live, null, early);
    for (const text of ["Build starting", "Build in progress", "Build complete", "No build trajectory"]) {
      expect(html).not.toContain(text);
    }
  });

  it("shows the finished banner with the cost the stage's finished event carries, and no step count", () => {
    const done = [stageEvent("build.started"),
                  stageEvent("build.finished", { cost_usd: 2.5, n_input_tokens: 7, n_output_tokens: 3 })];
    const html = buildTab(live, null, done);
    expect(html).toContain("Build complete");
    expect(html).toContain("$2.5000");
    expect(html).not.toContain("steps");
  });

  it("asks for the trajectory again while the finished build's outputs are being published", () => {
    // The stage logs its finished event before its manifest publishes the trajectory, and the
    // instance runs on for hours: the page cannot wait for it to end.
    /** What the trajectory query of the running instance, given *events*, does when asked whether
     * to refetch with *data* in its cache. */
    function refetch(events: RolloutEvent[], data: unknown): unknown {
      const client = new QueryClient();
      buildTab(live, null, events, client);
      // Nothing subscribes in a static render, so the query holds the options it was made with.
      const options = client.getQueryCache().find({ queryKey: trajectoryKey(live) })?.options;
      const interval = (options as { refetchInterval?: unknown } | undefined)?.refetchInterval;
      return (interval as (query: unknown) => unknown)({ state: { data } });
    }
    const done = [stageEvent("build.started"), stageEvent("build.finished", { cost_usd: 2.5 })];
    expect(refetch(done, undefined)).toBeGreaterThan(0);
    expect(refetch(done, { steps: [] })).toBe(false);
    expect(refetch([stageEvent("build.started")], undefined)).toBe(false);
  });

  it("reads the trajectory's images from the transcript's directory, where harbor wrote its trajectory", () => {
    const withImage = { ...step, message: [{ type: "text", text: "see" },
      { type: "image", source: { media_type: "image/png", path: "images/step_1.png" } }] };
    const html = buildTab(store, { schema_version: "1", session_id: "s", steps: [withImage] }, [],
                          new QueryClient(), "?expand=1");
    expect(html).toContain(`src="${FILES}/${OUTPUTS}/transcript/images/step_1.png"`);
  });

  it("says why an image cannot be shown when the record names no transcript to read it from", () => {
    const withImage = { ...step, message: [{ type: "image",
      source: { media_type: "image/png", path: "images/step_1.png" } }] };
    const html = buildTab(published, { schema_version: "1", session_id: "s", steps: [withImage] }, [],
                          new QueryClient(), "?expand=1");
    expect(html).not.toContain("<img");
    expect(html).toContain("Image unavailable");
    expect(html).toContain("images/step_1.png");
    expect(html).toContain(esc(ABSENT.publishedBuild));
  });

  it("says only why there is no build for a rollout whose record holds none", () => {
    const html = buildTab({ ...rec, build: null }, null);
    expect(html).toContain(esc(`No build shown: ${ABSENT.noBuild}.`));
    expect(html).not.toContain("No build trajectory was captured.");
  });

  it("says where a store record's build fails to validate, when the scanner served it as none for that", () => {
    const where = ["build.cost_usd", "build.ended_at"];
    const html = buildTab({ ...rec, build: null, build_invalid: where } as RolloutRecord, null);
    expect(html).toContain(esc(`No build shown: ${invalidBuild(where)}.`));
  });

  it("says why there is no build for a published rollout, whose build the viewer does not read", () => {
    const html = buildTab(published, null);
    expect(html).toContain(esc(`No build shown: ${ABSENT.publishedBuild}.`));
    expect(html).not.toContain("No build trajectory was captured.");
  });

  it("asks the endpoint once: a missing trajectory is an answer, not an error to retry", () => {
    const client = new QueryClient();
    buildTab(rec, null, [], client);
    expect(client.getQueryCache().find({ queryKey: trajectoryKey(rec) })?.options.retry).toBe(false);
  });
});

describe("the source tab's submission bar", () => {
  const html = renderToStaticMarkup(<SubmissionModeBar build={BUILD} />);

  it("reads the commit and the image from the package record", () => {
    expect(html).toContain(COMMIT);
    expect(html).toContain(`…${IMAGE.slice(-56)}`);
  });

  it("shows the source files absent with the reason", () => {
    expect(html).toContain(absent(ABSENT.sourceFiles));
  });
});
