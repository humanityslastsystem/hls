// The tabs that read a build's files (Prompt, Deploy, Source) on a rollout the stage pipeline
// sealed. The fixture is the rollout.json finalize writes for a rollout with a build
// (tests/finalize/test_rollout_records.py pins it to finalize's). The record names each file by
// the store key of a build output; the files endpoint serves it by its path below the instance
// (RolloutRecord.instance, which the scanner adds), and a record that names none shows the file
// absent with the reason, never reading a path of the layouts the stage pipeline replaced.

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { renderToStaticMarkup } from "react-dom/server";
import { afterEach, describe, expect, it, vi } from "vitest";

import { DeployPanel } from "~/components/rollout/deploy-panel";
import { PromptPanel } from "~/components/rollout/prompt-panel";
import { BundleNote, SourcePanel } from "~/components/rollout/source-panel";
import { ABSENT, absentOutput, absentRoleLog } from "~/lib/build";
import grading from "~/lib/fixtures/stage-grading.json";
import fixture from "~/lib/fixtures/stage-rollout.json";
import type { FileInfo, RolloutBuild, RolloutRecord } from "~/lib/types";

const rec = { ...fixture, evals: [] } as RolloutRecord;
const BUILD = rec.build as RolloutBuild;
// A store instance's record carries the instance's own key; its build's output keys lie below it.
const INSTANCE = `runs/${fixture.task_id}/${fixture.rollout_id}`;
const OUTPUTS = "stages/agent_build/attempts/1/outputs";
const FILES = `/api/rollouts/${fixture.task_id}/${fixture.rollout_id}/files`;
const store = { ...rec, instance: INSTANCE } as RolloutRecord;
// A store record that holds no build, as one whose rollout ran none.
const unbuilt = { ...store, build: null } as RolloutRecord;
// A published record as the scanner serves it: it names no instance, and its build and the arm that
// built it are in the old recorder's shape, which the scanner serves as none.
const published = { ...rec, build: null, conditions: null } as RolloutRecord;

/** *text* as React writes it into markup, in an element or an attribute. */
function esc(text: string): string {
  return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;")
    .replace(/'/g, "&#x27;");
}

/** *r* with its build's output *name* gone. */
function without(r: RolloutRecord, name: string): RolloutRecord {
  const outputs = { ...BUILD.outputs };
  delete outputs[name];
  return { ...r, build: { ...BUILD, outputs } };
}

describe("the prompt tab's brief", () => {
  afterEach(() => vi.unstubAllGlobals());

  /** The prompt tab of *r* as first drawn (*first*) and after its queries ran (*html*), and the
   * URLs of the brief's provenance they asked the files endpoint for, the endpoint answering
   * *status*. */
  async function promptTab(r: RolloutRecord, status = 200): Promise<{ first: string; html: string; asked: string[] }> {
    const asked: string[] = [];
    vi.stubGlobal("fetch", async (url: string) => {
      asked.push(url);
      return new Response(JSON.stringify({ sections: [] }), { status });
    });
    const client = new QueryClient();
    const view = () => renderToStaticMarkup(
      <QueryClientProvider client={client}>
        <PromptPanel rec={r} />
      </QueryClientProvider>);
    const first = view();
    for (const q of client.getQueryCache().findAll({ queryKey: ["brief-provenance"] })) await q.fetch();
    return { first, html: view(), asked: asked.filter((url) => url.includes("brief_provenance")) };
  }

  it("is read from the build's brief_provenance output, and from nowhere else", async () => {
    const { asked } = await promptTab(store);
    expect(asked).toEqual([`${FILES}/${OUTPUTS}/brief_provenance.json`]);
  });

  it("is not looked for at an old path by a published record, whose build the viewer does not read", async () => {
    const { first, html, asked } = await promptTab(published);
    expect(asked).toEqual([]);
    // Nothing is waited for: the reason is there at first.
    expect(first).not.toContain("Loading brief");
    expect(first).toContain(esc(ABSENT.publishedBuild));
    expect(html).toContain("No build brief was captured for this rollout.");
  });

  it("is absent with the reason for a store record that holds no build", async () => {
    const { first, asked } = await promptTab(unbuilt);
    expect(asked).toEqual([]);
    expect(first).toContain(esc(ABSENT.noBuild));
  });

  it("is absent with the reason when the build recorded no brief_provenance output", async () => {
    const { first, asked } = await promptTab(without(store, "brief_provenance"));
    expect(asked).toEqual([]);
    expect(first).not.toContain("Loading brief");
    expect(first).toContain(esc(absentOutput("brief_provenance")));
  });

  it("says the file could not be read when the record names it and the endpoint has none", async () => {
    const { html } = await promptTab(store, 404);
    expect(html).toContain("No build brief was captured for this rollout.");
    expect(html).toContain(esc(ABSENT.briefUnreadable));
  });
});

describe("the deploy tab's producer log", () => {
  /** The deploy tab of *r*, and the keys of the log queries its roles hold. */
  function deployTab(r: RolloutRecord): { html: string; keys: unknown[][] } {
    const client = new QueryClient();
    const html = renderToStaticMarkup(
      <QueryClientProvider client={client}>
        <DeployPanel rec={r} />
      </QueryClientProvider>);
    const keys = client.getQueryCache().findAll({ queryKey: ["container-log"] }).map((q) => [...q.queryKey]);
    return { html, keys };
  }

  it("is the build agent's log in the transcript's copy of the agent's logs", () => {
    const { keys } = deployTab(store);
    expect(keys).toContainEqual(["container-log", fixture.task_id, fixture.rollout_id,
                                 `${OUTPUTS}/transcript/logs/claude-code.txt`]);
  });

  it("is shown not captured, with the reason, when the record names no arm to read the log of", () => {
    const { html, keys } = deployTab({ ...store, conditions: null } as RolloutRecord);
    expect(html).toContain(`title="${esc(ABSENT.noConditions)}">log not captured`);
    expect(JSON.stringify(keys)).not.toContain(".txt");
  });

  it("is shown not captured, with the reason, for a build that recorded no transcript", () => {
    const { html } = deployTab(without(store, "transcript"));
    expect(html).toContain(`title="${esc(absentOutput("transcript"))}">log not captured`);
  });

  it("is not a row of a record with no build", () => {
    expect(deployTab(published).html).not.toContain(">producer<");
  });
});

describe("the deploy tab's role logs", () => {
  // The grading's episode, as finalize writes it, names each role log by its file name and its key.
  const graded = { ...store, evals: [grading.episode] } as RolloutRecord;
  const DRIVE = "stages/deploy_drive/attempts/1/logs";

  /** The deploy tab of *r*, and the paths of the role logs its roles hold queries for. */
  function deployTab(r: RolloutRecord): { html: string; paths: string[] } {
    const client = new QueryClient();
    const html = renderToStaticMarkup(
      <QueryClientProvider client={client}>
        <DeployPanel rec={r} />
      </QueryClientProvider>);
    const paths = client.getQueryCache().findAll({ queryKey: ["container-log"] }).map((q) => String(q.queryKey[3]));
    return { html, paths: paths.filter((p) => !p.includes("/outputs/")) };
  }

  it("are each role's out and err files, by the keys the episode names below the instance", () => {
    const { paths, html } = deployTab(graded);
    expect(paths.sort()).toEqual([
      `${DRIVE}/artifact.err`, `${DRIVE}/artifact.out`, `${DRIVE}/driver.err`, `${DRIVE}/driver.out`,
      "stages/verify/attempts/1/logs/verify.err"]);
    // A button for each: the artifact and the driver wrote both, the verifier only its stderr.
    expect(html.split("out</button>").length - 1).toBe(2);
    expect(html.split("err</button>").length - 1).toBe(3);
  });

  it("are shown not captured, with the reason, for a role the eval's stages left no log of", () => {
    const { html } = deployTab(graded);
    expect(html).toContain(`title="${esc(absentRoleLog("sa-base"))}">log not captured`);
    expect(html).toContain(`title="${esc(absentRoleLog("submission"))}">log not captured`);
  });

  it("are not read for a published record: every role is shown not captured, with the reason", () => {
    // A published episode names each role's log by the role's name and its path in the old layout.
    const episode = { ...grading.episode, container_logs: { artifact: "evals/terminal/artifact.log" } };
    const { html, paths } = deployTab({ ...published, evals: [episode] } as RolloutRecord);
    expect(paths).toEqual([]);
    expect(html).toContain(`title="${esc(ABSENT.publishedLogs)}">log not captured`);
    expect(html).not.toContain(">producer<");
  });
});

describe("the source tab's artifact", () => {
  /** A listing of the rollout's files as the files endpoint gives it: *paths* and the directories
   * above them. */
  function listing(...paths: string[]): FileInfo[] {
    const dirs = new Set<string>();
    for (const p of paths) {
      const parts = p.split("/");
      for (let i = 1; i < parts.length; i++) dirs.add(parts.slice(0, i).join("/"));
    }
    const name = (p: string) => p.split("/").pop() as string;
    return [...[...dirs].map((p) => ({ path: p, name: name(p), is_dir: true, size: null })),
            ...paths.map((p) => ({ path: p, name: name(p), is_dir: false, size: 15 }))];
  }

  function sourceTab(r: RolloutRecord, files: FileInfo[]): string {
    const client = new QueryClient();
    client.setQueryData(["files", r.task_id, r.rollout_id], files);
    return renderToStaticMarkup(
      <QueryClientProvider client={client}>
        <SourcePanel rec={r} />
      </QueryClientProvider>);
  }

  /** Whether the toolbar's Artifact button of *html* is disabled. */
  function artifactDisabled(html: string): boolean {
    const button = html.split("<button").find((chunk) => chunk.includes("Artifact</button>")) ?? "";
    return button.slice(0, button.indexOf(">")).includes('disabled=""');
  }

  const outputs = listing(`${OUTPUTS}/build.json`, `${OUTPUTS}/brief.md`, `${OUTPUTS}/source.bundle`);

  it("is the source bundle the build's output names, and no other file of the outputs", () => {
    const html = sourceTab(store, outputs);
    expect(artifactDisabled(html)).toBe(false);
    expect(html).toContain(">source.bundle<");
    expect(html).not.toContain(">build.json<");
    expect(html).not.toContain(">brief.md<");
  });

  it("is selected at first and shown as a download, never fetched as text", () => {
    const html = sourceTab(store, outputs);
    expect(html).toContain("git bundle");
    expect(html).not.toContain("Loading…");
    expect(html).toContain(`href="${FILES}/${OUTPUTS}/source.bundle"`);
  });

  it("is not the source of an old layout: files below build/source show in the folder view alone", () => {
    const html = sourceTab(store, listing(`${OUTPUTS}/source.bundle`, "build/source/server.py"));
    expect(html).toContain(">source.bundle<");
    expect(html).not.toContain(">server.py<");
  });

  it("is disabled, with the reason, for a published record, whose build the viewer does not read", () => {
    const html = sourceTab(published, listing("build/source/server.py", "rollout.json"));
    expect(artifactDisabled(html)).toBe(true);
    expect(html).toContain(esc(`absent: ${ABSENT.publishedBuild}`));
    // The folder view lists the rollout's files as they are.
    expect(html).toContain(">rollout.json<");
  });

  it("is disabled, with the reason, for a store record that holds no build", () => {
    const html = sourceTab(unbuilt, outputs);
    expect(artifactDisabled(html)).toBe(true);
    expect(html).toContain(esc(`absent: ${ABSENT.noBuild}`));
  });

  it("is disabled, with the reason, when the build left no bundle", () => {
    const html = sourceTab(without(store, "source_bundle"), outputs);
    expect(artifactDisabled(html)).toBe(true);
    expect(html).toContain(esc(`absent: ${absentOutput("source_bundle")}`));
  });

  it("is disabled, with the reason, when the rollout's files do not list the bundle", () => {
    const html = sourceTab(store, listing(`${OUTPUTS}/build.json`));
    expect(artifactDisabled(html)).toBe(true);
    expect(html).toContain(esc(`absent: ${ABSENT.notListed}`));
  });

  it("is not previewed: a git bundle is a pack, which the page offers as a download", () => {
    const html = renderToStaticMarkup(<BundleNote size={1234} />);
    expect(html).toContain("git bundle");
    expect(html).toContain("1.2 KB");
  });
});
