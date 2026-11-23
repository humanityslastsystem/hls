// The Eval tab of a running store instance, whose grading has no episode yet: its banner follows
// the events the deploy_drive stage logs (hls.stages.deploy_drive), the only stage with events of its
// own that mark the eval's progress.

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import { EvalPanel } from "~/components/rollout/eval-panel";
import type { RolloutEvent } from "~/components/rollout/use-rollout-events";
import fixture from "~/lib/fixtures/stage-rollout.json";
import type { RolloutRecord } from "~/lib/types";

// A running instance's record, as the scanner serves it before finalize writes the records.
const live = { ...fixture, status: "running", build: null, conditions: null, evals: [],
               instance: `runs/${fixture.task_id}/${fixture.rollout_id}` } as RolloutRecord;

/** An event of the deploy_drive stage's log, as the page's event stream sends it. */
function ev(type: string, data: Record<string, unknown> = {}): RolloutEvent {
  return { type, ts: 1, data: { source: "stage deploy_drive attempt 1", ...data } };
}

/** The Eval tab of the running instance, given *events*. */
function evalTab(events: RolloutEvent[]): string {
  return renderToStaticMarkup(
    <QueryClientProvider client={new QueryClient()}>
      <EvalPanel rec={live} liveEvents={events} />
    </QueryClientProvider>,
  );
}

const NONE = "No eval episodes found for this rollout.";

describe("the eval tab's banner on a running instance", () => {
  it("shows no banner before the drive stage has logged an event of its own", () => {
    const html = evalTab([ev("role_started", { role: "extract" })]);
    expect(html).toContain(NONE);
    expect(html).not.toContain("Eval starting");
    expect(html).not.toContain("Eval in progress");
  });

  it("shows no banner for the events of a pipeline that no longer exists", () => {
    const html = evalTab([ev("eval.started", { eval_id: "terminal" }), ev("eval.scored")]);
    expect(html).toContain(NONE);
    expect(html).not.toContain("Eval in progress");
  });

  it("says the artifact is starting from the stage's first drive event", () => {
    const html = evalTab([ev("drive.process_ceiling"), ev("drive.seal_probe")]);
    expect(html).toContain("Eval starting");
    expect(html).not.toContain(NONE);
  });

  it("says the artifact is being driven once it is ready", () => {
    const html = evalTab([ev("drive.process_ceiling"), ev("artifact_ready", { url: "u" })]);
    expect(html).toContain("Eval in progress");
    expect(html).toContain("Driving the artifact");
  });

  it("names the drive's outcome once its result is logged, and that verification and scoring follow", () => {
    const html = evalTab([ev("drive.process_ceiling"), ev("artifact_ready"), ev("drive_result", { outcome: "served" })]);
    expect(html).toContain("Eval in progress");
    expect(html).toContain("Drive ended (served)");
    expect(html).toContain("verification and scoring follow");
  });
});
