// How far a running instance's eval has got, from the events its deploy_drive stage logs. The stage
// logs "drive.<name>" events from the start of its attempt (its process ceiling is the first),
// `artifact_ready` once the served artifact answers, and `drive_result` with the drive's outcome.

import { describe, expect, it } from "vitest";

import { evalProgress } from "./eval-progress";

/** A page event of a stage's log. */
function ev(type: string, data: Record<string, unknown> = {}, attempt = 1) {
  return { type, ts: 1, data: { source: `stage deploy_drive attempt ${attempt}`, ...data } };
}

describe("an eval's progress, from the events its drive stage logs", () => {
  it("is none before the drive stage logs an event of its own, whatever else was logged", () => {
    expect(evalProgress([])).toEqual({ phase: "none", outcome: null });
    const others = [ev("build.started"), ev("role_started", { role: "extract" }), ev("stage_started"),
                    ev("eval.started"), ev("eval.scored")];
    expect(evalProgress(others).phase).toBe("none");
  });

  it("is starting from the stage's first drive event until the artifact answers", () => {
    expect(evalProgress([ev("drive.process_ceiling")]).phase).toBe("starting");
    expect(evalProgress([ev("drive.process_ceiling"), ev("drive.seal_probe")]).phase).toBe("starting");
  });

  it("is driving once the artifact is ready", () => {
    expect(evalProgress([ev("drive.process_ceiling"), ev("artifact_ready", { url: "u" })]).phase).toBe("driving");
  });

  it("is driven at the drive's result, with the outcome it names", () => {
    const events = [ev("drive.process_ceiling"), ev("artifact_ready"), ev("drive_result", { outcome: "served" })];
    expect(evalProgress(events)).toEqual({ phase: "driven", outcome: "served" });
  });

  it("takes no outcome from a result that names none", () => {
    expect(evalProgress([ev("drive_result")])).toEqual({ phase: "driven", outcome: null });
  });

  it("is the newest attempt's when the stage was retried: an attempt that failed does not show as driving", () => {
    const events = [ev("drive.process_ceiling"), ev("artifact_ready"), ev("drive.process_ceiling", {}, 2)];
    expect(evalProgress(events).phase).toBe("starting");
    expect(evalProgress([...events, ev("artifact_ready", {}, 2)]).phase).toBe("driving");
  });
});
