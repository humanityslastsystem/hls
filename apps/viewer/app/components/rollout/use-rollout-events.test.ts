// The page's events of a store instance across the stream's connections. Each connection restarts
// every log of the instance from its first line, in time order, and leaves out a log a finalize scan
// has quarantined since; each event names its line by its log (`data.source`) and where the line
// begins there (`offset`), which the page keeps each line once by.

import { describe, expect, it } from "vitest";

import { isNew, type RolloutEvent } from "./use-rollout-events";

/** An event of line *offset* of log *source*, of kind *type* at time *ts*. */
function line(source: string, offset: number, type: string, ts: number): RolloutEvent {
  return { type, ts, offset, data: { source } };
}

describe("the page's events across the stream's connections", () => {
  const ORCH = "orchestrator gen 1";
  const VERIFY = "stage verify attempt 1";

  it("are each line once, whatever order a later connection sends them in and whatever log it leaves out", () => {
    const held = new Set<string>();
    const first = [line(ORCH, 0, "instance_created", 1), line(VERIFY, 0, "role_started", 5)];
    expect(first.filter((ev) => isNew(held, ev))).toEqual(first);
    // Since: a build line written late with an earlier time, the verify log quarantined, and the
    // orchestrator's next line.
    const late = line("stage agent_build attempt 1", 0, "build.started", 3);
    const next = line(ORCH, 120, "instance_ended", 9);
    expect([first[0], late, next].filter((ev) => isNew(held, ev))).toEqual([late, next]);
    expect([first[0], late, next, first[1]].filter((ev) => isNew(held, ev))).toEqual([]);
  });

  it("are told apart by their log and their place in it alike", () => {
    const held = new Set<string>();
    expect(isNew(held, line(ORCH, 0, "a", 1))).toBe(true);
    expect(isNew(held, line(VERIFY, 0, "b", 1))).toBe(true);
    expect(isNew(held, line(ORCH, 40, "c", 1))).toBe(true);
    expect(isNew(held, line(ORCH, 0, "a", 1))).toBe(false);
  });

  it("keep the closing event, which names no line", () => {
    const held = new Set<string>();
    const closing: RolloutEvent = { type: "rollout.finished", ts: 1, data: { source: "viewer" } };
    expect(isNew(held, closing)).toBe(true);
    expect(isNew(held, closing)).toBe(true);
  });
});
