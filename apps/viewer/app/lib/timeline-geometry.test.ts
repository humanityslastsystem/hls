import { describe, expect, it } from "vitest";
import {
  clampWin,
  laneClamp,
  laneFloor,
  sloClass,
  spanVisible,
  xOf,
} from "./timeline-geometry";
import type { TimelineRequest } from "./types";

const req = (o: Partial<TimelineRequest>): TimelineRequest => ({
  ti: 0, sid: "s", turn: 0, send: 0, recv: 1, plan: null, ttft: 0.3,
  ntok: 100, err: false, ...o,
});

describe("clampWin", () => {
  it("keeps a window inside [tMin,tMax]", () => {
    expect(clampWin(-5, 10, 0, 100, 0.2)).toEqual({ a: 0, b: 15 });
  });
  it("enforces a minimum span", () => {
    const w = clampWin(5, 5.05, 0, 100, 0.2);
    expect(w.b - w.a).toBeCloseTo(0.2, 6);
  });
  it("never exceeds the full range", () => {
    expect(clampWin(-10, 200, 0, 100, 0.2)).toEqual({ a: 0, b: 100 });
  });
});

describe("xOf", () => {
  it("maps window edges to plot edges", () => {
    expect(xOf(0, 0, 10, 150, 1000)).toBeCloseTo(150, 6);
    expect(xOf(10, 0, 10, 150, 1000)).toBeCloseTo(1150, 6);
    expect(xOf(5, 0, 10, 150, 1000)).toBeCloseTo(650, 6);
  });
});

describe("laneFloor + laneClamp", () => {
  it("floors at the fill-the-viewport height when rows fit", () => {
    // 330 sessions, 3 headers(22), plotH 938 -> (938-66)/330 = 2.6424
    expect(laneFloor(330, 3, 938, 22)).toBeCloseTo(2.6424, 3);
  });
  it("floors at 1px and caps at 40px", () => {
    expect(laneFloor(13200, 3, 938, 22)).toBe(1); // can't fit -> 1
    expect(laneFloor(3, 0, 938, 22)).toBe(40); // huge fit height -> capped
  });
  it("clamps height between floor and 40", () => {
    expect(laneClamp(100, 2.64, 40)).toBe(40);
    expect(laneClamp(1, 2.64, 40)).toBeCloseTo(2.64, 6);
    expect(laneClamp(6, 2.64, 40)).toBe(6);
  });
});

describe("spanVisible", () => {
  it("culls spans outside the time window", () => {
    expect(spanVisible(0.4, 1.8, 1, 5)).toBe(true);
    expect(spanVisible(6, 7, 1, 5)).toBe(false);
    expect(spanVisible(-2, -1, 1, 5)).toBe(false);
  });
});

describe("sloClass", () => {
  const streaming = { kind: "streaming", ttft_ms: 6000, tpot_ms: 100 };
  it("errors win regardless of timing", () => {
    expect(sloClass(req({ err: true }), streaming)).toBe("err");
  });
  it("flags a TTFT violation", () => {
    // ttft duration = (7 - 0)*1000 = 7000ms > 6000
    expect(sloClass(req({ send: 0, ttft: 7, recv: 10 }), streaming)).toBe("violated");
  });
  it("flags a TPOT violation", () => {
    // decode = (recv - ttft) = 0.5s over (ntok-1)=99 -> ~5ms? choose big: ntok 2 -> 500ms > 100
    expect(sloClass(req({ send: 0, ttft: 0.1, recv: 0.7, ntok: 2 }), streaming)).toBe("violated");
  });
  it("passes a healthy streaming request", () => {
    expect(sloClass(req({ send: 0, ttft: 0.3, recv: 1, ntok: 100 }), streaming)).toBe("met");
  });
  it("uses latency target for latency SLOs", () => {
    const lat = { kind: "latency", target_ms: 500 };
    expect(sloClass(req({ send: 0, recv: 0.6 }), lat)).toBe("violated");
    expect(sloClass(req({ send: 0, recv: 0.4 }), lat)).toBe("met");
  });
  it("treats non-errored as met when no SLO target", () => {
    expect(sloClass(req({}), null)).toBe("met");
    expect(sloClass(req({ err: true }), null)).toBe("err");
  });
});
