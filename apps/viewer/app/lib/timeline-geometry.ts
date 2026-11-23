// Pure, DOM-free geometry + classification for the Timeline canvas.
// Unit-tested in timeline-geometry.test.ts; the canvas renderer consumes these.
import type { TimelineRequest, TimelineSLO } from "./types";

/** Clamp a time window to [tMin,tMax] preserving span, enforcing a minimum span. */
export function clampWin(
  a: number, b: number, tMin: number, tMax: number, minSpan: number,
): { a: number; b: number } {
  const full = tMax - tMin;
  let span = Math.max(Math.min(minSpan, full), Math.min(full, b - a));
  if (a < tMin) a = tMin;
  if (a + span > tMax) a = tMax - span;
  return { a, b: a + span };
}

/** Map a time t to an x pixel given the window and plot geometry. */
export function xOf(t: number, winA: number, winB: number, gut: number, plotW: number): number {
  return gut + ((t - winA) / (winB - winA)) * plotW;
}

/**
 * Lane-height floor = the height at which every visible lane fills the viewport.
 * Zooming out can shrink lanes until they fill the screen, then stops (no hairlines).
 * Capped at 40px; floored at 1px when rows cannot all fit.
 */
export function laneFloor(nSess: number, nHead: number, plotH: number, headerH: number): number {
  const fitH = nSess > 0 ? (plotH - nHead * headerH) / nSess : 40;
  return Math.max(1, Math.min(40, fitH));
}

export function laneClamp(h: number, minH: number, maxH: number): number {
  return Math.max(minH, Math.min(maxH, h));
}

/** A request span [send,recv] overlaps the visible time window [winA,winB]. */
export function spanVisible(send: number, recv: number | null, winA: number, winB: number): boolean {
  const end = recv == null ? send : recv;
  return end >= winA && send <= winB;
}

/** SLO classification for coloring: error first, then violated vs met. */
export function sloClass(
  req: TimelineRequest, slo: TimelineSLO | null | undefined,
): "err" | "violated" | "met" {
  if (req.err) return "err";
  if (!slo) return "met";
  if (slo.kind === "streaming") {
    if (req.ttft != null && slo.ttft_ms != null && (req.ttft - req.send) * 1000 > slo.ttft_ms) {
      return "violated";
    }
    // TPOT end-of-window: prefer the last streamed packet's timestamp (tlast),
    // matching the server's StreamingLatencySLO._meets formula
    // ((last_packet_ts - first_packet_ts) / (n-1)). Falling back to `recv` (t_recv,
    // stamped after the SSE stream's trailing `data: [DONE]`) over-counts, since
    // recv is strictly >= the last packet -- that inflates TPOT and can flag a
    // request red that the scorer credited green. Only fall back when tlast isn't
    // available (older trace data, or no per-packet timing captured).
    const tpotEnd = req.tlast ?? req.recv;
    if (
      slo.tpot_ms != null && req.ttft != null && tpotEnd != null &&
      req.ntok != null && req.ntok > 1 &&
      ((tpotEnd - req.ttft) * 1000) / (req.ntok - 1) > slo.tpot_ms
    ) {
      return "violated";
    }
    return "met";
  }
  if (slo.kind === "latency" && slo.target_ms != null && req.recv != null) {
    return (req.recv - req.send) * 1000 > slo.target_ms ? "violated" : "met";
  }
  return "met";
}
