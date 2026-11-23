// Framework-agnostic canvas renderer for the Timeline tab.
//
// Port of a tested design mock, adapted to: (1) real TimelinePayload data instead
// of the mock's synthetic gen(),
// (2) the unit-tested ~/lib/timeline-geometry module for all window/x/lane/cull/
// SLO math, (3) Ink CSS color tokens read at runtime (light/dark aware) instead
// of the mock's hardcoded hex, and (4) no U+00B7 middot anywhere (project rule).
//
// The factory builds its own <canvas> elements + tooltip inside `host`, wires
// every wheel/pointer listener, and returns destroy()/setData(). The toolbar
// (group/color/tenant toggles) lives in the React panel, which re-creates the
// controller when those options change.

import {
  clampWin,
  laneClamp,
  laneFloor,
  sloClass,
  spanVisible,
  xOf,
} from "~/lib/timeline-geometry";
import type { TimelinePayload, TimelineSLO } from "~/lib/types";

export interface TimelineController {
  destroy(): void;
  setData(p: TimelinePayload): void;
}

export interface TimelineOpts {
  group: "tenant" | "flat";
  color: "slo" | "tenant" | "lat";
  tenantOn: boolean[];
  onReadout?: (text: string) => void;
}

// Fixed tenant palette (hex so alpha tints are cheap). Cycled mod length.
const TENANT_COLORS = [
  "#356aa0",
  "#7d52a8",
  "#b06a2c",
  "#2f7d4f",
  "#a23c6e",
  "#3c7d8f",
  "#8f6a2c",
  "#5a52a8",
];

// Layout constants (verbatim from the mock).
const HEADER_H = 22;
const GUT = 150;
const MINI = 46;
const PLOT_BOTTOM = 18;

interface Span {
  ti: number;
  sid: string;
  turn: number;
  tSend: number;
  tRecv: number; // never null at render time (falls back to tSend)
  tPlan: number | null;
  tFirst: number; // TTFT instant (falls back to tSend)
  ttftMs: number | null;
  nTok: number | null;
  err: boolean;
  cls: "err" | "violated" | "met";
}

interface Session {
  ti: number;
  sid: string;
  t0: number; // arrival = min send
  spanIdx: number[];
  viol: number;
  err: number;
  n: number;
}

interface RowHead {
  h: true;
  ti: number;
}
interface RowLane {
  h: false;
  se: Session;
}
type Row = RowHead | RowLane;

export function createTimeline(
  host: HTMLDivElement,
  payload: TimelinePayload,
  opts: TimelineOpts,
): TimelineController {
  // ---------------- DOM ----------------
  const ovWrap = document.createElement("div");
  ovWrap.style.cssText =
    "height:60px;border-bottom:1.5px solid var(--border);position:relative;flex:none";
  const ov = document.createElement("canvas");
  ov.style.cssText = "display:block;width:100%;height:100%;cursor:ew-resize";
  ovWrap.appendChild(ov);

  const mainWrap = document.createElement("div");
  mainWrap.style.cssText = "position:relative;flex:1;min-height:0";
  const cv = document.createElement("canvas");
  cv.style.cssText = "display:block;width:100%;height:100%";
  mainWrap.appendChild(cv);

  const frame = document.createElement("div");
  frame.style.cssText =
    "position:absolute;inset:0;display:flex;flex-direction:column";
  frame.appendChild(ovWrap);
  frame.appendChild(mainWrap);

  const tipEl = document.createElement("div");
  tipEl.style.cssText =
    "position:fixed;pointer-events:none;z-index:40;background:var(--foreground);" +
    "color:var(--background);font-family:var(--font-mono),ui-monospace,monospace;" +
    "font-size:11px;line-height:1.5;padding:8px 10px;border:1.5px solid var(--foreground);" +
    "box-shadow:3px 3px 0 rgba(0,0,0,0.25);display:none;max-width:340px;white-space:nowrap";

  host.appendChild(frame);
  host.appendChild(tipEl);

  const octx = ov.getContext("2d")!;
  const ctx = cv.getContext("2d")!;

  // ---------------- CSS color tokens (light/dark aware) ----------------
  const css = (v: string) =>
    getComputedStyle(document.documentElement).getPropertyValue(v).trim();
  // mock token -> real Ink token
  const C = {
    bg: () => css("--background"),
    panel: () => css("--card"),
    ink: () => css("--foreground"),
    muted: () => css("--muted-foreground"),
    hair: () => css("--border"),
    block: () => css("--block-border"),
    pass: () => css("--ok"),
    fail: () => css("--destructive"),
    err: () => css("--foreground"),
  };
  // Alpha tint for a fixed hex tenant color (mock's hexA).
  const hexA = (hex: string, a: number) => {
    const n = parseInt(hex.slice(1), 16);
    return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
  };
  const tenColor = (ti: number) => TENANT_COLORS[ti % TENANT_COLORS.length];

  // ---------------- data ----------------
  let data: TimelinePayload = payload;
  let spans: Span[] = [];
  let sessions: Session[] = [];
  let tMin = 0;
  let tMax = 1;
  let TN = 1; // tenant count
  // per-tenant derived stats (req count + SLO pass rate), filled in build().
  let tenReqN: number[] = [];
  let tenPassRate: number[] = [];

  function buildData() {
    spans = [];
    sessions = [];
    TN = Math.max(1, data.tenants.length);
    tMin = 0;
    tMax = Math.max(0.001, data.duration);

    // sessions grouped by sid, preserving arrival order via min send.
    const byKey = new Map<string, Session>();
    for (const req of data.requests) {
      const tSend = req.send;
      const tRecv = req.recv == null ? tSend : req.recv;
      let tFirst = req.ttft == null ? tSend : req.ttft;
      if (tFirst < tSend) tFirst = tSend;
      if (tFirst > tRecv) tFirst = tRecv;
      const ttftMs =
        req.ttft == null ? null : Math.max(0, (req.ttft - tSend) * 1000);
      const tenant = data.tenants[req.ti] as
        | { slo?: TimelineSLO | null }
        | undefined;
      const cls = sloClass(req, tenant?.slo);
      const span: Span = {
        ti: req.ti,
        sid: req.sid ?? "(no session)",
        turn: req.turn,
        tSend,
        tRecv,
        tPlan: req.plan,
        tFirst,
        ttftMs,
        nTok: req.ntok,
        err: req.err,
        cls,
      };
      const i = spans.push(span) - 1;
      // session key namespaced by tenant so identical sids across tenants stay distinct
      const key = `${req.ti} ${span.sid}`;
      let se = byKey.get(key);
      if (!se) {
        se = {
          ti: req.ti,
          sid: span.sid,
          t0: tSend,
          spanIdx: [],
          viol: 0,
          err: 0,
          n: 0,
        };
        byKey.set(key, se);
        sessions.push(se);
      }
      se.spanIdx.push(i);
      se.n++;
      if (span.err) se.err++;
      else if (cls === "violated") se.viol++;
      if (tSend < se.t0) se.t0 = tSend;
    }

    // per-tenant req count + pass rate
    tenReqN = new Array(TN).fill(0);
    const tenPass = new Array(TN).fill(0);
    for (const s of spans) {
      if (s.ti < 0 || s.ti >= TN) continue;
      tenReqN[s.ti]++;
      if (s.cls === "met") tenPass[s.ti]++;
    }
    tenPassRate = tenReqN.map((n, i) =>
      n ? Math.round((100 * tenPass[i]) / n) : 0,
    );
    calcConc();
  }

  // ---------------- state ----------------
  const st = {
    win: { a: 0, b: 1 },
    scrollY: 0,
    autoLane: true,
    laneH: 6,
  };
  let ROW_H = 6;
  const tenOn = () => opts.tenantOn;
  const anyTenant = () => tenOn().some(Boolean);

  let rows: Row[] = [];
  let rowY: number[] = [];
  let totalH = 0;

  let W = 1;
  let H = 1;
  let OW = 1;
  let OH = 1;
  let dpr = 1;

  const plotW = () => Math.max(1, W - GUT - MINI - 12);
  const plotH = () => Math.max(1, H - PLOT_BOTTOM);
  const ovW = () => Math.max(1, OW - GUT - MINI - 12);
  const x = (t: number) => xOf(t, st.win.a, st.win.b, GUT, plotW());
  const maxScroll = () => Math.max(0, totalH - plotH() + 30);
  const clampScroll = (s: number) =>
    Math.max(0, Math.min(maxScroll(), s));

  function setWin(a: number, b: number) {
    const minSpan = Math.min(0.2, tMax - tMin);
    const w = clampWin(a, b, tMin, tMax, minSpan);
    st.win.a = w.a;
    st.win.b = w.b;
  }

  const laneAuto = (n: number) =>
    n > 4000 ? 2 : n > 1500 ? 3 : n > 700 ? 4 : 6;

  function buildRows() {
    const on = sessions.filter((se) => tenOn()[se.ti]);
    const nHead =
      opts.group === "flat" ? 0 : tenOn().filter(Boolean).length;
    const minH = laneFloor(on.length, nHead, plotH(), HEADER_H);
    if (st.autoLane) {
      ROW_H = Math.min(40, Math.max(minH, laneAuto(on.length)));
    } else {
      st.laneH = laneClamp(st.laneH, minH, 40);
      ROW_H = st.laneH;
    }
    rows = [];
    if (opts.group === "flat") {
      for (const se of [...on].sort((a, b) => a.t0 - b.t0)) {
        rows.push({ h: false, se });
      }
    } else {
      for (let ti = 0; ti < TN; ti++) {
        if (!tenOn()[ti]) continue;
        rows.push({ h: true, ti });
        for (const se of on
          .filter((se) => se.ti === ti)
          .sort((a, b) => a.t0 - b.t0)) {
          rows.push({ h: false, se });
        }
      }
    }
    rowY = new Array(rows.length);
    let y = 0;
    for (let i = 0; i < rows.length; i++) {
      rowY[i] = y;
      y += rows[i].h ? HEADER_H : ROW_H;
    }
    totalH = y;
    st.scrollY = clampScroll(st.scrollY);
  }

  // ---------------- colors ----------------
  function colorFor(s: Span): string {
    if (opts.color === "tenant") return tenColor(s.ti);
    if (opts.color === "lat") {
      const slo = data.tenants[s.ti]?.slo;
      const target =
        slo && slo.kind === "streaming" ? slo.ttft_ms : slo?.target_ms;
      const r =
        s.ttftMs != null && target
          ? Math.min(1, s.ttftMs / target)
          : s.cls === "met"
            ? 0
            : 1;
      return `rgb(${Math.round(47 + 165 * r)},${Math.round(125 - 90 * r)},60)`;
    }
    return s.err ? C.err() : s.cls === "met" ? C.pass() : C.fail();
  }
  const laneColor = (se: Session) => {
    const f = se.n ? (se.viol + se.err) / se.n : 0;
    const g = Math.min(1, f * 1.6);
    return `rgb(${Math.round(47 + 175 * g)},${Math.round(125 - 95 * g)},65)`;
  };

  // ---------------- concurrency (overview) ----------------
  let conc: Float32Array[] = [];
  function calcConc() {
    const N = 600;
    conc = Array.from({ length: TN }, () => new Float32Array(N));
    const dt = tMax - tMin || 1;
    for (const s of spans) {
      if (s.ti < 0 || s.ti >= TN) continue;
      const a = Math.max(0, Math.floor(((s.tSend - tMin) / dt) * N));
      const b = Math.min(N - 1, Math.floor(((s.tRecv - tMin) / dt) * N));
      const arr = conc[s.ti];
      for (let i = a; i <= b; i++) arr[i]++;
    }
  }

  // ---------------- rendering ----------------
  function niceStep(v: number) {
    const p = Math.pow(10, Math.floor(Math.log10(v)));
    const f = v / p;
    return (f < 1.5 ? 1 : f < 3.5 ? 2 : f < 7.5 ? 5 : 10) * p;
  }

  function draw() {
    drawOverview();
    drawMain();
  }

  function drawOverview() {
    octx.clearRect(0, 0, OW, OH);
    octx.fillStyle = C.panel();
    octx.fillRect(0, 0, OW, OH);
    const N = conc[0]?.length ?? 0;
    if (N === 0) return;
    const ox = GUT;
    const ow = ovW();
    let mx = 1;
    for (let i = 0; i < N; i++) {
      let s = 0;
      for (let t = 0; t < TN; t++) if (tenOn()[t]) s += conc[t][i];
      if (s > mx) mx = s;
    }
    const base = new Float32Array(N);
    for (let t = 0; t < TN; t++) {
      if (!tenOn()[t]) continue;
      octx.beginPath();
      for (let i = 0; i < N; i++) {
        const px = ox + (i / (N - 1)) * ow;
        const py = OH - 5 - (base[i] / mx) * (OH - 13);
        i ? octx.lineTo(px, py) : octx.moveTo(px, py);
      }
      for (let i = N - 1; i >= 0; i--) {
        const px = ox + (i / (N - 1)) * ow;
        const py = OH - 5 - ((base[i] + conc[t][i]) / mx) * (OH - 13);
        octx.lineTo(px, py);
      }
      octx.closePath();
      octx.fillStyle = hexA(tenColor(t), 0.55);
      octx.fill();
      for (let i = 0; i < N; i++) base[i] += conc[t][i];
    }
    octx.fillStyle = C.muted();
    octx.font = "10px ui-monospace,monospace";
    octx.fillText("in-flight", 8, 13);
    octx.fillText("peak " + Math.round(mx), 8, OH - 6);
    // time brush
    const dt = tMax - tMin || 1;
    const bx = ox + ((st.win.a - tMin) / dt) * ow;
    const bw = Math.max(3, ((st.win.b - st.win.a) / dt) * ow);
    octx.fillStyle = "rgba(127,120,108,0.12)";
    octx.fillRect(bx, 0, bw, OH);
    octx.strokeStyle = C.ink();
    octx.lineWidth = 1.5;
    octx.strokeRect(bx + 0.5, 1, bw, OH - 2);
    octx.fillStyle = C.ink();
    octx.fillRect(bx - 1, 0, 3, OH);
    octx.fillRect(bx + bw - 2, 0, 3, OH);
  }

  function drawMain() {
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = C.bg();
    ctx.fillRect(0, 0, W, H);
    // swimlane bars (clipped to plot)
    ctx.save();
    ctx.beginPath();
    ctx.rect(GUT, 0, plotW(), plotH());
    ctx.clip();
    for (let i = 0; i < rows.length; i++) {
      const r = rows[i];
      const y = rowY[i] - st.scrollY;
      const rh = r.h ? HEADER_H : ROW_H;
      if (y + rh < 0) continue;
      if (y > plotH()) break;
      if (r.h) {
        ctx.fillStyle = hexA(tenColor(r.ti), 0.1);
        ctx.fillRect(GUT, y, plotW(), rh);
        ctx.strokeStyle = hexA(tenColor(r.ti), 0.5);
        ctx.beginPath();
        ctx.moveTo(GUT, y + rh - 0.5);
        ctx.lineTo(W - MINI, y + rh - 0.5);
        ctx.stroke();
        continue;
      }
      const pad = ROW_H > 3 ? 1 : 0;
      const hh = rh - 2 * pad;
      for (const idx of r.se.spanIdx) {
        const s = spans[idx];
        if (!spanVisible(s.tSend, s.tRecv, st.win.a, st.win.b)) continue;
        const x0 = x(s.tSend);
        const x1 = x(s.tRecv);
        let xf = x(s.tFirst);
        if (xf < x0) xf = x0;
        if (xf > x1) xf = x1;
        const w = Math.max(ROW_H > 3 ? 1.6 : 1, x1 - x0);
        // scheduler slack lead-in (grey) from plan -> send
        if (ROW_H > 3 && s.tPlan != null && s.tSend - s.tPlan > 0.05) {
          const px = x(s.tPlan);
          if (px > GUT) {
            ctx.fillStyle = "rgba(0,0,0,0.14)";
            ctx.fillRect(px, y + pad, Math.max(1, x0 - px), hh);
          }
        }
        const col = colorFor(s);
        if (w < 2.4 || hh < 3) {
          ctx.fillStyle = col;
          ctx.fillRect(x0, y + pad, w, hh);
        } else {
          // pale lead = waiting for first token (TTFT); solid = decoding (TPOT)
          ctx.globalAlpha = 0.4;
          ctx.fillStyle = col;
          ctx.fillRect(x0, y + pad, Math.max(1, xf - x0), hh);
          ctx.globalAlpha = 1;
          ctx.fillRect(xf, y + pad, Math.max(1, x1 - xf), hh);
          if (hh >= 4) {
            ctx.fillStyle = "rgba(27,25,22,0.5)";
            ctx.fillRect(xf, y + pad, 0.8, hh);
          }
        }
      }
    }
    ctx.restore();
    drawAxis();
    drawGutter();
    drawMinimap();
    if (!anyTenant()) {
      ctx.fillStyle = C.muted();
      ctx.font = "12px ui-monospace,monospace";
      ctx.textAlign = "center";
      ctx.fillText(
        "no tenants selected",
        (GUT + W - MINI) / 2,
        plotH() / 2,
      );
      ctx.textAlign = "left";
    }
    updateReadout();
  }

  function drawAxis() {
    const span = st.win.b - st.win.a;
    const step = niceStep(span / 8);
    ctx.strokeStyle = C.hair();
    ctx.beginPath();
    ctx.moveTo(GUT, H - 16.5);
    ctx.lineTo(W - MINI, H - 16.5);
    ctx.stroke();
    ctx.fillStyle = C.muted();
    ctx.font = "10px ui-monospace,monospace";
    ctx.textAlign = "center";
    for (
      let t = Math.ceil(st.win.a / step) * step;
      t < st.win.b;
      t += step
    ) {
      const px = x(t);
      if (px < GUT || px > W - MINI) continue;
      ctx.fillText(t.toFixed(span < 4 ? 1 : 0) + "s", px, H - 3);
    }
    ctx.textAlign = "left";
  }

  function drawGutter() {
    ctx.fillStyle = C.panel();
    ctx.fillRect(0, 0, GUT, H);
    ctx.strokeStyle = C.hair();
    ctx.beginPath();
    ctx.moveTo(GUT + 0.5, 0);
    ctx.lineTo(GUT + 0.5, H);
    ctx.stroke();
    ctx.textAlign = "left";
    for (let i = 0; i < rows.length; i++) {
      const r = rows[i];
      if (!r.h) continue;
      const y = rowY[i] - st.scrollY;
      if (y + HEADER_H < 0 || y > plotH()) continue;
      const tenant = data.tenants[r.ti];
      const tid = tenant?.id ?? `t${r.ti}`;
      const tname = tenant?.name ?? "";
      ctx.fillStyle = hexA(tenColor(r.ti), 0.1);
      ctx.fillRect(0, y, GUT, HEADER_H);
      ctx.fillStyle = C.ink();
      ctx.font = "bold 11px ui-monospace,monospace";
      ctx.fillText(tname ? `${tid} ${tname}` : tid, 8, y + 14);
      ctx.font = "9px ui-monospace,monospace";
      ctx.fillStyle = tenPassRate[r.ti] > 85 ? C.pass() : C.fail();
      ctx.textAlign = "right";
      ctx.fillText(tenPassRate[r.ti] + "%", GUT - 8, y + 14);
      ctx.textAlign = "left";
    }
    if (opts.group === "flat" && anyTenant()) {
      ctx.fillStyle = C.muted();
      ctx.font = "9px ui-monospace,monospace";
      ctx.fillText("all sessions", 8, 12);
      ctx.fillText("by arrival", 8, 24);
    }
  }

  function drawMinimap() {
    const x0 = W - MINI;
    const mh = plotH();
    ctx.fillStyle = C.panel();
    ctx.fillRect(x0, 0, MINI, H);
    ctx.strokeStyle = C.hair();
    ctx.beginPath();
    ctx.moveTo(x0 + 0.5, 0);
    ctx.lineTo(x0 + 0.5, H);
    ctx.stroke();
    const sc = mh / Math.max(totalH, 1);
    for (let i = 0; i < rows.length; i++) {
      const r = rows[i];
      if (r.h) continue;
      const y = rowY[i] * sc;
      const h = Math.max(1, ROW_H * sc);
      ctx.fillStyle = laneColor(r.se);
      ctx.fillRect(x0 + 6, y, MINI - 11, h);
      ctx.fillStyle = hexA(tenColor(r.se.ti), 0.9);
      ctx.fillRect(x0 + 3, y, 2.5, h);
    }
    const vy = st.scrollY * sc;
    const vh = Math.min(mh, plotH() * sc);
    ctx.strokeStyle = C.ink();
    ctx.lineWidth = 1.5;
    ctx.strokeRect(x0 + 2.5, vy + 0.5, MINI - 5, Math.max(4, vh));
    ctx.fillStyle = "rgba(127,120,108,0.08)";
    ctx.fillRect(x0 + 2.5, vy, MINI - 5, Math.max(4, vh));
    if (gesture && gesture.type === "laneBrush" && gesture.moved) {
      const a = Math.min(gesture.y0, gesture.y1);
      const b = Math.max(gesture.y0, gesture.y1);
      ctx.fillStyle = "rgba(53,106,160,0.18)";
      ctx.fillRect(x0 + 2.5, a, MINI - 5, b - a);
      ctx.strokeStyle = tenColor(0);
      ctx.lineWidth = 1.5;
      ctx.strokeRect(x0 + 2.5, a + 0.5, MINI - 5, b - a);
    }
  }

  function updateReadout() {
    if (!opts.onReadout) return;
    let vis = 0;
    for (const s of spans) if (tenOn()[s.ti]) vis++;
    let ns = 0;
    for (const se of sessions) if (tenOn()[se.ti]) ns++;
    opts.onReadout(
      `${vis.toLocaleString()} requests, ${ns.toLocaleString()} sessions, lane ${Math.round(
        ROW_H,
      )}px${st.autoLane ? " (auto)" : ""}`,
    );
  }

  // ---------------- vertical zoom ----------------
  function rowAt(contentY: number) {
    for (let i = 0; i < rows.length; i++) {
      const rh = rows[i].h ? HEADER_H : ROW_H;
      if (rowY[i] + rh > contentY) return i;
    }
    return Math.max(0, rows.length - 1);
  }
  function setLaneH(
    newH: number,
    anchorIdx: number | null,
    anchorScreenY: number,
  ) {
    st.autoLane = false;
    st.laneH = newH;
    buildRows();
    if (anchorIdx != null) {
      st.scrollY = clampScroll(
        rowY[Math.min(anchorIdx, rows.length - 1)] - (anchorScreenY || 0),
      );
    }
  }
  function zoomLanesToContent(p0: number, p1: number) {
    let k = 0;
    let h = 0;
    let first = -1;
    for (let i = 0; i < rows.length; i++) {
      const rh = rows[i].h ? HEADER_H : ROW_H;
      const top = rowY[i];
      const bot = top + rh;
      if (bot > p0 && top < p1) {
        if (first < 0) first = i;
        rows[i].h ? h++ : k++;
      }
    }
    if (k < 1) return;
    setLaneH((plotH() - h * HEADER_H) / k, first, 0);
  }
  function laneZoomBy(f: number) {
    const a = rowAt(st.scrollY);
    setLaneH((st.autoLane ? ROW_H : st.laneH) * f, a, 0);
  }
  function laneZoomAround(f: number, screenY: number) {
    const cY = st.scrollY + screenY;
    const i = rowAt(cY);
    const off = cY - rowY[i];
    const oldH = rows[i].h ? HEADER_H : ROW_H;
    setLaneH((st.autoLane ? ROW_H : st.laneH) * f, null, 0);
    const newH = rows[i].h ? HEADER_H : ROW_H;
    const newOff = rows[i].h ? off : off * (newH / oldH);
    st.scrollY = clampScroll(rowY[i] + newOff - screenY);
  }

  // ---------------- time zoom / pan ----------------
  function timeZoom(f: number, screenX: number) {
    const fr = Math.max(0, Math.min(1, (screenX - GUT) / plotW()));
    const span = st.win.b - st.win.a;
    const tAt = st.win.a + fr * span;
    setWin(tAt - fr * span * f, tAt + (1 - fr) * span * f);
  }
  function timePan(dt: number) {
    setWin(st.win.a + dt, st.win.b + dt);
  }

  // ---------------- input: wheel ----------------
  function relMain(e: MouseEvent) {
    const r = cv.getBoundingClientRect();
    return { mx: e.clientX - r.left, my: e.clientY - r.top };
  }
  function onMainWheel(e: WheelEvent) {
    e.preventDefault();
    const { mx } = relMain(e);
    const d = e.deltaY;
    if (mx >= W - MINI) {
      laneZoomAround(d > 0 ? 1 / 1.2 : 1.2, plotH() / 2); // right strip -> zoom lanes
    } else if (e.ctrlKey || e.metaKey) {
      timeZoom(d > 0 ? 1.15 : 0.87, mx); // pinch -> zoom time
    } else if (e.shiftKey) {
      timePan((d / plotW()) * (st.win.b - st.win.a)); // shift -> pan time
    } else if (Math.abs(e.deltaX) > Math.abs(d)) {
      timePan((e.deltaX / plotW()) * (st.win.b - st.win.a));
    } else {
      st.scrollY = clampScroll(st.scrollY + d); // plain -> scroll lanes
    }
    draw();
  }
  function onOvWheel(e: WheelEvent) {
    e.preventDefault();
    const r = ov.getBoundingClientRect();
    timeZoom(e.deltaY > 0 ? 1.15 : 0.87, e.clientX - r.left);
    draw();
  }

  // ---------------- input: pointer gestures ----------------
  type Gesture =
    | { type: "pan"; cx: number; cy: number; a: number; b: number; sy: number }
    | { type: "laneBrush"; y0: number; y1: number; moved: boolean }
    | { type: "timeBrush"; start: number };
  let gesture: Gesture | null = null;

  function tAtOv(e: MouseEvent) {
    const r = ov.getBoundingClientRect();
    const fr = Math.max(
      0,
      Math.min(1, (e.clientX - r.left - GUT) / ovW()),
    );
    return tMin + fr * (tMax - tMin);
  }

  function onMainDown(e: MouseEvent) {
    const { mx, my } = relMain(e);
    if (mx < GUT) return;
    if (mx >= W - MINI) {
      gesture = { type: "laneBrush", y0: my, y1: my, moved: false };
    } else {
      gesture = {
        type: "pan",
        cx: e.clientX,
        cy: e.clientY,
        a: st.win.a,
        b: st.win.b,
        sy: st.scrollY,
      };
      cv.style.cursor = "grabbing";
    }
  }
  function onOvDown(e: MouseEvent) {
    const r = ov.getBoundingClientRect();
    if (e.clientX - r.left < GUT) return;
    const start = tAtOv(e);
    gesture = { type: "timeBrush", start };
    const cur = tAtOv(e);
    setWin(Math.min(start, cur), Math.max(start, cur));
    draw();
  }
  function onMove(e: MouseEvent) {
    if (!gesture) {
      hover(e);
      return;
    }
    if (gesture.type === "pan") {
      const dx = e.clientX - gesture.cx;
      const dy = e.clientY - gesture.cy;
      setWin(
        gesture.a - (dx / plotW()) * (gesture.b - gesture.a),
        gesture.b - (dx / plotW()) * (gesture.b - gesture.a),
      );
      st.scrollY = clampScroll(gesture.sy - dy);
      draw();
    } else if (gesture.type === "laneBrush") {
      const { my } = relMain(e);
      gesture.y1 = my;
      if (Math.abs(gesture.y1 - gesture.y0) > 4) gesture.moved = true;
      draw();
    } else if (gesture.type === "timeBrush") {
      const cur = tAtOv(e);
      setWin(
        Math.min(gesture.start, cur),
        Math.max(gesture.start, cur),
      );
      draw();
    }
  }
  function onUp() {
    if (gesture && gesture.type === "laneBrush") {
      const sc = plotH() / Math.max(totalH, 1);
      if (gesture.moved) {
        const a = Math.min(gesture.y0, gesture.y1);
        const b = Math.max(gesture.y0, gesture.y1);
        zoomLanesToContent(a / sc, b / sc);
      } else {
        st.scrollY = clampScroll(gesture.y0 / sc - plotH() / 2);
      }
    }
    gesture = null;
    cv.style.cursor = "";
    draw();
  }
  function onMainHoverCursor(e: MouseEvent) {
    if (gesture) return;
    const { mx } = relMain(e);
    cv.style.cursor =
      mx >= W - MINI ? "ns-resize" : mx >= GUT ? "grab" : "default";
  }
  function onMainLeave() {
    if (!gesture) tipEl.style.display = "none";
  }

  function hover(e: MouseEvent) {
    const r = cv.getBoundingClientRect();
    const mx = e.clientX - r.left;
    const my = e.clientY - r.top;
    if (mx < GUT || mx >= W - MINI || my < 0 || my > plotH()) {
      tipEl.style.display = "none";
      return;
    }
    let hit: Span | null = null;
    for (let i = 0; i < rows.length; i++) {
      const row = rows[i];
      const y = rowY[i] - st.scrollY;
      const rh = row.h ? HEADER_H : ROW_H;
      if (my >= y && my < y + rh) {
        if (!row.h) {
          for (const idx of row.se.spanIdx) {
            const s = spans[idx];
            const x0 = x(s.tSend);
            const x1 = Math.max(x0 + 1.6, x(s.tRecv));
            if (mx >= x0 - 2 && mx <= x1 + 2) {
              hit = s;
              break;
            }
          }
        }
        break;
      }
    }
    if (!hit) {
      tipEl.style.display = "none";
      return;
    }
    const tenant = data.tenants[hit.ti];
    const tid = tenant?.id ?? `t${hit.ti}`;
    const tname = tenant?.name ?? "";
    const slo = tenant?.slo;
    const target =
      slo && slo.kind === "streaming"
        ? slo.ttft_ms
        : (slo?.target_ms ?? null);
    const sloLine =
      hit.cls === "err"
        ? '<span style="color:#ffd479">ERROR</span>'
        : hit.cls === "violated"
          ? '<span style="color:#ff9b9b">VIOLATED</span>'
          : '<span style="color:#9fe0a8">met</span>';
    const lat = ((hit.tRecv - hit.tSend) * 1000).toFixed(0);
    const ttft = hit.ttftMs != null ? `${hit.ttftMs.toFixed(0)}ms` : "n/a";
    const planLine =
      hit.tPlan != null
        ? `scheduled ${hit.tPlan.toFixed(2)}s, slack ${(
            (hit.tSend - hit.tPlan) *
            1000
          ).toFixed(0)}ms<br>`
        : "";
    tipEl.innerHTML =
      `<b style="color:#ffd479">${escapeHtml(hit.sid)}</b> turn ${hit.turn}<br>` +
      `${escapeHtml(tname ? `${tid} ${tname}` : tid)}<br>` +
      `send ${hit.tSend.toFixed(2)}s to recv ${hit.tRecv.toFixed(2)}s<br>` +
      `latency <b style="color:#ffd479">${lat}ms</b>, TTFT ${ttft}<br>` +
      planLine +
      `SLO ${sloLine}${target != null ? ` (target ${target}ms)` : ""}`;
    tipEl.style.display = "block";
    tipEl.style.left = e.clientX + 14 + "px";
    tipEl.style.top = e.clientY + 14 + "px";
  }

  function escapeHtml(s: string) {
    return s
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;");
  }

  // ---------------- resize ----------------
  function resize() {
    dpr = window.devicePixelRatio || 1;
    const pairs: [HTMLCanvasElement, CanvasRenderingContext2D, boolean][] = [
      [ov, octx, true],
      [cv, ctx, false],
    ];
    for (const [c, xctx, isOv] of pairs) {
      const r = c.getBoundingClientRect();
      c.width = Math.max(1, Math.round(r.width * dpr));
      c.height = Math.max(1, Math.round(r.height * dpr));
      xctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      if (isOv) {
        OW = r.width;
        OH = r.height;
      } else {
        W = r.width;
        H = r.height;
      }
    }
    buildRows();
    draw();
  }

  // ---------------- wiring ----------------
  cv.addEventListener("wheel", onMainWheel, { passive: false });
  ov.addEventListener("wheel", onOvWheel, { passive: false });
  cv.addEventListener("mousedown", onMainDown);
  ov.addEventListener("mousedown", onOvDown);
  window.addEventListener("mousemove", onMove);
  window.addEventListener("mouseup", onUp);
  cv.addEventListener("mousemove", onMainHoverCursor);
  cv.addEventListener("mouseleave", onMainLeave);

  const ro = new ResizeObserver(() => resize());
  ro.observe(mainWrap);
  ro.observe(ovWrap);
  window.addEventListener("resize", resize);

  // ---------------- boot ----------------
  buildData();
  setWin(tMin, tMax);
  // initial layout happens via the ResizeObserver's first callback; also size now.
  resize();

  function setData(p: TimelinePayload) {
    data = p;
    buildData();
    st.scrollY = 0;
    st.autoLane = true;
    setWin(tMin, tMax);
    buildRows();
    draw();
  }

  function destroy() {
    cv.removeEventListener("wheel", onMainWheel);
    ov.removeEventListener("wheel", onOvWheel);
    cv.removeEventListener("mousedown", onMainDown);
    ov.removeEventListener("mousedown", onOvDown);
    window.removeEventListener("mousemove", onMove);
    window.removeEventListener("mouseup", onUp);
    cv.removeEventListener("mousemove", onMainHoverCursor);
    cv.removeEventListener("mouseleave", onMainLeave);
    window.removeEventListener("resize", resize);
    ro.disconnect();
    frame.remove();
    tipEl.remove();
  }

  return { destroy, setData };
}
