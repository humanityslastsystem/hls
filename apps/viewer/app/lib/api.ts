import type {
  BriefProvenance,
  FileInfo,
  PaginatedResponse,
  RolloutProvenance,
  RolloutRecord,
  RolloutSummary,
  TimelinePayload,
  TracePage,
  TraceRow,
  TraceSummary,
} from "./types";

export const API_BASE = import.meta.env.VITE_API_URL ?? "";

function enc(s: string): string {
  return encodeURIComponent(s);
}

export async function fetchRollouts(
  page = 1,
  pageSize = 100,
): Promise<PaginatedResponse<RolloutSummary>> {
  const r = await fetch(
    `${API_BASE}/api/rollouts?page=${page}&page_size=${pageSize}`,
  );
  if (!r.ok) throw new Error(`fetchRollouts: ${r.status}`);
  return r.json();
}

/** The URL of the record the scanner serves for a rollout, in whichever layout it is stored. */
export function recordUrl(task: string, id: string): string {
  return `${API_BASE}/api/rollouts/${enc(task)}/${enc(id)}`;
}

export async function fetchRollout(
  task: string,
  id: string,
): Promise<RolloutRecord> {
  const r = await fetch(recordUrl(task, id));
  if (!r.ok) throw new Error(`fetchRollout: ${r.status}`);
  return r.json();
}

export async function fetchConfig(
  task: string,
  id: string,
): Promise<RolloutProvenance> {
  const r = await fetch(
    `${API_BASE}/api/rollouts/${enc(task)}/${enc(id)}/config`,
  );
  if (!r.ok) throw new Error(`fetchConfig: ${r.status}`);
  return r.json();
}

export async function fetchTrajectory(
  task: string,
  id: string,
): Promise<unknown> {
  const r = await fetch(
    `${API_BASE}/api/rollouts/${enc(task)}/${enc(id)}/trajectory`,
  );
  if (!r.ok) throw new Error(`fetchTrajectory: ${r.status}`);
  return r.json();
}

export async function fetchTraceSummary(
  task: string,
  id: string,
  evalId: string,
): Promise<TraceSummary> {
  const r = await fetch(
    `${API_BASE}/api/rollouts/${enc(task)}/${enc(id)}/evals/${enc(evalId)}/trace/summary`,
  );
  if (!r.ok) throw new Error(`fetchTraceSummary: ${r.status}`);
  return r.json();
}

export async function fetchTracePage(
  task: string,
  id: string,
  evalId: string,
  {
    offset = 0,
    limit = 50,
    only,
  }: { offset?: number; limit?: number; only?: string },
): Promise<TracePage> {
  const params = new URLSearchParams({
    offset: String(offset),
    limit: String(limit),
  });
  if (only) params.set("only", only);
  const r = await fetch(
    `${API_BASE}/api/rollouts/${enc(task)}/${enc(id)}/evals/${enc(evalId)}/trace?${params}`,
  );
  if (!r.ok) throw new Error(`fetchTracePage: ${r.status}`);
  return r.json();
}

export async function fetchTraceRow(
  task: string,
  id: string,
  evalId: string,
  requestId: string,
): Promise<TraceRow> {
  const r = await fetch(
    `${API_BASE}/api/rollouts/${enc(task)}/${enc(id)}/evals/${enc(evalId)}/trace/row/${enc(requestId)}`,
  );
  if (!r.ok) throw new Error(`fetchTraceRow: ${r.status}`);
  return r.json();
}

export async function fetchTimeline(
  task: string,
  id: string,
  evalId: string,
): Promise<TimelinePayload> {
  const r = await fetch(
    `${API_BASE}/api/rollouts/${enc(task)}/${enc(id)}/evals/${enc(evalId)}/timeline`,
  );
  if (!r.ok) throw new Error(`fetchTimeline: ${r.status}`);
  return r.json();
}

export async function fetchFiles(
  task: string,
  id: string,
): Promise<FileInfo[]> {
  const r = await fetch(`${API_BASE}/api/rollouts/${enc(task)}/${enc(id)}/files`);
  if (!r.ok) throw new Error(`fetchFiles: ${r.status}`);
  return r.json();
}

export function fileUrl(task: string, id: string, path: string): string {
  const encodedPath = path.split("/").map(encodeURIComponent).join("/");
  return `${API_BASE}/api/rollouts/${enc(task)}/${enc(id)}/files/${encodedPath}`;
}

export async function fetchAllRollouts(): Promise<RolloutSummary[]> {
  const PAGE_SIZE = 200;
  let page = 1;
  let all: RolloutSummary[] = [];
  while (true) {
    const res = await fetchRollouts(page, PAGE_SIZE);
    all = all.concat(res.items);
    if (all.length >= res.total || page >= res.total_pages) break;
    page++;
  }
  return all;
}

export function rolloutEventsUrl(task: string, id: string): string {
  return `${API_BASE}/api/rollouts/${enc(task)}/${enc(id)}/events`;
}

export function rolloutsStreamUrl(): string {
  return `${API_BASE}/api/rollouts/stream`;
}
