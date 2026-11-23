import { useEffect, useRef, useState } from "react";
import { rolloutEventsUrl } from "~/lib/api";
import { useEventSource } from "./use-event-source";

export interface RolloutEvent {
  type: string;
  ts: number;
  // Where the event's line begins in its log (data.source), in bytes; with the log it names the
  // event on every connection. The viewer's closing event names no line and has none.
  offset?: number;
  data: Record<string, unknown>;
}

/** Whether *ev* is an event the page does not hold yet, by its log and where its line begins there,
 * and so adds to *held*, the names of the lines it holds. The closing event names no line and is
 * always new. */
export function isNew(held: Set<string>, ev: RolloutEvent): boolean {
  if (ev.offset == null) return true;
  const name = `${String(ev.data.source)}\u0000${ev.offset}`;
  if (held.has(name)) return false;
  held.add(name);
  return true;
}

export function useRolloutEvents(
  task: string,
  id: string,
  enabled = true,
): { events: RolloutEvent[]; done: boolean } {
  const [events, setEvents] = useState<RolloutEvent[]>([]);
  const [done, setDone] = useState(false);
  // The lines of the events held so far for this stream identity (isNew).
  const heldRef = useRef(new Set<string>());

  // A new stream (rollout change / enable) starts with no line held.
  useEffect(() => {
    heldRef.current = new Set();
  }, [task, id, enabled]);

  useEventSource(rolloutEventsUrl(task, id), enabled, {
    resetBackoffOnStart: true,
    // Each connection restarts every log from its first line, in time order (_store_events in
    // src/hls/viewer/server.py): a line written since with an earlier time comes before lines
    // already held, and a log a finalize scan quarantined since is left out. So the page keeps
    // each event once by the line it names, never by its place in the stream.
    onMessage: (e, { stop }) => {
      try {
        const ev = JSON.parse(e.data) as RolloutEvent;
        if (!isNew(heldRef.current, ev)) return;
        setEvents((prev) => [...prev, ev]);
        if (ev.type === "rollout.finished") {
          setDone(true);
          stop();
        }
      } catch {
        /* ignore malformed line */
      }
    },
  });

  return { events, done };
}
