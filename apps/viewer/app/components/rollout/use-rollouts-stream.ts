import { useEffect, useState } from "react";
import { rolloutsStreamUrl } from "~/lib/api";
import { useEventSource } from "./use-event-source";

export interface StreamDelta {
  task_id: string;
  rollout_id: string;
  [key: string]: unknown;
}

export interface RolloutsStreamState {
  deltas: Record<string, StreamDelta>;
  /** True while the SSE connection is open and healthy. */
  connected: boolean;
}

export function useRolloutsStream(enabled = true): RolloutsStreamState {
  const [deltas, setDeltas] = useState<Record<string, StreamDelta>>({});
  const [connected, setConnected] = useState(false);

  // Disabling the stream means "not connected" (callers gate polling on this).
  useEffect(() => {
    if (!enabled) setConnected(false);
  }, [enabled]);

  useEventSource(rolloutsStreamUrl(), enabled, {
    onOpen: () => setConnected(true),
    onError: () => setConnected(false),
    onMessage: (e) => {
      try {
        const delta = JSON.parse(e.data) as StreamDelta;
        const key = `${delta.task_id}/${delta.rollout_id}`;
        setDeltas((prev) => ({ ...prev, [key]: delta }));
      } catch {
        /* ignore malformed line */
      }
    },
  });

  return { deltas, connected };
}
