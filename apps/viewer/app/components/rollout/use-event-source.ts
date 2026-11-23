import { useEffect, useRef } from "react";

const BASE_BACKOFF_MS = 1000;
const MAX_BACKOFF_MS = 30000;

export interface EventSourceControls {
  /** Closes the current connection and prevents further automatic reconnects. */
  stop: () => void;
}

export interface UseEventSourceOptions {
  /** Called for every message received on the connection. */
  onMessage: (event: MessageEvent, controls: EventSourceControls) => void;
  /** Called when the connection is (re)established, after the internal backoff reset. */
  onOpen?: () => void;
  /** Called when the connection drops (before the reconnect is scheduled). */
  onError?: () => void;
  /**
   * Reset the reconnect backoff to the base delay whenever this effect (re)starts,
   * e.g. because `url` changed. Defaults to false (backoff persists across restarts).
   */
  resetBackoffOnStart?: boolean;
}

/**
 * Owns the lifecycle of a single EventSource connection: connect, reconnect with
 * exponential backoff on error, and teardown on unmount. Callers keep their own
 * state (e.g. accumulated events) and message-handling semantics in `onMessage`.
 */
export function useEventSource(
  url: string,
  enabled: boolean,
  { onMessage, onOpen, onError, resetBackoffOnStart = false }: UseEventSourceOptions,
): void {
  const esRef = useRef<EventSource | null>(null);
  const timeoutRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const backoffRef = useRef(BASE_BACKOFF_MS);
  const unmountedRef = useRef(false);
  const stoppedRef = useRef(false);

  // Keep latest callbacks in refs so the effect doesn't need them in its
  // dependency array (avoids reconnecting on every render when callers pass
  // freshly-created inline functions).
  const onMessageRef = useRef(onMessage);
  onMessageRef.current = onMessage;
  const onOpenRef = useRef(onOpen);
  onOpenRef.current = onOpen;
  const onErrorRef = useRef(onError);
  onErrorRef.current = onError;

  useEffect(() => {
    if (!enabled) return;
    unmountedRef.current = false;
    stoppedRef.current = false;
    if (resetBackoffOnStart) backoffRef.current = BASE_BACKOFF_MS;

    function connect() {
      if (unmountedRef.current || stoppedRef.current) return;
      const es = new EventSource(url);
      esRef.current = es;

      es.onopen = () => {
        backoffRef.current = BASE_BACKOFF_MS;
        onOpenRef.current?.();
      };

      es.onmessage = (e) => {
        onMessageRef.current(e, {
          stop: () => {
            stoppedRef.current = true;
            es.close();
            if (esRef.current === es) esRef.current = null;
          },
        });
      };

      es.onerror = () => {
        es.close();
        esRef.current = null;
        onErrorRef.current?.();
        if (unmountedRef.current || stoppedRef.current) return;
        const delay = backoffRef.current;
        backoffRef.current = Math.min(delay * 2, MAX_BACKOFF_MS);
        timeoutRef.current = setTimeout(connect, delay);
      };
    }

    connect();

    return () => {
      unmountedRef.current = true;
      if (timeoutRef.current != null) clearTimeout(timeoutRef.current);
      esRef.current?.close();
    };
  }, [url, enabled, resetBackoffOnStart]);
}
