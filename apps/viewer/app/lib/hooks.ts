import { useCallback, useEffect, useRef, useState } from "react";
import { useHotkeys } from "react-hotkeys-hook";

/**
 * Returns a debounced version of the value that only updates
 * after the specified delay has passed without changes.
 */
export function useDebouncedValue<T>(value: T, delay: number): T {
  const [debouncedValue, setDebouncedValue] = useState(value);

  useEffect(() => {
    const timer = setTimeout(() => setDebouncedValue(value), delay);
    return () => clearTimeout(timer);
  }, [value, delay]);

  return debouncedValue;
}

/**
 * Guard: returns true if the event target is an editable field.
 * Used internally by hooks that must not hijack text input.
 */
function isEditableTarget(e: KeyboardEvent): boolean {
  const tag = (e.target as HTMLElement)?.tagName;
  return tag === "INPUT" || tag === "TEXTAREA" || tag === "SELECT";
}

/**
 * Vim-style keyboard navigation for virtualised data tables.
 *
 * Keys (only when focus is NOT in a form field):
 *   j      — move cursor down
 *   k      — move cursor up
 *   Enter  — navigate to the highlighted row
 *   Escape — clear cursor, or call onEscapeUnhighlighted if already clear
 *
 * The caller is responsible for scrolling the virtualiser to the cursor
 * index via the returned `cursorIndex` and an attached ref/effect.
 */
export function useKeyboardTableNavigation<T>({
  rows,
  onNavigate,
  onEscapeUnhighlighted,
  enabled = true,
}: {
  rows: T[];
  onNavigate: (row: T) => void;
  onEscapeUnhighlighted?: () => void;
  enabled?: boolean;
}) {
  const [cursorIndex, setCursorIndex] = useState(-1);

  // Reset cursor when the row count changes (e.g. filters applied).
  // Dep is rows.length, not rows, to avoid resetting on every SSE tick
  // where a fresh array reference is passed but the set is identical.
  useEffect(() => {
    setCursorIndex(-1);
  }, [rows.length]);

  const handleNavigate = useCallback(() => {
    if (cursorIndex >= 0 && cursorIndex < rows.length) {
      onNavigate(rows[cursorIndex]);
    }
  }, [cursorIndex, rows, onNavigate]);

  const handleEscape = useCallback(() => {
    if (cursorIndex >= 0) {
      setCursorIndex(-1);
    } else {
      onEscapeUnhighlighted?.();
    }
  }, [cursorIndex, onEscapeUnhighlighted]);

  useHotkeys(
    "j",
    () => setCursorIndex((i) => Math.min(i + 1, rows.length - 1)),
    { enableOnFormTags: false, enabled },
    [rows.length],
  );

  useHotkeys(
    "k",
    () => setCursorIndex((i) => Math.max(i - 1, 0)),
    { enableOnFormTags: false, enabled },
    [],
  );

  useHotkeys("enter", handleNavigate, { enableOnFormTags: false, enabled }, [
    handleNavigate,
  ]);

  useHotkeys("escape", handleEscape, { enableOnFormTags: false, enabled }, [
    handleEscape,
  ]);

  return { cursorIndex, setCursorIndex };
}

/**
 * Registers a `g l` chord shortcut.
 *
 * The sequence is: press `g`, then press `l` within 800 ms.
 * Only fires when focus is NOT in a form field.
 * Calls `onActivate` when the sequence completes.
 */
export function useGotoList({ onActivate }: { onActivate: () => void }) {
  const pendingG = useRef(false);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  useEffect(() => {
    function handler(e: KeyboardEvent) {
      if (isEditableTarget(e)) return;
      if (e.metaKey || e.ctrlKey || e.altKey) return;

      if (e.key === "g") {
        pendingG.current = true;
        if (timerRef.current) clearTimeout(timerRef.current);
        timerRef.current = setTimeout(() => {
          pendingG.current = false;
        }, 800);
        return;
      }

      if (e.key === "l" && pendingG.current) {
        pendingG.current = false;
        if (timerRef.current) clearTimeout(timerRef.current);
        onActivate();
        return;
      }

      // Any other key resets the chord
      pendingG.current = false;
    }

    window.addEventListener("keydown", handler);
    return () => {
      window.removeEventListener("keydown", handler);
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, [onActivate]);
}

/**
 * Registers number key shortcuts `1`–`N` (where N = keys.length) to
 * switch between tabs/views. Only fires when focus is NOT in a form field.
 *
 * Example:
 *   useNumberKeyTabs({ keys: ["overview","prompt","build"], onSelect })
 *   → 1 selects "overview", 2 selects "prompt", 3 selects "build"
 */
export function useNumberKeyTabs<T extends string>({
  keys,
  onSelect,
  enabled = true,
}: {
  keys: T[];
  onSelect: (key: T) => void;
  enabled?: boolean;
}) {
  useEffect(() => {
    if (!enabled) return;
    function handler(e: KeyboardEvent) {
      if (isEditableTarget(e)) return;
      if (e.metaKey || e.ctrlKey || e.altKey) return;
      const n = parseInt(e.key, 10);
      if (!Number.isNaN(n) && n >= 1 && n <= keys.length) {
        const target = keys[n - 1];
        if (target !== undefined) onSelect(target);
      }
    }
    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [keys, onSelect, enabled]);
}
