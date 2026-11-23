// Renders ATIF message / observation content (string or multimodal parts) for
// the build trajectory. Ink styling: hairline borders, square corners, mono for
// data. Images resolve against an imageBase: the URL of the directory the trajectory's image paths
// are relative to, or the reason the record names none.

import { useState } from "react";
import { ImageOff } from "lucide-react";

import type { BuildFact } from "~/lib/build";
import type {
  ContentPart,
  MessageContent,
  ObservationContent,
} from "~/lib/types";

// ---------------------------------------------------------------------------
// Text extraction helpers (shared with step-trigger for previews)
// ---------------------------------------------------------------------------

export function getTextFromContent(
  content: MessageContent | ObservationContent,
): string {
  if (content == null) return "";
  if (typeof content === "string") return content;
  return content
    .filter((p): p is ContentPart & { type: "text" } => p.type === "text")
    .map((p) => p.text ?? "")
    .join("\n");
}

export function getFirstLine(
  content: MessageContent | ObservationContent,
): string | null {
  const text = getTextFromContent(content);
  return text?.split("\n").find((l) => l.trim().length > 0)?.trim() || null;
}

// ---------------------------------------------------------------------------
// Image with error fallback
// ---------------------------------------------------------------------------

interface ImageError {
  status: number;
  message: string;
}

function ImageUnavailable({ path, error }: { path: string; error: ImageError }) {
  return (
    <div className="my-2 border border-dashed border-border bg-muted/30 p-3">
      <div className="mb-1 flex items-center gap-2 text-muted-foreground">
        <ImageOff className="h-4 w-4" />
        <span className="font-mono text-xs font-medium">Image unavailable</span>
        {error.status > 0 && (
          <span className="bg-muted px-1.5 py-0.5 font-mono text-[10px]">
            {error.status}
          </span>
        )}
      </div>
      <div className="break-all font-mono text-[11px] text-muted-foreground/80">
        {path}
      </div>
      <div className="mt-1 font-mono text-[11px] text-muted-foreground/60">
        {error.message}
      </div>
    </div>
  );
}

function ImageWithFallback({ src, path }: { src: string; path: string }) {
  const [error, setError] = useState<ImageError | null>(null);

  async function handleError() {
    try {
      const res = await fetch(src);
      let message = res.statusText || "Failed to load image";
      if (!res.ok) {
        try {
          const json = await res.json();
          message = json.detail || message;
        } catch {
          // not JSON — keep status text
        }
      }
      setError({ status: res.status, message });
    } catch {
      setError({ status: 0, message: "Network error" });
    }
  }

  if (error) return <ImageUnavailable path={path} error={error} />;

  return (
    <div className="my-2">
      <img
        src={src}
        alt={`Image: ${path}`}
        className="h-auto max-w-full border border-border"
        style={{ maxHeight: "400px" }}
        loading="lazy"
        onError={handleError}
      />
      <div className="mt-1 font-mono text-[11px] text-muted-foreground">{path}</div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Content renderer
// ---------------------------------------------------------------------------

export function ContentRenderer({
  content,
  imageBase,
  className = "",
}: {
  content: MessageContent | ObservationContent;
  imageBase: BuildFact<string>;
  className?: string;
}) {
  if (content == null) {
    return <span className="font-mono text-xs italic text-muted-foreground">(empty)</span>;
  }

  if (typeof content === "string") {
    return (
      <div className={`whitespace-pre-wrap break-words text-sm leading-relaxed ${className}`}>
        {content || (
          <span className="font-mono text-xs italic text-muted-foreground">(empty)</span>
        )}
      </div>
    );
  }

  return (
    <div className={`space-y-2 ${className}`}>
      {content.map((part, idx) => {
        if (part.type === "text") {
          return (
            <div key={idx} className="whitespace-pre-wrap break-words text-sm leading-relaxed">
              {part.text}
            </div>
          );
        }
        if (part.type === "image" && part.source) {
          if ("absent" in imageBase) {
            return (
              <ImageUnavailable
                key={idx}
                path={part.source.path}
                error={{ status: 0, message: imageBase.absent }}
              />
            );
          }
          const encodedPath = part.source.path
            .split("/")
            .map(encodeURIComponent)
            .join("/");
          const url = `${imageBase.value}/${encodedPath}`;
          return <ImageWithFallback key={idx} src={url} path={part.source.path} />;
        }
        return null;
      })}
    </div>
  );
}
