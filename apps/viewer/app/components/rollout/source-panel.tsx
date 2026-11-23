import { useQuery } from "@tanstack/react-query";
import { ChevronRight, Copy, File as FileIcon, FileCode2, Folder, FolderTree } from "lucide-react";
import { useEffect, useMemo, useState } from "react";

import { fetchFiles, fileUrl } from "~/lib/api";
import { Fact } from "~/components/rollout/build-fact";
import { Segmented } from "~/components/rollout/segmented";
import { ABSENT, buildFacts, buildFiles, buildOf, valueOf } from "~/lib/build";
import { highlight, isImageFile, langForFile } from "~/lib/highlighter";
import { cn } from "~/lib/utils";
import type { FileInfo, RolloutBuild, RolloutRecord } from "~/lib/types";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function parentOf(path: string): string {
  const i = path.lastIndexOf("/");
  return i === -1 ? "" : path.slice(0, i);
}

function fmtBytes(n: number | null | undefined): string {
  if (n == null) return "";
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / 1024 / 1024).toFixed(1)} MB`;
}

// ---------------------------------------------------------------------------
// Copy button
// ---------------------------------------------------------------------------

function CopyBtn({
  getText,
  className,
}: {
  getText: () => string;
  className?: string;
}) {
  const [copied, setCopied] = useState(false);
  function copy() {
    void navigator.clipboard.writeText(getText()).then(() => {
      setCopied(true);
      setTimeout(() => setCopied(false), 1500);
    });
  }
  return (
    <button
      type="button"
      onClick={copy}
      className={cn(
        "inline-flex items-center gap-0.5 px-1 py-0.5 text-[10px] text-muted-foreground hover:text-foreground transition-colors",
        className,
      )}
      title="Copy"
    >
      <Copy className="h-3 w-3" />
      {copied && <span>✓</span>}
    </button>
  );
}

// ---------------------------------------------------------------------------
// File tree
// ---------------------------------------------------------------------------

function TreeNode({
  item,
  allFiles,
  selected,
  onSelect,
  depth,
}: {
  item: FileInfo;
  allFiles: FileInfo[];
  selected: string | null;
  onSelect: (p: string) => void;
  depth: number;
}) {
  const sortFn = (a: FileInfo, b: FileInfo) =>
    a.is_dir !== b.is_dir
      ? a.is_dir
        ? -1
        : 1
      : a.name.localeCompare(b.name);

  if (!item.is_dir) {
    const isSel = selected === item.path;
    return (
      <button
        type="button"
        onClick={() => onSelect(item.path)}
        style={{ paddingLeft: `${depth * 12 + 8}px` }}
        className={cn(
          "flex w-full items-center gap-1.5 px-2 py-1 text-left text-xs transition-colors",
          isSel ? "bg-accent text-foreground" : "hover:bg-accent/50",
        )}
      >
        <FileIcon className="h-3 w-3 shrink-0 text-muted-foreground" />
        <span className="flex-1 truncate font-mono">{item.name}</span>
        {item.size != null && (
          <span className="shrink-0 text-[10px] text-muted-foreground">
            {fmtBytes(item.size)}
          </span>
        )}
      </button>
    );
  }

  const children = allFiles
    .filter((f) => parentOf(f.path) === item.path)
    .sort(sortFn);

  // Collapsible directory
  return <DirNode item={item} children={children} allFiles={allFiles} selected={selected} onSelect={onSelect} depth={depth} />;
}

function DirNode({
  item,
  children,
  allFiles,
  selected,
  onSelect,
  depth,
}: {
  item: FileInfo;
  children: FileInfo[];
  allFiles: FileInfo[];
  selected: string | null;
  onSelect: (p: string) => void;
  depth: number;
}) {
  const [open, setOpen] = useState(true);
  return (
    <div>
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        style={{ paddingLeft: `${depth * 12 + 8}px` }}
        className="flex w-full items-center gap-1.5 px-2 py-1 text-left text-xs transition-colors hover:bg-accent/50"
      >
        <ChevronRight
          className={cn(
            "h-3 w-3 shrink-0 transition-transform",
            open && "rotate-90",
          )}
        />
        <Folder className="h-3 w-3 shrink-0 text-muted-foreground" />
        <span className="font-mono">{item.name}</span>
      </button>
      {open && children.length > 0 && (
        <div>
          {children.map((c) => (
            <TreeNode
              key={c.path}
              item={c}
              allFiles={allFiles}
              selected={selected}
              onSelect={onSelect}
              depth={depth + 1}
            />
          ))}
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Shiki code viewer
// ---------------------------------------------------------------------------

function ShikiViewer({ code, filename, path }: { code: string; filename: string; path?: string }) {
  const lang = langForFile(filename);
  const [html, setHtml] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    void highlight(code, { lang, path }).then((h) => {
      if (!cancelled) setHtml(h);
    });
    return () => {
      cancelled = true;
    };
  }, [code, lang, path]);

  return (
    <div className="overflow-auto">
      {html != null ? (
        <div
          className={cn(
            "shiki-wrapper font-mono text-xs leading-5",
            // Make Shiki pre transparent so our bg shows
            "[&_.shiki]:bg-transparent [&_pre]:m-0 [&_pre]:bg-transparent [&_pre]:p-0",
          )}
          // eslint-disable-next-line react/no-danger
          dangerouslySetInnerHTML={{ __html: html }}
        />
      ) : (
        <pre className="font-mono text-xs leading-5 whitespace-pre text-foreground">
          <code>{code}</code>
        </pre>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// File viewer
// ---------------------------------------------------------------------------

function FileViewer({
  task,
  id,
  path,
}: {
  task: string;
  id: string;
  path: string;
}) {
  const url = fileUrl(task, id, path);
  const filename = path.split("/").pop() ?? path;

  // Image files — direct img tag
  if (isImageFile(path)) {
    return (
      <div className="p-4">
        <img src={url} alt={filename} className="max-w-full" />
      </div>
    );
  }

  return <TextViewer task={task} id={id} path={path} />;
}

function TextViewer({
  task,
  id,
  path,
}: {
  task: string;
  id: string;
  path: string;
}) {
  const { data, isLoading, error, status } = useQuery<string, Error>({
    queryKey: ["file", task, id, path],
    queryFn: async () => {
      const url = fileUrl(task, id, path);
      const r = await fetch(url);
      if (r.status === 413) {
        throw Object.assign(new Error("File too large to preview (> 1 MB)"), {
          code: "TOO_LARGE",
        });
      }
      if (!r.ok) throw new Error(`Failed to load file: ${r.status}`);
      return r.text();
    },
    retry: false,
  });

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-10">
        <span className="font-mono text-xs text-muted-foreground animate-pulse">
          Loading…
        </span>
      </div>
    );
  }

  if (error) {
    const msg =
      (error as Error & { code?: string }).code === "TOO_LARGE"
        ? "File too large to preview (> 1 MB)."
        : `Failed to load file: ${(error as Error).message}`;
    return (
      <div className="border border-border bg-muted/20 p-4">
        <p className="font-mono text-xs text-muted-foreground">{msg}</p>
      </div>
    );
  }

  const code = data ?? "";
  const filename = path.split("/").pop() ?? path;

  return (
    <div className="relative">
      <div className="absolute right-2 top-1 z-10">
        <CopyBtn getText={() => code} />
      </div>
      <ShikiViewer code={code} filename={filename} path={path} />
    </div>
  );
}

// ---------------------------------------------------------------------------
// The bundle: shown in place of a preview
// ---------------------------------------------------------------------------

// The submission is a git bundle, a pack that is not text: the page offers it as a download.
export function BundleNote({ size }: { size?: number | null }) {
  return (
    <div className="border border-border bg-muted/20 p-4">
      <p className="font-mono text-xs text-muted-foreground">
        The submission is a git bundle{size != null && ` (${fmtBytes(size)})`}, which the viewer does not preview.
        Download it with raw ↗ and clone it.
      </p>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Submission mode info bar
// ---------------------------------------------------------------------------

export function SubmissionModeBar({ build }: { build: RolloutBuild }) {
  const facts = buildFacts(build);
  return (
    <div className="flex flex-wrap items-center gap-x-4 gap-y-1 text-[11px] font-mono text-muted-foreground">
      <span className="flex items-center gap-0.5">
        commit{" "}
        <Fact
          fact={facts.commit}
          format={(commit) => (
            <>
              {commit.slice(0, 10)}
              <CopyBtn getText={() => commit} />
            </>
          )}
        />
      </span>
      <span>
        files <Fact fact={facts.sourceFiles} format={String} />
      </span>
      <span className="flex min-w-0 items-center gap-0.5">
        image{" "}
        <Fact
          fact={facts.image}
          format={(img) => (
            <>
              <span className="truncate" title={`${img.key}\nsha256 ${img.sha256}`}>
                {img.key.length > 60 ? `…${img.key.slice(-56)}` : img.key}
              </span>
              <CopyBtn getText={() => img.key} />
            </>
          )}
        />
      </span>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main panel
// ---------------------------------------------------------------------------

export function SourcePanel({ rec }: { rec: RolloutRecord }) {
  const task = rec.task_id;
  const id = rec.rollout_id;
  const build = valueOf(buildOf(rec));

  const { data: files, isLoading } = useQuery({
    queryKey: ["files", task, id],
    queryFn: () => fetchFiles(task, id),
  });

  // "artifact" = the source bundle the build's output names; "folder" = whole rollout dir
  const [mode, setMode] = useState<"artifact" | "folder">("artifact");
  const [picked, setPicked] = useState<string | null>(null);

  // The bundle, when the record names one and the rollout's files list it; else why there is none.
  const bundle = buildFiles(rec).sourceBundle;
  const bundleFile = useMemo(() => {
    const path = valueOf(bundle);
    return path == null ? undefined : files?.find((f) => f.path === path && !f.is_dir);
  }, [files, bundle]);
  const bundlePath = bundleFile?.path ?? null;
  const artifactAbsent = bundleFile != null ? null : "absent" in bundle ? bundle.absent : ABSENT.notListed;

  // Effective mode: if no bundle, fall back to folder
  const effectiveMode =
    mode === "artifact" && bundlePath == null ? "folder" : mode;
  const rootPrefix = effectiveMode === "artifact" && bundlePath != null ? parentOf(bundlePath) : "";

  // Visible files for the current mode
  const visible = useMemo(() => {
    if (!files) return [];
    if (effectiveMode === "folder") return files;
    return files.filter((f) => f.path === bundlePath);
  }, [files, effectiveMode, bundlePath]);

  // The file shown: the one picked while the mode still shows it, else the bundle, else the
  // first file shown.
  const selected = useMemo(() => {
    if (picked != null && visible.some((f) => f.path === picked && !f.is_dir)) return picked;
    return bundlePath ?? visible.find((f) => !f.is_dir)?.path ?? null;
  }, [picked, visible, bundlePath]);

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-10">
        <span className="font-mono text-xs text-muted-foreground animate-pulse">
          Loading files…
        </span>
      </div>
    );
  }

  if (!files || files.length === 0) {
    return (
      <div className="w-full px-6 py-4">
        <div className="border border-border bg-muted/20 p-4">
          <p className="font-mono text-xs text-muted-foreground">
            No files recorded for this rollout.
          </p>
        </div>
      </div>
    );
  }

  const sortFn = (a: FileInfo, b: FileInfo) =>
    a.is_dir !== b.is_dir ? (a.is_dir ? -1 : 1) : a.name.localeCompare(b.name);

  const topLevel = visible
    .filter((f) => parentOf(f.path) === rootPrefix)
    .sort(sortFn);

  const fileCount = visible.filter((f) => !f.is_dir).length;

  return (
    <div className="flex h-full flex-col" style={{ minHeight: "calc(100vh - 140px)" }}>
      {/* Toolbar */}
      <div className="flex flex-wrap items-center gap-x-4 gap-y-2 border-b border-border bg-card px-4 py-2">
        {/* Artifact / Folder toggle */}
        <Segmented
          value={effectiveMode}
          onChange={setMode}
          options={[
            {
              value: "artifact",
              label: "Artifact",
              icon: FileCode2,
              disabled: bundlePath == null,
              title: artifactAbsent ?? undefined,
            },
            { value: "folder", label: "Folder", icon: FolderTree },
          ]}
        />

        {/* Mode info */}
        {effectiveMode === "artifact" && build ? (
          <SubmissionModeBar build={build} />
        ) : (
          <span className="font-mono text-[11px] text-muted-foreground">
            {`${fileCount} file${fileCount === 1 ? "" : "s"} in rollout directory`}
            {artifactAbsent != null && ` · artifact absent: ${artifactAbsent}`}
          </span>
        )}
      </div>

      {/* Split: tree + code viewer */}
      <div
        className="grid min-h-0 flex-1"
        style={{ gridTemplateColumns: "minmax(180px, 240px) 1fr" }}
      >
        {/* File tree */}
        <div className="overflow-y-auto border-r border-border bg-muted/10 py-1">
          {topLevel.map((item) => (
            <TreeNode
              key={item.path}
              item={item}
              allFiles={visible}
              selected={selected}
              onSelect={setPicked}
              depth={0}
            />
          ))}
        </div>

        {/* Code/image viewer */}
        <div className="min-h-0 overflow-auto">
          {selected ? (
            <>
              {/* File path bar */}
              <div className="flex items-center gap-2 border-b border-border px-3 py-1.5">
                <span className="flex-1 truncate font-mono text-[11px] text-muted-foreground">
                  {selected}
                </span>
                <CopyBtn getText={() => selected} />
                <a
                  href={fileUrl(task, id, selected)}
                  target="_blank"
                  rel="noreferrer"
                  className="font-mono text-[10px] text-muted-foreground hover:text-foreground transition-colors"
                >
                  raw ↗
                </a>
              </div>
              <div className="p-3">
                {selected === bundlePath ? (
                  <BundleNote size={bundleFile?.size} />
                ) : (
                  <FileViewer task={task} id={id} path={selected} />
                )}
              </div>
            </>
          ) : (
            <div className="flex items-center justify-center py-10">
              <span className="font-mono text-xs text-muted-foreground">
                Select a file.
              </span>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
