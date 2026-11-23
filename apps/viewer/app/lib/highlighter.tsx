import { createHighlighterCore } from "shiki/core";
import { createJavaScriptRegexEngine } from "shiki/engine/javascript";
import type { HighlighterCore } from "shiki/core";

// Languages
import langBash from "shiki/langs/bash.mjs";
import langJson from "shiki/langs/json.mjs";
import langMarkdown from "shiki/langs/markdown.mjs";
import langPython from "shiki/langs/python.mjs";
import langToml from "shiki/langs/toml.mjs";
import langYaml from "shiki/langs/yaml.mjs";
import langTypescript from "shiki/langs/typescript.mjs";
import langJavascript from "shiki/langs/javascript.mjs";

// Themes
import themeGithubDark from "shiki/themes/github-dark.mjs";
import themeGithubLight from "shiki/themes/github-light.mjs";

export const defaultThemes = {
  light: "github-light",
  dark: "github-dark",
} as const;

const langs = [
  langBash,
  langJson,
  langMarkdown,
  langPython,
  langToml,
  langYaml,
  langTypescript,
  langJavascript,
];

// Extension → shiki language ID
const EXT_MAP: Record<string, string> = {
  py: "python",
  sh: "bash",
  bash: "bash",
  zsh: "bash",
  js: "javascript",
  jsx: "javascript",
  ts: "typescript",
  tsx: "typescript",
  json: "json",
  jsonl: "json",
  toml: "toml",
  yaml: "yaml",
  yml: "yaml",
  md: "markdown",
  markdown: "markdown",
};

export function langForFile(filename: string): string {
  const ext = filename.split(".").pop()?.toLowerCase() ?? "";
  return EXT_MAP[ext] ?? "text";
}

export function isImageFile(filename: string): boolean {
  return /\.(png|jpe?g|gif|webp|svg)$/i.test(filename);
}

// Singleton highlighter promise
let highlighterPromise: Promise<HighlighterCore> | null = null;

export function getHighlighter(): Promise<HighlighterCore> {
  if (!highlighterPromise) {
    highlighterPromise = createHighlighterCore({
      themes: [themeGithubLight, themeGithubDark],
      langs,
      engine: createJavaScriptRegexEngine(),
    });
  }
  return highlighterPromise;
}

export interface HighlightOptions {
  lang: string;
  /** File path used as cache key — avoids storing large code strings as keys. */
  path?: string;
  themes?: { light: string; dark: string };
}

// Cache keyed by path (or lang):themes — avoids large code strings as map keys
const cache = new Map<string, Promise<string>>();

/** Returns an HTML string from Shiki (safe — Shiki-generated, not user-controlled HTML). */
export async function highlight(
  code: string,
  options: HighlightOptions,
): Promise<string> {
  const themes = options.themes ?? defaultThemes;
  // Use path when available for a compact, stable key; fall back to a short
  // hash of the code so we still deduplicate anonymous snippets.
  const keyBase = options.path
    ? options.path
    : String(
        code.split("").reduce((h, c) => (Math.imul(31, h) + c.charCodeAt(0)) | 0, 0),
      );
  const cacheKey = `${options.lang}:${themes.light}:${themes.dark}:${keyBase}`;

  const cached = cache.get(cacheKey);
  if (cached) return cached;

  const promise = (async () => {
    const highlighter = await getHighlighter();

    let lang = options.lang;
    const loadedLangs = highlighter.getLoadedLanguages();
    if (!loadedLangs.includes(lang)) lang = "text";

    if (lang === "text") {
      // Escape and return plain pre/code
      const escaped = code
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;");
      return `<pre class="shiki-plain"><code>${escaped}</code></pre>`;
    }

    return highlighter.codeToHtml(code, {
      lang,
      themes: { light: themes.light, dark: themes.dark },
      defaultColor: false,
    });
  })();

  cache.set(cacheKey, promise);
  return promise;
}
