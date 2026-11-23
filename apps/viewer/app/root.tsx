import "@fontsource/inter/400.css";
import "@fontsource/inter/500.css";
import "@fontsource/inter/600.css";
import "@fontsource/inter/700.css";
import "@fontsource/jetbrains-mono/400.css";
import "@fontsource/jetbrains-mono/500.css";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { Moon, Sun } from "lucide-react";
import { useTheme } from "next-themes";
import { NuqsAdapter } from "nuqs/adapters/react-router/v7";
import * as React from "react";
import {
  isRouteErrorResponse,
  Link,
  Links,
  Meta,
  Outlet,
  Scripts,
  ScrollRestoration,
  useSearchParams,
} from "react-router";
import { Toaster } from "sonner";

import type { Route } from "./+types/root";
import "./app.css";
import { ThemeProvider } from "~/components/theme-provider";
import { Button } from "~/components/ui/button";
import { cn } from "~/lib/utils";

const queryClient = new QueryClient();

export function Layout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <meta charSet="utf-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
        <Meta />
        <Links />
      </head>
      <body>
        {children}
        <ScrollRestoration />
        <Scripts />
      </body>
    </html>
  );
}

function ThemeToggle() {
  const { theme, setTheme } = useTheme();
  return (
    <Button
      variant="ghost"
      size="icon"
      onClick={() => setTheme(theme === "dark" ? "light" : "dark")}
      aria-label="Toggle theme"
    >
      <Sun className="h-4 w-4 rotate-0 scale-100 transition-all dark:-rotate-90 dark:scale-0" />
      <Moon className="absolute h-4 w-4 rotate-90 scale-0 transition-all dark:rotate-0 dark:scale-100" />
    </Button>
  );
}

function AppShell() {
  const [searchParams] = useSearchParams();
  const { setTheme } = useTheme();

  React.useEffect(() => {
    const themeParam = searchParams.get("theme");
    if (themeParam === "dark" || themeParam === "light") {
      setTheme(themeParam);
    }
  }, [searchParams, setTheme]);

  return (
    <div className="min-h-screen bg-background">
      <header className="border-b border-border">
        <div className="w-full px-6 h-12 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <Link to="/" className="flex items-center gap-2 transition-opacity hover:opacity-80">
              <div className={cn(
                "w-5 h-5 bg-primary border-[1.5px] border-primary shadow-[var(--shadow-sm)] flex items-center justify-center",
              )}>
                <span className="text-primary-foreground text-[9px] font-mono font-bold leading-none">H</span>
              </div>
              <span className="font-mono text-xs font-semibold uppercase tracking-widest text-foreground">
                HLS ROLLOUTS
              </span>
            </Link>
            <div id="breadcrumb-slot" className="text-muted-foreground text-xs font-mono" />
          </div>
          <ThemeToggle />
        </div>
      </header>
      <main className="w-full">
        <Outlet />
      </main>
    </div>
  );
}

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <ThemeProvider attribute="class" defaultTheme="light" enableSystem={false}>
        <NuqsAdapter>
          <AppShell />
          <Toaster />
        </NuqsAdapter>
      </ThemeProvider>
    </QueryClientProvider>
  );
}

export function ErrorBoundary({ error }: Route.ErrorBoundaryProps) {
  let message = "Oops!";
  let details = "An unexpected error occurred.";
  let stack: string | undefined;

  if (isRouteErrorResponse(error)) {
    message = error.status === 404 ? "404" : "Error";
    details =
      error.status === 404
        ? "The requested page could not be found."
        : error.statusText || details;
  } else if (import.meta.env.DEV && error && error instanceof Error) {
    details = error.message;
    stack = error.stack;
  }

  return (
    <main className="pt-16 px-6">
      <h1 className="font-mono text-2xl font-bold">{message}</h1>
      <p className="text-muted-foreground mt-2">{details}</p>
      {stack && (
        <pre className="w-full mt-4 p-4 bg-muted overflow-x-auto text-xs font-mono">
          <code>{stack}</code>
        </pre>
      )}
    </main>
  );
}
