import { useEffect, useState } from "react";

export const pct = (n, digits = 2) =>
  Number.isFinite(n)
    ? new Intl.NumberFormat("en-US", {
        style: "percent",
        maximumFractionDigits: digits,
        minimumFractionDigits: digits,
      }).format(n)
    : "Unavailable";
export const num = (n, digits = 2) =>
  Number.isFinite(n)
    ? n.toLocaleString("en-US", {
        maximumFractionDigits: digits,
        minimumFractionDigits: digits,
      })
    : "Unavailable";
export const date = (value) =>
  value
    ? new Intl.DateTimeFormat("en-US", {
        month: "short",
        day: "numeric",
        year: "numeric",
        timeZone: "UTC",
      }).format(new Date(value))
    : "Unavailable";
export const colors = [
  "var(--chart-1)",
  "var(--chart-2)",
  "var(--chart-3)",
  "var(--chart-4)",
  "var(--chart-5)",
];

async function read(url, signal) {
  const response = await fetch(url, { signal, cache: "no-cache" });
  if (!response.headers.get("content-type")?.includes("application/json"))
    throw new Error(
      "The data service is unavailable. Showing the exported snapshot.",
    );
  const data = await response.json();
  if (!response.ok)
    throw Object.assign(new Error(data.error || "Connection unavailable"), {
      retryAt: data.retryAt,
    });
  return data;
}

export function useProjectData() {
  const [state, setState] = useState(() => ({
    status: globalThis.__PROJECT_SNAPSHOT__ ? "snapshot" : "loading",
    data: globalThis.__PROJECT_SNAPSHOT__ || null,
    error: null,
  }));
  useEffect(() => {
    const controller = new AbortController();
    let timer,
      current,
      stopped = false,
      refreshing = false;
    const signal = controller.signal;
    async function refresh() {
      if (refreshing || stopped) return;
      refreshing = true;
      let delay = 25_000;
      try {
        if (!navigator.onLine)
          throw new Error("Offline. Last successful export retained.");
        if (document.hidden) return;
        const live = await read("/api/live", signal);
        if (live.status !== "live")
          throw new Error(live.error || "Live connection unavailable");
        if (live.manifest.version !== current.manifest.version) {
          const values = await Promise.all(
            ["results", "history"].map((file) =>
              read(`/api/live?file=${file}&ref=${live.head}`, signal),
            ),
          );
          current = {
            manifest: live.manifest,
            results: values[0],
            history: values[1],
          };
        }
        setState({
          status: "live",
          data: current,
          activity: live.activity,
          head: live.head,
          checkedAt: live.checkedAt,
          error: null,
        });
      } catch (error) {
        if (!stopped)
          setState((s) => ({
            ...s,
            status: navigator.onLine ? "snapshot" : "offline",
            error: error.message,
          }));
        delay = Math.max(60_000, (error.retryAt || 0) - Date.now());
      } finally {
        refreshing = false;
        if (!stopped) timer = setTimeout(refresh, delay);
      }
    }
    async function start() {
      try {
        const values = await Promise.all(
          ["manifest", "results", "history"].map((file) =>
            read(`/data/${file}.json`, signal),
          ),
        );
        current = {
          manifest: values[0],
          results: values[1],
          history: values[2],
        };
        if (stopped) return;
        setState({ status: "snapshot", data: current, error: null });
        await refresh();
      } catch (error) {
        if (!stopped)
          setState((s) => ({
            ...s,
            status: s.data ? "snapshot" : "error",
            error: error.message,
          }));
      }
    }
    const wake = () => {
      clearTimeout(timer);
      if (current) refresh();
    };
    start();
    window.addEventListener("online", wake);
    window.addEventListener("offline", wake);
    document.addEventListener("visibilitychange", wake);
    return () => {
      stopped = true;
      clearTimeout(timer);
      controller.abort();
      window.removeEventListener("online", wake);
      window.removeEventListener("offline", wake);
      document.removeEventListener("visibilitychange", wake);
    };
  }, []);
  return state;
}
