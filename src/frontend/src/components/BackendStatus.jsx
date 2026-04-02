// BackendStatus: probes the backend /health endpoint and displays a pill
// in the navbar indicating whether the Render service is online, sleeping
// (spun down), or currently waking up.
//
// Render free-tier services spin down after ~15 min of inactivity.  The first
// request after spin-down can take 30-60 seconds.  This component gives users
// a clear, actionable status so they know why responses might be slow.

import { useEffect, useRef, useState } from "react";
import { apiUrl } from "../apiBase";

// How long to wait for a single health-check request before giving up (ms).
const CHECK_TIMEOUT_MS = 8_000;

// Interval between background health checks when the backend is known online (ms).
const POLL_ONLINE_MS = 60_000;

// Interval between retries while the backend is waking up (ms).
const POLL_WAKING_MS = 5_000;

/**
 * Fetch /health with an AbortController timeout.
 * Resolves `true` when the endpoint returns HTTP 200, `false` otherwise.
 */
async function pingHealth() {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), CHECK_TIMEOUT_MS);
    try {
        const res = await fetch(apiUrl("/health"), {
            signal: controller.signal,
            cache: "no-store",
        });
        return res.ok;
    } catch {
        return false;
    } finally {
        clearTimeout(timer);
    }
}

/**
 * BackendStatus pill component.
 *
 * Renders in four states:
 *  • checking — grey pulsing dot while the initial probe is in flight
 *  • online   — green dot + "Backend online"
 *  • waking   — amber spinner + "Waking up…"
 *  • sleeping — red dot + "Backend sleeping" + "Wake up" button
 */
export default function BackendStatus() {
    // "checking" | "online" | "waking" | "sleeping"
    const [backendState, setBackendState] = useState("checking");

    // Refs so async callbacks never capture stale state.
    const mountedRef = useRef(true);
    const pollTimerRef = useRef(null);
    // Expose wakeUp handler from the main effect so the button can call it.
    const wakeUpRef = useRef(null);

    // -----------------------------------------------------------------------
    // Main polling effect
    // -----------------------------------------------------------------------
    useEffect(() => {
        mountedRef.current = true;

        function clearPoll() {
            clearTimeout(pollTimerRef.current);
            pollTimerRef.current = null;
        }

        function scheduleNext(delayMs, nextState) {
            clearPoll();
            pollTimerRef.current = setTimeout(async () => {
                if (!mountedRef.current) return;
                const ok = await pingHealth();
                if (!mountedRef.current) return;
                if (ok) {
                    setBackendState("online");
                    scheduleNext(POLL_ONLINE_MS, "online");
                } else if (nextState === "waking") {
                    scheduleNext(POLL_WAKING_MS, "waking");
                } else {
                    setBackendState("sleeping");
                    scheduleNext(POLL_ONLINE_MS, "sleeping");
                }
            }, delayMs);
        }

        async function wakeUp() {
            if (!mountedRef.current) return;
            clearPoll();
            setBackendState("waking");
            const ok = await pingHealth();
            if (!mountedRef.current) return;
            if (ok) {
                setBackendState("online");
                scheduleNext(POLL_ONLINE_MS, "online");
            } else {
                scheduleNext(POLL_WAKING_MS, "waking");
            }
        }

        // Expose so the button JSX can trigger a wake-up.
        wakeUpRef.current = wakeUp;

        // Initial check — slight delay to avoid blocking first paint.
        pollTimerRef.current = setTimeout(async () => {
            if (!mountedRef.current) return;
            const ok = await pingHealth();
            if (!mountedRef.current) return;
            if (ok) {
                setBackendState("online");
                scheduleNext(POLL_ONLINE_MS, "online");
            } else {
                setBackendState("sleeping");
                scheduleNext(POLL_ONLINE_MS, "sleeping");
            }
        }, 500);

        return () => {
            mountedRef.current = false;
            clearPoll();
        };
    }, []);

    // -----------------------------------------------------------------------
    // Render
    // -----------------------------------------------------------------------

    if (backendState === "checking") {
        return (
            <span className="flex items-center gap-1.5 text-xs text-slate-400 px-3 py-1">
                <span className="inline-block w-2 h-2 rounded-full bg-slate-500 animate-pulse" />
                Checking backend…
            </span>
        );
    }

    if (backendState === "online") {
        return (
            <span className="flex items-center gap-1.5 text-xs text-emerald-400 px-3 py-1">
                <span className="inline-block w-2 h-2 rounded-full bg-emerald-400" />
                Backend online
            </span>
        );
    }

    if (backendState === "waking") {
        return (
            <span className="flex items-center gap-1.5 text-xs text-amber-400 px-3 py-1">
                <svg
                    className="animate-spin w-3 h-3"
                    xmlns="http://www.w3.org/2000/svg"
                    fill="none"
                    viewBox="0 0 24 24"
                >
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8v8H4z" />
                </svg>
                Waking up… (may take ~30s)
            </span>
        );
    }

    // backendState === "sleeping"
    return (
        <span className="flex items-center gap-2 text-xs px-3 py-1">
            <span className="flex items-center gap-1.5 text-slate-400">
                <span className="inline-block w-2 h-2 rounded-full bg-red-500" />
                Backend sleeping
            </span>
            <button
                onClick={() => wakeUpRef.current?.()}
                className="px-2 py-0.5 rounded bg-amber-600 hover:bg-amber-500 text-white font-semibold transition text-xs"
            >
                Wake up
            </button>
        </span>
    );
}


