import { useEffect, useMemo, useState } from "react";

function cn(...xs) {
  return xs.filter(Boolean).join(" ");
}

export default function App() {
  const [version, setVersion] = useState("v1");
  const [data, setData] = useState(null);
  const [err, setErr] = useState("");
  const [loading, setLoading] = useState(false);

  const endpoint = useMemo(() => `/api/${version}/hello`, [version]);

  const load = async () => {
    setLoading(true);
    setErr("");
    try {
      const r = await fetch(endpoint);
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      setData(await r.json());
    } catch (e) {
      setData(null);
      setErr(String(e));
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    load();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [endpoint]);

  return (
    <div className="min-h-screen overflow-hidden">
      <style>{`
        @keyframes gradientShift {
          0% { background-position: 0% 50%; }
          50% { background-position: 100% 50%; }
          100% { background-position: 0% 50%; }
        }
        @keyframes floatBlob {
          0% { transform: translate(0px, 0px) scale(1); }
          33% { transform: translate(38px, -28px) scale(1.06); }
          66% { transform: translate(-26px, 22px) scale(0.98); }
          100% { transform: translate(0px, 0px) scale(1); }
        }
        @keyframes shimmer {
          0% { background-position: 200% 0; }
          100% { background-position: -200% 0; }
        }
        @keyframes popIn {
          0% { transform: translateY(12px); opacity: 0; }
          100% { transform: translateY(0px); opacity: 1; }
        }
      `}</style>

      {/* Animated gradient background */}
      <div
        className="fixed inset-0 -z-10"
        style={{
          background:
            "linear-gradient(120deg, #60a5fa, #a78bfa, #fb7185, #34d399, #fbbf24)",
          backgroundSize: "320% 320%",
          animation: "gradientShift 12s ease-in-out infinite",
        }}
      />

      {/* Floating color blobs */}
      <div className="pointer-events-none fixed inset-0 -z-10">
        <div
          className="absolute -top-24 -left-24 h-[360px] w-[360px] rounded-full blur-3xl opacity-55"
          style={{
            background:
              "radial-gradient(circle at 30% 30%, rgba(255,255,255,.95), rgba(96,165,250,.9) 55%, rgba(255,255,255,0) 72%)",
            animation: "floatBlob 10s ease-in-out infinite",
          }}
        />
        <div
          className="absolute top-8 -right-32 h-[460px] w-[460px] rounded-full blur-3xl opacity-50"
          style={{
            background:
              "radial-gradient(circle at 35% 35%, rgba(255,255,255,.95), rgba(251,113,133,.9) 55%, rgba(255,255,255,0) 72%)",
            animation: "floatBlob 12s ease-in-out infinite",
          }}
        />
        <div
          className="absolute -bottom-44 left-1/3 h-[520px] w-[520px] rounded-full blur-3xl opacity-45"
          style={{
            background:
              "radial-gradient(circle at 35% 35%, rgba(255,255,255,.95), rgba(52,211,153,.9) 55%, rgba(255,255,255,0) 72%)",
            animation: "floatBlob 14s ease-in-out infinite",
          }}
        />
      </div>

      {/* A subtle “scrim” behind text to improve contrast */}
      <div className="pointer-events-none fixed inset-0 -z-10 bg-black/10" />

      <div className="mx-auto flex min-h-screen max-w-3xl flex-col justify-center px-6 py-10">
        {/* Top pill (no 4dt907) */}
        <div className="inline-flex w-fit items-center gap-2 rounded-full border border-white/40 bg-white/35 px-3 py-1 text-xs font-medium text-slate-900 shadow-sm backdrop-blur-xl">
          <span className="h-2 w-2 rounded-full bg-emerald-500 shadow-[0_0_12px_rgba(16,185,129,.55)]" />
          Full-Stack Demo • Frontend + Backend
        </div>

        <h1 className="mt-5 text-4xl font-semibold tracking-tight text-slate-950 drop-shadow-[0_2px_12px_rgba(255,255,255,.25)]">
          Hello World
        </h1>

        <p className="mt-2 text-slate-950/80">
          React UI calling{" "}
          <code className="rounded-lg border border-white/50 bg-white/45 px-2 py-0.5 font-mono text-[0.95em] text-slate-950 shadow-sm backdrop-blur-xl">
            {endpoint}
          </code>
        </p>

        {/* controls */}
        <div className="mt-6 flex flex-wrap items-center gap-3">
          <div className="inline-flex rounded-2xl border border-white/45 bg-white/30 p-1 shadow-sm backdrop-blur-xl">
            {["v1", "v2"].map((v) => (
              <button
                key={v}
                onClick={() => setVersion(v)}
                className={cn(
                  "rounded-xl px-4 py-2 text-sm font-semibold transition",
                  v === version
                    ? "bg-white/80 text-slate-950 shadow"
                    : "text-slate-950/80 hover:bg-white/35",
                )}
              >
                {v.toUpperCase()}
              </button>
            ))}
          </div>

          <button
            onClick={load}
            disabled={loading}
            className="rounded-2xl border border-white/45 bg-white/30 px-4 py-2 text-sm font-semibold text-slate-950 shadow-sm backdrop-blur-xl transition hover:bg-white/40 disabled:opacity-60"
          >
            {loading ? "Refreshing…" : "Refresh"}
          </button>

          <span
            className={cn(
              "ml-auto rounded-full border px-3 py-1 text-xs font-semibold shadow-sm backdrop-blur-xl",
              loading
                ? "border-white/45 bg-white/30 text-slate-950"
                : err
                  ? "border-white/45 bg-rose-500/30 text-slate-950"
                  : "border-white/45 bg-emerald-500/25 text-slate-950",
            )}
          >
            {loading ? "loading" : err ? "error" : "ok"}
          </span>
        </div>

        {/* glass card */}
        <div
          className="relative mt-7 overflow-hidden rounded-[28px] border border-white/45 bg-white/25 shadow-[0_28px_80px_rgba(0,0,0,.20)] backdrop-blur-2xl"
          style={{ animation: "popIn 420ms ease-out both" }}
        >
          {/* Apple-ish glass reflections */}
          <div className="pointer-events-none absolute inset-0">
            {/* top soft highlight */}
            <div className="absolute -top-24 left-0 right-0 h-56 bg-gradient-to-b from-white/45 via-white/15 to-transparent" />
            {/* diagonal sheen */}
            <div
              className="absolute -left-40 top-10 h-72 w-[520px] rotate-12"
              style={{
                background:
                  "linear-gradient(90deg, transparent 0%, rgba(255,255,255,.18) 35%, rgba(255,255,255,.40) 50%, rgba(255,255,255,.18) 65%, transparent 100%)",
                filter: "blur(1px)",
                opacity: 0.9,
              }}
            />
            {/* inner glow edge */}
            <div className="absolute inset-0 rounded-[28px] shadow-[inset_0_1px_0_rgba(255,255,255,.55),inset_0_-1px_0_rgba(255,255,255,.18)]" />
          </div>

          <div className="relative border-b border-white/35 px-6 py-4">
            <div className="text-xs font-semibold text-slate-950/85">
              Response
            </div>
            <div className="text-xs text-slate-950/70">
              Version:{" "}
              <span className="font-semibold text-slate-950">{version}</span>
            </div>
          </div>

          <div className="relative px-6 py-6">
            {loading && (
              <div className="space-y-3">
                {["w-2/3", "w-5/6", "w-1/2"].map((w) => (
                  <div
                    key={w}
                    className={cn("h-4 rounded-xl", w)}
                    style={{
                      backgroundImage:
                        "linear-gradient(90deg, rgba(255,255,255,.25), rgba(255,255,255,.60), rgba(255,255,255,.25))",
                      backgroundSize: "200% 100%",
                      animation: "shimmer 1.1s linear infinite",
                    }}
                  />
                ))}
              </div>
            )}

            {!loading && err && (
              <div className="rounded-2xl border border-white/45 bg-rose-500/25 p-4 text-sm text-slate-950">
                <div className="font-semibold">Couldn’t reach backend</div>
                <div className="mt-1 opacity-90">{err}</div>
              </div>
            )}

            {!loading && !err && data && (
              <div className="space-y-4">
                <div className="flex items-center gap-3 text-xl text-slate-950">
                  <span className="h-3 w-3 rounded-full bg-yellow-300 shadow-[0_0_18px_rgba(253,224,71,.65)]" />
                  <span className="font-semibold">{data.message}</span>
                </div>

                <pre className="overflow-auto rounded-2xl border border-white/40 bg-white/20 p-4 text-sm text-slate-950 shadow-sm backdrop-blur-xl">
                  {JSON.stringify(data, null, 2)}
                </pre>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
