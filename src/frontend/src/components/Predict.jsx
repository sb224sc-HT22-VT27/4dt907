// src/frontend/src/components/Predict.jsx

import { useEffect, useMemo, useState } from "react";
import FeatureBuilder from "./FeatureBuilder";
import { FEATURE_GROUPS, EXAMPLE_41 } from "../featuresSchema";

function parseFeatures(input) {
  const parts = input
    .split(",")
    .map((s) => s.trim())
    .filter(Boolean);

  if (parts.length === 0)
    return { ok: false, error: "Paste comma-separated numbers." };

  const nums = parts.map(Number);
  if (nums.some((n) => Number.isNaN(n))) {
    return { ok: false, error: "Use comma-separated numbers only." };
  }
  return { ok: true, value: nums };
}

function buildOrderedFeatureNames() {
  const names = [];
  for (const g of FEATURE_GROUPS) {
    if (g.single) {
      names.push(g.single);
      continue;
    }
    const start = g.startIndex ?? 1;
    for (let i = 0; i < g.count; i++) {
      const idx = start + i;
      names.push(`${g.prefix}${idx}${g.suffix}`);
    }
  }
  return names;
}

const ORDERED_NAMES = buildOrderedFeatureNames();
const TOTAL_FEATURES = ORDERED_NAMES.length;

function fromArray(arr) {
  const obj = {};
  ORDERED_NAMES.forEach((n, i) => (obj[n] = Number(arr?.[i] ?? 0)));
  return obj;
}

function toOrderedArray(valuesObj) {
  return ORDERED_NAMES.map((n) => Number(valuesObj?.[n] ?? 0));
}

export default function Predict() {
  const [task, setTask] = useState("score");
  // Default to prod
  const [variant, setVariant] = useState("champion");

  // IMPORTANT: no hard-coded “expected” fallback — rely purely on backend
  const [expectedCount, setExpectedCount] = useState(null); // number | null
  const [expectedLoading, setExpectedLoading] = useState(true);
  const [modelInfoError, setModelInfoError] = useState("");

  const hasExpected = Number.isFinite(expectedCount);
  const activeExpected = hasExpected ? expectedCount : null;

  const [featureValues, setFeatureValues] = useState(() => fromArray(EXAMPLE_41));

  const [importOpen, setImportOpen] = useState(false);
  const [importText, setImportText] = useState("");

  const [hint, setHint] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  const endpoint = useMemo(() => {
    return task === "weakest"
      ? `/api/v1/weakest-link/${variant}`
      : `/api/v1/predict/${variant}`;
  }, [task, variant]);

  const modelInfoUrl = useMemo(() => {
    return task === "weakest"
      ? `/api/v1/model-info/weakest-link/${variant}`
      : `/api/v1/model-info/${variant}`;
  }, [task, variant]);

  // Load expected feature count from backend ONLY (no cache, no fallback)
  useEffect(() => {
    const ctrl = new AbortController();
    let cancelled = false;

    setExpectedLoading(true);
    setExpectedCount(null);
    setModelInfoError("");

    async function loadModelInfo() {
      try {
        const res = await fetch(modelInfoUrl, {
          signal: ctrl.signal,
          cache: "no-store",
          headers: {
            "Cache-Control": "no-cache",
            Pragma: "no-cache",
          },
        });

        const data = await res.json().catch(() => ({}));

        const count =
          res.ok && typeof data?.expected_features === "number"
            ? data.expected_features
            : null;

        if (!cancelled) {
          setExpectedCount(count);
          if (!res.ok || typeof count !== "number") {
            setModelInfoError(
              data?.detail ||
                "Could not load model info (expected_features missing)."
            );
          }
        }
      } catch (e) {
        if (!cancelled) {
          setExpectedCount(null);
          setModelInfoError(String(e));
        }
      } finally {
        if (!cancelled) setExpectedLoading(false);
      }
    }

    loadModelInfo();

    return () => {
      cancelled = true;
      ctrl.abort();
    };
  }, [modelInfoUrl]);

  const flashHint = (msg) => {
    setHint(msg);
    window.clearTimeout(flashHint._t);
    flashHint._t = window.setTimeout(() => setHint(""), 1200);
  };

  // Modal behavior
  useEffect(() => {
    if (!importOpen) return;

    const onKeyDown = (e) => {
      if (e.key === "Escape") setImportOpen(false);
    };

    document.addEventListener("keydown", onKeyDown);
    const prevOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";

    return () => {
      document.removeEventListener("keydown", onKeyDown);
      document.body.style.overflow = prevOverflow;
    };
  }, [importOpen]);

  const controlsDisabled = loading || expectedLoading || !hasExpected;

  function loadExample() {
    setError("");
    setResult(null);

    if (!hasExpected) {
      setError("Model info is still loading — try again in a moment.");
      return;
    }

    // Keep internal state size stable (TOTAL_FEATURES) but only “use” activeExpected when sending
    const slice = EXAMPLE_41.slice(0, activeExpected);
    const padded = slice.concat(new Array(TOTAL_FEATURES - slice.length).fill(0));
    setFeatureValues(fromArray(padded));

    flashHint("Example loaded.");
  }

  function zeroAll() {
    setError("");
    setResult(null);

    // Zeroing doesn't need expectedCount; it just sets all known UI fields to 0
    const zeros = new Array(TOTAL_FEATURES).fill(0);
    setFeatureValues(fromArray(zeros));
    flashHint("All values set to 0.");
  }

  async function copyValues() {
    setError("");
    if (!hasExpected) {
      setError("Model info is still loading — try again in a moment.");
      return;
    }

    try {
      const text = toOrderedArray(featureValues)
        .slice(0, activeExpected)
        .join(", ");
      await navigator.clipboard.writeText(text);
      flashHint("Copied.");
    } catch {
      setImportText(
        toOrderedArray(featureValues).slice(0, activeExpected).join(", ")
      );
      setImportOpen(true);
      flashHint("Clipboard blocked — copy from the box.");
    }
  }

  async function pasteValues() {
    setError("");
    if (!hasExpected) {
      setError("Model info is still loading — try again in a moment.");
      return;
    }

    setImportOpen(true);
    try {
      const text = await navigator.clipboard.readText();
      setImportText(text);
      flashHint("Pasted. Click Apply.");
    } catch {
      flashHint("Paste into the box.");
    }
  }

  function openImport() {
    setError("");
    if (!hasExpected) {
      setError("Model info is still loading — try again in a moment.");
      return;
    }
    setImportOpen(true);
    flashHint("Paste values and click Apply.");
  }

  function applyImport() {
    setError("");
    if (!hasExpected) {
      setError("Model info is still loading — try again in a moment.");
      return;
    }

    const parsed = parseFeatures(importText);
    if (!parsed.ok) return setError(parsed.error);

    // Allow pasting more than expected: take the first activeExpected values.
    let values = parsed.value;
    if (values.length > activeExpected) values = values.slice(0, activeExpected);

    if (values.length !== activeExpected) {
      return setError(
        `Expected ${activeExpected} features, but got ${values.length}.`
      );
    }

    // Store in TOTAL_FEATURES-sized structure for UI
    const padded = values.concat(new Array(TOTAL_FEATURES - values.length).fill(0));
    setFeatureValues(fromArray(padded));

    setResult(null);
    setImportOpen(false);
    flashHint("Imported.");
  }

  async function onPredict() {
    setError("");
    setResult(null);

    if (!hasExpected) {
      setError("Model info is still loading — try again in a moment.");
      return;
    }

    const features = toOrderedArray(featureValues).slice(0, activeExpected);

    if (features.length !== activeExpected) {
      return setError(
        `Expected ${activeExpected} features, but got ${features.length}.`
      );
    }

    setLoading(true);
    try {
      const res = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ features }),
      });

      const data = await res.json().catch(() => ({}));
      if (!res.ok)
        return setError(data?.detail || `Request failed (${res.status})`);

      setResult(data);
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="ios-bg min-h-screen">
      <div className="mx-auto max-w-5xl px-4 py-10">
        <div className="ios-card rounded-[28px]">
          <div className="p-6 md:p-8">
            <div className="flex flex-col gap-2 md:flex-row md:items-end md:justify-between">
              <div>
                <h1 className="text-4xl font-semibold tracking-tight text-slate-900">
                  Model Prediction
                </h1>
                <p className="mt-2 text-slate-600">
                  Adjust values. Predict. Clean and simple.
                </p>
              </div>

              <div className="text-sm text-slate-600">
                Expects{" "}
                <span className="font-semibold text-slate-900">
                  {hasExpected ? activeExpected : "…"}
                </span>{" "}
                features
                {expectedLoading ? (
                  <span className="ml-2 text-slate-400">(loading)</span>
                ) : null}
              </div>
            </div>

            {/* Controls */}
            <div className="mt-6 mx-auto max-w-4xl">
              <div className="flex flex-col gap-3 md:flex-row md:items-center md:gap-6">
                <div className="w-full md:w-[420px]">
                  <div className="text-sm text-slate-700 mb-2">Task</div>
                  <select
                    value={task}
                    onChange={(e) => {
                      setError("");
                      setResult(null);
                      setTask(e.target.value);
                    }}
                    className="ios-input w-full rounded-2xl px-4 py-3 text-slate-900 outline-none focus:ring-4 focus:ring-sky-200/50"
                  >
                    <option value="score">Score prediction (A2)</option>
                    <option value="weakest">
                      Weakest link classification (A3)
                    </option>
                  </select>
                </div>

                <div className="w-full md:w-[420px]">
                  <div className="text-sm text-slate-700 mb-2">Model</div>
                  <select
                    value={variant}
                    onChange={(e) => {
                      setError("");
                      setResult(null);
                      setVariant(e.target.value);
                    }}
                    className="ios-input w-full rounded-2xl px-4 py-3 text-slate-900 outline-none focus:ring-4 focus:ring-sky-200/50"
                  >
                    <option value="champion">Champion (prod)</option>
                    <option value="latest">Latest (dev)</option>
                  </select>
                </div>

                <div className="flex flex-wrap gap-2 md:ml-auto">
                  <button
                    type="button"
                    onClick={loadExample}
                    disabled={controlsDisabled}
                    className="ios-btn rounded-full px-4 py-3 text-sm font-medium text-slate-800 disabled:opacity-50"
                  >
                    Example
                  </button>
                  <button
                    type="button"
                    onClick={zeroAll}
                    className="ios-btn rounded-full px-4 py-3 text-sm font-medium text-slate-800"
                  >
                    Zero
                  </button>
                  <button
                    type="button"
                    onClick={copyValues}
                    disabled={controlsDisabled}
                    className="ios-btn rounded-full px-4 py-3 text-sm font-medium text-slate-800 disabled:opacity-50"
                  >
                    Copy
                  </button>
                  <button
                    type="button"
                    onClick={openImport}
                    disabled={controlsDisabled}
                    className="ios-btn rounded-full px-4 py-3 text-sm font-medium text-slate-800 disabled:opacity-50"
                  >
                    Import
                  </button>

                  <button
                    onClick={onPredict}
                    disabled={controlsDisabled}
                    className="ios-btn ios-btn-primary rounded-full px-5 py-3 text-sm font-semibold disabled:opacity-50"
                  >
                    {loading
                      ? "Predicting…"
                      : expectedLoading
                      ? "Loading model…"
                      : "Predict"}
                  </button>
                </div>
              </div>

              <div className="mt-2 h-5 text-xs text-slate-500">{hint || ""}</div>
            </div>

            {/* Model-info error (separate from prediction errors) */}
            {modelInfoError ? (
              <div className="mt-4 rounded-2xl border border-amber-200 bg-amber-50 p-4 text-sm text-amber-900">
                <b>Model info warning:</b> {modelInfoError}
              </div>
            ) : null}

            {/* Builder */}
            <div className="mt-6">
              {hasExpected ? (
                <FeatureBuilder
                  values={featureValues}
                  setValues={setFeatureValues}
                  maxFeatures={activeExpected}
                />
              ) : (
                <div className="ios-card rounded-[28px] p-5 text-sm text-slate-600">
                  Loading model info…
                </div>
              )}
            </div>

            <div className="mt-4 text-xs text-slate-500">
              Endpoint: <code className="text-slate-700">{endpoint}</code>
            </div>

            {error ? (
              <div className="mt-4 rounded-2xl border border-red-200 bg-red-50 p-4 text-sm text-red-800">
                <b>Error:</b> {error}
              </div>
            ) : null}

            {result ? (
              <div className="mt-4 ios-pill rounded-[28px] p-4 text-sm text-slate-800">
                <div>
                  <span className="text-slate-500">Prediction:</span>{" "}
                  <code className="font-semibold">
                    {String(result?.prediction ?? "—")}
                  </code>
                </div>
                <div className="mt-2">
                  <span className="text-slate-500">Model URI:</span>{" "}
                  <code className="break-all">{result?.model_uri ?? "—"}</code>
                </div>
                <div className="mt-2">
                  <span className="text-slate-500">Run ID:</span>{" "}
                  <code className="break-all">{result?.run_id ?? "—"}</code>
                </div>
              </div>
            ) : null}
          </div>
        </div>
      </div>

      {/* Import Modal */}
      {importOpen ? (
        <div className="fixed inset-0 z-50">
          <div
            className="absolute inset-0 bg-slate-900/10 backdrop-blur-[2px]"
            onClick={() => setImportOpen(false)}
          />
          <div className="relative mx-auto mt-20 w-[min(92vw,720px)] px-4">
            <div className="ios-card rounded-[28px] p-5">
              <div className="flex items-start justify-between gap-4">
                <div>
                  <div className="text-lg font-semibold text-slate-900">
                    Import values
                  </div>
                  <div className="text-sm text-slate-600">
                    {hasExpected
                      ? `Paste ${activeExpected} comma-separated numbers.`
                      : "Loading model info…"}
                  </div>
                </div>

                <button
                  type="button"
                  onClick={() => setImportOpen(false)}
                  className="ios-btn rounded-full px-4 py-2 text-sm font-medium text-slate-800"
                >
                  Close
                </button>
              </div>

              <textarea
                value={importText}
                onChange={(e) => setImportText(e.target.value)}
                rows={6}
                placeholder="0.1, 0.2, 0.3, ..."
                className="ios-input mt-4 w-full rounded-2xl px-4 py-3 text-slate-900 outline-none focus:ring-4 focus:ring-sky-200/50"
              />

              <div className="mt-4 flex flex-wrap gap-2">
                <button
                  type="button"
                  onClick={applyImport}
                  disabled={!hasExpected}
                  className="ios-btn ios-btn-primary rounded-full px-5 py-2 text-sm font-semibold disabled:opacity-50"
                >
                  Apply
                </button>
                <button
                  type="button"
                  onClick={() => {
                    if (!hasExpected) return;
                    setImportText(
                      toOrderedArray(featureValues)
                        .slice(0, activeExpected)
                        .join(", ")
                    );
                  }}
                  disabled={!hasExpected}
                  className="ios-btn rounded-full px-4 py-2 text-sm font-medium text-slate-800 disabled:opacity-50"
                >
                  Fill current
                </button>
                <button
                  type="button"
                  onClick={pasteValues}
                  disabled={!hasExpected}
                  className="ios-btn rounded-full px-4 py-2 text-sm font-medium text-slate-800 disabled:opacity-50"
                >
                  Paste
                </button>
              </div>

              <div className="mt-3 text-xs text-slate-500">
                Tip: press <span className="font-semibold">Esc</span> to close.
              </div>
            </div>
          </div>
        </div>
      ) : null}
    </div>
  );
}
