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
const DEFAULT_EXPECTED = ORDERED_NAMES.length;

// Cache expected feature counts per model-info URL (prevents repeated loading)
const expectedCountCache = new Map();

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

  const [expectedCount, setExpectedCount] = useState(DEFAULT_EXPECTED);
  const activeExpected = expectedCount ?? DEFAULT_EXPECTED;

  const [featureValues, setFeatureValues] = useState(() =>
    fromArray(EXAMPLE_41),
  );

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

  // Load expected feature count (cached) and update on Task/Model change
  useEffect(() => {
    let cancelled = false;

    const cached = expectedCountCache.get(modelInfoUrl);
    if (typeof cached === "number") {
      setExpectedCount(cached);
      return () => {
        cancelled = true;
      };
    }

    // Avoid showing stale count from previous selection while we load.
    setExpectedCount(DEFAULT_EXPECTED);

    async function loadModelInfo() {
      try {
        const res = await fetch(modelInfoUrl);
        const data = await res.json().catch(() => ({}));

        const count =
          res.ok && typeof data?.expected_features === "number"
            ? data.expected_features
            : null;

        if (!cancelled) {
          if (typeof count === "number") {
            expectedCountCache.set(modelInfoUrl, count);
            setExpectedCount(count);
          } else {
            setExpectedCount(DEFAULT_EXPECTED);
          }
        }
      } catch {
        if (!cancelled) setExpectedCount(DEFAULT_EXPECTED);
      }
    }

    loadModelInfo();
    return () => {
      cancelled = true;
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

  function loadExample() {
    setError("");
    setResult(null);
    setFeatureValues(fromArray(EXAMPLE_41));
    flashHint("Example loaded.");
  }

  function zeroAll() {
    setError("");
    setResult(null);
    const zeros = new Array(ORDERED_NAMES.length).fill(0);
    setFeatureValues(fromArray(zeros));
    flashHint("All values set to 0.");
  }

  async function copyValues() {
    setError("");
    try {
      const text = toOrderedArray(featureValues)
        .slice(0, activeExpected)
        .join(", ");
      await navigator.clipboard.writeText(text);
      flashHint("Copied.");
    } catch {
      setImportText(
        toOrderedArray(featureValues).slice(0, activeExpected).join(", "),
      );
      setImportOpen(true);
      flashHint("Clipboard blocked — copy from the box.");
    }
  }

  async function pasteValues() {
    setError("");
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
    setImportOpen(true);
    flashHint("Paste values and click Apply.");
  }

  function applyImport() {
    setError("");
    const parsed = parseFeatures(importText);
    if (!parsed.ok) return setError(parsed.error);

    // Allow pasting more than expected: take the first activeExpected values.
    let values = parsed.value;
    if (values.length > activeExpected)
      values = values.slice(0, activeExpected);

    if (values.length !== activeExpected) {
      return setError(
        `Expected ${activeExpected} features, but got ${values.length}.`,
      );
    }

    setFeatureValues(fromArray(values));
    setResult(null);
    setImportOpen(false);
    flashHint("Imported.");
  }

  async function onPredict() {
    setError("");
    setResult(null);

    const features = toOrderedArray(featureValues).slice(0, activeExpected);

    if (features.length !== activeExpected) {
      return setError(
        `Expected ${activeExpected} features, but got ${features.length}.`,
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
                  {activeExpected}
                </span>{" "}
                features
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
                    className="ios-btn rounded-full px-4 py-3 text-sm font-medium text-slate-800"
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
                    className="ios-btn rounded-full px-4 py-3 text-sm font-medium text-slate-800"
                  >
                    Copy
                  </button>
                  <button
                    type="button"
                    onClick={openImport}
                    className="ios-btn rounded-full px-4 py-3 text-sm font-medium text-slate-800"
                  >
                    Import
                  </button>

                  <button
                    onClick={onPredict}
                    disabled={loading}
                    className="ios-btn ios-btn-primary rounded-full px-5 py-3 text-sm font-semibold"
                  >
                    {loading ? "Predicting…" : "Predict"}
                  </button>
                </div>
              </div>

              <div className="mt-2 h-5 text-xs text-slate-500">
                {hint || ""}
              </div>
            </div>

            {/* Builder */}
            <div className="mt-6">
              <FeatureBuilder
                values={featureValues}
                setValues={setFeatureValues}
                maxFeatures={activeExpected}
              />
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
                    Paste {activeExpected} comma-separated numbers.
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
                  className="ios-btn ios-btn-primary rounded-full px-5 py-2 text-sm font-semibold"
                >
                  Apply
                </button>
                <button
                  type="button"
                  onClick={() => {
                    setImportText(
                      toOrderedArray(featureValues)
                        .slice(0, activeExpected)
                        .join(", "),
                    );
                  }}
                  className="ios-btn rounded-full px-4 py-2 text-sm font-medium text-slate-800"
                >
                  Fill current
                </button>
                <button
                  type="button"
                  onClick={pasteValues}
                  className="ios-btn rounded-full px-4 py-2 text-sm font-medium text-slate-800"
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
