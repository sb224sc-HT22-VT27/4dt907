import React, { useMemo, useState } from "react";

function parseFeatures(input) {
  const parts = input.split(",").map((s) => s.trim()).filter(Boolean);
  if (parts.length === 0) return { ok: false, error: "Enter at least one number." };

  const nums = parts.map(Number);
  if (nums.some((n) => Number.isNaN(n))) return { ok: false, error: "Use comma-separated numbers only." };
  return { ok: true, value: nums };
}

export default function Predict() {
  const [variant, setVariant] = useState("latest");
  const [featuresText, setFeaturesText] = useState("1, 1");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  const endpoint = useMemo(
    () => (variant === "latest" ? "/api/v1/predict/latest" : "/api/v1/predict/champion"),
    [variant]
  );

  async function onPredict() {
    setError("");
    setResult(null);

    const parsed = parseFeatures(featuresText);
    if (!parsed.ok) return setError(parsed.error);

    setLoading(true);
    try {
      const res = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ features: parsed.value }),
      });

      const data = await res.json().catch(() => ({}));
      if (!res.ok) return setError(data?.detail || `Request failed (${res.status})`);
      setResult(data);
    } catch (e) {
      setError(String(e));
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="mx-auto max-w-2xl px-4 py-14">
      <div className="rounded-3xl border border-slate-200 bg-white/70 backdrop-blur-md shadow-xl shadow-slate-200/60 transition hover:shadow-2xl">
        <div className="p-6 md:p-8">
          <h1 className="text-3xl font-semibold tracking-tight text-slate-900">Model Prediction</h1>
          <p className="mt-2 text-slate-600">
            Choose <span className="font-semibold text-slate-900">Champion</span> or{" "}
            <span className="font-semibold text-slate-900">Latest</span>, enter feature values, and get a prediction.
          </p>

          <div className="mt-8 grid gap-4">
            <label className="grid gap-2">
              <span className="text-sm text-slate-700">Model</span>
              <select
                value={variant}
                onChange={(e) => setVariant(e.target.value)}
                className="rounded-2xl border border-slate-200 bg-white px-4 py-3 text-slate-900 outline-none transition focus:border-sky-300 focus:ring-4 focus:ring-sky-200/60"
              >
                <option value="champion">Champion (Best_Production_Model)</option>
                <option value="latest">Latest (Latest_Model)</option>
              </select>
            </label>

            <label className="grid gap-2">
              <span className="text-sm text-slate-700">Features (comma-separated)</span>
              <input
                value={featuresText}
                onChange={(e) => setFeaturesText(e.target.value)}
                placeholder="e.g. 1, 1"
                className="rounded-2xl border border-slate-200 bg-white px-4 py-3 text-slate-900 outline-none transition focus:border-sky-300 focus:ring-4 focus:ring-sky-200/60"
              />
            </label>

            <button
              onClick={onPredict}
              disabled={loading}
              className="mt-2 rounded-2xl bg-slate-900 px-5 py-3 font-medium text-white shadow-sm transition hover:bg-slate-800 active:scale-[0.99] disabled:opacity-60 disabled:cursor-not-allowed"
            >
              {loading ? "Predicting..." : "Predict"}
            </button>

            <div className="text-xs text-slate-500">
              Endpoint: <code className="text-slate-700">{endpoint}</code>
            </div>

            {error ? (
              <div className="rounded-2xl border border-red-200 bg-red-50 p-4 text-sm text-red-800">
                <b>Error:</b> {error}
              </div>
            ) : null}

            {result ? (
              <div className="rounded-2xl border border-slate-200 bg-white p-4 text-sm text-slate-800">
                <div>
                  <span className="text-slate-500">Prediction:</span>{" "}
                  <code className="font-semibold">{String(result.prediction)}</code>
                </div>
                <div className="mt-2">
                  <span className="text-slate-500">Model URI:</span>{" "}
                  <code className="break-all">{result.model_uri}</code>
                </div>
              </div>
            ) : null}
          </div>
        </div>
      </div>
    </div>
  );
}
