import { useEffect, useMemo, useRef, useState, useId } from "react";
import { FEATURE_GROUPS } from "../featuresSchema";

function clamp(x, lo, hi) {
    return Math.max(lo, Math.min(hi, x));
}

function groupKeyFromSchema(g) {
    if (g.single) return "Estimated";
    if (String(g.suffix || "").includes("Angle")) return "Angle";
    if (String(g.suffix || "").includes("NASM")) return "NASM";
    if (String(g.suffix || "").includes("Time")) return "Time";
    return g.title || "Other";
}

function shortLabel(name) {
    if (name.includes("_Angle_Deviation"))
        return `A${name.match(/No_(\d+)_/)?.[1] ?? ""}`;
    if (name.includes("_NASM_Deviation"))
        return `N${name.match(/No_(\d+)_/)?.[1] ?? ""}`;
    if (name.includes("_Time_Deviation"))
        return `T${name.match(/No_(\d+)_/)?.[1] ?? ""}`;
    if (name === "EstimatedScore") return "E";
    return name;
}

/**
 * Light glossy iOS knob with a subtle arc.
 */
function Knob({ name, value, onChange, min = 0, max = 1, step = 0.001 }) {
    const ref = useRef(null);
    const gid = useId();

    const norm = max === min ? 0 : clamp((value - min) / (max - min), 0, 1);

    // Arc geometry
    const SIZE = 64;
    const CX = SIZE / 2;
    const CY = SIZE / 2;
    const R = 24;
    const CIRC = 2 * Math.PI * R;
    const ARC_FRAC = 0.38; // tasteful arc
    const ARC_LEN = CIRC * ARC_FRAC;
    const START_ANGLE = 215;

    const trackDash = `${ARC_LEN} ${CIRC}`;
    const valueDash = `${Math.max(0.0001, ARC_LEN * norm)} ${CIRC}`;

    // Knob drag mapping
    const SWEEP_MIN = -135;
    const SWEEP_MAX = 135;

    const setFromPointer = (clientX, clientY) => {
        const el = ref.current;
        if (!el) return;

        const rect = el.getBoundingClientRect();
        const cx = rect.left + rect.width / 2;
        const cy = rect.top + rect.height / 2;

        const dx = clientX - cx;
        const dy = clientY - cy;

        let deg = (Math.atan2(dy, dx) * 180) / Math.PI;
        let fromUp = deg + 90;
        if (fromUp > 180) fromUp -= 360;
        if (fromUp < -180) fromUp += 360;

        fromUp = clamp(fromUp, SWEEP_MIN, SWEEP_MAX);
        const t = (fromUp - SWEEP_MIN) / (SWEEP_MAX - SWEEP_MIN);

        let v = min + t * (max - min);
        if (step > 0) v = Math.round(v / step) * step;
        v = Number(v.toFixed(6));
        onChange(v);
    };

    const onPointerDown = (e) => {
        e.preventDefault();
        e.currentTarget.setPointerCapture?.(e.pointerId);
        setFromPointer(e.clientX, e.clientY);
    };

    const onPointerMove = (e) => {
        if (!e.currentTarget.hasPointerCapture?.(e.pointerId)) return;
        setFromPointer(e.clientX, e.clientY);
    };

    const onPointerUp = (e) => {
        try {
            e.currentTarget.releasePointerCapture?.(e.pointerId);
        } catch {
            // ignore
        }
    };

    return (
        <div className="flex flex-col items-center gap-2">
            <div
                ref={ref}
                className="ios-knob"
                title={name}
                onPointerDown={onPointerDown}
                onPointerMove={onPointerMove}
                onPointerUp={onPointerUp}
                onPointerCancel={onPointerUp}
                role="slider"
                tabIndex={0}
                aria-label={name}
                aria-valuemin={min}
                aria-valuemax={max}
                aria-valuenow={value}
            >
                <svg
                    className="ios-knob__ring"
                    viewBox={`0 0 ${SIZE} ${SIZE}`}
                    aria-hidden="true"
                >
                    <defs>
                        <linearGradient
                            id={`iosgrad-${gid}`}
                            x1="0"
                            y1="0"
                            x2="1"
                            y2="1"
                        >
                            <stop
                                offset="0%"
                                stopColor="#38bdf8"
                                stopOpacity="0.95"
                            />
                            <stop
                                offset="45%"
                                stopColor="#a855f7"
                                stopOpacity="0.88"
                            />
                            <stop
                                offset="100%"
                                stopColor="#fb7185"
                                stopOpacity="0.80"
                            />
                        </linearGradient>
                    </defs>

                    <g transform={`rotate(${START_ANGLE} ${CX} ${CY})`}>
                        <circle
                            className="ios-knob__track"
                            cx={CX}
                            cy={CY}
                            r={R}
                            strokeDasharray={trackDash}
                            strokeDashoffset="0"
                        />
                        <circle
                            className="ios-knob__value"
                            cx={CX}
                            cy={CY}
                            r={R}
                            stroke={`url(#iosgrad-${gid})`}
                            strokeDasharray={valueDash}
                            strokeDashoffset="0"
                        />
                    </g>
                </svg>

                <div className="ios-knob__center" />
            </div>

            <div className="text-xs font-semibold text-slate-800">
                {shortLabel(name)}
            </div>

            <input
                className="ios-input w-20 rounded-xl px-2 py-1 text-center text-xs text-slate-900 outline-none focus:ring-2 focus:ring-sky-200/60"
                type="number"
                min={min}
                max={max}
                step={step}
                value={Number.isFinite(value) ? value : 0}
                onChange={(e) => {
                    const v = Number(e.target.value);
                    if (Number.isNaN(v)) return;
                    onChange(clamp(v, min, max));
                }}
            />
        </div>
    );
}

export default function FeatureBuilder({ values, setValues, maxFeatures }) {
    const groups = useMemo(() => {
        const map = new Map();
        const limit = typeof maxFeatures === "number" ? maxFeatures : Infinity;
        let used = 0;

        for (const g of FEATURE_GROUPS) {
            if (used >= limit) break;

            const key = groupKeyFromSchema(g);
            if (!map.has(key)) map.set(key, []);

            if (g.single) {
                if (used < limit) {
                    map.get(key).push(g.single);
                    used += 1;
                }
                continue;
            }

            const start = g.startIndex ?? 1;
            for (let i = 0; i < g.count; i++) {
                if (used >= limit) break;
                const idx = start + i;
                map.get(key).push(`${g.prefix}${idx}${g.suffix}`);
                used += 1;
            }
        }

        const orderedKeys = [];
        for (const g of FEATURE_GROUPS) {
            const k = groupKeyFromSchema(g);
            if (!orderedKeys.includes(k)) orderedKeys.push(k);
        }

        // Drop empty groups so tabs don’t show when maxFeatures cuts them off
        return orderedKeys
            .map((k) => ({ key: k, names: map.get(k) || [] }))
            .filter((g) => g.names.length > 0);
    }, [maxFeatures]);

    const [active, setActive] = useState(groups[0]?.key || "Angle");

    // If the active tab disappears after maxFeatures changes, move to first valid tab
    useEffect(() => {
        if (!groups.some((g) => g.key === active)) {
            setActive(groups[0]?.key || "Angle");
        }
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [groups]);

    const activeGroup = groups.find((g) => g.key === active) || groups[0];

    const setOne = (name, v) => setValues((prev) => ({ ...prev, [name]: v }));

    return (
        <div className="ios-card rounded-[28px] p-4">
            {/* Minimal tabs */}
            <div className="flex flex-wrap gap-2">
                {groups.map((g) => (
                    <button
                        key={g.key}
                        type="button"
                        onClick={() => setActive(g.key)}
                        className={`ios-btn rounded-full px-4 py-2 text-sm font-medium text-slate-800 ${
                            g.key === active ? "bg-white/80" : ""
                        }`}
                        title={`${g.key} (${g.names.length})`}
                    >
                        {g.key}{" "}
                        <span className="opacity-60">({g.names.length})</span>
                    </button>
                ))}
            </div>

            {/* Compact contained scroll */}
            <div className="mt-4 max-h-[520px] overflow-auto pr-1">
                <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-6 gap-4">
                    {activeGroup?.names?.map((name) => (
                        <div key={name} className="ios-pill rounded-[24px] p-3">
                            <Knob
                                name={name}
                                value={Number(values?.[name] ?? 0)}
                                onChange={(v) => setOne(name, v)}
                                min={0}
                                max={1}
                                step={0.001}
                            />
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );
}
