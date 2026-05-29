// SquatAnalyzer: uses MediaPipe Pose (tasks-vision) to detect squat keypoints
// from a live webcam feed, an uploaded video file, or a static image, then
// sends them to the Python backend for angle calculation and depth
// classification (Deep / Shallow / Invalid).
// Accumulated frames can be saved to Supabase or exported as CSV on demand.

import { useEffect, useRef, useState, useCallback, useMemo } from "react";
import { apiUrl } from "../apiBase";
import supabase from "../supabaseClient";

// MediaPipe keypoint indices used for squat analysis
// https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker
// Landmark set aligned with the normalised Kinect CSV files
const SQUAT_LANDMARK_NAMES = {
    0: "nose",
    11: "left_shoulder",
    12: "right_shoulder",
    13: "left_elbow",
    14: "right_elbow",
    15: "left_wrist",
    16: "right_wrist",
    23: "left_hip",
    24: "right_hip",
    25: "left_knee",
    26: "right_knee",
    27: "left_ankle",
    28: "right_ankle",
};

// All 33 MediaPipe landmark names in index order.
const ALL_LANDMARK_NAMES = [
    "nose", // 0
    "left_eye_inner", // 1
    "left_eye", // 2
    "left_eye_outer", // 3
    "right_eye_inner", // 4
    "right_eye", // 5
    "right_eye_outer", // 6
    "left_ear", // 7
    "right_ear", // 8
    "mouth_left", // 9
    "mouth_right", // 10
    "left_shoulder", // 11
    "right_shoulder", // 12
    "left_elbow", // 13
    "right_elbow", // 14
    "left_wrist", // 15
    "right_wrist", // 16
    "left_pinky", // 17
    "right_pinky", // 18
    "left_index", // 19
    "right_index", // 20
    "left_thumb", // 21
    "right_thumb", // 22
    "left_hip", // 23
    "right_hip", // 24
    "left_knee", // 25
    "right_knee", // 26
    "left_ankle", // 27
    "right_ankle", // 28
    "left_heel", // 29
    "right_heel", // 30
    "left_foot_index", // 31
    "right_foot_index", // 32
];

// Display/viewer joint order (used only for 3-D skeleton rendering).
const SQUAT_JOINT_ORDER = [
    { name: "nose", idx: 0 },
    { name: "left_shoulder", idx: 11 },
    { name: "left_elbow", idx: 13 },
    { name: "right_shoulder", idx: 12 },
    { name: "right_elbow", idx: 14 },
    { name: "left_wrist", idx: 15 },
    { name: "right_wrist", idx: 16 },
    { name: "left_hip", idx: 23 },
    { name: "right_hip", idx: 24 },
    { name: "left_knee", idx: 25 },
    { name: "right_knee", idx: 26 },
    { name: "left_ankle", idx: 27 },
    { name: "right_ankle", idx: 28 },
];
// Full body skeleton connections (MediaPipe BlazePose topology).
const POSE_CONNECTIONS = [
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 7],
    [0, 4],
    [4, 5],
    [5, 6],
    [6, 8],
    [9, 10],
    [11, 12],
    [11, 23],
    [12, 24],
    [23, 24],
    [11, 13],
    [13, 15],
    [15, 17],
    [15, 19],
    [15, 21],
    [17, 19],
    [12, 14],
    [14, 16],
    [16, 18],
    [16, 20],
    [16, 22],
    [18, 20],
    [23, 25],
    [25, 27],
    [27, 29],
    [27, 31],
    [29, 31],
    [24, 26],
    [26, 28],
    [28, 30],
    [28, 32],
    [30, 32],
];

// Indices that are squat-relevant (highlighted brighter).
const SQUAT_INDICES = new Set([
    0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28,
]);

// Skeleton connections between squat joints only (for 3-D viewer).
// Derived from POSE_CONNECTIONS filtered to SQUAT_INDICES and mapped to names.
const _IDX_TO_SQUAT_NAME = Object.fromEntries(
    SQUAT_JOINT_ORDER.map(({ name, idx }) => [idx, name]),
);
const SQUAT_CONNECTIONS_BY_NAME = POSE_CONNECTIONS.filter(
    ([a, b]) => SQUAT_INDICES.has(a) && SQUAT_INDICES.has(b),
)
    .map(([a, b]) => [_IDX_TO_SQUAT_NAME[a], _IDX_TO_SQUAT_NAME[b]])
    .filter(([a, b]) => a && b);

const MODEL_URL =
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task";
const WASM_URL =
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm";
const ESM_URL =
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/+esm";
const MAX_SQUAT_SCORE = 4;

/**
 * Create a new PoseLandmarker for continuous video/webcam detection.
 */
async function createVideoLandmarker() {
    const { PoseLandmarker, FilesetResolver } = await import(ESM_URL);
    const vision = await FilesetResolver.forVisionTasks(WASM_URL);
    return PoseLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath: MODEL_URL, delegate: "CPU" },
        runningMode: "VIDEO",
        numPoses: 1,
    });
}

/**
 * Create a one-shot PoseLandmarker for static image detection.
 */
async function createImageLandmarker() {
    const { PoseLandmarker, FilesetResolver } = await import(ESM_URL);
    const vision = await FilesetResolver.forVisionTasks(WASM_URL);
    return PoseLandmarker.createFromOptions(vision, {
        baseOptions: { modelAssetPath: MODEL_URL, delegate: "CPU" },
        runningMode: "IMAGE",
        numPoses: 1,
    });
}

/**
 * Filter a MediaPipe world-landmarks array to only the 3-D joints needed for a squat.
 */
function filterSquatKeypoints3d(landmarks) {
    if (!landmarks) return [];
    return Object.entries(SQUAT_LANDMARK_NAMES).flatMap(([idx, name]) => {
        const lm = landmarks[Number(idx)];
        if (!lm) return [];
        return [{ name, x: lm.x, y: lm.y, z: lm.z ?? 0 }];
    });
}

/**
 * Draw the full-body skeleton on a canvas.
 * Landmark coordinates are already in display space (MediaPipe received an
 * orientation-corrected frame via the capture canvas).
 */
function drawSkeleton(canvas, w, h, detection, off = { x: 0, y: 0 }) {
    if (!detection?.landmarks?.length) return;
    const ctx = canvas.getContext("2d");
    const lms = detection.landmarks[0];

    for (const [a, b] of POSE_CONNECTIONS) {
        const la = lms[a];
        const lb = lms[b];
        if (!la || !lb) continue;
        const isSquatBone = SQUAT_INDICES.has(a) && SQUAT_INDICES.has(b);
        ctx.beginPath();
        ctx.moveTo(off.x + la.x * w, off.y + la.y * h);
        ctx.lineTo(off.x + lb.x * w, off.y + lb.y * h);
        ctx.strokeStyle = isSquatBone
            ? "rgba(244,114,182,0.92)"
            : "rgba(148,163,184,0.55)";
        ctx.lineWidth = isSquatBone ? 3 : 1.5;
        ctx.stroke();
    }

    for (let idx = 0; idx < lms.length; idx++) {
        const lm = lms[idx];
        if (!lm) continue;
        const isSquat = SQUAT_INDICES.has(idx);
        ctx.beginPath();
        ctx.arc(
            off.x + lm.x * w,
            off.y + lm.y * h,
            isSquat ? 6 : 3,
            0,
            Math.PI * 2,
        );
        ctx.fillStyle = isSquat ? "#38bdf8" : "rgba(255,255,255,0.65)";
        ctx.fill();
    }
}

// 3-D Skeleton Viewer

/**
 * Rotate a 3-D point by yaw (ry, around Y) then pitch (rx, around X).
 * Returns projected (px, py) and depth for optional sorting.
 */
function project3D(x, y, z, rxDeg, ryDeg, scale, cx, cy) {
    const rx = (rxDeg * Math.PI) / 180;
    const ry = (ryDeg * Math.PI) / 180;
    // Yaw
    const x1 = x * Math.cos(ry) + z * Math.sin(ry);
    const y1 = y;
    const z1 = -x * Math.sin(ry) + z * Math.cos(ry);
    // Pitch
    const x2 = x1;
    const y2 = y1 * Math.cos(rx) - z1 * Math.sin(rx);
    const z2 = y1 * Math.sin(rx) + z1 * Math.cos(rx);
    // MediaPipe world y is positive-down, so map directly (not negated).
    return { px: cx + x2 * scale, py: cy + y2 * scale, depth: z2 };
}

function Skeleton3DViewer({ frames, liveFrameRef }) {
    const canvasRef = useRef(null);
    const [playing, setPlaying] = useState(false);
    const [frameIdx, setFrameIdx] = useState(0);
    // true = always show the newest frame (live follow); false = user has control
    const [liveMode, setLiveMode] = useState(true);
    const [prevFramesLength, setPrevFramesLength] = useState(frames.length);
    const rotation = useRef({ x: 15, y: 25 });
    const dragRef = useRef(null);
    const zoomRef = useRef(260);
    const playTimerRef = useRef(null);
    const frameIdxRef = useRef(0);

    // Derived state updates during render (avoids setState-in-effect lint error).
    if (prevFramesLength !== frames.length) {
        setPrevFramesLength(frames.length);
        if (frames.length === 0) {
            setLiveMode(true);
        } else if (liveMode && !playing) {
            setFrameIdx(frames.length - 1);
        }
    }

    const displayedFrameIdx = useMemo(() => {
        if (frames.length === 0) return 0;
        return Math.min(frameIdx, frames.length - 1);
    }, [frames.length, frameIdx]);

    // Keep ref in sync with state for rAF callbacks
    useEffect(() => {
        frameIdxRef.current = displayedFrameIdx;
    }, [displayedFrameIdx]);

    // Play timer
    useEffect(() => {
        if (playing) {
            playTimerRef.current = setInterval(() => {
                setFrameIdx((i) => {
                    if (i >= frames.length - 1) {
                        setPlaying(false);
                        return frames.length - 1;
                    }
                    return i + 1;
                });
            }, 150); // ~6 fps replay
        } else {
            clearInterval(playTimerRef.current);
        }
        return () => clearInterval(playTimerRef.current);
    }, [playing, frames.length]);

    function drawFrame(frameData, isLive = false) {
        const canvas = canvasRef.current;
        if (!canvas || !frameData) return;
        const ctx = canvas.getContext("2d");
        const W = canvas.width;
        const H = canvas.height;
        ctx.clearRect(0, 0, W, H);

        const rx = rotation.current.x;
        const ry = rotation.current.y;
        const scale = zoomRef.current;
        const cx = W / 2;
        const cy = H * 0.82;

        // Lock rotation pivot at ankle midpoint so the model spins around the feet
        const la = frameData.joints?.["left_ankle"];
        const ra = frameData.joints?.["right_ankle"];
        const anchor =
            la && ra
                ? {
                      x: (la.x + ra.x) / 2,
                      y: (la.y + ra.y) / 2,
                      z: (la.z + ra.z) / 2,
                  }
                : (la ?? ra ?? { x: 0, y: 0, z: 0 });

        // Project all joints
        const pos = {};
        for (const { name } of SQUAT_JOINT_ORDER) {
            const j = frameData.joints?.[name];
            if (!j) continue;
            pos[name] = project3D(
                j.x - anchor.x,
                j.y - anchor.y,
                j.z - anchor.z,
                rx,
                ry,
                scale,
                cx,
                cy,
            );
        }

        // Ground plane (projected circle at y=0 in foot-anchored space)
        const GR = 0.5; // radius in world units
        const SEGS = 72;
        const rim = [];
        for (let i = 0; i <= SEGS; i++) {
            const t = (i / SEGS) * Math.PI * 2;
            rim.push(
                project3D(
                    GR * Math.cos(t),
                    0,
                    GR * Math.sin(t),
                    rx,
                    ry,
                    scale,
                    cx,
                    cy,
                ),
            );
        }
        // Filled disc
        ctx.beginPath();
        ctx.moveTo(rim[0].px, rim[0].py);
        for (let i = 1; i <= SEGS; i++) ctx.lineTo(rim[i].px, rim[i].py);
        ctx.closePath();
        ctx.fillStyle = "rgba(30,41,59,0.55)";
        ctx.fill();
        // Outer ring
        ctx.strokeStyle = "rgba(148,163,184,0.3)";
        ctx.lineWidth = 1;
        ctx.stroke();
        // Grid lines (4 across each axis)
        ctx.lineWidth = 0.5;
        ctx.strokeStyle = "rgba(148,163,184,0.12)";
        for (let i = -3; i <= 3; i++) {
            const s = (i / 3) * GR;
            const clamp = Math.sqrt(Math.max(0, GR * GR - s * s));
            const xa = project3D(-clamp, 0, s, rx, ry, scale, cx, cy);
            const xb = project3D(clamp, 0, s, rx, ry, scale, cx, cy);
            ctx.beginPath();
            ctx.moveTo(xa.px, xa.py);
            ctx.lineTo(xb.px, xb.py);
            ctx.stroke();
            const za = project3D(s, 0, -clamp, rx, ry, scale, cx, cy);
            const zb = project3D(s, 0, clamp, rx, ry, scale, cx, cy);
            ctx.beginPath();
            ctx.moveTo(za.px, za.py);
            ctx.lineTo(zb.px, zb.py);
            ctx.stroke();
        }

        // Draw bones
        for (const [aName, bName] of SQUAT_CONNECTIONS_BY_NAME) {
            const pa = pos[aName];
            const pb = pos[bName];
            if (!pa || !pb) continue;
            ctx.beginPath();
            ctx.moveTo(pa.px, pa.py);
            ctx.lineTo(pb.px, pb.py);
            const isLeft = aName.includes("left") || bName.includes("left");
            ctx.strokeStyle = isLeft
                ? "rgba(56,189,248,0.85)"
                : "rgba(244,114,182,0.85)";
            ctx.lineWidth = 2.5;
            ctx.stroke();
        }

        // Draw joints
        for (const { name } of SQUAT_JOINT_ORDER) {
            const p = pos[name];
            if (!p) continue;
            const isLeg = ["knee", "hip", "ankle"].some((k) =>
                name.includes(k),
            );
            ctx.beginPath();
            ctx.arc(p.px, p.py, isLeg ? 6 : 4, 0, Math.PI * 2);
            ctx.fillStyle = name.includes("left")
                ? "#38bdf8"
                : name.includes("right")
                  ? "#f472b6"
                  : "#a3e635";
            ctx.fill();
        }

        // Classification label
        if (frameData.classification) {
            ctx.font = "bold 14px system-ui, sans-serif";
            const cls = frameData.classification;
            ctx.fillStyle =
                cls === "Deep"
                    ? "#16a34a"
                    : cls === "Shallow"
                      ? "#d97706"
                      : cls === "NotExercise"
                        ? "#64748b"
                        : "#dc2626";
            ctx.textAlign = "center";
            ctx.fillText(cls === "NotExercise" ? "not exercise" : cls, cx, 18);
        }

        // Frame counter / live indicator
        ctx.font = "11px system-ui, sans-serif";
        ctx.fillStyle = "rgba(148,163,184,0.8)";
        ctx.textAlign = "right";
        ctx.fillText(
            isLive ? "● LIVE" : `${displayedFrameIdx + 1} / ${frames.length}`,
            W - 6,
            H - 6,
        );
    }

    // Non-live: draw whenever anything changes (frame scrub, drag, zoom).
    useEffect(() => {
        if (liveFrameRef) return; // live rAF loop handles drawing
        drawFrame(frames[displayedFrameIdx], false);
    }); // no deps — runs on every render

    // Live: own rAF so detection updates never trigger React re-renders.
    useEffect(() => {
        if (!liveFrameRef) return;
        let rafId;
        function tick() {
            drawFrame(liveFrameRef.current, true);
            rafId = requestAnimationFrame(tick);
        }
        rafId = requestAnimationFrame(tick);
        return () => cancelAnimationFrame(rafId);
    }, [liveFrameRef]); // re-runs only when live mode toggled

    // Drag to rotate

    function onPointerDown(e) {
        dragRef.current = { x: e.clientX, y: e.clientY };
        canvasRef.current?.setPointerCapture(e.pointerId);
    }

    function onPointerMove(e) {
        if (!dragRef.current) return;
        const dx = e.clientX - dragRef.current.x;
        const dy = e.clientY - dragRef.current.y;
        rotation.current.y += dx * 0.4;
        rotation.current.x += dy * 0.4;
        dragRef.current = { x: e.clientX, y: e.clientY };
        drawFrame(
            liveFrameRef?.current ?? frames[frameIdxRef.current],
            !!liveFrameRef?.current,
        );
    }

    function onPointerUp() {
        dragRef.current = null;
    }

    function onWheel(e) {
        e.preventDefault();
        zoomRef.current = Math.max(
            80,
            Math.min(600, zoomRef.current - e.deltaY * 0.3),
        );
        drawFrame(
            liveFrameRef?.current ?? frames[frameIdxRef.current],
            !!liveFrameRef?.current,
        );
    }

    if (frames.length === 0 && !liveFrameRef) return null;

    return (
        <div className="ios-card rounded-2xl p-4 w-full flex flex-col items-center gap-3">
            <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider self-start">
                3-D Skeleton Replay
                <span className="ml-2 normal-case font-normal text-slate-300">
                    drag to rotate · scroll or ± to zoom
                </span>
            </p>

            <canvas
                ref={canvasRef}
                width={520}
                height={400}
                className="rounded-xl bg-slate-950 cursor-grab active:cursor-grabbing touch-none"
                style={{ maxWidth: "100%" }}
                onPointerDown={onPointerDown}
                onPointerMove={onPointerMove}
                onPointerUp={onPointerUp}
                onPointerLeave={onPointerUp}
                onWheel={onWheel}
            />

            {/* Controls */}
            <div className="flex items-center gap-2 w-full flex-wrap">
                <button
                    onClick={() => {
                        setLiveMode(false);
                        setFrameIdx(0);
                        setPlaying(true);
                    }}
                    className="ios-btn ios-btn-primary px-4 py-1.5 rounded-full text-xs font-semibold"
                >
                    ▶ Replay
                </button>
                {playing && (
                    <button
                        onClick={() => setPlaying(false)}
                        className="ios-btn px-4 py-1.5 rounded-full text-xs font-semibold text-slate-600"
                    >
                        ⏸ Pause
                    </button>
                )}
                {/* Step prev/next */}
                <button
                    onClick={() => {
                        setPlaying(false);
                        setLiveMode(false);
                        setFrameIdx((i) => Math.max(0, i - 1));
                    }}
                    className="ios-btn w-8 h-7 flex items-center justify-center rounded-full text-sm font-bold text-slate-600"
                    title="Previous frame"
                >
                    ‹‹
                </button>
                <button
                    onClick={() => {
                        setPlaying(false);
                        setLiveMode(false);
                        setFrameIdx((i) => Math.min(frames.length - 1, i + 1));
                    }}
                    className="ios-btn w-8 h-7 flex items-center justify-center rounded-full text-sm font-bold text-slate-600"
                    title="Next frame"
                >
                    ››
                </button>
                <input
                    type="range"
                    min={0}
                    max={frames.length - 1}
                    value={displayedFrameIdx}
                    onChange={(e) => {
                        setPlaying(false);
                        setLiveMode(false);
                        setFrameIdx(Number(e.target.value));
                    }}
                    className="flex-1 accent-sky-400 min-w-0"
                />
                {/* Zoom buttons */}
                <button
                    onClick={() => {
                        zoomRef.current = Math.min(600, zoomRef.current + 30);
                        drawFrame(
                            liveFrameRef?.current ??
                                frames[frameIdxRef.current],
                            !!liveFrameRef?.current,
                        );
                    }}
                    className="ios-btn w-7 h-7 flex items-center justify-center rounded-full text-sm font-bold text-slate-600"
                    title="Zoom in"
                >
                    +
                </button>
                <button
                    onClick={() => {
                        zoomRef.current = Math.max(80, zoomRef.current - 30);
                        drawFrame(
                            liveFrameRef?.current ??
                                frames[frameIdxRef.current],
                            !!liveFrameRef?.current,
                        );
                    }}
                    className="ios-btn w-7 h-7 flex items-center justify-center rounded-full text-sm font-bold text-slate-600"
                    title="Zoom out"
                >
                    −
                </button>
            </div>

            <p className="text-xs text-slate-400 self-start">
                Frame {displayedFrameIdx + 1} of {frames.length}
                {" · exercise frames only · z from DL model"}
            </p>
        </div>
    );
}

/**
 * Assess per-frame video quality for squat analysis.
 * Returns { blocking, warnings } where:
 *   blocking — issues that prevent reliable analysis (pipeline should be gated)
 *   warnings — advisory hints shown live but never block the pipeline
 */
function assessVideoQuality(detection, captureCanvas) {
    if (!detection?.landmarks?.length) {
        return { blocking: ["No person detected in frame"], warnings: [] };
    }

    const lms = detection.landmarks[0];
    const blocking = [];
    const warnings = [];
    const CORE = [11, 12, 23, 24, 25, 26, 27, 28];
    const vis = (i) => lms[i]?.visibility ?? 0;

    // BLOCKING: frame brightness
    if (captureCanvas?.width > 0 && captureCanvas?.height > 0) {
        try {
            const ctx = captureCanvas.getContext("2d");
            const { width: w, height: h } = captureCanvas;
            const { data } = ctx.getImageData(
                Math.floor(w * 0.25),
                Math.floor(h * 0.15),
                Math.floor(w * 0.5),
                Math.floor(h * 0.7),
            );
            let lum = 0;
            for (let i = 0; i < data.length; i += 4) {
                lum +=
                    0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
            }
            const avg = lum / (data.length / 4);
            // TODO: Improve values for better preformance
            if (avg < 28)
                blocking.push("Frame too dark: improve room lighting");
            else if (avg > 238)
                blocking.push(
                    "Frame overexposed: reduce direct lighting behind the subject",
                );
        } catch {
            /* cross-origin / security errors ignored */
        }
    }

    // WARNINGS: advisory hints — never stop the pipeline

    // Upside down
    const nose = lms[0],
        la = lms[27],
        ra = lms[28];
    if (nose && la && ra && nose.y > (la.y + ra.y) / 2 + 0.1) {
        warnings.push("Video appears upside down: rotate the camera 180°");
    }

    // Hands at or below hip level — warn when arms hang at sides instead of
    // being raised (normal squat starting position).
    const lw = lms[15],
        rw = lms[16],
        lh = lms[23],
        rh = lms[24];
    if (lw && rw && lh && rh) {
        const hipY = (lh.y + rh.y) / 2;
        const wristY = (lw.y + rw.y) / 2;
        if (wristY > hipY - 0.05) {
            warnings.push(
                "Hands are not raised: lift your arms into squat position",
            );
        }
    }

    // Key joints clipped at frame edges
    const coreLms = CORE.map((i) => lms[i]).filter(Boolean);
    if (
        coreLms.filter(
            (lm) => lm.x < 0.04 || lm.x > 0.96 || lm.y < 0.04 || lm.y > 0.96,
        ).length >= 2
    ) {
        warnings.push(
            "Person is partially out of frame: move camera back or reposition",
        );
    }

    // Knees hidden while shoulders clearly visible (facing camera head-on)
    if (vis(25) < 0.35 && vis(26) < 0.35 && vis(11) > 0.6 && vis(12) > 0.6) {
        if (Math.abs((lms[12]?.x ?? 0) - (lms[11]?.x ?? 0)) > 0.15) {
            warnings.push(
                "Knees not visible: try a side-on camera angle for better depth measurement",
            );
        }
    }

    return { blocking, warnings };
}

// Main component

export default function SquatAnalyzer() {
    const videoRef = useRef(null);
    const canvasRef = useRef(null);
    const fileInputRef = useRef(null);
    const imageInputRef = useRef(null);
    const landmarkerRef = useRef(null);
    const rafRef = useRef(null);
    const lastTimestampRef = useRef(-1);
    const videoBlobUrlRef = useRef(null);
    const lastDetectionRef = useRef(null);
    const sessionLogRef = useRef([]);
    // All kp3d frames collected during a webcam recording session
    const recordedKpFramesRef = useRef([]);
    // Normalized image-space landmarks [0,1] collected in parallel — fed to start-stop model
    const recordedNormFramesRef = useRef([]);
    // Offscreen canvas used to feed orientation-corrected frames to MediaPipe
    const captureCanvasRef = useRef(null);
    // AbortController for in-flight analyze-session fetch — aborted when a new session starts
    const analysisAbortRef = useRef(null);

    const [inputMode, setInputMode] = useState("webcam");
    const [sessionName, setSessionName] = useState("");
    const [status, setStatus] = useState("idle");
    const [result, setResult] = useState(null);
    const [goodBadThreshold, setGoodBadThreshold] = useState(0.5);
    const [PIPELINE_TIME, setPipelineTime] = useState(null); // * Updated base name for lint
    const [pipelineTimings, setPipelineTimings] = useState(null);
    const [errorMsg, setErrorMsg] = useState("");
    const [uploadedFileName, setUploadedFileName] = useState("");
    const [videoPaused, setVideoPaused] = useState(false);
    const [ALL_KEYPOINTS, setAllKeypoints] = useState([]); // * Updated base name for lint
    const [sessionLog, setSessionLog] = useState([]);
    const [isSaving, setIsSaving] = useState(false);
    const liveFrame3dRef = useRef(null);
    const [startStopMetrics, setStartStopMetrics] = useState(null);
    const [qualityIssues, setQualityIssues] = useState([]);
    const [qualityError, setQualityError] = useState(null);

    const sessionNameRef = useRef("");
    const qualityIssueCountsRef = useRef({});
    const qualityFrameCountRef = useRef(0);

    // Fetch model metrics on mount
    useEffect(() => {
        fetch(apiUrl("/api/v1/model-info/start-stop"))
            .then((r) => (r.ok ? r.json() : null))
            .then((d) => d && setStartStopMetrics(d))
            .catch(() => {});
    }, []);

    // Preload VIDEO landmarker on mount
    useEffect(() => {
        let cancelled = false;
        createVideoLandmarker()
            .then((lm) => {
                if (cancelled) {
                    lm.close?.();
                    return;
                }
                landmarkerRef.current = lm;
            })
            .catch(() => {});

        return () => {
            cancelled = true;
            landmarkerRef.current?.close?.();
            landmarkerRef.current = null;
        };
    }, []);

    // Helpers
    function revokeBlobUrl() {
        if (videoBlobUrlRef.current) {
            URL.revokeObjectURL(videoBlobUrlRef.current);
            videoBlobUrlRef.current = null;
        }
    }

    // Stops camera/detection and clears live UI, but keeps accumulated session data.
    const stopCapture = useCallback((nextStatus = "idle") => {
        cancelAnimationFrame(rafRef.current);
        const video = videoRef.current;
        if (video) {
            if (video.srcObject) {
                video.srcObject.getTracks().forEach((t) => t.stop());
                video.srcObject = null;
            }
            video.pause();
            video.src = "";
            video.load();
        }
        revokeBlobUrl();
        lastTimestampRef.current = -1;
        lastDetectionRef.current = null;
        const canvas = canvasRef.current;
        if (canvas) {
            canvas
                .getContext("2d")
                .clearRect(0, 0, canvas.width, canvas.height);
        }
        setAllKeypoints([]);
        setResult(null);
        setVideoPaused(false);
        liveFrame3dRef.current = null;
        qualityIssueCountsRef.current = {};
        qualityFrameCountRef.current = 0;
        setQualityIssues([]);
        setStatus(nextStatus);
    }, []);

    // Full reset: stop capture + discard data (used when switching modes).
    const stopAll = useCallback(() => {
        // Cancel any in-flight backend fetch so it can't overwrite the next session's result.
        analysisAbortRef.current?.abort();
        analysisAbortRef.current = null;
        stopCapture("idle");
        recordedKpFramesRef.current = [];
        recordedNormFramesRef.current = [];
        sessionLogRef.current = [];
        setSessionLog([]);
        setUploadedFileName("");
        setErrorMsg("");
        setQualityError(null);
        setPipelineTime(null);
        setPipelineTimings(null);
    }, [stopCapture]);

    // Discards accumulated session data (call explicitly for "new session").
    const clearSession = useCallback(() => {
        stopAll();
    }, [stopAll]);

    function mapAllKeypoints(landmarks) {
        return landmarks.map((lm, i) => ({
            index: i,
            name: ALL_LANDMARK_NAMES[i] ?? `landmark_${i}`,
            x: lm.x,
            y: lm.y,
            predictedZ: lm.z ?? null,
        }));
    }

    const switchMode = useCallback(
        (newMode) => {
            stopAll();
            setInputMode(newMode);
            setResult(null);
            setErrorMsg("");
            setUploadedFileName("");
            setAllKeypoints([]);
            const canvas = canvasRef.current;
            if (canvas)
                canvas
                    .getContext("2d")
                    .clearRect(0, 0, canvas.width, canvas.height);
        },
        [stopAll],
    );

    // Batch analyze (webcam stop or video end)
    // Sends all frames at once to the full pipeline: Cut → Z-pred → Classify.

    const stopAndAnalyze = useCallback(async () => {
        const frames = recordedKpFramesRef.current;
        const normFrames = recordedNormFramesRef.current;
        // Snapshot quality data before stopCapture resets it
        const totalQFrames = qualityFrameCountRef.current;
        const issueCounts = { ...qualityIssueCountsRef.current };
        stopCapture("analyzing");
        if (frames.length === 0) {
            setStatus("idle");
            return;
        }
        // Quality gate: abort if any issue persisted in more than 50% of frames
        if (totalQFrames > 10) {
            const persistentIssues = Object.entries(issueCounts)
                .filter(([, n]) => n / totalQFrames > 0.65)
                .sort(([, a], [, b]) => b - a)
                .map(([issue]) => issue);
            if (persistentIssues.length > 0) {
                setQualityError({ issues: persistentIssues });
                setStatus("idle");
                return;
            }
        }
        // Register a fresh AbortController so stopAll() can cancel this fetch if the
        // user starts a new session before this response arrives.
        const abort = new AbortController();
        analysisAbortRef.current = abort;
        try {
            const body = { frames };
            if (normFrames.length === frames.length)
                body.norm_frames = normFrames;
            const t0 = Date.now();
            const res = await fetch(apiUrl("/api/v1/squat/analyze-session"), {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(body),
                signal: abort.signal,
            });
            if (!res.ok) throw new Error("Session analysis failed");
            const data = await res.json();
            const elapsed = Date.now() - t0;
            setPipelineTime(elapsed);
            setPipelineTimings(
                data.timings
                    ? { ...data.timings, round_trip_ms: elapsed }
                    : null,
            );
            const entries = frames.map((kp3d, i) => ({
                timestamp: Date.now() + i,
                keypoints3d: kp3d,
                predictedZ: data.results[i]?.predicted_z ?? {},
                startStop: data.results[i]?.start_stop ?? 1,
            }));
            sessionLogRef.current = entries;
            setSessionLog(entries);
            const firstWithScore = data.results.find(
                (r) =>
                    r.start_stop === 1 &&
                    (r.good_bad_score != null || r.squat_score != null),
            );
            if (firstWithScore) {
                setResult({
                    goodBadScore: firstWithScore.good_bad_score,
                    squatScore: firstWithScore.squat_score,
                });
            }
            setStatus("finished");
        } catch (err) {
            // AbortError means a new session started — don't touch UI state.
            if (err?.name === "AbortError") return;
            setErrorMsg("Failed to analyze recording. Try again.");
            setStatus("error");
        }
    }, [stopCapture]);

    // Video-ended auto-finish

    useEffect(() => {
        if (status !== "running" || inputMode !== "upload") return;
        const video = videoRef.current;
        if (!video) return;
        const handleEnded = () => stopAndAnalyze();
        video.addEventListener("ended", handleEnded);
        return () => video.removeEventListener("ended", handleEnded);
    }, [status, inputMode, stopAndAnalyze]);

    // Webcam

    const startCamera = useCallback(async () => {
        recordedKpFramesRef.current = [];
        recordedNormFramesRef.current = [];
        sessionLogRef.current = [];
        setSessionLog([]);
        setResult(null);
        setStatus("loading");
        setErrorMsg("");
        setQualityError(null);
        try {
            if (!landmarkerRef.current) {
                landmarkerRef.current = await createVideoLandmarker();
            }
            const stream = await navigator.mediaDevices.getUserMedia({
                video: { width: 640, height: 480, frameRate: { ideal: 60 } },
            });
            videoRef.current.srcObject = stream;
            await videoRef.current.play();
            setStatus("running");
        } catch (err) {
            setStatus("error");
            setErrorMsg(String(err));
        }
    }, []);

    // Video upload

    const handleVideoChange = useCallback(
        async (e) => {
            const file = e.target.files?.[0];
            if (!file) return;
            if (!file.type.startsWith("video/")) {
                setErrorMsg("Please select a video file.");
                setStatus("error");
                return;
            }
            stopAll();
            setResult(null);
            setErrorMsg("");
            setUploadedFileName(file.name);
            setStatus("loading");
            try {
                // Always recreate the landmarker so MediaPipe's temporal filter
                // (which smooths landmarks across frames) starts clean for each video.
                // Without this, pose keypoints for the first ~5 frames of a new video
                // are blended with the previous video's final pose, skewing the score.
                if (landmarkerRef.current) {
                    landmarkerRef.current.close();
                    landmarkerRef.current = null;
                }
                landmarkerRef.current = await createVideoLandmarker();
                revokeBlobUrl();
                const blobUrl = URL.createObjectURL(file);
                videoBlobUrlRef.current = blobUrl;
                const video = videoRef.current;
                video.srcObject = null;
                video.src = blobUrl;
                video.load();
                await video.play();
                setStatus("running");
                setVideoPaused(false);
            } catch (err) {
                setStatus("error");
                setErrorMsg(String(err));
            }
            // Reset file input so re-uploading same file triggers onChange
            if (fileInputRef.current) fileInputRef.current.value = "";
        },
        [stopAll],
    );

    // Image upload

    const handleImageChange = useCallback(
        async (e) => {
            const file = e.target.files?.[0];
            if (!file) return;
            if (!file.type.startsWith("image/")) {
                setErrorMsg("Please select an image file.");
                setStatus("error");
                return;
            }
            stopAll();
            setResult(null);
            setErrorMsg("");
            setAllKeypoints([]);
            setUploadedFileName(file.name);
            setStatus("loading");

            const blobUrl = URL.createObjectURL(file);
            let imageLandmarker = null;
            try {
                imageLandmarker = await createImageLandmarker();

                const img = new Image();
                img.src = blobUrl;
                await new Promise((res, rej) => {
                    img.onload = res;
                    img.onerror = rej;
                });

                const DISPLAY_W = 640,
                    DISPLAY_H = 480;
                const scale = Math.min(
                    DISPLAY_W / img.naturalWidth,
                    DISPLAY_H / img.naturalHeight,
                );
                const drawW = img.naturalWidth * scale;
                const drawH = img.naturalHeight * scale;
                const offsetX = (DISPLAY_W - drawW) / 2;
                const offsetY = (DISPLAY_H - drawH) / 2;

                const canvas = canvasRef.current;
                canvas.width = DISPLAY_W;
                canvas.height = DISPLAY_H;
                const ctx = canvas.getContext("2d");
                ctx.fillStyle = "#000";
                ctx.fillRect(0, 0, DISPLAY_W, DISPLAY_H);
                ctx.drawImage(img, offsetX, offsetY, drawW, drawH);

                const detection = imageLandmarker.detect(canvas);

                if (detection.landmarks?.length > 0) {
                    // Quality check — canvas has raw image pixels before skeleton draw
                    const { blocking, warnings } = assessVideoQuality(
                        detection,
                        canvas,
                    );
                    drawSkeleton(canvas, DISPLAY_W, DISPLAY_H, detection, {
                        x: 0,
                        y: 0,
                    });
                    setAllKeypoints(mapAllKeypoints(detection.landmarks[0]));
                    if (warnings.length > 0) setQualityIssues(warnings);
                    if (blocking.length > 0) {
                        setQualityError({ issues: blocking });
                    }

                    const kp3d = filterSquatKeypoints3d(
                        detection.worldLandmarks?.[0] ?? [],
                    );
                    const normKp = filterSquatKeypoints3d(
                        detection.landmarks?.[0] ?? [],
                    );
                    if (kp3d.length > 0) {
                        const body = { frames: [kp3d] };
                        if (normKp.length === kp3d.length) {
                            body.norm_frames = [normKp];
                        }

                        const t0 = Date.now();
                        const res = await fetch(
                            apiUrl("/api/v1/squat/analyze-session"),
                            {
                                method: "POST",
                                headers: { "Content-Type": "application/json" },
                                body: JSON.stringify(body),
                            },
                        );
                        if (!res.ok) throw new Error("Image analysis failed");
                        const data = await res.json();
                        const elapsed = Date.now() - t0;
                        setPipelineTime(elapsed);
                        setPipelineTimings(
                            data.timings
                                ? { ...data.timings, round_trip_ms: elapsed }
                                : null,
                        );

                        const firstResult =
                            data.results?.find(
                                (r) =>
                                    r.good_bad_score != null ||
                                    r.squat_score != null,
                            ) ??
                            data.results?.[0] ??
                            {};
                        const entry = {
                            timestamp: Date.now(),
                            keypoints3d: kp3d,
                            predictedZ: firstResult.predicted_z ?? {},
                            startStop: firstResult.start_stop ?? 1,
                            classification: null,
                            confidence: null,
                        };
                        sessionLogRef.current = [entry];
                        setSessionLog([entry]);

                        if (
                            (firstResult.good_bad_score !== null &&
                                firstResult.good_bad_score !== undefined) ||
                            (firstResult.squat_score !== null &&
                                firstResult.squat_score !== undefined)
                        ) {
                            setResult({
                                goodBadScore:
                                    firstResult.good_bad_score ?? null,
                                squatScore: firstResult.squat_score ?? null,
                            });
                        }
                        setStatus("finished");
                        return;
                    }
                }
                setStatus("idle");
            } catch (err) {
                setStatus("error");
                setErrorMsg(String(err));
            } finally {
                imageLandmarker?.close?.();
                URL.revokeObjectURL(blobUrl);
                if (imageInputRef.current) imageInputRef.current.value = "";
            }
        },
        [stopAll],
    );

    const togglePlayPause = useCallback(() => {
        const video = videoRef.current;
        if (!video) return;
        if (video.paused) {
            video.play();
            setVideoPaused(false);
        } else {
            video.pause();
            setVideoPaused(true);
        }
    }, []);

    // Detection loop
    //
    // Drawing and detection are intentionally separated:
    // - rAF loop: draws the skeleton overlay at 60 fps, never blocked by inference.
    // - setInterval loop: runs MediaPipe inference at ~20 Hz independently.
    //   Because detectForVideo is synchronous (~20-50 ms), putting it in the rAF
    //   callback would block painting and cause visible lag.

    useEffect(() => {
        if (status !== "running") return;

        let lastDetectedVideoTime = -1;

        // --- rAF: draw overlay only (fast, never blocked by inference) ---
        function draw() {
            const video = videoRef.current;
            const canvas = canvasRef.current;
            if (!video || !canvas) {
                rafRef.current = requestAnimationFrame(draw);
                return;
            }
            const vw = video.videoWidth;
            const vh = video.videoHeight;
            const DISPLAY_W = 640;
            const DISPLAY_H = 480;
            if (canvas.width !== DISPLAY_W || canvas.height !== DISPLAY_H) {
                canvas.width = DISPLAY_W;
                canvas.height = DISPLAY_H;
            }
            const scale =
                vw && vh ? Math.min(DISPLAY_W / vw, DISPLAY_H / vh) : 1;
            const drawW = vw * scale;
            const drawH = vh * scale;
            canvas.getContext("2d").clearRect(0, 0, DISPLAY_W, DISPLAY_H);
            const detection = lastDetectionRef.current;
            if (detection)
                drawSkeleton(canvas, drawW, drawH, detection, {
                    x: (DISPLAY_W - drawW) / 2,
                    y: (DISPLAY_H - drawH) / 2,
                });
            rafRef.current = requestAnimationFrame(draw);
        }
        rafRef.current = requestAnimationFrame(draw);

        // --- setInterval: inference at ~20 Hz, decoupled from paint ---
        function runDetection() {
            const video = videoRef.current;
            const landmarker = landmarkerRef.current;
            if (
                !video ||
                !landmarker ||
                video.readyState < 2 ||
                video.paused ||
                video.ended ||
                !video.videoWidth ||
                !video.videoHeight
            )
                return;

            const isVideoUpload = inputMode === "upload";
            const videoTime = video.currentTime;
            if (isVideoUpload && videoTime === lastDetectedVideoTime) return;
            if (isVideoUpload) lastDetectedVideoTime = videoTime;

            const timestamp = performance.now();
            const vw = video.videoWidth;
            const vh = video.videoHeight;

            qualityFrameCountRef.current += 1;
            const qualityDue =
                qualityFrameCountRef.current > 15 &&
                qualityFrameCountRef.current % 20 === 0;

            try {
                const cap = captureCanvasRef.current;
                if ((isVideoUpload || qualityDue) && cap && vw > 0 && vh > 0) {
                    if (cap.width !== vw || cap.height !== vh) {
                        cap.width = vw;
                        cap.height = vh;
                    }
                    cap.getContext("2d").drawImage(video, 0, 0, vw, vh);
                    lastDetectionRef.current = landmarker.detectForVideo(
                        cap,
                        timestamp,
                    );
                } else {
                    lastDetectionRef.current = landmarker.detectForVideo(
                        video,
                        timestamp,
                    );
                }
            } catch {
                return;
            }

            const det = lastDetectionRef.current;
            if (det?.worldLandmarks?.length > 0) {
                const joints = {};
                for (const { name, idx } of SQUAT_JOINT_ORDER) {
                    const lm = det.worldLandmarks[0][idx];
                    if (!lm) continue;
                    joints[name] = {
                        x: lm.x,
                        y: lm.y,
                        z: lm.z ?? 0,
                    };
                }
                liveFrame3dRef.current = { classification: null, joints };

                recordedKpFramesRef.current.push(
                    filterSquatKeypoints3d(det.worldLandmarks[0]),
                );
                if (det.landmarks?.[0]) {
                    recordedNormFramesRef.current.push(
                        filterSquatKeypoints3d(det.landmarks[0]),
                    );
                }
            }

            if (qualityDue && det?.landmarks?.length > 0) {
                const { blocking, warnings } = assessVideoQuality(
                    det,
                    captureCanvasRef.current,
                );
                blocking.forEach((issue) => {
                    qualityIssueCountsRef.current[issue] =
                        (qualityIssueCountsRef.current[issue] || 0) + 1;
                });
                setQualityIssues([...blocking, ...warnings]);
            }
        }

        // 50 ms ≈ 20 Hz; if inference takes longer the interval self-throttles.
        const detectionInterval = setInterval(runDetection, 50);

        return () => {
            cancelAnimationFrame(rafRef.current);
            clearInterval(detectionInterval);
        };
    }, [status, inputMode]);

    // Backend + session log

    sessionNameRef.current = sessionName;

    // Manual save to Supabase

    async function handleSave() {
        if (!supabase || sessionLogRef.current.length === 0) return;
        setIsSaving(true);
        const idName = sessionName.trim() || null;
        const rows = sessionLogRef.current.map((e) => ({
            id_name: idName,
            raw_keypoints: {
                "3d": e.keypoints3d.map(({ x, y, z }) => [x, y, z ?? 0]),
            },
            score: e.confidence,
            classification: e.classification,
        }));
        const { error } = await supabase.from("squat_keypoints").insert(rows);
        if (error) console.warn("Supabase insert error:", error.message);
        setIsSaving(false);
    }

    // CSV export

    function downloadCSV() {
        const log = sessionLogRef.current;
        if (log.length === 0) return;

        const kp3dNames = log[0].keypoints3d.map((kp) => kp.name);

        const header = [
            "timestamp_ms",
            "classification",
            "confidence",
            ...kp3dNames.flatMap((n) => [
                `${n}_3d_x`,
                `${n}_3d_y`,
                `${n}_3d_z`,
                `${n}_pred_z`,
            ]),
        ].join(",");

        const csvRows = log.map((e) =>
            [
                e.timestamp,
                e.classification ?? "",
                e.confidence?.toFixed(3) ?? "",
                ...e.keypoints3d.flatMap(({ name, x, y, z }) => [
                    x.toFixed(4),
                    y.toFixed(4),
                    (z ?? 0).toFixed(4),
                    (e.predictedZ?.[name] ?? "").toString(),
                ]),
            ].join(","),
        );

        const csv = [header, ...csvRows].join("\n");
        const blob = new Blob([csv], { type: "text/csv" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = `squat_session_${new Date().toISOString().slice(0, 19).replace(/:/g, "-")}.csv`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }

    // Viewer frames (memoised)
    // Builds the 3-D viewer data from session log.

    const viewerFrames = useMemo(() => {
        const toViewerFrame = (entry) => {
            const kpMap = {};
            entry.keypoints3d.forEach((kp) => {
                kpMap[kp.name] = kp;
            });
            const joints = {};
            for (const { name } of SQUAT_JOINT_ORDER) {
                const kp = kpMap[name];
                if (!kp) continue;
                joints[name] = {
                    x: kp.x,
                    y: kp.y,
                    z: entry.predictedZ?.[name] ?? kp.z ?? 0,
                };
            }
            return { classification: entry.classification, joints };
        };

        // Strict filter: only replay frames the model labelled as exercise.
        // No fallback slice — if the model returns no 1s the viewer is empty,
        // which makes a broken model visible rather than hiding it.
        return sessionLog.filter((e) => e.startStop === 1).map(toViewerFrame);
    }, [sessionLog]);

    // Render

    const formScore = result?.goodBadScore ?? null;
    const formIsGood = formScore != null && formScore >= goodBadThreshold;
    const formPct = formScore != null ? Math.round(formScore * 100) : 0;
    const threshPct = Math.round(goodBadThreshold * 100);
    const squatScore = result?.squatScore ?? null;

    return (
        <div className="flex flex-col items-center gap-5 px-6 py-8 max-w-3xl mx-auto">
            {/* Header */}
            <div className="text-center">
                <h2 className="text-2xl font-bold text-slate-800">
                    Squat Analyzer
                </h2>
                <p className="text-slate-500 text-sm mt-1">
                    MediaPipe detects all 33 body landmarks. Squat depth is
                    classified by the Python backend.
                </p>
            </div>

            {/* Session name */}
            {supabase && (
                <div className="flex flex-col items-center gap-1">
                    <label
                        htmlFor="session-name"
                        className="text-xs font-semibold text-slate-500 uppercase tracking-wider"
                    >
                        Your name
                    </label>
                    <input
                        id="session-name"
                        type="text"
                        value={sessionName}
                        onChange={(e) => setSessionName(e.target.value)}
                        placeholder="Your name (optional)"
                        disabled={status === "running"}
                        className="ios-card rounded-xl px-4 py-2 text-sm text-slate-700 placeholder-slate-400 border border-slate-200 focus:outline-none focus:ring-2 focus:ring-sky-400 w-72 disabled:opacity-50"
                    />
                </div>
            )}

            {/* Mode toggle */}
            <div className="ios-pill flex rounded-full p-0.5 gap-px">
                {[
                    { id: "webcam", label: "Webcam (Record)" },
                    { id: "upload", label: "Upload Video" },
                    { id: "image", label: "Upload Image" },
                ].map(({ id, label }) => (
                    <button
                        key={id}
                        onClick={() => switchMode(id)}
                        className={`px-5 py-1.5 rounded-full text-sm font-semibold transition-all duration-150 ${
                            inputMode === id
                                ? "bg-white text-slate-800 shadow-sm"
                                : "text-slate-500 hover:text-slate-700"
                        }`}
                    >
                        {label}
                    </button>
                ))}
            </div>

            {/* Hidden capture canvas — receives video frames so the browser
                applies rotation metadata before MediaPipe processes them */}
            <canvas ref={captureCanvasRef} style={{ display: "none" }} />

            {/* Video + canvas overlay */}
            <div
                className="ios-card relative rounded-2xl overflow-hidden bg-black w-full"
                style={{ maxWidth: 640, aspectRatio: "4/3" }}
            >
                <video
                    ref={videoRef}
                    className="absolute inset-0 bg-black w-full h-full"
                    style={{
                        objectFit: "contain",
                        visibility:
                            inputMode === "image" ? "hidden" : "visible",
                    }}
                    muted
                    playsInline
                />
                <canvas
                    ref={canvasRef}
                    width={640}
                    height={480}
                    className="absolute inset-0 pointer-events-none w-full h-full"
                />
                {/* Live quality warning overlay */}
                {qualityIssues.length > 0 && status === "running" && (
                    <div className="absolute top-2 inset-x-2">
                        <div
                            className="rounded-xl px-3 py-2.5"
                            style={{
                                background: "rgba(245,158,11,0.88)",
                                backdropFilter: "blur(14px)",
                                border: "1px solid rgba(251,191,36,0.35)",
                                boxShadow: "0 4px 20px rgba(245,158,11,0.28)",
                            }}
                        >
                            <p className="text-white text-xs font-bold uppercase tracking-wide mb-1">
                                Video quality
                            </p>
                            {qualityIssues.map((issue, i) => (
                                <p
                                    key={i}
                                    className="text-white/95 text-xs leading-snug"
                                >
                                    {issue}
                                </p>
                            ))}
                        </div>
                    </div>
                )}

                {(status === "loading" || status === "analyzing") && (
                    <div className="absolute inset-0 flex items-center justify-center bg-black/60">
                        <p className="text-white text-sm font-semibold animate-pulse">
                            {status === "analyzing"
                                ? `Analyzing ${recordedKpFramesRef.current.length} frames…`
                                : "Loading MediaPipe model…"}
                        </p>
                    </div>
                )}
            </div>

            {/* Controls — webcam */}
            {inputMode === "webcam" && (
                <div className="flex flex-col items-center gap-2">
                    <div className="flex gap-3">
                        {status !== "running" ? (
                            <button
                                onClick={startCamera}
                                disabled={
                                    status === "loading" ||
                                    status === "analyzing"
                                }
                                className="ios-btn ios-btn-primary px-6 py-2 rounded-full text-sm font-semibold disabled:opacity-50"
                            >
                                {status === "loading"
                                    ? "Loading…"
                                    : status === "analyzing"
                                      ? "Analyzing…"
                                      : "Start Recording"}
                            </button>
                        ) : (
                            <button
                                onClick={stopAndAnalyze}
                                className="ios-btn px-6 py-2 rounded-full text-sm font-semibold text-red-600"
                            >
                                Stop & Analyze
                            </button>
                        )}
                    </div>
                </div>
            )}

            {/* Controls — video upload */}
            {inputMode === "upload" && (
                <div className="flex flex-col items-center gap-2">
                    <input
                        ref={fileInputRef}
                        type="file"
                        accept="video/*"
                        className="hidden"
                        onChange={handleVideoChange}
                    />
                    <div className="flex gap-3 items-center flex-wrap justify-center">
                        {status === "running" ? (
                            <>
                                <button
                                    onClick={togglePlayPause}
                                    className="ios-btn px-5 py-2 rounded-full text-sm font-semibold text-slate-700"
                                >
                                    {videoPaused ? "▶ Play" : "⏸ Pause"}
                                </button>
                                <button
                                    onClick={stopAndAnalyze}
                                    className="ios-btn px-5 py-2 rounded-full text-sm font-semibold text-red-600"
                                >
                                    ✕ Stop & Analyze
                                </button>
                            </>
                        ) : (
                            <button
                                onClick={() => fileInputRef.current?.click()}
                                disabled={
                                    status === "loading" ||
                                    status === "analyzing"
                                }
                                className="ios-btn ios-btn-primary px-6 py-2 rounded-full text-sm font-semibold disabled:opacity-50"
                            >
                                {status === "loading" || status === "analyzing"
                                    ? "Loading…"
                                    : "Choose Video…"}
                            </button>
                        )}
                    </div>
                    {status === "finished" && (
                        <p className="text-green-600 text-sm font-semibold">
                            ✓ Analysis complete · review results below
                        </p>
                    )}
                    {uploadedFileName && (
                        <p className="text-slate-400 text-xs">
                            {uploadedFileName}
                        </p>
                    )}
                </div>
            )}

            {/* Controls — image upload */}
            {inputMode === "image" && (
                <div className="flex flex-col items-center gap-2">
                    <input
                        ref={imageInputRef}
                        type="file"
                        accept="image/*"
                        className="hidden"
                        onChange={handleImageChange}
                    />
                    <button
                        onClick={() => imageInputRef.current?.click()}
                        disabled={status === "loading"}
                        className="ios-btn ios-btn-primary px-6 py-2 rounded-full text-sm font-semibold disabled:opacity-50"
                    >
                        {status === "loading" ? "Analysing…" : "Choose Image…"}
                    </button>
                    {uploadedFileName && (
                        <p className="text-slate-400 text-xs">
                            {uploadedFileName}
                        </p>
                    )}
                </div>
            )}

            {/* Error */}
            {status === "error" && (
                <p className="text-red-500 text-sm">
                    {errorMsg || "An error occurred."}
                </p>
            )}

            {/* Quality error card */}
            {qualityError && (
                <div
                    className="ios-card rounded-2xl p-5 w-full"
                    style={{
                        border: "1px solid rgba(251,191,36,0.4)",
                        boxShadow: "0 8px 32px rgba(245,158,11,0.12)",
                    }}
                >
                    <div className="flex items-start gap-3 mb-3">
                        <div
                            className="w-9 h-9 rounded-full flex items-center justify-center text-base font-bold shrink-0"
                            style={{
                                background:
                                    "radial-gradient(circle at 35% 30%, rgba(254,243,199,0.95), rgba(253,230,138,0.85))",
                                border: "1px solid rgba(251,191,36,0.4)",
                                color: "#92400e",
                            }}
                        >
                            !
                        </div>
                        <div>
                            <p className="text-sm font-bold text-amber-700">
                                Video quality too poor to analyze
                            </p>
                            <p className="text-xs text-slate-500 mt-0.5">
                                Fix the issues below and record again
                            </p>
                        </div>
                    </div>
                    <ul className="space-y-2 pl-1">
                        {qualityError.issues.map((issue, i) => (
                            <li key={i} className="flex items-start gap-2">
                                <span className="text-amber-400 font-bold shrink-0 mt-0.5">
                                    ·
                                </span>
                                <span className="text-sm text-slate-600">
                                    {issue}
                                </span>
                            </li>
                        ))}
                    </ul>
                </div>
            )}

            {/* Model quality indicator — Start/Stop MAE from DagsHub */}
            {startStopMetrics?.mae_total_average != null && (
                <p className="text-xs text-slate-400">
                    Start/Stop MAE:{" "}
                    <span className="font-semibold text-slate-600">
                        {startStopMetrics.mae_total_average.toFixed(4)}
                    </span>
                </p>
            )}

            {/* Form quality + threshold */}
            {(formScore != null || squatScore != null) && (
                <div className="ios-card rounded-2xl p-5 w-full">
                    {formScore != null && (
                        <>
                            {/* Header row */}
                            <div className="flex items-center justify-between mb-4">
                                <div>
                                    <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-0.5">
                                        Form Quality
                                    </p>
                                    <p
                                        className={`text-2xl font-bold ${formIsGood ? "text-green-500" : "text-red-500"}`}
                                    >
                                        {formIsGood ? "Good form" : "Bad form"}
                                    </p>
                                    {squatScore != null && (
                                        <p className="text-sm text-slate-500 mt-1">
                                            Score:{" "}
                                            <span className="font-semibold text-slate-700 tabular-nums">
                                                {squatScore.toFixed(2)} / 4
                                            </span>{" "}
                                            (0 good, 4 bad)
                                        </p>
                                    )}
                                </div>
                                <div
                                    className={`w-14 h-14 rounded-full flex items-center justify-center text-xl font-bold ios-card ${
                                        formIsGood
                                            ? "text-green-500"
                                            : "text-red-500"
                                    }`}
                                    style={{
                                        background: formIsGood
                                            ? "radial-gradient(circle at 35% 30%, rgba(220,252,231,0.95), rgba(187,247,208,0.85))"
                                            : "radial-gradient(circle at 35% 30%, rgba(254,226,226,0.95), rgba(252,165,165,0.70))",
                                        border: formIsGood
                                            ? "1px solid rgba(74,222,128,0.35)"
                                            : "1px solid rgba(248,113,113,0.35)",
                                    }}
                                >
                                    {formIsGood ? "✓" : "✗"}
                                </div>
                            </div>

                            {/* Score bar with threshold marker */}
                            <div className="mb-5">
                                <div className="flex justify-between text-xs mb-2">
                                    <span className="text-slate-400">
                                        Score
                                    </span>
                                    <span className="font-bold text-slate-600 tabular-nums">
                                        {formPct}%
                                    </span>
                                </div>
                                <div className="relative h-3 rounded-full bg-slate-100 overflow-visible">
                                    <div
                                        className="h-full rounded-full transition-all duration-700 ease-out"
                                        style={{
                                            width: `${formPct}%`,
                                            background: formIsGood
                                                ? "linear-gradient(to right, #fbbf24, #4ade80)"
                                                : "linear-gradient(to right, #f87171, #fbbf24)",
                                            boxShadow: formIsGood
                                                ? "0 0 12px rgba(74,222,128,0.4)"
                                                : "0 0 12px rgba(248,113,113,0.35)",
                                        }}
                                    />
                                    {/* Threshold marker line */}
                                    <div
                                        className="absolute top-1/2 -translate-y-1/2 w-0.5 h-5 rounded-full bg-slate-500 opacity-60"
                                        style={{
                                            left: `calc(${threshPct}% - 1px)`,
                                        }}
                                    />
                                </div>
                                <p className="text-xs text-slate-300 text-right mt-1 tabular-nums">
                                    threshold at {threshPct}%
                                </p>
                            </div>

                            {/* Threshold slider */}
                            <div>
                                <div className="flex justify-between text-xs mb-2">
                                    <span className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
                                        Threshold
                                    </span>
                                    <span className="font-bold text-slate-600 tabular-nums">
                                        {threshPct}%
                                    </span>
                                </div>
                                <input
                                    type="range"
                                    min="0"
                                    max="100"
                                    step="1"
                                    value={threshPct}
                                    onChange={(e) =>
                                        setGoodBadThreshold(
                                            Number(e.target.value) / 100,
                                        )
                                    }
                                    className="ios-slider"
                                    style={{ width: "100%", display: "block" }}
                                />
                                <div className="flex justify-between text-xs text-slate-300 mt-2">
                                    <span>Bad</span>
                                    <span>Good</span>
                                </div>
                            </div>
                        </>
                    )}
                </div>
            )}

            {/* Session actions */}
            {sessionLog.length > 0 && (
                <div className="flex flex-col items-center gap-2">
                    <p className="text-xs text-slate-400">
                        {sessionLog.length} frame
                        {sessionLog.length !== 1 ? "s" : ""} recorded
                    </p>
                    <div className="flex gap-3 flex-wrap justify-center">
                        {supabase && (
                            <button
                                onClick={handleSave}
                                disabled={isSaving}
                                className="ios-btn ios-btn-primary px-5 py-2 rounded-full text-sm font-semibold disabled:opacity-50"
                            >
                                {isSaving ? "Saving…" : "Save to Database"}
                            </button>
                        )}
                        <button
                            onClick={downloadCSV}
                            className="ios-btn px-5 py-2 rounded-full text-sm font-semibold text-slate-700"
                        >
                            Download CSV
                        </button>
                        <button
                            onClick={clearSession}
                            disabled={status === "running"}
                            className="ios-btn px-5 py-2 rounded-full text-sm font-semibold text-slate-400 disabled:opacity-40"
                        >
                            New Session
                        </button>
                    </div>
                </div>
            )}

            {/* Pipeline timing breakdown */}
            {pipelineTimings != null &&
                (() => {
                    const steps = [
                        {
                            key: "network_ms",
                            label: "Network (round-trip)",
                            sub: "client ↔ server",
                            value: Math.max(
                                0,
                                pipelineTimings.round_trip_ms -
                                    pipelineTimings.total_ms,
                            ),
                        },
                        {
                            key: "start_stop_ms",
                            label: "Start/Stop model",
                            sub: "exercise detection",
                            value: pipelineTimings.start_stop_ms ?? 0,
                        },
                        {
                            key: "z_prediction_ms",
                            label: "MediaPipe Z mapping",
                            sub: "depth reuse",
                            value: pipelineTimings.z_prediction_ms ?? 0,
                        },
                        {
                            key: "goodbad_ms",
                            label: "GoodBad model",
                            sub: "form quality",
                            value: pipelineTimings.goodbad_ms ?? 0,
                        },
                        {
                            key: "scoring_ms",
                            label: "Scoring model",
                            sub: "0 good → 4 bad",
                            value: pipelineTimings.scoring_ms ?? 0,
                        },
                        {
                            key: "feature_build_ms",
                            label: "Feature extraction",
                            sub: "preprocessing",
                            value: pipelineTimings.feature_build_ms ?? 0,
                        },
                    ];
                    const maxVal = Math.max(...steps.map((s) => s.value), 1);
                    return (
                        <div className="ios-card rounded-2xl p-5 w-full">
                            <div className="flex items-baseline justify-between mb-4">
                                <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider">
                                    Pipeline Timing
                                </p>
                                <span className="text-xs font-mono text-slate-400">
                                    Total:{" "}
                                    <span className="text-sky-500 font-semibold">
                                        {(
                                            pipelineTimings.round_trip_ms / 1000
                                        ).toFixed(2)}
                                        s
                                    </span>
                                </span>
                            </div>
                            <div className="space-y-3">
                                {steps.map(({ key, label, sub, value }) => (
                                    <div key={key}>
                                        <div className="flex items-baseline justify-between mb-1">
                                            <div>
                                                <span className="text-xs font-semibold text-slate-600">
                                                    {label}
                                                </span>
                                                <span className="ml-2 text-xs text-slate-400">
                                                    {sub}
                                                </span>
                                            </div>
                                            <span className="text-xs font-mono font-bold text-slate-600 tabular-nums">
                                                {value < 1
                                                    ? "<1"
                                                    : Math.round(value)}
                                                ms
                                            </span>
                                        </div>
                                        <div className="h-1.5 rounded-full bg-slate-100 overflow-hidden">
                                            <div
                                                className="h-full rounded-full transition-all duration-700 ease-out"
                                                style={{
                                                    width: `${Math.max(1, (value / maxVal) * 100)}%`,
                                                    background:
                                                        key === "network_ms"
                                                            ? "linear-gradient(to right, #94a3b8, #64748b)"
                                                            : key ===
                                                                "start_stop_ms"
                                                              ? "linear-gradient(to right, #38bdf8, #0ea5e9)"
                                                              : key ===
                                                                  "z_prediction_ms"
                                                                ? "linear-gradient(to right, #a78bfa, #7c3aed)"
                                                                : key ===
                                                                    "goodbad_ms"
                                                                  ? "linear-gradient(to right, #4ade80, #16a34a)"
                                                                  : key ===
                                                                      "scoring_ms"
                                                                    ? "linear-gradient(to right, #f97316, #ea580c)"
                                                                    : "linear-gradient(to right, #fbbf24, #d97706)",
                                                }}
                                            />
                                        </div>
                                    </div>
                                ))}
                            </div>
                            <div className="mt-4 pt-3 border-t border-slate-100 flex justify-between text-xs text-slate-400">
                                <span>Backend processing</span>
                                <span className="font-mono font-semibold text-slate-600">
                                    {Math.round(pipelineTimings.total_ms)}ms
                                </span>
                            </div>
                        </div>
                    );
                })()}

            {/* 3-D interactive skeleton viewer */}
            <Skeleton3DViewer
                frames={viewerFrames}
                liveFrameRef={status === "running" ? liveFrame3dRef : null}
            />
        </div>
    );
}
