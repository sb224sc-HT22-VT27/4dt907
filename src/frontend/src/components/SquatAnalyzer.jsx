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

// Canonical joint order used by the GRU z-predictor model (matches Kinect CSV
// column order after normalisation). Each entry is { name, idx }.
// Features per frame: [j0_x, j0_y, j1_x, j1_y, … ] → 13 × 2 = 26 floats.
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
const SQUAT_JOINT_NAMES_SET = new Set(SQUAT_JOINT_ORDER.map((j) => j.name));

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

const CLASSIFICATION_COLORS = {
    Deep: "text-green-600",
    Shallow: "text-amber-600",
    Invalid: "text-red-600",
};

const MODEL_URL =
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task";
const WASM_URL =
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm";
const ESM_URL =
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/+esm";

// Number of frames the GRU model expects as a sequence window.
const SEQ_LEN = 30;

/**
 * Build a 26-float feature vector for one frame from MediaPipe world landmarks.
 * Columns: [j0_x, j0_y, j1_x, j1_y, …] in SQUAT_JOINT_ORDER.
 */
function buildFrameFeatures(worldLandmarks) {
    return SQUAT_JOINT_ORDER.flatMap(({ idx }) => {
        const lm = worldLandmarks?.[idx];
        return lm ? [lm.x, lm.y] : [0, 0];
    });
}

/**
 * Pad a frame buffer to exactly targetLen by repeating the first frame,
 * or slice the last targetLen frames if the buffer is too long.
 */
function padOrSlice(buffer, targetLen) {
    if (buffer.length >= targetLen) return buffer.slice(-targetLen);
    const pad = buffer[0] ?? Array(SQUAT_JOINT_ORDER.length * 2).fill(0);
    return [...Array(targetLen - buffer.length).fill(pad), ...buffer];
}

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

// ── 3-D Skeleton Viewer ──────────────────────────────────────────────────────

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

function Skeleton3DViewer({ frames }) {
    const canvasRef = useRef(null);
    const [playing, setPlaying] = useState(false);
    const [frameIdx, setFrameIdx] = useState(0);
    const rotation = useRef({ x: 15, y: 25 });
    const dragRef = useRef(null);
    const zoomRef = useRef(260);
    const playTimerRef = useRef(null);
    const frameIdxRef = useRef(0);
    const displayedFrameIdx = useMemo(() => {
        if (frames.length === 0) return 0;
        if (!playing) return frames.length - 1; // follow latest when not replaying
        return Math.min(frameIdx, frames.length - 1); // clamp while replaying
    }, [frames.length, playing, frameIdx]);

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

    function drawFrame(frameData) {
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
        const cy = H / 2 + 20;

        // Project all joints
        const pos = {};
        for (const { name } of SQUAT_JOINT_ORDER) {
            const j = frameData.joints?.[name];
            if (!j) continue;
            pos[name] = project3D(j.x, j.y, j.z, rx, ry, scale, cx, cy);
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
                      : "#dc2626";
            ctx.textAlign = "center";
            ctx.fillText(cls, cx, 18);
        }

        // Frame counter
        ctx.font = "11px system-ui, sans-serif";
        ctx.fillStyle = "rgba(148,163,184,0.8)";
        ctx.textAlign = "right";
        ctx.fillText(
            `${displayedFrameIdx + 1} / ${frames.length}`,
            W - 6,
            H - 6,
        );
    }

    // Draw whenever frameIdx, rotation or zoom changes
    useEffect(() => {
        drawFrame(frames[displayedFrameIdx]);
    }); // run on every render — canvas state is imperative

    // ── Drag to rotate ───────────────────────────────────────────────────────

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
        drawFrame(frames[frameIdxRef.current]);
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
        drawFrame(frames[frameIdxRef.current]);
    }

    if (frames.length === 0) return null;

    return (
        <div className="ios-card rounded-2xl p-4 w-full flex flex-col items-center gap-3">
            <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider self-start">
                3-D Skeleton Replay
                <span className="ml-2 normal-case font-normal text-slate-300">
                    — drag to rotate · scroll to zoom
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
            <div className="flex items-center gap-3 w-full">
                <button
                    onClick={() => {
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
                <input
                    type="range"
                    min={0}
                    max={frames.length - 1}
                    value={displayedFrameIdx}
                    onChange={(e) => {
                        setPlaying(false);
                        setFrameIdx(Number(e.target.value));
                    }}
                    className="flex-1 accent-sky-400"
                />
            </div>

            <p className="text-xs text-slate-400 self-start">
                Frame {frameIdx + 1} of {frames.length} · z from DL model
            </p>
        </div>
    );
}

// ── Main component ────────────────────────────────────────────────────────────

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
    // Ring buffer: array of 26-float feature vectors, oldest → newest
    const frameBufferRef = useRef([]);

    const [inputMode, setInputMode] = useState("webcam");
    const [sessionName, setSessionName] = useState("");
    const [status, setStatus] = useState("idle");
    const [result, setResult] = useState(null);
    const [errorMsg, setErrorMsg] = useState("");
    const [uploadedFileName, setUploadedFileName] = useState("");
    const [videoPaused, setVideoPaused] = useState(false);
    const [allKeypoints, setAllKeypoints] = useState([]);
    const [sessionLog, setSessionLog] = useState([]);
    const [isSaving, setIsSaving] = useState(false);
    const [predictedZByName, setPredictedZByName] = useState({});

    const sessionNameRef = useRef("");
    const predictedZByNameRef = useRef({});

    // ── Preload VIDEO landmarker on mount ────────────────────────────────────
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

    // ── Helpers ──────────────────────────────────────────────────────────────

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
        frameBufferRef.current = [];
        const canvas = canvasRef.current;
        if (canvas) {
            canvas
                .getContext("2d")
                .clearRect(0, 0, canvas.width, canvas.height);
        }
        setAllKeypoints([]);
        setResult(null);
        setVideoPaused(false);
        setPredictedZByName({});
        predictedZByNameRef.current = {};
        setStatus(nextStatus);
    }, []);

    // Discards accumulated session data (call explicitly for "new session").
    const clearSession = useCallback(() => {
        sessionLogRef.current = [];
        setSessionLog([]);
        setUploadedFileName("");
        setErrorMsg("");
    }, []);

    // Full reset: stop capture + discard data (used when switching modes).
    const stopAll = useCallback(() => {
        stopCapture("idle");
        sessionLogRef.current = [];
        setSessionLog([]);
        setUploadedFileName("");
        setErrorMsg("");
    }, [stopCapture]);

    function mapAllKeypoints(landmarks) {
        return landmarks.map((lm, i) => ({
            index: i,
            name: ALL_LANDMARK_NAMES[i] ?? `landmark_${i}`,
            x: lm.x,
            y: lm.y,
            predictedZ:
                predictedZByNameRef.current[
                    ALL_LANDMARK_NAMES[i] ?? `landmark_${i}`
                ] ?? null,
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

    // ── Video-ended auto-finish ──────────────────────────────────────────────

    useEffect(() => {
        if (status !== "running" || inputMode !== "upload") return;
        const video = videoRef.current;
        if (!video) return;
        const handleEnded = () => stopCapture("finished");
        video.addEventListener("ended", handleEnded);
        return () => video.removeEventListener("ended", handleEnded);
    }, [status, inputMode, stopCapture]);

    // ── Webcam ───────────────────────────────────────────────────────────────

    const startCamera = useCallback(async () => {
        setStatus("loading");
        setErrorMsg("");
        try {
            if (!landmarkerRef.current) {
                landmarkerRef.current = await createVideoLandmarker();
            }
            const stream = await navigator.mediaDevices.getUserMedia({
                video: { width: 640, height: 480 },
            });
            videoRef.current.srcObject = stream;
            await videoRef.current.play();
            setStatus("running");
        } catch (err) {
            setStatus("error");
            setErrorMsg(String(err));
        }
    }, []);

    // ── Video upload ─────────────────────────────────────────────────────────

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
                if (!landmarkerRef.current) {
                    landmarkerRef.current = await createVideoLandmarker();
                }
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

    // ── Image upload ─────────────────────────────────────────────────────────

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

                const detection = imageLandmarker.detect(img);

                if (detection.landmarks?.length > 0) {
                    drawSkeleton(canvas, drawW, drawH, detection, {
                        x: offsetX,
                        y: offsetY,
                    });
                    setAllKeypoints(mapAllKeypoints(detection.landmarks[0]));
                    const kp3d = filterSquatKeypoints3d(
                        detection.worldLandmarks?.[0] ?? [],
                    );
                    sendFrame(kp3d);
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

    // ── Detection loop ───────────────────────────────────────────────────────
    //
    // Detection: ~20 Hz (every 3 rAF frames).
    // Ring buffer update: every detection frame (~20 Hz).
    // Backend call + z-prediction: ~2 Hz (every 15 detection frames).

    useEffect(() => {
        if (status !== "running") return;

        let frameCounter = 0;
        const DETECT_EVERY = 3; // ~20 Hz detection
        const DEBUG_EVERY = 6; // ~10 Hz debug panel update
        const BACKEND_EVERY = 15; // ~2 Hz backend + z-prediction

        function detect(timestamp) {
            const video = videoRef.current;
            const canvas = canvasRef.current;
            const landmarker = landmarkerRef.current;

            if (!video || !canvas || !landmarker || video.readyState < 2) {
                rafRef.current = requestAnimationFrame(detect);
                return;
            }
            if (video.paused || video.ended) {
                rafRef.current = requestAnimationFrame(detect);
                return;
            }
            if (video.videoWidth === 0 || video.videoHeight === 0) {
                rafRef.current = requestAnimationFrame(detect);
                return;
            }
            if (timestamp <= lastTimestampRef.current) {
                rafRef.current = requestAnimationFrame(detect);
                return;
            }
            lastTimestampRef.current = timestamp;
            frameCounter++;

            // ── Run MediaPipe detection ──
            if (frameCounter % DETECT_EVERY === 0) {
                try {
                    lastDetectionRef.current = landmarker.detectForVideo(
                        video,
                        timestamp,
                    );
                } catch {
                    // Transient error — skip detection, keep loop alive.
                }

                // Update ring buffer from world landmarks at detection rate (~20 Hz)
                const det = lastDetectionRef.current;
                if (det?.worldLandmarks?.length > 0) {
                    const frame = buildFrameFeatures(det.worldLandmarks[0]);
                    const buf = frameBufferRef.current;
                    buf.push(frame);
                    if (buf.length > SEQ_LEN) buf.shift();
                }
            }

            // ── Redraw canvas at full rAF rate ──
            const detection = lastDetectionRef.current;
            const vw = video.videoWidth;
            const vh = video.videoHeight;
            if (canvas.width !== vw || canvas.height !== vh) {
                canvas.width = vw;
                canvas.height = vh;
            }
            canvas.getContext("2d").clearRect(0, 0, vw, vh);
            if (detection) drawSkeleton(canvas, vw, vh, detection);

            // ── Debug panel (~10 Hz) ──
            if (frameCounter % DEBUG_EVERY === 0) {
                if (detection?.landmarks?.length > 0) {
                    setAllKeypoints(mapAllKeypoints(detection.landmarks[0]));
                } else {
                    setAllKeypoints([]);
                }
            }

            // ── Backend call + z-prediction (~2 Hz) ──
            if (
                frameCounter % BACKEND_EVERY === 0 &&
                detection?.landmarks?.length > 0
            ) {
                const kp3d = filterSquatKeypoints3d(
                    detection.worldLandmarks?.[0] ?? [],
                );
                sendFrame(kp3d);

                setAllKeypoints(mapAllKeypoints(detection.landmarks[0]));

                // Sequence-based z prediction (does NOT call per-keypoint endpoint)
                const worldLms = detection.worldLandmarks?.[0];
                if (worldLms) {
                    const buf = frameBufferRef.current;
                    if (buf.length > 0) {
                        const sequence = padOrSlice(buf, SEQ_LEN);
                        fetch(apiUrl("/api/v1/z-predictor/predict-sequence"), {
                            method: "POST",
                            headers: { "Content-Type": "application/json" },
                            body: JSON.stringify({ sequence }),
                        })
                            .then((r) => (r.ok ? r.json() : null))
                            .then((data) => {
                                if (!data) return;
                                const preds = data?.predictions ?? [];
                                const map = Object.fromEntries(
                                    SQUAT_JOINT_ORDER.map(({ name }, i) => [
                                        name,
                                        Number(preds[i] ?? 0),
                                    ]),
                                );
                                predictedZByNameRef.current = map;
                                setPredictedZByName(map);
                            })
                            .catch(() => {});
                    }
                }
            }

            rafRef.current = requestAnimationFrame(detect);
        }

        rafRef.current = requestAnimationFrame(detect);
        return () => cancelAnimationFrame(rafRef.current);
    }, [status]);

    useEffect(() => {
        predictedZByNameRef.current = predictedZByName;
    }, [predictedZByName]);

    // ── Backend + session log ────────────────────────────────────────────────

    sessionNameRef.current = sessionName;

    async function sendFrame(keypoints3d) {
        let data = null;
        try {
            const res = await fetch(apiUrl("/api/v1/squat/classify"), {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ keypoints_3d: keypoints3d }),
            });
            if (res.ok) data = await res.json();
        } catch {
            /* network errors silently ignored */
        }

        if (data) setResult(data);

        const entry = {
            timestamp: Date.now(),
            keypoints3d,
            predictedZ: { ...predictedZByNameRef.current },
            classification: data?.classification ?? null,
            confidence: data?.confidence ?? null,
        };
        sessionLogRef.current = [...sessionLogRef.current, entry];
        setSessionLog(sessionLogRef.current);
    }

    // ── Manual save to Supabase ───────────────────────────────────────────────

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

    // ── CSV export ────────────────────────────────────────────────────────────

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

    // ── Viewer frames (memoised) ──────────────────────────────────────────────
    // Builds the 3-D viewer data from session log: x/y from MediaPipe world coords,
    // z from DL model prediction (falls back to MediaPipe world z if unavailable).

    const viewerFrames = useMemo(
        () =>
            sessionLog.map((entry) => {
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
            }),
        [sessionLog],
    );

    // ── Render ────────────────────────────────────────────────────────────────

    const colorClass = result
        ? (CLASSIFICATION_COLORS[result.classification] ?? "text-slate-700")
        : "";

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
                    { id: "webcam", label: "Live Camera" },
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

            {/* Video + canvas overlay */}
            <div
                className="ios-card relative rounded-2xl overflow-hidden bg-black"
                style={{ width: 640, height: 480 }}
            >
                <video
                    ref={videoRef}
                    className="block bg-black"
                    style={{
                        width: 640,
                        height: 480,
                        objectFit: "contain",
                        visibility:
                            inputMode === "image" ? "hidden" : "visible",
                    }}
                    muted
                    playsInline
                />
                <canvas
                    ref={canvasRef}
                    className="absolute inset-0 pointer-events-none"
                    style={{ width: 640, height: 480 }}
                />
                {status === "loading" && (
                    <div className="absolute inset-0 flex items-center justify-center bg-black/60">
                        <p className="text-white text-sm font-semibold animate-pulse">
                            Loading MediaPipe model…
                        </p>
                    </div>
                )}
            </div>

            {/* Controls — webcam */}
            {inputMode === "webcam" && (
                <div className="flex gap-3">
                    {status !== "running" ? (
                        <button
                            onClick={startCamera}
                            disabled={status === "loading"}
                            className="ios-btn ios-btn-primary px-6 py-2 rounded-full text-sm font-semibold disabled:opacity-50"
                        >
                            {status === "loading" ? "Loading…" : "Start Camera"}
                        </button>
                    ) : (
                        <button
                            onClick={() => stopCapture("idle")}
                            className="ios-btn px-6 py-2 rounded-full text-sm font-semibold text-red-600"
                        >
                            Stop
                        </button>
                    )}
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
                                    onClick={() => stopCapture("idle")}
                                    className="ios-btn px-5 py-2 rounded-full text-sm font-semibold text-red-600"
                                >
                                    ✕ Stop
                                </button>
                            </>
                        ) : (
                            <button
                                onClick={() => fileInputRef.current?.click()}
                                disabled={status === "loading"}
                                className="ios-btn ios-btn-primary px-6 py-2 rounded-full text-sm font-semibold disabled:opacity-50"
                            >
                                {status === "loading"
                                    ? "Loading…"
                                    : "Choose Video…"}
                            </button>
                        )}
                    </div>
                    {status === "finished" && (
                        <p className="text-green-600 text-sm font-semibold">
                            ✓ Analysis complete — review results below
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

            {/* Classification result */}
            {result && (
                <div className="ios-card rounded-2xl p-5 text-center w-72">
                    <p className="text-slate-400 text-xs uppercase tracking-wider mb-1">
                        Classification
                    </p>
                    <p className={`text-4xl font-bold mb-4 ${colorClass}`}>
                        {result.classification}
                    </p>
                    <div className="flex justify-around text-sm">
                        <div>
                            <p className="text-slate-400 text-xs">Left knee</p>
                            <p className="font-mono text-slate-700 font-semibold">
                                {result.left_knee_angle?.toFixed(1)}°
                            </p>
                        </div>
                        <div>
                            <p className="text-slate-400 text-xs">Right knee</p>
                            <p className="font-mono text-slate-700 font-semibold">
                                {result.right_knee_angle?.toFixed(1)}°
                            </p>
                        </div>
                        {result.confidence != null && (
                            <div>
                                <p className="text-slate-400 text-xs">
                                    Confidence
                                </p>
                                <p className="font-mono text-slate-700 font-semibold">
                                    {(result.confidence * 100).toFixed(0)}%
                                </p>
                            </div>
                        )}
                    </div>
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

            {/* Debug panel — 33 keypoints with x/y (MediaPipe) + predicted z (DL) */}
            {(status === "running" || allKeypoints.length > 0) && (
                <div className="ios-card rounded-2xl p-4 w-full">
                    <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-1">
                        Keypoints
                        {status === "running" && (
                            <span className="ml-2 normal-case font-normal text-slate-300">
                                — live
                            </span>
                        )}
                    </p>
                    <p className="text-xs text-slate-300 mb-3">
                        x · y from MediaPipe &nbsp;·&nbsp; z from DL model
                        <span className="ml-2 text-sky-400 font-semibold">
                            (squat joints highlighted)
                        </span>
                    </p>
                    {allKeypoints.length === 0 ? (
                        <p className="text-xs text-slate-400 italic">
                            Waiting for pose detection…
                        </p>
                    ) : (
                        <div className="grid grid-cols-2 sm:grid-cols-3 gap-y-0.5 gap-x-4 text-xs font-mono max-h-56 overflow-y-auto pr-1">
                            {allKeypoints.map((kp) => {
                                const isSquat = SQUAT_INDICES.has(kp.index);
                                return (
                                    <div
                                        key={kp.index}
                                        className={`flex items-baseline gap-1.5 py-0.5 ${
                                            isSquat
                                                ? "text-sky-600 font-bold"
                                                : "text-slate-400"
                                        }`}
                                    >
                                        <span className="w-5 text-right text-slate-300">
                                            {kp.index}
                                        </span>
                                        <span className="w-28 truncate">
                                            {kp.name}
                                        </span>
                                        <span className="tabular-nums">
                                            {kp.x.toFixed(3)}
                                        </span>
                                        <span className="tabular-nums">
                                            {kp.y.toFixed(3)}
                                        </span>
                                        {isSquat && (
                                            <span
                                                className={`tabular-nums ${
                                                    kp.predictedZ == null
                                                        ? "text-slate-500 italic"
                                                        : "text-emerald-500"
                                                }`}
                                            >
                                                {kp.predictedZ == null
                                                    ? "—"
                                                    : kp.predictedZ.toFixed(3)}
                                            </span>
                                        )}
                                    </div>
                                );
                            })}
                        </div>
                    )}
                </div>
            )}

            {/* 3-D interactive skeleton viewer */}
            <Skeleton3DViewer frames={viewerFrames} />
        </div>
    );
}
