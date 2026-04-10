// SquatAnalyzer: uses MediaPipe Pose (tasks-vision) to detect squat keypoints
// from a live webcam feed, an uploaded video file, or a static image, then
// sends them to the Python backend for angle calculation and depth
// classification (Deep / Shallow / Invalid).
// Accumulated frames can be saved to Supabase or exported as CSV on demand.

import { useEffect, useRef, useState, useCallback } from "react";
import { apiUrl } from "../apiBase";
import supabase from "../supabaseClient";

// MediaPipe keypoint indices used for squat analysis
// https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker
const SQUAT_LANDMARK_NAMES = {
    // TODO: Take updated version from provided A11 .csv files
    11: "left_shoulder",
    12: "right_shoulder",
    13: "left_elbow",
    14: "right_elbow",
    23: "left_hip",
    24: "right_hip",
    25: "left_knee",
    26: "right_knee",
    27: "left_ankle",
    28: "right_ankle",
    29: "left_heel",
    30: "right_heel",
    31: "left_foot_index",
    32: "right_foot_index",
};

// All 33 MediaPipe landmark names in index order.
const ALL_LANDMARK_NAMES = [
    "nose",            // 0
    "left_eye_inner",  // 1
    "left_eye",        // 2
    "left_eye_outer",  // 3
    "right_eye_inner", // 4
    "right_eye",       // 5
    "right_eye_outer", // 6
    "left_ear",        // 7
    "right_ear",       // 8
    "mouth_left",      // 9
    "mouth_right",     // 10
    "left_shoulder",   // 11
    "right_shoulder",  // 12
    "left_elbow",      // 13
    "right_elbow",     // 14
    "left_wrist",      // 15
    "right_wrist",     // 16
    "left_pinky",      // 17
    "right_pinky",     // 18
    "left_index",      // 19
    "right_index",     // 20
    "left_thumb",      // 21
    "right_thumb",     // 22
    "left_hip",        // 23
    "right_hip",       // 24
    "left_knee",       // 25
    "right_knee",      // 26
    "left_ankle",      // 27
    "right_ankle",     // 28
    "left_heel",       // 29
    "right_heel",      // 30
    "left_foot_index", // 31
    "right_foot_index",// 32
];

// Full body skeleton connections (MediaPipe BlazePose topology).
const POSE_CONNECTIONS = [
    [0, 1], [1, 2], [2, 3], [3, 7],
    [0, 4], [4, 5], [5, 6], [6, 8],
    [9, 10],
    [11, 12], [11, 23], [12, 24], [23, 24],
    [11, 13], [13, 15], [15, 17], [15, 19], [15, 21], [17, 19],
    [12, 14], [14, 16], [16, 18], [16, 20], [16, 22], [18, 20],
    [23, 25], [25, 27], [27, 29], [27, 31], [29, 31],
    [24, 26], [26, 28], [28, 30], [28, 32], [30, 32],
];

// Indices that are squat-relevant (highlighted brighter).
// TODO: Take updated version from provided A11 .csv files
const SQUAT_INDICES = new Set([11, 12, 13, 14, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32]);

const CLASSIFICATION_COLORS = {
    Deep:    "text-green-600",
    Shallow: "text-amber-600",
    Invalid: "text-red-600",
};

const MODEL_URL =
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task";
const WASM_URL =
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm";
const ESM_URL =
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/+esm";

/**
 * Create a new PoseLandmarker for continuous video/webcam detection.
 * No module-level caching — instances are owned and cleaned up by the component.
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
 * Caller is responsible for calling .close() immediately after use.
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
 * Filter a MediaPipe landmarks array to only the joints needed for a squat.
 */
function filterSquatKeypoints(landmarks, is3d = false) {
    if (!landmarks) return [];
    return Object.entries(SQUAT_LANDMARK_NAMES).flatMap(([idx, name]) => {
        const lm = landmarks[Number(idx)];
        if (!lm) return [];
        const kp = { name, x: lm.x, y: lm.y, score: lm.visibility ?? null };
        if (is3d) kp.z = lm.z ?? 0;
        return [kp];
    });
}

/**
 * Draw the full-body skeleton on a canvas.
 * @param {HTMLCanvasElement} canvas
 * @param {number}  w        - Draw-area width in pixels.
 * @param {number}  h        - Draw-area height in pixels.
 * @param {object}  detection - MediaPipe detection result.
 * @param {{ x: number, y: number }} [off] - Top-left offset for letterboxed images.
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
        ctx.strokeStyle = isSquatBone ? "rgba(244,114,182,0.92)" : "rgba(148,163,184,0.55)";
        ctx.lineWidth   = isSquatBone ? 3 : 1.5;
        ctx.stroke();
    }

    for (let idx = 0; idx < lms.length; idx++) {
        const lm = lms[idx];
        if (!lm) continue;
        const isSquat = SQUAT_INDICES.has(idx);
        ctx.beginPath();
        ctx.arc(off.x + lm.x * w, off.y + lm.y * h, isSquat ? 6 : 3, 0, Math.PI * 2);
        ctx.fillStyle = isSquat ? "#38bdf8" : "rgba(255,255,255,0.65)";
        ctx.fill();
    }
}

export default function SquatAnalyzer() {
    const videoRef         = useRef(null);
    const canvasRef        = useRef(null);
    const fileInputRef     = useRef(null);
    const imageInputRef    = useRef(null);
    const landmarkerRef    = useRef(null);   // VIDEO-mode landmarker owned here
    const rafRef           = useRef(null);
    const lastTimestampRef = useRef(-1);
    const videoBlobUrlRef  = useRef(null);
    const lastDetectionRef = useRef(null);   // cached between rAF frames
    const sessionLogRef    = useRef([]);     // readable inside async callbacks

    const [inputMode,        setInputMode]        = useState("webcam"); // webcam | upload | image
    const [sessionName,      setSessionName]      = useState("");
    const [status,           setStatus]           = useState("idle");   // idle | loading | running | error
    const [result,           setResult]           = useState(null);
    const [errorMsg,         setErrorMsg]         = useState("");
    const [uploadedFileName, setUploadedFileName] = useState("");
    const [videoPaused,      setVideoPaused]      = useState(false);
    const [allKeypoints,     setAllKeypoints]     = useState([]);
    const [sessionLog,       setSessionLog]       = useState([]);
    const [isSaving,         setIsSaving]         = useState(false);

    // ── Preload VIDEO landmarker on mount; clean up on unmount ───────────────
    useEffect(() => {
        let cancelled = false;
        createVideoLandmarker()
            .then((lm) => {
                if (cancelled) { lm.close?.(); return; }
                landmarkerRef.current = lm;
            })
            .catch(() => {}); // CDN unavailable — will retry on first startCamera

        return () => {
            cancelled = true;
            if (landmarkerRef.current) {
                landmarkerRef.current.close?.();
                landmarkerRef.current = null;
            }
        };
    }, []);

    // ── Helpers ──────────────────────────────────────────────────────────────

    function revokeBlobUrl() {
        if (videoBlobUrlRef.current) {
            URL.revokeObjectURL(videoBlobUrlRef.current);
            videoBlobUrlRef.current = null;
        }
    }

    const stopAll = useCallback(() => {
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
        setStatus("idle");
        setVideoPaused(false);
    }, []);

    const switchMode = useCallback(
        (newMode) => {
            stopAll();
            setInputMode(newMode);
            setResult(null);
            setErrorMsg("");
            setUploadedFileName("");
            setAllKeypoints([]);
            const canvas = canvasRef.current;
            if (canvas) canvas.getContext("2d").clearRect(0, 0, canvas.width, canvas.height);
        },
        [stopAll],
    );

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
                // Create a temporary IMAGE-mode landmarker, use it, then close it.
                // Never reuse the VIDEO landmarker for image mode — setOptions
                // switching corrupts the inference graph.
                imageLandmarker = await createImageLandmarker();

                const img = new Image();
                img.src = blobUrl;
                await new Promise((res, rej) => { img.onload = res; img.onerror = rej; });

                const DISPLAY_W = 640, DISPLAY_H = 480;
                const scale   = Math.min(DISPLAY_W / img.naturalWidth, DISPLAY_H / img.naturalHeight);
                const drawW   = img.naturalWidth  * scale;
                const drawH   = img.naturalHeight * scale;
                const offsetX = (DISPLAY_W - drawW) / 2;
                const offsetY = (DISPLAY_H - drawH) / 2;

                const canvas = canvasRef.current;
                canvas.width  = DISPLAY_W;
                canvas.height = DISPLAY_H;
                const ctx = canvas.getContext("2d");
                ctx.fillStyle = "#000";
                ctx.fillRect(0, 0, DISPLAY_W, DISPLAY_H);
                ctx.drawImage(img, offsetX, offsetY, drawW, drawH);

                const detection = imageLandmarker.detect(img);

                if (detection.landmarks?.length > 0) {
                    drawSkeleton(canvas, drawW, drawH, detection, { x: offsetX, y: offsetY });
                    setAllKeypoints(detection.landmarks[0].map((lm, i) => ({
                        index: i,
                        name:  ALL_LANDMARK_NAMES[i] ?? `landmark_${i}`,
                        x: lm.x, y: lm.y, z: lm.z ?? 0,
                        visibility: lm.visibility ?? 0,
                    })));
                    const kp2d = filterSquatKeypoints(detection.landmarks[0], false);
                    const kp3d = filterSquatKeypoints(detection.worldLandmarks?.[0] ?? [], true);
                    sendFrame(kp2d, kp3d);
                }
                setStatus("idle");
            } catch (err) {
                setStatus("error");
                setErrorMsg(String(err));
            } finally {
                imageLandmarker?.close?.(); // release GPU resources immediately
                URL.revokeObjectURL(blobUrl);
                if (imageInputRef.current) imageInputRef.current.value = "";
            }
        },
        [stopAll], // eslint-disable-line react-hooks/exhaustive-deps
    );

    const togglePlayPause = useCallback(() => {
        const video = videoRef.current;
        if (!video) return;
        if (video.paused) { video.play(); setVideoPaused(false); }
        else              { video.pause(); setVideoPaused(true); }
    }, []);

    // ── Detection loop ───────────────────────────────────────────────────────
    //
    // Detection rate is capped at ~20 fps (every 3 rAF frames) so the CPU
    // inference doesn't starve the main thread. The canvas is redrawn on every
    // rAF using the last cached detection result, keeping lines smooth.

    useEffect(() => {
        if (status !== "running") return;

        let frameCounter = 0;
        const DETECT_EVERY     = 3;  // ~20 Hz detection
        const DEBUG_EVERY      = 6;  // ~10 Hz debug panel update
        const BACKEND_EVERY    = 15; // ~2 Hz backend + session log

        function detect(timestamp) {
            const video      = videoRef.current;
            const canvas     = canvasRef.current;
            const landmarker = landmarkerRef.current;

            if (!video || !canvas || !landmarker || video.readyState < 2) {
                rafRef.current = requestAnimationFrame(detect);
                return;
            }
            if (video.paused || video.ended) {
                rafRef.current = requestAnimationFrame(detect);
                return;
            }
            // Guard: MediaPipe crashes when video dimensions are 0.
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

            // ── Run detection at reduced rate ──
            if (frameCounter % DETECT_EVERY === 0) {
                try {
                    lastDetectionRef.current = landmarker.detectForVideo(video, timestamp);
                } catch {
                    // Transient GPU error — skip this detection, keep loop alive.
                }
            }

            // ── Always redraw canvas at full rAF rate for smooth lines ──
            const detection = lastDetectionRef.current;
            const vw = video.videoWidth;
            const vh = video.videoHeight;
            if (canvas.width !== vw || canvas.height !== vh) {
                canvas.width  = vw;
                canvas.height = vh;
            }
            canvas.getContext("2d").clearRect(0, 0, vw, vh);
            if (detection) drawSkeleton(canvas, vw, vh, detection);

            // ── Debug panel (~10 Hz) ──
            if (frameCounter % DEBUG_EVERY === 0) {
                if (detection?.landmarks?.length > 0) {
                    setAllKeypoints(detection.landmarks[0].map((lm, i) => ({
                        index: i,
                        name:  ALL_LANDMARK_NAMES[i] ?? `landmark_${i}`,
                        x: lm.x, y: lm.y, z: lm.z ?? 0,
                        visibility: lm.visibility ?? 0,
                    })));
                } else {
                    setAllKeypoints([]);
                }
            }

            // ── Throttled backend call + session log (~2 Hz) ──
            if (frameCounter % BACKEND_EVERY === 0 && detection?.landmarks?.length > 0) {
                const kp2d = filterSquatKeypoints(detection.landmarks[0], false);
                const kp3d = filterSquatKeypoints(detection.worldLandmarks?.[0] ?? [], true);
                sendFrame(kp2d, kp3d);
            }

            rafRef.current = requestAnimationFrame(detect);
        }

        rafRef.current = requestAnimationFrame(detect);
        return () => cancelAnimationFrame(rafRef.current);
    }, [status]); // eslint-disable-line react-hooks/exhaustive-deps

    // ── Backend + session log ────────────────────────────────────────────────

    async function sendFrame(keypoints2d, keypoints3d) {
        let data = null;
        try {
            const res = await fetch(apiUrl("/api/v1/squat/classify"), {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ keypoints_2d: keypoints2d, keypoints_3d: keypoints3d }),
            });
            if (res.ok) data = await res.json();
        } catch { /* network errors silently ignored */ }

        if (data) setResult(data);

        const entry = {
            timestamp:        Date.now(),
            classification:   data?.classification   ?? null,
            left_knee_angle:  data?.left_knee_angle  ?? null,
            right_knee_angle: data?.right_knee_angle ?? null,
            confidence:       data?.confidence       ?? null,
            keypoints2d,
            keypoints3d,
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
            id_name:       idName,
            raw_keypoints: {
                "2d": e.keypoints2d.map(({ x, y }) => [x, y]),
                "3d": e.keypoints3d.map(({ x, y, z }) => [x, y, z ?? 0]),
            },
            score:          e.confidence,
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

        const kp2dNames = log[0].keypoints2d.map((kp) => kp.name);
        const kp3dNames = log[0].keypoints3d.map((kp) => kp.name);

        const header = [
            "timestamp_ms", "classification",
            "left_knee_angle_deg", "right_knee_angle_deg", "confidence",
            ...kp2dNames.flatMap((n) => [`${n}_x`, `${n}_y`]),
            ...kp3dNames.flatMap((n) => [`${n}_3d_x`, `${n}_3d_y`, `${n}_3d_z`]),
        ].join(",");

        const csvRows = log.map((e) => [
            e.timestamp,
            e.classification   ?? "",
            e.left_knee_angle?.toFixed(2)  ?? "",
            e.right_knee_angle?.toFixed(2) ?? "",
            e.confidence?.toFixed(3)       ?? "",
            ...e.keypoints2d.flatMap(({ x, y }) => [x.toFixed(4), y.toFixed(4)]),
            ...e.keypoints3d.flatMap(({ x, y, z }) => [
                x.toFixed(4), y.toFixed(4), (z ?? 0).toFixed(4),
            ]),
        ].join(","));

        const csv  = [header, ...csvRows].join("\n");
        const blob = new Blob([csv], { type: "text/csv" });
        const url  = URL.createObjectURL(blob);
        const a    = document.createElement("a");
        a.href     = url;
        a.download = `squat_session_${new Date().toISOString().slice(0,19).replace(/:/g,"-")}.csv`;
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
    }

    // ── Render ────────────────────────────────────────────────────────────────

    const colorClass = result
        ? (CLASSIFICATION_COLORS[result.classification] ?? "text-slate-700")
        : "";

    return (
        <div className="flex flex-col items-center gap-5 px-6 py-8 max-w-3xl mx-auto">
            {/* Header */}
            <div className="text-center">
                <h2 className="text-2xl font-bold text-slate-800">Squat Analyzer</h2>
                <p className="text-slate-500 text-sm mt-1">
                    MediaPipe detects all 33 body landmarks. Squat depth is classified by the Python backend.
                </p>
            </div>

            {/* Session name */}
            {supabase && (
                <div className="flex flex-col items-center gap-1">
                    <label htmlFor="session-name" className="text-xs font-semibold text-slate-500 uppercase tracking-wider">
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
                    { id: "image",  label: "Upload Image" },
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
            <div className="ios-card relative rounded-2xl overflow-hidden bg-black" style={{ width: 640, height: 480 }}>
                <video
                    ref={videoRef}
                    className="block bg-black"
                    style={{
                        width: 640, height: 480, objectFit: "contain",
                        visibility: inputMode === "image" ? "hidden" : "visible",
                    }}
                    muted
                    playsInline
                />
                <canvas
                    ref={canvasRef}
                    className="absolute inset-0 pointer-events-none"
                    style={{ width: 640, height: 480 }}
                />
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
                        <button onClick={stopAll} className="ios-btn px-6 py-2 rounded-full text-sm font-semibold text-red-600">
                            Stop
                        </button>
                    )}
                </div>
            )}

            {/* Controls — video upload */}
            {inputMode === "upload" && (
                <div className="flex flex-col items-center gap-2">
                    <div className="flex gap-3 items-center">
                        <input ref={fileInputRef} type="file" accept="video/*" className="hidden" onChange={handleVideoChange} />
                        <button
                            onClick={() => fileInputRef.current?.click()}
                            disabled={status === "loading"}
                            className="ios-btn ios-btn-primary px-6 py-2 rounded-full text-sm font-semibold disabled:opacity-50"
                        >
                            {status === "loading" ? "Loading…" : "Choose Video…"}
                        </button>
                        {status === "running" && (
                            <>
                                <button onClick={togglePlayPause} className="ios-btn px-5 py-2 rounded-full text-sm font-semibold text-slate-700">
                                    {videoPaused ? "▶ Play" : "⏸ Pause"}
                                </button>
                                <button onClick={stopAll} className="ios-btn px-5 py-2 rounded-full text-sm font-semibold text-red-600">
                                    ✕ Stop
                                </button>
                            </>
                        )}
                    </div>
                    {uploadedFileName && <p className="text-slate-400 text-xs">{uploadedFileName}</p>}
                </div>
            )}

            {/* Controls — image upload */}
            {inputMode === "image" && (
                <div className="flex flex-col items-center gap-2">
                    <input ref={imageInputRef} type="file" accept="image/*" className="hidden" onChange={handleImageChange} />
                    <button
                        onClick={() => imageInputRef.current?.click()}
                        disabled={status === "loading"}
                        className="ios-btn ios-btn-primary px-6 py-2 rounded-full text-sm font-semibold disabled:opacity-50"
                    >
                        {status === "loading" ? "Analysing…" : "Choose Image…"}
                    </button>
                    {uploadedFileName && <p className="text-slate-400 text-xs">{uploadedFileName}</p>}
                </div>
            )}

            {/* Error */}
            {status === "error" && (
                <p className="text-red-500 text-sm">{errorMsg || "An error occurred."}</p>
            )}

            {/* Classification result */}
            {result && (
                <div className="ios-card rounded-2xl p-5 text-center w-72">
                    <p className="text-slate-400 text-xs uppercase tracking-wider mb-1">Classification</p>
                    <p className={`text-4xl font-bold mb-4 ${colorClass}`}>{result.classification}</p>
                    <div className="flex justify-around text-sm">
                        <div>
                            <p className="text-slate-400 text-xs">Left knee</p>
                            <p className="font-mono text-slate-700 font-semibold">{result.left_knee_angle?.toFixed(1)}°</p>
                        </div>
                        <div>
                            <p className="text-slate-400 text-xs">Right knee</p>
                            <p className="font-mono text-slate-700 font-semibold">{result.right_knee_angle?.toFixed(1)}°</p>
                        </div>
                        {result.confidence != null && (
                            <div>
                                <p className="text-slate-400 text-xs">Confidence</p>
                                <p className="font-mono text-slate-700 font-semibold">{(result.confidence * 100).toFixed(0)}%</p>
                            </div>
                        )}
                    </div>
                </div>
            )}

            {/* Session actions */}
            {sessionLog.length > 0 && (
                <div className="flex flex-col items-center gap-2">
                    <p className="text-xs text-slate-400">
                        {sessionLog.length} frame{sessionLog.length !== 1 ? "s" : ""} recorded
                    </p>
                    <div className="flex gap-3">
                        {supabase && (
                            <button
                                onClick={handleSave}
                                disabled={isSaving}
                                className="ios-btn ios-btn-primary px-5 py-2 rounded-full text-sm font-semibold disabled:opacity-50"
                            >
                                {isSaving ? "Saving…" : "Save to Database"}
                            </button>
                        )}
                        <button onClick={downloadCSV} className="ios-btn px-5 py-2 rounded-full text-sm font-semibold text-slate-700">
                            Download CSV
                        </button>
                    </div>
                </div>
            )}

            {/* Debug panel */}
            {(status === "running" || allKeypoints.length > 0) && (
                <div className="ios-card rounded-2xl p-4 w-full">
                    <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3">
                        All 33 Keypoints
                        {status === "running" && (
                            <span className="ml-2 normal-case font-normal text-slate-300">— live</span>
                        )}
                    </p>
                    {allKeypoints.length === 0 ? (
                        <p className="text-xs text-slate-400 italic">Waiting for pose detection…</p>
                    ) : (
                        <div className="grid grid-cols-2 sm:grid-cols-3 gap-y-0.5 gap-x-4 text-xs font-mono max-h-56 overflow-y-auto pr-1">
                            {allKeypoints.map((kp) => (
                                <div
                                    key={kp.index}
                                    className={`flex items-baseline gap-1.5 py-0.5 ${
                                        SQUAT_INDICES.has(kp.index) ? "text-sky-600 font-bold" : "text-slate-400"
                                    }`}
                                >
                                    <span className="w-5 text-right text-slate-300">{kp.index}</span>
                                    <span className="w-28 truncate">{kp.name}</span>
                                    <span className="tabular-nums">{kp.x.toFixed(3)}</span>
                                    <span className="tabular-nums">{kp.y.toFixed(3)}</span>
                                    <span className="text-slate-300 tabular-nums">{(kp.visibility * 100).toFixed(0)}%</span>
                                </div>
                            ))}
                        </div>
                    )}
                </div>
            )}
        </div>
    );
}
