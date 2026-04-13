// SquatAnalyzer: uses MediaPipe Pose (tasks-vision) to detect squat keypoints
// from a live webcam feed or an uploaded video file, then sends them to the
// Python backend for angle calculation and depth classification (Deep / Shallow / Invalid).
// Keypoints are also stored in Supabase (public.squat_keypoints table) in parallel.

// TODO: Add image upload aswell

import { useEffect, useRef, useState, useCallback } from "react";
import { apiUrl } from "../apiBase";
import supabase from "../supabaseClient";

// MediaPipe keypoint indices used for squat analysis
// https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker
// Landmark set aligned with the normalized Kinect CSV files
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

// Full body skeleton connections (MediaPipe BlazePose topology).
const POSE_CONNECTIONS = [
    // Face
    [0, 1],
    [1, 2],
    [2, 3],
    [3, 7],
    [0, 4],
    [4, 5],
    [5, 6],
    [6, 8],
    [9, 10],
    // Torso
    [11, 12],
    [11, 23],
    [12, 24],
    [23, 24],
    // Left arm
    [11, 13],
    [13, 15],
    [15, 17],
    [15, 19],
    [15, 21],
    [17, 19],
    // Right arm
    [12, 14],
    [14, 16],
    [16, 18],
    [16, 20],
    [16, 22],
    [18, 20],
    // Left leg
    [23, 25],
    [25, 27],
    [27, 29],
    [27, 31],
    [29, 31],
    // Right leg
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

const CLASSIFICATION_COLORS = {
    Deep: "text-green-600",
    Shallow: "text-amber-600",
    Invalid: "text-red-600",
};

/**
 * Filter a MediaPipe world-landmarks array to only the 3-D joints needed for a squat.
 *
 * @param {Array} landmarks - Full array of 33 world landmark objects from MediaPipe.
 * @returns {Array} Filtered array of { name, x, y, z, score } objects.
 */
function filterSquatKeypoints3d(landmarks) {
    if (!landmarks) return [];
    return Object.entries(SQUAT_LANDMARK_NAMES).flatMap(([idx, name]) => {
        const lm = landmarks[Number(idx)];
        if (!lm) return [];
        return [{ name, x: lm.x, y: lm.y, z: lm.z ?? 0, score: lm.visibility ?? null }];
    });
}

/**
 * Load the MediaPipe PoseLandmarker from CDN at runtime so the React bundle
 * stays small (no npm install required on the frontend).
 *
 * @returns {Promise<object>} Resolved PoseLandmarker instance.
 */
async function loadPoseLandmarker() {
    const { PoseLandmarker, FilesetResolver } =
        await import("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/+esm");

    const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm",
    );

    return PoseLandmarker.createFromOptions(vision, {
        baseOptions: {
            modelAssetPath:
                "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task",
            delegate: "GPU",
        },
        runningMode: "VIDEO",
        numPoses: 1,
    });
}

export default function SquatAnalyzer() {
    const videoRef = useRef(null);
    const canvasRef = useRef(null);
    const fileInputRef = useRef(null);
    const landmarkerRef = useRef(null);
    const rafRef = useRef(null);
    const lastTimestampRef = useRef(-1);
    const videoBlobUrlRef = useRef(null);
    // Mirror of sessionName state — readable inside async rAF callbacks.
    const sessionNameRef = useRef("");

    // "webcam" | "upload"
    const [inputMode, setInputMode] = useState("webcam");
    const [sessionName, setSessionName] = useState("");
    const [status, setStatus] = useState("idle"); // idle | loading | running | error
    const [result, setResult] = useState(null); // last classification response
    const [errorMsg, setErrorMsg] = useState("");
    const [uploadedFileName, setUploadedFileName] = useState("");
    const [videoPaused, setVideoPaused] = useState(false);
    // All 33 raw landmarks from the last processed frame (for debug display).
    const [allKeypoints, setAllKeypoints] = useState([]);

    // Helpers
    /** Release any blob URL previously created for an uploaded video. */
    function revokeBlobUrl() {
        if (videoBlobUrlRef.current) {
            URL.revokeObjectURL(videoBlobUrlRef.current);
            videoBlobUrlRef.current = null;
        }
    }

    /** Fully stop all media and cancel the detection loop. */
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
        setStatus("idle");
        setVideoPaused(false);
    }, []);

    // Switch mode: reset everything first.
    const switchMode = useCallback(
        (newMode) => {
            stopAll();
            setInputMode(newMode);
            setResult(null);
            setErrorMsg("");
            setUploadedFileName("");
        },
        [stopAll],
    );

    // Webcam start
    const startCamera = useCallback(async () => {
        setStatus("loading");
        setErrorMsg("");
        try {
            if (!landmarkerRef.current) {
                landmarkerRef.current = await loadPoseLandmarker();
            }

            const stream = await navigator.mediaDevices.getUserMedia({
                video: { width: 640, height: 480 },
            });

            const video = videoRef.current;
            video.srcObject = stream;
            await video.play();

            setStatus("running");
        } catch (err) {
            setStatus("error");
            setErrorMsg(String(err));
        }
    }, []);

    // Video upload
    const handleFileChange = useCallback(
        async (e) => {
            const file = e.target.files?.[0];
            if (!file) return;

            // Only accept video files.
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
                    landmarkerRef.current = await loadPoseLandmarker();
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

    /** Toggle play / pause for an uploaded video. */
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

    // Detection loop (shared by both webcam and uploaded video)
    useEffect(() => {
        if (status !== "running") return;

        let frameCounter = 0;
        const SEND_EVERY_N_FRAMES = 15; // ~2 Hz at 30 fps

        async function detect(timestamp) {
            const video = videoRef.current;
            const canvas = canvasRef.current;
            const landmarker = landmarkerRef.current;

            if (!video || !canvas || !landmarker || video.readyState < 2) {
                rafRef.current = requestAnimationFrame(detect);
                return;
            }

            // Don't process while paused.
            if (video.paused || video.ended) {
                rafRef.current = requestAnimationFrame(detect);
                return;
            }

            // Avoid re-processing the same video frame.
            if (timestamp <= lastTimestampRef.current) {
                rafRef.current = requestAnimationFrame(detect);
                return;
            }
            lastTimestampRef.current = timestamp;

            const detection = landmarker.detectForVideo(video, timestamp);
            drawOverlay(canvas, video, detection);

            // Throttle backend calls and state updates to ~2 Hz.
            frameCounter++;
            if (
                frameCounter % SEND_EVERY_N_FRAMES === 0 &&
                detection.landmarks?.length > 0
            ) {
                const kp3d = filterSquatKeypoints3d(
                    detection.worldLandmarks?.[0] ?? [],
                );
                sendFrame(kp3d);

                // Collect all 33 landmarks for the debug panel.
                const all = detection.landmarks[0].map((lm, i) => ({
                    index: i,
                    name: ALL_LANDMARK_NAMES[i] ?? `landmark_${i}`,
                    x: lm.x,
                    y: lm.y,
                    z: lm.z ?? 0,
                    visibility: lm.visibility ?? 0,
                }));
                setAllKeypoints(all);
            }

            rafRef.current = requestAnimationFrame(detect);
        }

        rafRef.current = requestAnimationFrame(detect);
        return () => cancelAnimationFrame(rafRef.current);
    }, [status]);

    // Backend + Supabase — combined per-frame dispatch
    //
    // Flow:
    //   1. POST 3-D keypoints to the Python backend for classification.
    //   2. Update the UI with the result.
    //   3. Insert one row to Supabase's public.squat_keypoints table using the
    //      classification result from step 1.
    //
    // raw_keypoints shape stored in DB:
    //   {
    //     "3d": [ [x, y, z], … ] // one [x, y, z] triple per squat keypoint
    //   }
    // Keep the ref in sync so the async callback always sees the latest name.
    sessionNameRef.current = sessionName;

    /**
     * Classification is handled by the Python backend (POST /api/v1/squat/classify).
     * The backend attempts PyTorch inference using the trained SquatNet model first
     * and automatically falls back to a rule-based threshold classifier when the
     * model file is absent or torch is unavailable — so the endpoint is always
     * responsive even without a trained checkpoint.
     *
     * Future work: train an improved model, publish it to DAGsHub / MLflow, and
     * register it at a new versioned endpoint. The rule-based path will remain the
     * fallback throughout.
     */
    async function sendFrame(keypoints3d) {
        // --- 1. Backend classification ---
        let classification = null;
        let confidence = null;
        try {
            const res = await fetch(apiUrl("/api/v1/squat/classify"), {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    keypoints_3d: keypoints3d,
                }),
            });
            if (res.ok) {
                const data = await res.json();
                setResult(data);
                classification = data.classification ?? null;
                confidence = data.confidence ?? null;
            }
        } catch {
            // Network errors are silently ignored during live detection.
        }

        // --- 2. Supabase insert ---
        if (!supabase) return;
        // id_name is nullable — send null when the user hasn't entered a name.
        const idName = sessionNameRef.current.trim() || null;

        // Shape raw_keypoints: only 3D points as [x, y, z]; 2D data is
        // used for classification only and is not persisted.
        const rawKeypoints = {
            "3d": keypoints3d.map(({ x, y, z }) => [x, y, z]),
        };

        // Supabase JS client returns { data, error } rather than throwing.
        const { error: dbError } = await supabase
            .from("squat_keypoints")
            .insert({
                id_name: idName,
                raw_keypoints: rawKeypoints,
                score: confidence,
                classification,
            });
        if (dbError) {
            console.warn("Supabase insert error:", dbError.message);
        }
    }

    // Canvas overlay — full body skeleton + highlighted squat joints
    function drawOverlay(canvas, video, detectionResult) {
        const ctx = canvas.getContext("2d");
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        if (!detectionResult.landmarks?.length) return;

        const lms = detectionResult.landmarks[0];
        const w = canvas.width;
        const h = canvas.height;

        // 1. Draw all skeleton connections.
        POSE_CONNECTIONS.forEach(([a, b]) => {
            const la = lms[a];
            const lb = lms[b];
            if (!la || !lb) return;
            const isSquatBone = SQUAT_INDICES.has(a) && SQUAT_INDICES.has(b);
            ctx.beginPath();
            ctx.moveTo(la.x * w, la.y * h);
            ctx.lineTo(lb.x * w, lb.y * h);
            ctx.strokeStyle = isSquatBone
                ? "rgba(244, 114, 182, 0.92)" // pink — squat legs
                : "rgba(148, 163, 184, 0.55)"; // slate — rest of body
            ctx.lineWidth = isSquatBone ? 3 : 1.5;
            ctx.stroke();
        });

        // 2. Draw dots for every landmark.
        lms.forEach((lm, idx) => {
            if (!lm) return;
            const isSquat = SQUAT_INDICES.has(idx);
            ctx.beginPath();
            ctx.arc(lm.x * w, lm.y * h, isSquat ? 6 : 3, 0, Math.PI * 2);
            ctx.fillStyle = isSquat ? "#38bdf8" : "rgba(255,255,255,0.65)";
            ctx.fill();
        });
    }

    // Render
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

            {/* Session name — stored as id_name in Supabase public.squat_keypoints */}
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

            {/* Mode toggle — iOS segmented control */}
            <div className="ios-pill flex rounded-full p-0.5 gap-px">
                <button
                    onClick={() => switchMode("webcam")}
                    className={`px-5 py-1.5 rounded-full text-sm font-semibold transition-all duration-150 ${
                        inputMode === "webcam"
                            ? "bg-white text-slate-800 shadow-sm"
                            : "text-slate-500 hover:text-slate-700"
                    }`}
                >
                    Live Camera
                </button>
                <button
                    onClick={() => switchMode("upload")}
                    className={`px-5 py-1.5 rounded-full text-sm font-semibold transition-all duration-150 ${
                        inputMode === "upload"
                            ? "bg-white text-slate-800 shadow-sm"
                            : "text-slate-500 hover:text-slate-700"
                    }`}
                >
                    Upload Video
                </button>
            </div>

            {/* Video + canvas overlay */}
            <div
                className="ios-card relative rounded-2xl overflow-hidden"
                style={{ width: 640, height: 480 }}
            >
                <video
                    ref={videoRef}
                    className="block bg-black"
                    style={{ width: 640, height: 480, objectFit: "contain" }}
                    muted
                    playsInline
                />
                <canvas
                    ref={canvasRef}
                    className="absolute inset-0 pointer-events-none"
                    style={{ width: 640, height: 480 }}
                />
            </div>

            {/* Controls — webcam mode */}
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
                            onClick={stopAll}
                            className="ios-btn px-6 py-2 rounded-full text-sm font-semibold text-red-600"
                        >
                            Stop
                        </button>
                    )}
                </div>
            )}

            {/* Controls — upload mode */}
            {inputMode === "upload" && (
                <div className="flex flex-col items-center gap-2">
                    <div className="flex gap-3 items-center">
                        <input
                            ref={fileInputRef}
                            type="file"
                            accept="video/*"
                            className="hidden"
                            onChange={handleFileChange}
                        />
                        <button
                            onClick={() => fileInputRef.current?.click()}
                            disabled={status === "loading"}
                            className="ios-btn ios-btn-primary px-6 py-2 rounded-full text-sm font-semibold disabled:opacity-50"
                        >
                            {status === "loading"
                                ? "Loading…"
                                : "Choose Video…"}
                        </button>

                        {status === "running" && (
                            <>
                                <button
                                    onClick={togglePlayPause}
                                    className="ios-btn px-5 py-2 rounded-full text-sm font-semibold text-slate-700"
                                >
                                    {videoPaused ? "▶ Play" : "⏸ Pause"}
                                </button>
                                <button
                                    onClick={stopAll}
                                    className="ios-btn px-5 py-2 rounded-full text-sm font-semibold text-red-600"
                                >
                                    ✕ Stop
                                </button>
                            </>
                        )}
                    </div>
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

            {/* 33 Keypoints debug panel */}
            {allKeypoints.length > 0 && (
                <div className="ios-card rounded-2xl p-4 w-full">
                    <p className="text-xs font-semibold text-slate-400 uppercase tracking-wider mb-3">
                        All 33 Keypoints — live
                    </p>
                    <div className="grid grid-cols-2 sm:grid-cols-3 gap-y-0.5 gap-x-4 text-xs font-mono max-h-56 overflow-y-auto pr-1">
                        {allKeypoints.map((kp) => (
                            <div
                                key={kp.index}
                                className={`flex items-baseline gap-1.5 py-0.5 ${
                                    SQUAT_INDICES.has(kp.index)
                                        ? "text-sky-600 font-bold"
                                        : "text-slate-400"
                                }`}
                            >
                                <span className="w-5 text-right text-slate-300">
                                    {kp.index}
                                </span>
                                <span className="w-28 truncate">{kp.name}</span>
                                <span className="tabular-nums">
                                    {kp.x.toFixed(3)}
                                </span>
                                <span className="tabular-nums">
                                    {kp.y.toFixed(3)}
                                </span>
                                <span className="text-slate-300 tabular-nums">
                                    {(kp.visibility * 100).toFixed(0)}%
                                </span>
                            </div>
                        ))}
                    </div>
                </div>
            )}
        </div>
    );
}
