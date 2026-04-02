// SquatAnalyzer: uses MediaPipe Pose (tasks-vision) to detect squat keypoints
// from a live webcam feed or an uploaded video file, then sends them to the
// Python backend for angle calculation and depth classification (Deep / Shallow / Invalid).

import { useEffect, useRef, useState, useCallback } from "react";
import { apiUrl } from "../apiBase";

// ---------------------------------------------------------------------------
// MediaPipe keypoint indices used for squat analysis
// https://ai.google.dev/edge/mediapipe/solutions/vision/pose_landmarker
// ---------------------------------------------------------------------------
const SQUAT_LANDMARK_NAMES = {
    23: "left_hip",
    24: "right_hip",
    25: "left_knee",
    26: "right_knee",
    27: "left_ankle",
    28: "right_ankle",
    29: "left_heel",
    30: "right_heel",
};

const CLASSIFICATION_COLORS = {
    Deep: "text-green-400",
    Shallow: "text-yellow-400",
    Invalid: "text-red-400",
};

/**
 * Filter a MediaPipe landmarks array to only the joints needed for a squat.
 *
 * @param {Array} landmarks - Full array of 33 landmark objects from MediaPipe.
 * @param {boolean} is3d - Whether the landmarks contain a `z` component.
 * @returns {Array} Filtered array of { name, x, y, z?, score } objects.
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
 * Load the MediaPipe PoseLandmarker from CDN at runtime so the React bundle
 * stays small (no npm install required on the frontend).
 *
 * @returns {Promise<object>} Resolved PoseLandmarker instance.
 */
async function loadPoseLandmarker() {
    const { PoseLandmarker, FilesetResolver } =
        await import("https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/+esm");

    const vision = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm"
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

    // "webcam" | "upload"
    const [inputMode, setInputMode] = useState("webcam");
    const [status, setStatus] = useState("idle"); // idle | loading | running | error
    const [result, setResult] = useState(null);   // last classification response
    const [errorMsg, setErrorMsg] = useState("");
    const [uploadedFileName, setUploadedFileName] = useState("");
    const [videoPaused, setVideoPaused] = useState(false);

    // -----------------------------------------------------------------------
    // Helpers
    // -----------------------------------------------------------------------

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
    const switchMode = useCallback((newMode) => {
        stopAll();
        setInputMode(newMode);
        setResult(null);
        setErrorMsg("");
        setUploadedFileName("");
    }, [stopAll]);

    // -----------------------------------------------------------------------
    // Webcam start
    // -----------------------------------------------------------------------
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

    // -----------------------------------------------------------------------
    // Video upload
    // -----------------------------------------------------------------------
    const handleFileChange = useCallback(async (e) => {
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
    }, [stopAll]);

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

    // -----------------------------------------------------------------------
    // Detection loop (shared by both webcam and uploaded video)
    // -----------------------------------------------------------------------
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

            // Throttle backend calls to avoid overwhelming the server.
            frameCounter++;
            if (frameCounter % SEND_EVERY_N_FRAMES === 0 && detection.landmarks?.length > 0) {
                const kp2d = filterSquatKeypoints(detection.landmarks[0], false);
                const kp3d = filterSquatKeypoints(detection.worldLandmarks?.[0] ?? [], true);
                sendToBackend(kp2d, kp3d);
            }

            rafRef.current = requestAnimationFrame(detect);
        }

        rafRef.current = requestAnimationFrame(detect);
        return () => cancelAnimationFrame(rafRef.current);
    }, [status]);

    // -----------------------------------------------------------------------
    // Backend request
    // -----------------------------------------------------------------------
    async function sendToBackend(keypoints2d, keypoints3d) {
        try {
            const res = await fetch(apiUrl("/api/v1/squat/classify"), {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    keypoints_2d: keypoints2d,
                    keypoints_3d: keypoints3d,
                }),
            });
            if (!res.ok) return;
            const data = await res.json();
            setResult(data);
        } catch {
            // Network errors are silently ignored during live detection.
        }
    }

    // -----------------------------------------------------------------------
    // Canvas overlay (skeleton lines for squat joints)
    // -----------------------------------------------------------------------
    function drawOverlay(canvas, video, detectionResult) {
        const ctx = canvas.getContext("2d");
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        if (!detectionResult.landmarks?.length) return;

        const lms = detectionResult.landmarks[0];
        const w = canvas.width;
        const h = canvas.height;

        // Draw dots for squat joints.
        Object.keys(SQUAT_LANDMARK_NAMES).forEach((idx) => {
            const lm = lms[Number(idx)];
            if (!lm) return;
            ctx.beginPath();
            ctx.arc(lm.x * w, lm.y * h, 6, 0, Math.PI * 2);
            ctx.fillStyle = "#38bdf8";
            ctx.fill();
        });

        // Draw connecting lines: hip-knee-ankle for each side.
        const connections = [
            [23, 25], [25, 27], // left hip→knee→ankle
            [24, 26], [26, 28], // right hip→knee→ankle
        ];
        ctx.lineWidth = 3;
        ctx.strokeStyle = "#f472b6";
        connections.forEach(([a, b]) => {
            const la = lms[a];
            const lb = lms[b];
            if (!la || !lb) return;
            ctx.beginPath();
            ctx.moveTo(la.x * w, la.y * h);
            ctx.lineTo(lb.x * w, lb.y * h);
            ctx.stroke();
        });
    }

    // -----------------------------------------------------------------------
    // Render
    // -----------------------------------------------------------------------
    const colorClass = result ? (CLASSIFICATION_COLORS[result.classification] ?? "text-white") : "";

    return (
        <div className="flex flex-col items-center gap-4 p-6">
            <h2 className="text-2xl font-bold text-white">Squat Analyzer</h2>
            <p className="text-slate-400 text-sm text-center max-w-lg">
                Detects hip, knee, and ankle positions with MediaPipe and classifies
                squat depth via the Python backend.
            </p>

            {/* Mode toggle */}
            <div className="flex rounded-lg overflow-hidden border border-slate-600">
                <button
                    onClick={() => switchMode("webcam")}
                    className={`px-5 py-2 text-sm font-semibold transition ${
                        inputMode === "webcam"
                            ? "bg-sky-600 text-white"
                            : "bg-slate-800 text-slate-400 hover:text-white"
                    }`}
                >
                    📷 Live Camera
                </button>
                <button
                    onClick={() => switchMode("upload")}
                    className={`px-5 py-2 text-sm font-semibold transition ${
                        inputMode === "upload"
                            ? "bg-sky-600 text-white"
                            : "bg-slate-800 text-slate-400 hover:text-white"
                    }`}
                >
                    🎬 Upload Video
                </button>
            </div>

            {/* Video + canvas overlay */}
            <div className="relative rounded-xl overflow-hidden border border-slate-700 bg-slate-900">
                <video
                    ref={videoRef}
                    className="block"
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
                            className="px-5 py-2 rounded-lg bg-sky-600 hover:bg-sky-500 disabled:opacity-50 text-white font-semibold transition"
                        >
                            {status === "loading" ? "Loading…" : "Start Camera"}
                        </button>
                    ) : (
                        <button
                            onClick={stopAll}
                            className="px-5 py-2 rounded-lg bg-rose-600 hover:bg-rose-500 text-white font-semibold transition"
                        >
                            Stop
                        </button>
                    )}
                </div>
            )}

            {/* Controls — upload mode */}
            {inputMode === "upload" && (
                <div className="flex flex-col items-center gap-3">
                    <div className="flex gap-3 items-center">
                        {/* Hidden file input, triggered by the button */}
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
                            className="px-5 py-2 rounded-lg bg-sky-600 hover:bg-sky-500 disabled:opacity-50 text-white font-semibold transition"
                        >
                            {status === "loading" ? "Loading…" : "Choose Video…"}
                        </button>

                        {status === "running" && (
                            <>
                                <button
                                    onClick={togglePlayPause}
                                    className="px-4 py-2 rounded-lg bg-slate-700 hover:bg-slate-600 text-white font-semibold transition"
                                >
                                    {videoPaused ? "▶ Play" : "⏸ Pause"}
                                </button>
                                <button
                                    onClick={stopAll}
                                    className="px-4 py-2 rounded-lg bg-rose-600 hover:bg-rose-500 text-white font-semibold transition"
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
                <p className="text-red-400 text-sm">{errorMsg || "An error occurred."}</p>
            )}

            {/* Classification result */}
            {result && (
                <div className="bg-slate-800 rounded-xl p-5 text-center min-w-64 border border-slate-700">
                    <p className="text-slate-400 text-xs mb-1 uppercase tracking-wider">Classification</p>
                    <p className={`text-4xl font-bold mb-3 ${colorClass}`}>
                        {result.classification}
                    </p>
                    <div className="flex justify-around text-sm text-slate-300">
                        <div>
                            <p className="text-slate-500 text-xs">Left knee</p>
                            <p className="font-mono">{result.left_knee_angle?.toFixed(1)}°</p>
                        </div>
                        <div>
                            <p className="text-slate-500 text-xs">Right knee</p>
                            <p className="font-mono">{result.right_knee_angle?.toFixed(1)}°</p>
                        </div>
                        {result.confidence != null && (
                            <div>
                                <p className="text-slate-500 text-xs">Confidence</p>
                                <p className="font-mono">{(result.confidence * 100).toFixed(0)}%</p>
                            </div>
                        )}
                    </div>
                </div>
            )}
        </div>
    );
}
