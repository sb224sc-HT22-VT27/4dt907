// SquatAnalyzer: uses MediaPipe Pose (tasks-vision) to detect squat keypoints
// from a live webcam feed, then sends them to the Python backend for angle
// calculation and depth classification (Deep / Shallow / Invalid).

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
    const landmarkerRef = useRef(null);
    const rafRef = useRef(null);
    const lastTimestampRef = useRef(-1);

    const [status, setStatus] = useState("idle"); // idle | loading | running | error
    const [result, setResult] = useState(null);   // last classification response
    const [errorMsg, setErrorMsg] = useState("");

    // -----------------------------------------------------------------------
    // Initialise MediaPipe and webcam
    // -----------------------------------------------------------------------
    const startCamera = useCallback(async () => {
        setStatus("loading");
        setErrorMsg("");
        try {
            landmarkerRef.current = await loadPoseLandmarker();

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

    const stopCamera = useCallback(() => {
        cancelAnimationFrame(rafRef.current);
        const video = videoRef.current;
        if (video?.srcObject) {
            video.srcObject.getTracks().forEach((t) => t.stop());
            video.srcObject = null;
        }
        setStatus("idle");
    }, []);

    // -----------------------------------------------------------------------
    // Detection loop
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

            // Avoid re-processing the same video frame.
            if (timestamp <= lastTimestampRef.current) {
                rafRef.current = requestAnimationFrame(detect);
                return;
            }
            lastTimestampRef.current = timestamp;

            const result = landmarker.detectForVideo(video, timestamp);
            drawOverlay(canvas, video, result);

            // Throttle backend calls to avoid overwhelming the server.
            frameCounter++;
            if (frameCounter % SEND_EVERY_N_FRAMES === 0 && result.landmarks?.length > 0) {
                const kp2d = filterSquatKeypoints(result.landmarks[0], false);
                const kp3d = filterSquatKeypoints(result.worldLandmarks?.[0] ?? [], true);
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

            {/* Video + canvas overlay */}
            <div className="relative rounded-xl overflow-hidden border border-slate-700">
                <video
                    ref={videoRef}
                    className="block"
                    style={{ width: 640, height: 480 }}
                    muted
                    playsInline
                />
                <canvas
                    ref={canvasRef}
                    className="absolute inset-0 pointer-events-none"
                    style={{ width: 640, height: 480 }}
                />
            </div>

            {/* Controls */}
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
                        onClick={stopCamera}
                        className="px-5 py-2 rounded-lg bg-rose-600 hover:bg-rose-500 text-white font-semibold transition"
                    >
                        Stop
                    </button>
                )}
            </div>

            {/* Error */}
            {status === "error" && (
                <p className="text-red-400 text-sm">{errorMsg || "Camera access failed."}</p>
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
