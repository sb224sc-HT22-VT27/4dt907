// SquatAnalyzer: uses MediaPipe Pose (tasks-vision) to detect squat keypoints
// from a live webcam feed, an uploaded video file, or a static image, then
// sends them to the Python backend for angle calculation and depth
// classification (Deep / Shallow / Invalid).
// Accumulated frames can be saved to Supabase or exported as CSV on demand.

import { useEffect, useRef, useState, useCallback, useMemo } from "react";
import { apiUrl } from "../apiBase";
import supabase from "../supabaseClient";
import Skeleton3DViewer from "./Skeleton3DViewer";
import {
    ALL_LANDMARK_NAMES,
    ANALYZE_SESSION_TIMEOUT_MS,
    SQUAT_JOINT_ORDER,
    assessVideoQuality,
    createImageLandmarker,
    createVideoLandmarker,
    drawSkeleton,
    filterSquatKeypoints3d,
} from "./squatAnalyzerUtils";

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
    const [showPipelineTimings, setShowPipelineTimings] = useState(false);
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

    const analyzeSession = useCallback(async (body) => {
        const ctrl = new AbortController();
        const timeoutId = setTimeout(
            () => ctrl.abort(),
            ANALYZE_SESSION_TIMEOUT_MS,
        );
        try {
            const res = await fetch(apiUrl("/api/v1/squat/analyze-session"), {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(body),
                signal: ctrl.signal,
            });
            if (!res.ok) throw new Error("Session analysis failed.");
            return await res.json();
        } catch (err) {
            if (err?.name === "AbortError") {
                throw new Error(
                    "Analysis timed out. Please retry, shorten the clip, or check backend availability.",
                );
            }
            throw err;
        } finally {
            clearTimeout(timeoutId);
        }
    }, []);

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
        setShowPipelineTimings(false);
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
            const data = await analyzeSession(body);
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
            setErrorMsg(
                err?.message || "Failed to analyze recording. Try again.",
            );
            setStatus("error");
        }
    }, [analyzeSession, stopCapture]);

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

    const handleVideoFile = useCallback(
        async (file) => {
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

    const handleVideoChange = useCallback(
        async (e) => {
            const file = e.target.files?.[0];
            if (!file) return;
            await handleVideoFile(file);
        },
        [handleVideoFile],
    );
    const handleVideoDrop = useCallback(
        async (e) => {
            e.preventDefault();
            const file = e.dataTransfer?.files?.[0];
            if (!file || status === "loading" || status === "analyzing") return;
            await handleVideoFile(file);
        },
        [handleVideoFile, status],
    );

    // Image upload

    const handleImageFile = useCallback(
        async (file) => {
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
                        const data = await analyzeSession(body);
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
                setErrorMsg(err?.message || String(err));
            } finally {
                imageLandmarker?.close?.();
                URL.revokeObjectURL(blobUrl);
                if (imageInputRef.current) imageInputRef.current.value = "";
            }
        },
        [analyzeSession, stopAll],
    );
    const handleImageChange = useCallback(
        async (e) => {
            const file = e.target.files?.[0];
            if (!file) return;
            await handleImageFile(file);
        },
        [handleImageFile],
    );
    const handleImageDrop = useCallback(
        async (e) => {
            e.preventDefault();
            const file = e.dataTransfer?.files?.[0];
            if (!file || status === "loading" || status === "analyzing") return;
            await handleImageFile(file);
        },
        [handleImageFile, status],
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
                    MediaPipe detects body landmarks. Classifies if squat and
                    gives a score based on these landmarks.
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
                <div
                    className="flex flex-col items-center gap-2"
                    onDragOver={(e) => e.preventDefault()}
                    onDrop={handleVideoDrop}
                >
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
                    {status !== "running" && (
                        <p className="text-slate-400 text-xs">
                            or drop a video file here
                        </p>
                    )}
                </div>
            )}

            {/* Controls — image upload */}
            {inputMode === "image" && (
                <div
                    className="flex flex-col items-center gap-2"
                    onDragOver={(e) => e.preventDefault()}
                    onDrop={handleImageDrop}
                >
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
                    <p className="text-slate-400 text-xs">
                        or drop an image file here
                    </p>
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
                                        Exercise Quality
                                    </p>
                                    <p
                                        className={`text-2xl font-bold ${formIsGood ? "text-green-500" : "text-red-500"}`}
                                    >
                                        {formIsGood ? "A squat" : "Not a squat"}
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
                                        Probability of squat
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
            {pipelineTimings != null && (
                <label
                    htmlFor="show-pipeline-timing"
                    className="mb-3 inline-flex items-center gap-2 text-sm text-slate-600 select-none"
                >
                    <input
                        id="show-pipeline-timing"
                        type="checkbox"
                        checked={showPipelineTimings}
                        onChange={(e) =>
                            setShowPipelineTimings(e.target.checked)
                        }
                    />
                    Show pipeline timing window
                </label>
            )}
            {pipelineTimings != null &&
                showPipelineTimings &&
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
