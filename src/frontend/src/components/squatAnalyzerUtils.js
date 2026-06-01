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
export const ALL_LANDMARK_NAMES = [
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
export const SQUAT_JOINT_ORDER = [
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
export const SQUAT_CONNECTIONS_BY_NAME = POSE_CONNECTIONS.filter(
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
export const MAX_SQUAT_SCORE = 4;
export const ANALYZE_SESSION_TIMEOUT_MS = 90_000;

/**
 * Create a new PoseLandmarker for continuous video/webcam detection.
 */
export async function createVideoLandmarker() {
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
export async function createImageLandmarker() {
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
export function filterSquatKeypoints3d(landmarks) {
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
export function drawSkeleton(canvas, w, h, detection, off = { x: 0, y: 0 }) {
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

/**
 * Assess per-frame video quality for squat analysis.
 * Returns { blocking, warnings } where:
 *   blocking — issues that prevent reliable analysis (pipeline should be gated)
 *   warnings — advisory hints shown live but never block the pipeline
 */
export function assessVideoQuality(detection, captureCanvas) {
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
