import { vi, beforeEach, afterEach, beforeAll } from "vitest";

export const fetchMock = vi.fn();
export const detectImageMock = vi
    .fn()
    .mockReturnValue({ landmarks: [], worldLandmarks: [] });

// Silence MediaPipe ESM import (not available in jsdom)
vi.mock("@mediapipe/tasks-vision", () => ({}));

// Mock dynamic ESM import used inside createVideoLandmarker / createImageLandmarker
vi.mock(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/+esm",
    () => ({
        PoseLandmarker: {
            createFromOptions: vi.fn().mockResolvedValue({
                detectForVideo: vi
                    .fn()
                    .mockReturnValue({ landmarks: [], worldLandmarks: [] }),
                detect: detectImageMock,
                close: vi.fn(),
            }),
        },
        FilesetResolver: {
            forVisionTasks: vi.fn().mockResolvedValue({}),
        },
    }),
    { virtual: true },
);

// Stub supabaseClient so the "Save to Database" button can be tested
vi.mock("../supabaseClient", () => ({
    default: {
        from: vi.fn(() => ({
            insert: vi.fn().mockResolvedValue({ error: null }),
        })),
    },
}));

// Stub apiBase
vi.mock("../apiBase", () => ({
    apiUrl: (path) => `http://localhost:8000${path}`,
}));

/** Return a minimal successful classify response */
export const classifyResponse = (classification = "Deep") =>
    Promise.resolve({
        ok: true,
        json: () =>
            Promise.resolve({
                classification,
                left_knee_angle: 95.2,
                right_knee_angle: 93.8,
                confidence: 0.87,
            }),
    });

export const analyzeSessionResponse = ({
    start_stop = 1,
    good_bad_score = 0.71,
    squat_score = 2,
    predicted_z = {},
} = {}) =>
    Promise.resolve({
        ok: true,
        json: () =>
            Promise.resolve({
                results: [
                    { start_stop, good_bad_score, squat_score, predicted_z },
                ],
                timings: {
                    feature_build_ms: 1,
                    start_stop_ms: 1,
                    smooth_ms: 0,
                    z_prediction_ms: 0,
                    goodbad_ms: 1,
                    total_ms: 3,
                },
            }),
    });

export function setupSquatAnalyzerMocks() {
    // Without this, switchMode's canvas.getContext("2d").clearRect(...) throws.
    beforeAll(() => {
        HTMLCanvasElement.prototype.getContext = vi.fn(() => ({
            clearRect: vi.fn(),
            beginPath: vi.fn(),
            moveTo: vi.fn(),
            lineTo: vi.fn(),
            closePath: vi.fn(),
            stroke: vi.fn(),
            arc: vi.fn(),
            fill: vi.fn(),
            fillRect: vi.fn(),
            drawImage: vi.fn(),
            fillText: vi.fn(),
        }));
    });

    beforeEach(() => {
        vi.clearAllMocks();
        detectImageMock.mockReturnValue({ landmarks: [], worldLandmarks: [] });
        fetchMock.mockImplementation((url) => {
            if (url.includes("/squat/classify")) return classifyResponse();
            if (url.includes("/squat/analyze-session"))
                return analyzeSessionResponse();
            if (url.includes("/model-info/start-stop"))
                return Promise.resolve({
                    ok: true,
                    json: () => Promise.resolve({ mae_total_average: 0.1 }),
                });
            return Promise.resolve({ ok: false });
        });
        vi.stubGlobal("fetch", fetchMock);
    });

    afterEach(() => {
        vi.restoreAllMocks();
    });
}
