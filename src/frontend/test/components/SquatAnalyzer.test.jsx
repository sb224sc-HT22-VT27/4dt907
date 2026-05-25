// src/frontend/test/components/SquatAnalyzer.test.jsx

import {render, screen, fireEvent, waitFor} from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import {vi, describe, it, expect, beforeEach, afterEach, beforeAll} from "vitest";
import SquatAnalyzer from "../../src/components/SquatAnalyzer";

// ── Module mocks ──────────────────────────────────────────────────────────────

const fetchMock = vi.fn();

// Silence MediaPipe ESM import (not available in jsdom)
vi.mock("@mediapipe/tasks-vision", () => ({}));

// Mock dynamic ESM import used inside createVideoLandmarker / createImageLandmarker
vi.mock(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/+esm",
    () => ({
        PoseLandmarker: {
            createFromOptions: vi.fn().mockResolvedValue({
                detectForVideo: vi.fn().mockReturnValue({landmarks: [], worldLandmarks: []}),
                detect: vi.fn().mockReturnValue({landmarks: [], worldLandmarks: []}),
                close: vi.fn(),
            }),
        },
        FilesetResolver: {
            forVisionTasks: vi.fn().mockResolvedValue({}),
        },
    }),
    {virtual: true}
);

// Stub supabaseClient so the "Save to Database" button can be tested
vi.mock("../supabaseClient", () => ({
    default: {
        from: vi.fn(() => ({
            insert: vi.fn().mockResolvedValue({error: null}),
        })),
    },
}));

// Stub apiBase
vi.mock("../apiBase", () => ({
    apiUrl: (path) => `http://localhost:8000${path}`,
}));

// Stub fetch globally
fetchMock.fetch = vi.fn();

// ── Helpers ───────────────────────────────────────────────────────────────────

/** Return a minimal successful classify response */
const classifyResponse = (classification = "Deep") =>
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


const analyzeSessionResponse = (goodBadScore = 0.82) =>
    Promise.resolve({
        ok: true,
        json: () =>
            Promise.resolve({
                results: [
                    {
                        start_stop: 0,
                        predicted_z: { left_knee: 0.12, right_knee: 0.11 },
                        good_bad_score: goodBadScore,
                    },
                ],
                timings: {
                    feature_build_ms: 1,
                    start_stop_ms: 1,
                    smooth_ms: 0,
                    z_prediction_ms: 1,
                    goodbad_ms: 1,
                    total_ms: 4,
                },
            }),
    });

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
    vi.stubGlobal("fetch", fetchMock.fetch);
    fetchMock.fetch.mockImplementation((url) => {
        if (url.includes("/squat/analyze-session")) return analyzeSessionResponse();
        if (url.includes("/squat/classify")) return classifyResponse();
        return Promise.resolve({ ok: false });
    });
});

afterEach(() => {
    vi.restoreAllMocks();
});

// ── Test suite ───────────────────────────────────────────────────────────────

describe("SquatAnalyzer - rendering", () => {
    it("renders the heading", () => {
        render(<SquatAnalyzer />);
        expect(
            screen.getByRole("heading", { name: /squat analyzer/i })
        ).toBeInTheDocument();
    });

    it("renders two mode buttons", () => {
        render(<SquatAnalyzer />);
        const buttons = screen.getAllByRole("button", { name: /camera|upload|snapshot/i });
        expect(buttons).toHaveLength(2);
    });
});

describe("SquatAnalyzer - switching modes", () => {
    it("switches to Upload Video mode when 'Upload Video' button is clicked", async () => {
        render(<SquatAnalyzer />);
        await userEvent.click(screen.getByRole("button", { name: /upload video/i }));
        expect(
            screen.getByRole("button", {name: /choose video/i})
        ).toBeInTheDocument();
    });
    
    it("switches to Upload Image mode when 'Upload Image' button is clicked", async () => {
        render(<SquatAnalyzer />);
        await userEvent.click(screen.getByRole("button", { name: /upload image/i }));
        expect(
            screen.getByRole("button", { name: /choose image/i })
        ).toBeInTheDocument();
    });
});

describe("SquatAnalyzer - CSV download", () => {
    it("does not show 'Download CSV' when no frames have been recorded", () => {
        render(<SquatAnalyzer />);
        expect(
            screen.queryByRole("button", { name: /download csv/i })
        ).not.toBeInTheDocument();
    });
});

describe("SquatAnalyzer - error display", () => {
    it("shows an error message when camera access fails", async () => {
        Object.defineProperty(window.navigator, "mediaDevices", {
            writable: true,
            value: {
                getUserMedia: vi.fn().mockRejectedValue(new Error("Permission denied")),
            },
        });

        render(<SquatAnalyzer />);
        await userEvent.click(screen.getByRole("button", { name: /start recording/i }));

        await waitFor(() => {
            expect(
                screen.getByText(/permission denied/i)
            ).toBeInTheDocument();
        });
    });
});

describe("SquatAnalyzer - video upload validation", () => {
    it("shows an error when a non-video file is selected", async () => {
        render(<SquatAnalyzer />);
        await userEvent.click(screen.getByText("Upload Video"));

        const input = document.querySelector("input[type='file'][accept='video/*']");
        const badFile = new File(["data"], "photo.png", { type: "image/png" });
        fireEvent.change(input, { target: { files: [badFile] } });

        await waitFor(() => {
            expect(
                screen.getByText(/please select a video file/i)
            ).toBeInTheDocument();
        });
    });
});

describe("SquatAnalyzer - image upload validation", () => {
    it("shows an error when a non-image file is selected", async () => {
        render(<SquatAnalyzer />);
        await userEvent.click(screen.getByText("Upload Image"));

        const input = document.querySelector("input[type='file'][accept='image/*']");
        const badFile = new File(["data"], "clip.mp4", { type: "video/mp4" });
        fireEvent.change(input, { target: { files: [badFile] } });

        await waitFor(() => {
            expect(
                screen.getByText(/please select an image file/i)
            ).toBeInTheDocument();
        });
    });

    it("analyzes uploaded image and shows form score + 3-D replay", async () => {
        const mp = await import(
            "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/+esm"
        );
        vi.spyOn(mp.PoseLandmarker, "createFromOptions").mockImplementation(
            async (_vision, opts) => {
                if (opts.runningMode === "IMAGE") {
                    const lms = Array.from({ length: 33 }, (_, i) => ({
                        x: 0.4 + i * 0.001,
                        y: 0.3 + i * 0.001,
                        z: 0.1,
                        visibility: 0.9,
                    }));
                    return {
                        detect: vi.fn().mockReturnValue({
                            landmarks: [lms],
                            worldLandmarks: [lms],
                        }),
                        close: vi.fn(),
                    };
                }
                return {
                    detectForVideo: vi.fn().mockReturnValue({
                        landmarks: [],
                        worldLandmarks: [],
                    }),
                    close: vi.fn(),
                };
            },
        );

        vi.spyOn(URL, "createObjectURL").mockReturnValue("blob:test-image");
        vi.spyOn(URL, "revokeObjectURL").mockImplementation(() => {});

        class MockImage {
            naturalWidth = 640;
            naturalHeight = 480;

            set src(_value) {
                setTimeout(() => this.onload?.(), 0);
            }
        }
        vi.stubGlobal("Image", MockImage);

        render(<SquatAnalyzer />);
        await userEvent.click(screen.getByText("Upload Image"));

        const input = document.querySelector("input[type='file'][accept='image/*']");
        const imageFile = new File(["image-data"], "pose.png", { type: "image/png" });
        fireEvent.change(input, { target: { files: [imageFile] } });

        await waitFor(() => {
            expect(
                fetchMock.fetch.mock.calls.some(
                    ([url, options]) =>
                        String(url).includes("/api/v1/squat/analyze-session") &&
                        options?.method === "POST",
                ),
            ).toBe(true);
        });

        expect(await screen.findByText(/good form/i)).toBeInTheDocument();
        expect(screen.getByText(/3-D Skeleton Replay/i)).toBeInTheDocument();
    });
});