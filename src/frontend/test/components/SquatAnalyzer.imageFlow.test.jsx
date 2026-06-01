import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, vi } from "vitest";
import SquatAnalyzer from "../../src/components/SquatAnalyzer";
import {
    analyzeSessionResponse,
    detectImageMock,
    fetchMock,
    setupSquatAnalyzerMocks,
    classifyResponse,
} from "./squatAnalyzerTestSetup";

setupSquatAnalyzerMocks();

describe("SquatAnalyzer - image upload validation", () => {
    it("runs analyze-session and shows 3-D replay for uploaded image with pose", async () => {
        const landmarks = Array.from({ length: 33 }, (_, i) => ({
            x: 0.25 + i * 0.005,
            y: 0.2 + i * 0.005,
            z: i * 0.001,
            visibility: 0.99,
        }));
        const worldLandmarks = Array.from({ length: 33 }, (_, i) => ({
            x: i * 0.01,
            y: i * 0.01,
            z: i * 0.01,
        }));
        detectImageMock.mockReturnValue({
            landmarks: [landmarks],
            worldLandmarks: [worldLandmarks],
        });

        const OriginalImage = globalThis.Image;
        class MockImage {
            constructor() {
                this.naturalWidth = 1280;
                this.naturalHeight = 720;
                this.onload = null;
                this.onerror = null;
                this._src = "";
            }
            set src(value) {
                this._src = value;
                Promise.resolve().then(() => this.onload?.());
            }
        }
        globalThis.Image = MockImage;

        const createObjectURLSpy = vi
            .spyOn(URL, "createObjectURL")
            .mockReturnValue("blob:mock-image");
        const revokeObjectURLSpy = vi
            .spyOn(URL, "revokeObjectURL")
            .mockImplementation(() => {});

        try {
            render(<SquatAnalyzer />);
            await userEvent.click(
                screen.getByRole("button", { name: /upload image/i }),
            );

            const input = document.querySelector(
                "input[type='file'][accept='image/*']",
            );
            const imageFile = new File(["img"], "pose.png", {
                type: "image/png",
            });
            fireEvent.change(input, { target: { files: [imageFile] } });

            await waitFor(() => {
                expect(fetchMock).toHaveBeenCalledWith(
                    expect.stringMatching(/\/api\/v1\/squat\/analyze-session$/),
                    expect.objectContaining({ method: "POST" }),
                );
            });
            expect(
                await screen.findByText(/3-d skeleton replay/i),
            ).toBeInTheDocument();
            expect(
                await screen.findByText(/1 frame recorded/i),
            ).toBeInTheDocument();
            expect(createObjectURLSpy).toHaveBeenCalled();
            expect(revokeObjectURLSpy).toHaveBeenCalled();
        } finally {
            globalThis.Image = OriginalImage;
        }
    });

    it("keeps pipeline timing window hidden until toggled on", async () => {
        const landmarks = Array.from({ length: 33 }, (_, i) => ({
            x: 0.25 + i * 0.005,
            y: 0.2 + i * 0.005,
            z: i * 0.001,
            visibility: 0.99,
        }));
        const worldLandmarks = Array.from({ length: 33 }, (_, i) => ({
            x: i * 0.01,
            y: i * 0.01,
            z: i * 0.01,
        }));
        detectImageMock.mockReturnValue({
            landmarks: [landmarks],
            worldLandmarks: [worldLandmarks],
        });

        const OriginalImage = globalThis.Image;
        class MockImage {
            constructor() {
                this.naturalWidth = 1280;
                this.naturalHeight = 720;
                this.onload = null;
                this.onerror = null;
                this._src = "";
            }
            set src(value) {
                this._src = value;
                Promise.resolve().then(() => this.onload?.());
            }
        }
        globalThis.Image = MockImage;

        try {
            render(<SquatAnalyzer />);
            await userEvent.click(
                screen.getByRole("button", { name: /upload image/i }),
            );

            const input = document.querySelector(
                "input[type='file'][accept='image/*']",
            );
            const imageFile = new File(["img"], "pose.png", {
                type: "image/png",
            });
            fireEvent.change(input, { target: { files: [imageFile] } });

            const toggle = await screen.findByRole("checkbox", {
                name: /show pipeline timing window/i,
            });
            expect(
                screen.queryByText(/network \(round-trip\)/i),
            ).not.toBeInTheDocument();

            await userEvent.click(toggle);
            expect(
                screen.getByText(/network \(round-trip\)/i),
            ).toBeInTheDocument();
        } finally {
            globalThis.Image = OriginalImage;
        }
    });

    it("shows rounded squat score for processed image results", async () => {
        fetchMock.mockImplementation((url) => {
            if (url.includes("/squat/classify")) return classifyResponse();
            if (url.includes("/squat/analyze-session")) {
                return analyzeSessionResponse({
                    start_stop: 0,
                    good_bad_score: null,
                    squat_score: 2.8,
                });
            }
            if (url.includes("/model-info/start-stop")) {
                return Promise.resolve({
                    ok: true,
                    json: () => Promise.resolve({ mae_total_average: 0.1 }),
                });
            }
            return Promise.resolve({ ok: false });
        });

        const landmarks = Array.from({ length: 33 }, (_, i) => ({
            x: 0.25 + i * 0.005,
            y: 0.2 + i * 0.005,
            z: i * 0.001,
            visibility: 0.99,
        }));
        const worldLandmarks = Array.from({ length: 33 }, (_, i) => ({
            x: i * 0.01,
            y: i * 0.01,
            z: i * 0.01,
        }));
        detectImageMock.mockReturnValue({
            landmarks: [landmarks],
            worldLandmarks: [worldLandmarks],
        });

        const OriginalImage = globalThis.Image;
        class MockImage {
            constructor() {
                this.naturalWidth = 1280;
                this.naturalHeight = 720;
                this.onload = null;
                this.onerror = null;
                this._src = "";
            }
            set src(value) {
                this._src = value;
                Promise.resolve().then(() => this.onload?.());
            }
        }
        globalThis.Image = MockImage;

        try {
            render(<SquatAnalyzer />);
            await userEvent.click(
                screen.getByRole("button", { name: /upload image/i }),
            );
            const input = document.querySelector(
                "input[type='file'][accept='image/*']",
            );
            const imageFile = new File(["img"], "pose.png", {
                type: "image/png",
            });
            fireEvent.change(input, { target: { files: [imageFile] } });
        } finally {
            globalThis.Image = OriginalImage;
        }
    });
});
