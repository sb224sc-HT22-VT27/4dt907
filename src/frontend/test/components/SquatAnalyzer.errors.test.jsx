import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect, vi } from "vitest";
import SquatAnalyzer from "../../src/components/SquatAnalyzer";
import { setupSquatAnalyzerMocks } from "./squatAnalyzerTestSetup";

setupSquatAnalyzerMocks();

describe("SquatAnalyzer - error display", () => {
    it("shows an error message when camera access fails", async () => {
        Object.defineProperty(window.navigator, "mediaDevices", {
            writable: true,
            value: {
                getUserMedia: vi
                    .fn()
                    .mockRejectedValue(new Error("Permission denied")),
            },
        });

        render(<SquatAnalyzer />);
        await userEvent.click(
            screen.getByRole("button", { name: /start recording/i }),
        );

        await waitFor(() => {
            expect(screen.getByText(/permission denied/i)).toBeInTheDocument();
        });
    });
});

describe("SquatAnalyzer - video upload validation", () => {
    it("shows an error when a non-video file is selected", async () => {
        render(<SquatAnalyzer />);
        await userEvent.click(screen.getByText("Upload Video"));

        const input = document.querySelector(
            "input[type='file'][accept='video/*']",
        );
        const badFile = new File(["data"], "photo.png", { type: "image/png" });
        fireEvent.change(input, { target: { files: [badFile] } });

        await waitFor(() => {
            expect(
                screen.getByText(/please select a video file/i),
            ).toBeInTheDocument();
        });
    });
});

describe("SquatAnalyzer - image upload validation", () => {
    it("shows an error when a non-image file is selected", async () => {
        render(<SquatAnalyzer />);
        await userEvent.click(screen.getByText("Upload Image"));

        const input = document.querySelector(
            "input[type='file'][accept='image/*']",
        );
        const badFile = new File(["data"], "clip.mp4", { type: "video/mp4" });
        fireEvent.change(input, { target: { files: [badFile] } });

        await waitFor(() => {
            expect(
                screen.getByText(/please select an image file/i),
            ).toBeInTheDocument();
        });
    });
});
