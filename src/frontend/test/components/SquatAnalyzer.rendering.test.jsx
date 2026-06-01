import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, it, expect } from "vitest";
import SquatAnalyzer from "../../src/components/SquatAnalyzer";
import { setupSquatAnalyzerMocks } from "./squatAnalyzerTestSetup";

setupSquatAnalyzerMocks();

describe("SquatAnalyzer - rendering", () => {
    it("renders the heading", () => {
        render(<SquatAnalyzer />);
        expect(
            screen.getByRole("heading", { name: /squat analyzer/i }),
        ).toBeInTheDocument();
    });

    it("renders two mode buttons", () => {
        render(<SquatAnalyzer />);
        const buttons = screen.getAllByRole("button", {
            name: /camera|upload|snapshot/i,
        });
        expect(buttons).toHaveLength(2);
    });
});

describe("SquatAnalyzer - switching modes", () => {
    it("switches to Upload Video mode when 'Upload Video' button is clicked", async () => {
        render(<SquatAnalyzer />);
        await userEvent.click(
            screen.getByRole("button", { name: /upload video/i }),
        );
        expect(
            screen.getByRole("button", { name: /choose video/i }),
        ).toBeInTheDocument();
    });

    it("switches to Upload Image mode when 'Upload Image' button is clicked", async () => {
        render(<SquatAnalyzer />);
        await userEvent.click(
            screen.getByRole("button", { name: /upload image/i }),
        );
        expect(
            screen.getByRole("button", { name: /choose image/i }),
        ).toBeInTheDocument();
    });
});

describe("SquatAnalyzer - CSV download", () => {
    it("does not show 'Download CSV' when no frames have been recorded", () => {
        render(<SquatAnalyzer />);
        expect(
            screen.queryByRole("button", { name: /download csv/i }),
        ).not.toBeInTheDocument();
    });
});
