// src/frontend/test/App.test.jsx
import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import App from "../src/App";

// Mock Predict so App tests stay isolated
vi.mock("../src/components/Predict", () => ({
    default: () => <div data-testid="predict-mock">Predict</div>,
}));

describe("App", () => {
    it("renders without crashing", () => {
        render(<App />);
        expect(screen.getByTestId("predict-mock")).toBeInTheDocument();
    });

    // it('renders the Predict component', () => {
    //   render(<App />)
    //   expect(screen.getByTestId('predict-mock')).toBeInTheDocument()
    // })

    it("has correct layout classes on the wrapper", () => {
        const { container } = render(<App />);
        expect(container.firstChild).toHaveClass("min-h-screen", "bg-aurora");
    });
});

// src/frontend/test/App.test.jsx
import { describe, it, expect, vi } from "vitest";
import { render, screen } from "@testing-library/react";
import App from "../src/App";

// Mock Predict so App tests stay isolated
vi.mock("../src/components/Predict", () => ({
    default: () => <div data-testid="predict-mock">Predict</div>,
}));

describe("App", () => {
    it("renders without crashing", () => {
        render(<App />);
        expect(screen.getByTestId("predict-mock")).toBeInTheDocument();
    });

    // it('renders the Predict component', () => {
    //   render(<App />)
    //   expect(screen.getByTestId('predict-mock')).toBeInTheDocument()
    // })

    it("has correct layout classes on the wrapper", () => {
        const { container } = render(<App />);
        expect(container.firstChild).toHaveClass("min-h-screen", "bg-aurora");
    });
});
