// src/frontend/test/components/Predict.test.jsx

import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import {
    render,
    screen,
    fireEvent,
    waitFor,
    act,
} from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import Predict from "../../src/components/Predict";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** Wait for the loading state (model-info) to resolve. */
async function waitForModelLoad() {
    await waitFor(() =>
        expect(screen.queryByText(/loading model/i)).not.toBeInTheDocument(),
    );
}

// ---------------------------------------------------------------------------
// Setup / teardown
// ---------------------------------------------------------------------------

const fetchMock = vi.fn();

beforeEach(() => {
    fetchMock.mockImplementation((url) => {
        if (url.includes("model-info")) {
            return Promise.resolve({
                ok: true,
                status: 200,
                json: () => Promise.resolve({ expected_features: 41 }),
            });
        }

        return Promise.resolve({
            ok: true,
            status: 200,
            json: () =>
                Promise.resolve({
                    prediction: 0.87,
                    model_uri: "models:/champion/1",
                    run_id: "abc123",
                }),
        });
    });

    vi.stubGlobal("fetch", fetchMock);
});

afterEach(() => {
    vi.restoreAllMocks();
});

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

describe("Predict — rendering", () => {
    it("renders the page heading", () => {
        render(<Predict />);
        expect(
            screen.getByRole("heading", { name: /model prediction/i }),
        ).toBeInTheDocument();
    });

    it("shows the sub-heading copy", () => {
        render(<Predict />);
        expect(
            screen.getByText(/adjust values\. predict\./i),
        ).toBeInTheDocument();
    });

    it("shows 'Loading model…' text in Predict button while model-info is loading", () => {
        vi.stubGlobal(
            "fetch",
            vi.fn(() => new Promise(() => {})),
        );
        render(<Predict />);
        expect(
            screen.getByRole("button", { name: /loading model/i }),
        ).toBeInTheDocument();
    });

    it("disables action buttons while model-info is loading", () => {
        vi.stubGlobal(
            "fetch",
            vi.fn(() => new Promise(() => {})),
        );
        render(<Predict />);
        expect(screen.getByRole("button", { name: /example/i })).toBeDisabled();
        expect(screen.getByRole("button", { name: /copy/i })).toBeDisabled();
        expect(screen.getByRole("button", { name: /import/i })).toBeDisabled();
        expect(
            screen.getByRole("button", { name: /loading model/i }),
        ).toBeDisabled();
    });

    it("displays feature count after model-info loads", async () => {
        render(<Predict />);
        await waitFor(() => expect(screen.getByText(/41/)).toBeInTheDocument());
    });

    it("renders the endpoint URL", async () => {
        render(<Predict />);
        await waitForModelLoad();
        expect(
            screen.getByText(/\/api\/v1\/predict\/champion/),
        ).toBeInTheDocument();
    });
});

// ---------------------------------------------------------------------------
// Model-info error handling
// ---------------------------------------------------------------------------

describe("Predict — model-info error handling", () => {
    it("shows a warning when model-info returns a non-OK status", async () => {
        vi.stubGlobal(
            "fetch",
            vi.fn(() =>
                Promise.resolve({
                    ok: false,
                    status: 500,
                    json: () => Promise.resolve({ detail: "Server exploded" }),
                }),
            ),
        );
        render(<Predict />);
        await waitFor(() =>
            expect(screen.getByText(/model info warning/i)).toBeInTheDocument(),
        );
    });

    it("shows a warning when expected_features is missing from model-info response", async () => {
        vi.stubGlobal(
            "fetch",
            vi.fn(() =>
                Promise.resolve({
                    ok: true,
                    status: 200,
                    json: () => Promise.resolve({}),
                }),
            ),
        );
        render(<Predict />);
        await waitFor(() =>
            expect(screen.getByText(/model info warning/i)).toBeInTheDocument(),
        );
    });

    it("shows an error if Predict is clicked while model-info is still loading", async () => {
        vi.stubGlobal(
            "fetch",
            vi.fn(() => new Promise(() => {})),
        );
        render(<Predict />);
        const btn = screen.getByRole("button", { name: /loading model/i });
        expect(btn).toBeDisabled();
    });
});

// ---------------------------------------------------------------------------
// Example / Zero buttons
// ---------------------------------------------------------------------------

describe("Predict — Example and Zero buttons", () => {
    it("shows 'Example loaded.' hint when Example is clicked", async () => {
        render(<Predict />);
        await waitForModelLoad();

        await userEvent.click(screen.getByRole("button", { name: /example/i }));
        expect(screen.getByText(/example loaded/i)).toBeInTheDocument();
    });

    it("shows 'All values set to 0.' hint when Zero is clicked", async () => {
        render(<Predict />);
        await waitForModelLoad();

        await userEvent.click(screen.getByRole("button", { name: /zero/i }));
        expect(screen.getByText(/all values set to 0/i)).toBeInTheDocument();
    });
});

// ---------------------------------------------------------------------------
// Import modal
// ---------------------------------------------------------------------------

describe("Predict — Import modal", () => {
    async function openImport() {
        render(<Predict />);
        await waitForModelLoad();
        await userEvent.click(screen.getByRole("button", { name: /import/i }));
    }

    it("opens the import modal when Import is clicked", async () => {
        await openImport();
        expect(screen.getByText(/import values/i)).toBeInTheDocument();
    });

    it("closes the import modal when Close is clicked", async () => {
        await openImport();
        await userEvent.click(screen.getByRole("button", { name: /close/i }));
        expect(screen.queryByText(/import values/i)).not.toBeInTheDocument();
    });

    it("closes the import modal on Escape key", async () => {
        await openImport();
        fireEvent.keyDown(document, { key: "Escape" });
        await waitFor(() =>
            expect(
                screen.queryByText(/import values/i),
            ).not.toBeInTheDocument(),
        );
    });

    it("shows an error when applying fewer values than expected", async () => {
        await openImport();
        const textarea = screen.getByRole("textbox");
        await userEvent.clear(textarea);
        await userEvent.type(textarea, "0.1, 0.2, 0.3");
        await userEvent.click(screen.getByRole("button", { name: /apply/i }));
        await waitFor(() =>
            expect(
                screen.getByText(/expected 41 features/i),
            ).toBeInTheDocument(),
        );
    });

    it("shows an error when pasting non-numeric values", async () => {
        await openImport();
        const textarea = screen.getByRole("textbox");
        await userEvent.clear(textarea);
        await userEvent.type(textarea, "a, b, c");
        await userEvent.click(screen.getByRole("button", { name: /apply/i }));
        await waitFor(() =>
            expect(
                screen.getByText(/use comma-separated numbers only/i),
            ).toBeInTheDocument(),
        );
    });

    it("accepts and applies exactly 41 values", async () => {
        await openImport();
        const values = Array(41).fill("0.5").join(", ");
        const textarea = screen.getByRole("textbox");
        await userEvent.clear(textarea);
        await userEvent.type(textarea, values);
        await userEvent.click(screen.getByRole("button", { name: /apply/i }));
        await waitFor(() =>
            expect(
                screen.queryByText(/import values/i),
            ).not.toBeInTheDocument(),
        );
        expect(screen.getByText(/imported\./i)).toBeInTheDocument();
    });

    it("truncates pasted values to expectedCount when more are provided", async () => {
        await openImport();
        const values = Array(60).fill("0.5").join(", ");
        const textarea = screen.getByRole("textbox");
        await userEvent.clear(textarea);
        await userEvent.type(textarea, values);
        await userEvent.click(screen.getByRole("button", { name: /apply/i }));

        await waitFor(() =>
            expect(
                screen.queryByText(/import values/i),
            ).not.toBeInTheDocument(),
        );
    });
});

// ---------------------------------------------------------------------------
// Prediction flow
// ---------------------------------------------------------------------------

describe("Predict — prediction flow", () => {
    it("calls the correct endpoint on Predict click", async () => {
        render(<Predict />);
        await waitForModelLoad();
        await userEvent.click(
            screen.getByRole("button", { name: /^predict$/i }),
        );

        await waitFor(() =>
            expect(fetchMock).toHaveBeenCalledWith(
                "/api/v1/predict/champion",
                expect.objectContaining({ method: "POST" }),
            ),
        );
    });

    it("sends the correct number of features in the request body", async () => {
        render(<Predict />);
        await waitForModelLoad();
        await userEvent.click(
            screen.getByRole("button", { name: /^predict$/i }),
        );

        await waitFor(() => {
            const call = fetchMock.mock.calls.find((c) =>
                c[0].includes("/predict/"),
            );
            expect(call).toBeDefined();
            const body = JSON.parse(call[1].body);
            expect(body.features).toHaveLength(41);
        });
    });

    it("displays prediction result after successful response", async () => {
        render(<Predict />);
        await waitForModelLoad();
        await userEvent.click(
            screen.getByRole("button", { name: /^predict$/i }),
        );

        await waitFor(() =>
            expect(screen.getByText(/0\.87/)).toBeInTheDocument(),
        );
        expect(screen.getByText(/models:\/champion\/1/)).toBeInTheDocument();
        expect(screen.getByText(/abc123/)).toBeInTheDocument();
    });

    it("displays response time after a successful prediction", async () => {
        render(<Predict />);
        await waitForModelLoad();
        await userEvent.click(
            screen.getByRole("button", { name: /^predict$/i }),
        );

        await waitFor(() =>
            expect(screen.getByText(/ms/i)).toBeInTheDocument(),
        );
    });

    it("displays an error message when the API returns a non-OK status", async () => {
        vi.stubGlobal(
            "fetch",
            vi.fn((url) => {
                if (url.includes("model-info"))
                    return Promise.resolve({
                        ok: true,
                        status: 200,
                        json: () => Promise.resolve({ expected_features: 41 }),
                    });
                return Promise.resolve({
                    ok: false,
                    status: 422,
                    json: () =>
                        Promise.resolve({ detail: "Unprocessable entity" }),
                });
            }),
        );

        render(<Predict />);
        await waitForModelLoad();
        await userEvent.click(
            screen.getByRole("button", { name: /^predict$/i }),
        );

        await waitFor(() =>
            expect(
                screen.getByText(/unprocessable entity/i),
            ).toBeInTheDocument(),
        );
    });

    it("displays a generic error when fetch throws (network failure)", async () => {
        vi.stubGlobal(
            "fetch",
            vi.fn((url) => {
                if (url.includes("model-info"))
                    return Promise.resolve({
                        ok: true,
                        status: 200,
                        json: () => Promise.resolve({ expected_features: 41 }),
                    });
                return Promise.reject(new Error("Network error"));
            }),
        );

        render(<Predict />);
        await waitForModelLoad();
        await userEvent.click(
            screen.getByRole("button", { name: /^predict$/i }),
        );

        await waitFor(() =>
            expect(screen.getByText(/network error/i)).toBeInTheDocument(),
        );
    });

    it("shows 'Predicting…' in the button while loading", async () => {
        let resolvePredict;
        vi.stubGlobal(
            "fetch",
            vi.fn((url) => {
                if (url.includes("model-info"))
                    return Promise.resolve({
                        ok: true,
                        status: 200,
                        json: () => Promise.resolve({ expected_features: 41 }),
                    });
                return new Promise((res) => {
                    resolvePredict = res;
                });
            }),
        );

        render(<Predict />);
        await waitForModelLoad();
        await userEvent.click(
            screen.getByRole("button", { name: /^predict$/i }),
        );

        expect(
            screen.getByRole("button", { name: /predicting/i }),
        ).toBeInTheDocument();

        act(() => {
            resolvePredict({
                ok: true,
                status: 200,
                json: () =>
                    Promise.resolve({
                        prediction: 0.5,
                        model_uri: "",
                        run_id: "",
                    }),
            });
        });
    });

    it("clears a previous error when Predict is clicked again", async () => {
        let callCount = 0;
        vi.stubGlobal(
            "fetch",
            vi.fn((url) => {
                if (url.includes("model-info"))
                    return Promise.resolve({
                        ok: true,
                        status: 200,
                        json: () => Promise.resolve({ expected_features: 41 }),
                    });
                callCount++;
                if (callCount === 1)
                    return Promise.resolve({
                        ok: false,
                        status: 500,
                        json: () => Promise.resolve({ detail: "oops" }),
                    });
                return Promise.resolve({
                    ok: true,
                    status: 200,
                    json: () =>
                        Promise.resolve({
                            prediction: 0.9,
                            model_uri: "",
                            run_id: "",
                        }),
                });
            }),
        );

        render(<Predict />);
        await waitForModelLoad();

        await userEvent.click(
            screen.getByRole("button", { name: /^predict$/i }),
        );
        await waitFor(() =>
            expect(screen.getByText(/oops/i)).toBeInTheDocument(),
        );

        await userEvent.click(
            screen.getByRole("button", { name: /^predict$/i }),
        );
        await waitFor(() =>
            expect(screen.queryByText(/oops/i)).not.toBeInTheDocument(),
        );
    });
});
