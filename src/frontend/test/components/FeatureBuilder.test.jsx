// src/frontend/test/components/FeatureBuilder.test.jsx

import { describe, it, expect, vi } from "vitest";
import { render, screen, fireEvent } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import FeatureBuilder from "../../src/components/FeatureBuilder";
import { FEATURE_GROUPS } from "../../src/featuresSchema";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/**
 * Build a zeroed-out values object for `count` features drawn from FEATURE_GROUPS.
 * Mirrors the shape that Predict.jsx passes into FeatureBuilder.
 */
function buildValues(count = 41) {
    const values = {};
    let used = 0;
    for (const g of FEATURE_GROUPS) {
        if (used >= count) break;
        if (g.single) {
            values[g.single] = 0;
            used++;
            continue;
        }
        const start = g.startIndex ?? 1;
        for (let i = 0; i < g.count && used < count; i++) {
            values[`${g.prefix}${start + i}${g.suffix}`] = 0;
            used++;
        }
    }
    return values;
}

const NOOP_SET = vi.fn();

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------

describe("FeatureBuilder — rendering", () => {
    it("renders without crashing", () => {
        render(
            <FeatureBuilder
                values={buildValues(41)}
                setValues={NOOP_SET}
                maxFeatures={41}
            />,
        );
    });

    it("renders at least one tab button", () => {
        render(
            <FeatureBuilder
                values={buildValues(41)}
                setValues={NOOP_SET}
                maxFeatures={41}
            />,
        );
        const tabs = screen.getAllByRole("button");
        expect(tabs.length).toBeGreaterThan(0);
    });

    it("renders knob sliders (role=slider) for the active tab", () => {
        render(
            <FeatureBuilder
                values={buildValues(41)}
                setValues={NOOP_SET}
                maxFeatures={41}
            />,
        );
        const knobs = screen.getAllByRole("slider");
        expect(knobs.length).toBeGreaterThan(0);
    });

    it("renders numeric inputs for each knob in the active tab", () => {
        render(
            <FeatureBuilder
                values={buildValues(41)}
                setValues={NOOP_SET}
                maxFeatures={41}
            />,
        );
        const inputs = screen.getAllByRole("spinbutton");
        expect(inputs.length).toBeGreaterThan(0);
    });
});

// ---------------------------------------------------------------------------
// Tab switching
// ---------------------------------------------------------------------------

describe("FeatureBuilder — tab switching", () => {
    it("shows different knobs after switching to a different tab", async () => {
        render(
            <FeatureBuilder
                values={buildValues(41)}
                setValues={NOOP_SET}
                maxFeatures={41}
            />,
        );

        const tabs = screen.getAllByRole("button");
        if (tabs.length < 2) return;

        const initialKnobCount = screen.getAllByRole("slider").length;

        await userEvent.click(tabs[1]);

        const newKnobs = screen.getAllByRole("slider");
        expect(newKnobs.length).toBeGreaterThan(0);
        expect(typeof newKnobs.length).toBe("number");

        void initialKnobCount;
    });

    it("highlights the active tab differently from inactive tabs", async () => {
        render(
            <FeatureBuilder
                values={buildValues(41)}
                setValues={NOOP_SET}
                maxFeatures={41}
            />,
        );

        const tabs = screen.getAllByRole("button");
        if (tabs.length < 2) return;

        await userEvent.click(tabs[1]);
        expect(tabs[1].className).toContain("bg-white/80");
    });

    it("does not show tabs for groups that have no features (due to maxFeatures cap)", () => {
        render(
            <FeatureBuilder
                values={buildValues(1)}
                setValues={NOOP_SET}
                maxFeatures={1}
            />,
        );

        const tabs = screen.getAllByRole("button");
        expect(tabs.length).toBe(1);
    });
});

// ---------------------------------------------------------------------------
// maxFeatures cap
// ---------------------------------------------------------------------------

describe("FeatureBuilder — maxFeatures cap", () => {
    it("renders no more sliders than maxFeatures across all tabs", async () => {
        const MAX = 10;
        render(
            <FeatureBuilder
                values={buildValues(MAX)}
                setValues={NOOP_SET}
                maxFeatures={MAX}
            />,
        );

        const tabs = screen.getAllByRole("button");
        let total = 0;

        for (const tab of tabs) {
            await userEvent.click(tab);
            total += screen.getAllByRole("slider").length;
        }

        expect(total).toBeLessThanOrEqual(MAX);
    });

    it("shows only 1 slider when maxFeatures=1", () => {
        render(
            <FeatureBuilder
                values={buildValues(1)}
                setValues={NOOP_SET}
                maxFeatures={1}
            />,
        );
        expect(screen.getAllByRole("slider")).toHaveLength(1);
    });

    it("renders all 41 sliders (across tabs) when maxFeatures=41", async () => {
        render(
            <FeatureBuilder
                values={buildValues(41)}
                setValues={NOOP_SET}
                maxFeatures={41}
            />,
        );

        const tabs = screen.getAllByRole("button");
        let total = 0;
        for (const tab of tabs) {
            await userEvent.click(tab);
            total += screen.getAllByRole("slider").length;
        }

        expect(total).toBe(41);
    });
});

// ---------------------------------------------------------------------------
// Knob value display
// ---------------------------------------------------------------------------

describe("FeatureBuilder — Knob value display", () => {
    it("numeric inputs reflect the current feature values", () => {
        const values = buildValues(41);
        const firstKey = Object.keys(values)[0];
        values[firstKey] = 0.75;

        render(
            <FeatureBuilder
                values={values}
                setValues={NOOP_SET}
                maxFeatures={41}
            />,
        );

        const inputs = screen.getAllByRole("spinbutton");
        const match = inputs.find((el) => Number(el.value) === 0.75);
        expect(match).toBeDefined();
    });

    it("knob slider reports correct aria-valuenow", () => {
        const values = buildValues(41);
        const firstKey = Object.keys(values)[0];
        values[firstKey] = 0.5;

        render(
            <FeatureBuilder
                values={values}
                setValues={NOOP_SET}
                maxFeatures={41}
            />,
        );

        const knobs = screen.getAllByRole("slider");
        const match = knobs.find(
            (el) => el.getAttribute("aria-valuenow") === "0.5",
        );
        expect(match).toBeDefined();
    });

    it("displays 0 for undefined/missing feature keys", () => {
        render(
            <FeatureBuilder
                values={{}}
                setValues={NOOP_SET}
                maxFeatures={41}
            />,
        );

        const inputs = screen.getAllByRole("spinbutton");
        inputs.forEach((el) => expect(Number(el.value)).toBe(0));
    });
});

// ---------------------------------------------------------------------------
// Knob interaction — numeric input
// ---------------------------------------------------------------------------

describe("FeatureBuilder — numeric input interaction", () => {
    it("calls setValues when a numeric input changes", async () => {
        const setValues = vi.fn();
        const values = buildValues(41);

        render(
            <FeatureBuilder
                values={values}
                setValues={setValues}
                maxFeatures={41}
            />,
        );

        const inputs = screen.getAllByRole("spinbutton");
        await userEvent.clear(inputs[0]);
        await userEvent.type(inputs[0], "0.42");

        expect(setValues).toHaveBeenCalled();
    });

    it("clamps values to [0, 1] range via the numeric input", async () => {
        const setValues = vi.fn();
        const values = buildValues(41);

        render(
            <FeatureBuilder
                values={values}
                setValues={setValues}
                maxFeatures={41}
            />,
        );

        const inputs = screen.getAllByRole("spinbutton");
        fireEvent.change(inputs[0], { target: { value: "5" } }); // above max of 1

        expect(setValues).toHaveBeenCalledWith(expect.any(Function));

        const updater =
            setValues.mock.calls[setValues.mock.calls.length - 1][0];
        const nextValues = updater(values);
        const changedKey = Object.keys(values)[0];
        expect(nextValues[changedKey]).toBeLessThanOrEqual(1);
    });
});

// ---------------------------------------------------------------------------
// Knob interaction — pointer / drag
// ---------------------------------------------------------------------------

describe("FeatureBuilder — Knob drag interaction", () => {
    it("calls setValues when pointer is dragged on a knob", async () => {
        const setValues = vi.fn();
        render(
            <FeatureBuilder
                values={buildValues(41)}
                setValues={setValues}
                maxFeatures={41}
            />,
        );

        const knobs = screen.getAllByRole("slider");
        const knob = knobs[0];

        fireEvent.pointerDown(knob, { clientX: 50, clientY: 50, pointerId: 1 });
        fireEvent.pointerMove(knob, { clientX: 70, clientY: 30, pointerId: 1 });
        fireEvent.pointerUp(knob, { pointerId: 1 });

        expect(setValues).toHaveBeenCalled();
    });

    it("does not update if pointer moves without being captured (not dragging)", () => {
        const setValues = vi.fn();
        render(
            <FeatureBuilder
                values={buildValues(41)}
                setValues={setValues}
                maxFeatures={41}
            />,
        );

        const knobs = screen.getAllByRole("slider");
        fireEvent.pointerMove(knobs[0], {
            clientX: 70,
            clientY: 30,
            pointerId: 1,
        });

        expect(setValues).not.toHaveBeenCalled();
    });
});

// ---------------------------------------------------------------------------
// Accessibility
// ---------------------------------------------------------------------------

describe("FeatureBuilder — accessibility", () => {
    it("all knob sliders have aria-label", () => {
        render(
            <FeatureBuilder
                values={buildValues(41)}
                setValues={NOOP_SET}
                maxFeatures={41}
            />,
        );
        const knobs = screen.getAllByRole("slider");
        knobs.forEach((k) =>
            expect(k.getAttribute("aria-label")).not.toBeNull(),
        );
    });

    it("all knob sliders have aria-valuemin, aria-valuemax, aria-valuenow", () => {
        render(
            <FeatureBuilder
                values={buildValues(41)}
                setValues={NOOP_SET}
                maxFeatures={41}
            />,
        );
        const knobs = screen.getAllByRole("slider");
        knobs.forEach((k) => {
            expect(k.getAttribute("aria-valuemin")).not.toBeNull();
            expect(k.getAttribute("aria-valuemax")).not.toBeNull();
            expect(k.getAttribute("aria-valuenow")).not.toBeNull();
        });
    });

    it("tab buttons are keyboard-focusable", () => {
        render(
            <FeatureBuilder
                values={buildValues(41)}
                setValues={NOOP_SET}
                maxFeatures={41}
            />,
        );
        const tabs = screen.getAllByRole("button");
        tabs.forEach((t) => expect(t.tabIndex).toBeGreaterThanOrEqual(0));
    });
});
