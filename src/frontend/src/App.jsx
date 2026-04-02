// App shell: provides the top-level layout wrapper and mounts the main feature components
import React, { useState } from "react";
import Predict from "./components/Predict";
import SquatAnalyzer from "./components/SquatAnalyzer";

const TABS = [
    { id: "predict", label: "Predict" },
    { id: "squat", label: "Squat Analyzer" },
];

/**
 * Root React component.
 * Keeps global layout/styling concerns.
 */
export default function App() {
    const [activeTab, setActiveTab] = useState("predict");

    return (
        <div className="min-h-screen bg-aurora">
            <nav className="flex gap-2 px-6 pt-4">
                {TABS.map((tab) => (
                    <button
                        key={tab.id}
                        onClick={() => setActiveTab(tab.id)}
                        className={`px-4 py-2 rounded-t-lg text-sm font-semibold transition ${
                            activeTab === tab.id
                                ? "bg-slate-800 text-white"
                                : "text-slate-400 hover:text-white"
                        }`}
                    >
                        {tab.label}
                    </button>
                ))}
            </nav>
            {activeTab === "predict" && <Predict />}
            {activeTab === "squat" && <SquatAnalyzer />}
        </div>
    );
}
