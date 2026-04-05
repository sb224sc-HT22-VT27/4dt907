// App shell: provides the top-level layout wrapper and mounts the main feature components
import React, { useState } from "react";
import Predict from "./components/Predict";
import SquatAnalyzer from "./components/SquatAnalyzer";
import BackendStatus from "./components/BackendStatus";

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
        <div className="min-h-screen ios-bg">
            {/* Navbar — glass strip pinned to the top */}
            <nav className="sticky top-0 z-50 flex items-center gap-3 px-6 py-3 ios-pill border-b border-black/5">
                {/* iOS-style segmented control */}
                <div className="flex rounded-full bg-black/[0.06] p-0.5 gap-px">
                    {TABS.map((tab) => (
                        <button
                            key={tab.id}
                            onClick={() => setActiveTab(tab.id)}
                            className={`px-5 py-1.5 rounded-full text-sm font-semibold transition-all duration-150 ${
                                activeTab === tab.id
                                    ? "bg-white text-slate-800 shadow-sm"
                                    : "text-slate-500 hover:text-slate-700"
                            }`}
                        >
                            {tab.label}
                        </button>
                    ))}
                </div>
                {/* Push the backend status pill to the right end of the nav bar */}
                <span className="ml-auto">
                    <BackendStatus />
                </span>
            </nav>
            {activeTab === "predict" && <Predict />}
            {activeTab === "squat" && <SquatAnalyzer />}
        </div>
    );
}
