// App shell: provides the top-level layout wrapper and mounts the main feature components
import React from "react";
import Predict from "./components/Predict";

/**
 * Root React component.
 * Keeps global layout/styling concerns.
 */
export default function App() {
  return (
    <div className="min-h-screen bg-aurora">
      <Predict />
    </div>
  );
}
