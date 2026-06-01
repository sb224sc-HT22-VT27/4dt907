import { useEffect, useMemo, useRef, useState } from "react";
import { SQUAT_CONNECTIONS_BY_NAME, SQUAT_JOINT_ORDER } from "./squatAnalyzerUtils";

/**
 * Rotate a 3-D point by yaw (ry, around Y) then pitch (rx, around X).
 * Returns projected (px, py) and depth for optional sorting.
 */
function project3D(x, y, z, rxDeg, ryDeg, scale, cx, cy) {
    const rx = (rxDeg * Math.PI) / 180;
    const ry = (ryDeg * Math.PI) / 180;
    // Yaw
    const x1 = x * Math.cos(ry) + z * Math.sin(ry);
    const y1 = y;
    const z1 = -x * Math.sin(ry) + z * Math.cos(ry);
    // Pitch
    const x2 = x1;
    const y2 = y1 * Math.cos(rx) - z1 * Math.sin(rx);
    const z2 = y1 * Math.sin(rx) + z1 * Math.cos(rx);
    // MediaPipe world y is positive-down, so map directly (not negated).
    return { px: cx + x2 * scale, py: cy + y2 * scale, depth: z2 };
}

export default function Skeleton3DViewer({ frames, liveFrameRef }) {
    const canvasRef = useRef(null);
    const [playing, setPlaying] = useState(false);
    const [frameIdx, setFrameIdx] = useState(0);
    // true = always show the newest frame (live follow); false = user has control
    const [liveMode, setLiveMode] = useState(true);
    const [prevFramesLength, setPrevFramesLength] = useState(frames.length);
    const rotation = useRef({ x: 15, y: 25 });
    const dragRef = useRef(null);
    const zoomRef = useRef(260);
    const playTimerRef = useRef(null);
    const frameIdxRef = useRef(0);

    // Derived state updates during render (avoids setState-in-effect lint error).
    if (prevFramesLength !== frames.length) {
        setPrevFramesLength(frames.length);
        if (frames.length === 0) {
            setLiveMode(true);
        } else if (liveMode && !playing) {
            setFrameIdx(frames.length - 1);
        }
    }

    const displayedFrameIdx = useMemo(() => {
        if (frames.length === 0) return 0;
        return Math.min(frameIdx, frames.length - 1);
    }, [frames.length, frameIdx]);

    // Keep ref in sync with state for rAF callbacks
    useEffect(() => {
        frameIdxRef.current = displayedFrameIdx;
    }, [displayedFrameIdx]);

    // Play timer
    useEffect(() => {
        if (playing) {
            playTimerRef.current = setInterval(() => {
                setFrameIdx((i) => {
                    if (i >= frames.length - 1) {
                        setPlaying(false);
                        return frames.length - 1;
                    }
                    return i + 1;
                });
            }, 150); // ~6 fps replay
        } else {
            clearInterval(playTimerRef.current);
        }
        return () => clearInterval(playTimerRef.current);
    }, [playing, frames.length]);

    function drawFrame(frameData, isLive = false) {
        const canvas = canvasRef.current;
        if (!canvas || !frameData) return;
        const ctx = canvas.getContext("2d");
        const W = canvas.width;
        const H = canvas.height;
        ctx.clearRect(0, 0, W, H);

        const rx = rotation.current.x;
        const ry = rotation.current.y;
        const scale = zoomRef.current;
        const cx = W / 2;
        const cy = H * 0.82;

        // Lock rotation pivot at ankle midpoint so the model spins around the feet
        const la = frameData.joints?.["left_ankle"];
        const ra = frameData.joints?.["right_ankle"];
        const anchor =
            la && ra
                ? {
                      x: (la.x + ra.x) / 2,
                      y: (la.y + ra.y) / 2,
                      z: (la.z + ra.z) / 2,
                  }
                : (la ?? ra ?? { x: 0, y: 0, z: 0 });

        // Project all joints
        const pos = {};
        for (const { name } of SQUAT_JOINT_ORDER) {
            const j = frameData.joints?.[name];
            if (!j) continue;
            pos[name] = project3D(
                j.x - anchor.x,
                j.y - anchor.y,
                j.z - anchor.z,
                rx,
                ry,
                scale,
                cx,
                cy,
            );
        }

        // Ground plane (projected circle at y=0 in foot-anchored space)
        const GR = 0.5; // radius in world units
        const SEGS = 72;
        const rim = [];
        for (let i = 0; i <= SEGS; i++) {
            const t = (i / SEGS) * Math.PI * 2;
            rim.push(
                project3D(
                    GR * Math.cos(t),
                    0,
                    GR * Math.sin(t),
                    rx,
                    ry,
                    scale,
                    cx,
                    cy,
                ),
            );
        }
        // Filled disc
        ctx.beginPath();
        ctx.moveTo(rim[0].px, rim[0].py);
        for (let i = 1; i <= SEGS; i++) ctx.lineTo(rim[i].px, rim[i].py);
        ctx.closePath();
        ctx.fillStyle = "rgba(30,41,59,0.55)";
        ctx.fill();
        // Outer ring
        ctx.strokeStyle = "rgba(148,163,184,0.3)";
        ctx.lineWidth = 1;
        ctx.stroke();
        // Grid lines (4 across each axis)
        ctx.lineWidth = 0.5;
        ctx.strokeStyle = "rgba(148,163,184,0.12)";
        for (let i = -3; i <= 3; i++) {
            const s = (i / 3) * GR;
            const clamp = Math.sqrt(Math.max(0, GR * GR - s * s));
            const xa = project3D(-clamp, 0, s, rx, ry, scale, cx, cy);
            const xb = project3D(clamp, 0, s, rx, ry, scale, cx, cy);
            ctx.beginPath();
            ctx.moveTo(xa.px, xa.py);
            ctx.lineTo(xb.px, xb.py);
            ctx.stroke();
            const za = project3D(s, 0, -clamp, rx, ry, scale, cx, cy);
            const zb = project3D(s, 0, clamp, rx, ry, scale, cx, cy);
            ctx.beginPath();
            ctx.moveTo(za.px, za.py);
            ctx.lineTo(zb.px, zb.py);
            ctx.stroke();
        }

        // Draw bones
        for (const [aName, bName] of SQUAT_CONNECTIONS_BY_NAME) {
            const pa = pos[aName];
            const pb = pos[bName];
            if (!pa || !pb) continue;
            ctx.beginPath();
            ctx.moveTo(pa.px, pa.py);
            ctx.lineTo(pb.px, pb.py);
            const isLeft = aName.includes("left") || bName.includes("left");
            ctx.strokeStyle = isLeft
                ? "rgba(56,189,248,0.9)"
                : "rgba(248,113,113,0.9)";
            ctx.lineWidth = 2.6;
            ctx.stroke();
        }

        // Draw joints
        const pts = Object.entries(pos).map(([name, pt]) => ({ name, ...pt }));
        pts.sort((a, b) => a.depth - b.depth);
        for (const p of pts) {
            const isLeft = p.name.includes("left");
            ctx.beginPath();
            ctx.arc(p.px, p.py, 5, 0, Math.PI * 2);
            ctx.fillStyle = isLeft ? "#0ea5e9" : "#f43f5e";
            ctx.fill();
        }

        // Optional classification label
        if (!isLive && frameData.classification) {
            ctx.fillStyle = "rgba(15,23,42,0.7)";
            ctx.fillRect(10, 10, 140, 28);
            ctx.fillStyle = "#fff";
            ctx.font = "12px sans-serif";
            ctx.fillText(frameData.classification, 18, 28);
        }
    }

    // Draw when frames/live change
    useEffect(() => {
        const canvas = canvasRef.current;
        if (!canvas) return;
        if (frames.length === 0 && !liveFrameRef?.current) {
            const ctx = canvas.getContext("2d");
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            return;
        }
        const frame = liveMode ? liveFrameRef?.current : frames[displayedFrameIdx];
        drawFrame(frame, liveMode && !!liveFrameRef?.current);
    }, [frames, displayedFrameIdx, liveMode, liveFrameRef]);

    // Live auto-update when in live mode
    useEffect(() => {
        if (!liveMode) return;
        let rafId = null;
        function tick() {
            if (liveFrameRef?.current) drawFrame(liveFrameRef.current, true);
            rafId = requestAnimationFrame(tick);
        }
        rafId = requestAnimationFrame(tick);
        return () => cancelAnimationFrame(rafId);
    }, [liveMode, liveFrameRef]);

    function onPointerDown(e) {
        dragRef.current = {
            x: e.clientX,
            y: e.clientY,
            rx: rotation.current.x,
            ry: rotation.current.y,
        };
    }
    function onPointerMove(e) {
        if (!dragRef.current) return;
        const dx = e.clientX - dragRef.current.x;
        const dy = e.clientY - dragRef.current.y;
        rotation.current = {
            x: dragRef.current.rx + dy * 0.35,
            y: dragRef.current.ry + dx * 0.35,
        };
        drawFrame(
            liveFrameRef?.current ?? frames[frameIdxRef.current],
            !!liveFrameRef?.current,
        );
    }
    function onPointerUp() {
        dragRef.current = null;
    }
    function onWheel(e) {
        e.preventDefault();
        const delta = Math.sign(e.deltaY);
        zoomRef.current = Math.max(80, zoomRef.current + delta * 20);
        drawFrame(
            liveFrameRef?.current ?? frames[frameIdxRef.current],
            !!liveFrameRef?.current,
        );
    }

    return (
        <div className="flex flex-col gap-2 items-center w-full">
            <div className="ios-card rounded-2xl p-3 w-full max-w-xl">
                <canvas
                    ref={canvasRef}
                    width={560}
                    height={360}
                    className="w-full rounded-xl bg-slate-900"
                    onPointerDown={onPointerDown}
                    onPointerMove={onPointerMove}
                    onPointerUp={onPointerUp}
                    onPointerLeave={onPointerUp}
                    onWheel={onWheel}
                />
                <div className="flex items-center justify-between mt-3">
                    <div className="flex items-center gap-2">
                        <button
                            onClick={() => {
                                setPlaying(false);
                                setLiveMode(true);
                            }}
                            className={`ios-btn px-3 py-1 rounded-full text-xs font-semibold ${
                                liveMode
                                    ? "bg-white text-slate-800 shadow"
                                    : "text-slate-500"
                            }`}
                        >
                            Live
                        </button>
                        <button
                            onClick={() => {
                                setLiveMode(false);
                                setPlaying((p) => !p);
                            }}
                            className={`ios-btn px-3 py-1 rounded-full text-xs font-semibold ${
                                !liveMode
                                    ? "bg-white text-slate-800 shadow"
                                    : "text-slate-500"
                            }`}
                        >
                            {playing ? "Pause" : "Play"}
                        </button>
                    </div>

                    <div className="flex items-center gap-2">
                        <button
                            onClick={() => {
                                setLiveMode(false);
                                setPlaying(false);
                                setFrameIdx((i) => Math.max(0, i - 1));
                            }}
                            className="ios-btn w-7 h-7 flex items-center justify-center rounded-full text-sm font-bold text-slate-600"
                            title="Previous frame"
                        >
                            ‹
                        </button>
                        <button
                            onClick={() => {
                                setLiveMode(false);
                                setPlaying(false);
                                setFrameIdx((i) =>
                                    Math.min(frames.length - 1, i + 1),
                                );
                            }}
                            className="ios-btn w-7 h-7 flex items-center justify-center rounded-full text-sm font-bold text-slate-600"
                            title="Next frame"
                        >
                            ›
                        </button>
                        <button
                            onClick={() => {
                                zoomRef.current = Math.min(
                                    560,
                                    zoomRef.current + 30,
                                );
                                drawFrame(
                                    liveFrameRef?.current ??
                                        frames[frameIdxRef.current],
                                    !!liveFrameRef?.current,
                                );
                            }}
                            className="ios-btn w-7 h-7 flex items-center justify-center rounded-full text-sm font-bold text-slate-600"
                            title="Zoom in"
                        >
                            +
                        </button>
                        <button
                            onClick={() => {
                                zoomRef.current = Math.max(
                                    80,
                                    zoomRef.current - 30,
                                );
                                drawFrame(
                                    liveFrameRef?.current ??
                                        frames[frameIdxRef.current],
                                    !!liveFrameRef?.current,
                                );
                            }}
                            className="ios-btn w-7 h-7 flex items-center justify-center rounded-full text-sm font-bold text-slate-600"
                            title="Zoom out"
                        >
                            −
                        </button>
                    </div>
                </div>

                <p className="text-xs text-slate-400 self-start">
                    Frame {displayedFrameIdx + 1} of {frames.length}
                    {" · exercise frames only · z from DL model"}
                </p>
            </div>
        </div>
    );
}
