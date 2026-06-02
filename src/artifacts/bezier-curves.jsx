import { useState, useEffect, useRef, useMemo, useCallback } from "react";

export const meta = {
  title: "Bezier Curves and de Casteljau",
  category: "Graphics",
  description:
    "Drag four dots and a smooth curve bends to follow. The trick is repeated averaging: split each line at the same fraction, then split the splits, until one point is left to trace.",
  date: "2026-01-16",
  tags: ["bezier", "de-casteljau", "splines", "geometry"],
};

const C = {
  bg: "#f7f5f2",
  card: "#ffffff",
  border: "#e6e1d9",
  ink: "#1b1916",
  muted: "#6f675e",
  faint: "#9b938a",
  accent: "#c0561f",
  accentSoft: "#f6ece5",
  grid: "#efebe4",
  hull: "rgba(155, 147, 138, 0.16)",
  control: "#b9a98f",
  curve: "#c0561f",
  point: "#2f6f9e",
  level1: "#3f7d52",
  level2: "#9a6b1f",
  level3: "#9e4b7e",
};

const EASE = "cubic-bezier(0.23, 1, 0.32, 1)";
const MONO = "'IBM Plex Mono', ui-monospace, SFMono-Regular, Menlo, monospace";
const LEVEL_COLORS = [C.level1, C.level2, C.level3];

function lerp(a, b, t) {
  return { x: a.x + (b.x - a.x) * t, y: a.y + (b.y - a.y) * t };
}

// de Casteljau: each pass replaces n points with the n-1 midpoints found by
// interpolating consecutive pairs at the same t. Repeating until one point
// remains gives the curve point at t, and every intermediate level is the
// scaffold the pen tool implicitly walks.
function deCasteljau(points, t) {
  const levels = [points];
  let current = points;
  while (current.length > 1) {
    const next = [];
    for (let i = 0; i < current.length - 1; i++) {
      next.push(lerp(current[i], current[i + 1], t));
    }
    levels.push(next);
    current = next;
  }
  return { point: current[0], levels };
}

function binom(n, k) {
  let result = 1;
  for (let i = 0; i < k; i++) {
    result = (result * (n - i)) / (i + 1);
  }
  return result;
}

// Bernstein form: B(t) = sum_i C(n,i) (1-t)^(n-i) t^i P_i. Same curve as the
// geometric construction, expressed as a weighted blend of the control points.
function bernstein(points, t) {
  const n = points.length - 1;
  let x = 0;
  let y = 0;
  for (let i = 0; i <= n; i++) {
    const w = binom(n, i) * Math.pow(1 - t, n - i) * Math.pow(t, i);
    x += w * points[i].x;
    y += w * points[i].y;
  }
  return { x, y };
}

function bernsteinWeights(n, t) {
  const w = [];
  for (let i = 0; i <= n; i++) {
    w.push(binom(n, i) * Math.pow(1 - t, n - i) * Math.pow(t, i));
  }
  return w;
}

function sampleCurve(points, count) {
  const out = [];
  for (let i = 0; i <= count; i++) {
    out.push(deCasteljau(points, i / count).point);
  }
  return out;
}

function usePrefersReducedMotion() {
  const [reduce, setReduce] = useState(false);
  useEffect(() => {
    if (typeof window === "undefined" || !window.matchMedia) return;
    const mq = window.matchMedia("(prefers-reduced-motion: reduce)");
    const apply = () => setReduce(mq.matches);
    apply();
    mq.addEventListener("change", apply);
    return () => mq.removeEventListener("change", apply);
  }, []);
  return reduce;
}

const VIEW = 360;

const QUADRATIC = [
  { x: 60, y: 280 },
  { x: 180, y: 70 },
  { x: 300, y: 280 },
];

const CUBIC = [
  { x: 56, y: 270 },
  { x: 120, y: 70 },
  { x: 250, y: 90 },
  { x: 308, y: 268 },
];

function clampToView(p) {
  const pad = 14;
  return {
    x: Math.max(pad, Math.min(VIEW - pad, p.x)),
    y: Math.max(pad, Math.min(VIEW - pad, p.y)),
  };
}

function convexHull(points) {
  const pts = points
    .map((p) => ({ ...p }))
    .sort((a, b) => (a.x === b.x ? a.y - b.y : a.x - b.x));
  if (pts.length < 3) return pts;
  const cross = (o, a, b) => (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
  const lower = [];
  for (const p of pts) {
    while (lower.length >= 2 && cross(lower[lower.length - 2], lower[lower.length - 1], p) <= 0) lower.pop();
    lower.push(p);
  }
  const upper = [];
  for (let i = pts.length - 1; i >= 0; i--) {
    const p = pts[i];
    while (upper.length >= 2 && cross(upper[upper.length - 2], upper[upper.length - 1], p) <= 0) upper.pop();
    upper.push(p);
  }
  upper.pop();
  lower.pop();
  return lower.concat(upper);
}

function toPath(samples) {
  return samples.map((p, i) => `${i === 0 ? "M" : "L"} ${p.x.toFixed(2)} ${p.y.toFixed(2)}`).join(" ");
}

function Canvas({ points, t, showScaffold, dragIndex, onDragStart, reduce }) {
  const ref = useRef(null);

  const curve = useMemo(() => sampleCurve(points, 90), [points]);
  const hull = useMemo(() => convexHull(points), [points]);
  const { point: cursor, levels } = useMemo(() => deCasteljau(points, t), [points, t]);

  const toLocal = useCallback((e) => {
    const rect = ref.current.getBoundingClientRect();
    const sx = VIEW / rect.width;
    const sy = VIEW / rect.height;
    return { x: (e.clientX - rect.left) * sx, y: (e.clientY - rect.top) * sy };
  }, []);

  const handleDown = useCallback(
    (e) => {
      const local = toLocal(e);
      let best = -1;
      let bestDist = 22;
      points.forEach((p, i) => {
        const d = Math.hypot(p.x - local.x, p.y - local.y);
        if (d < bestDist) {
          bestDist = d;
          best = i;
        }
      });
      if (best >= 0) {
        e.currentTarget.setPointerCapture(e.pointerId);
        onDragStart(best);
      }
    },
    [points, toLocal, onDragStart]
  );

  const handleMove = useCallback(
    (e) => {
      if (dragIndex == null) return;
      onDragStart(dragIndex, clampToView(toLocal(e)));
    },
    [dragIndex, toLocal, onDragStart]
  );

  const handleUp = useCallback(
    (e) => {
      if (dragIndex != null) {
        try {
          e.currentTarget.releasePointerCapture(e.pointerId);
        } catch {
          // pointer may already be freed when a drag ends off-canvas
        }
        onDragStart(null);
      }
    },
    [dragIndex, onDragStart]
  );

  const gridLines = [];
  for (let g = 0; g <= VIEW; g += VIEW / 6) {
    gridLines.push(<line key={`v${g}`} x1={g} y1={0} x2={g} y2={VIEW} stroke={C.grid} strokeWidth={1} />);
    gridLines.push(<line key={`h${g}`} x1={0} y1={g} x2={VIEW} y2={g} stroke={C.grid} strokeWidth={1} />);
  }

  return (
    <svg
      ref={ref}
      viewBox={`0 0 ${VIEW} ${VIEW}`}
      onPointerDown={handleDown}
      onPointerMove={handleMove}
      onPointerUp={handleUp}
      onPointerCancel={handleUp}
      style={{
        width: VIEW,
        height: VIEW,
        maxWidth: "100%",
        borderRadius: 12,
        background: C.bg,
        border: `1px solid ${C.border}`,
        display: "block",
        touchAction: "none",
        cursor: dragIndex != null ? "grabbing" : "crosshair",
      }}
      role="img"
      aria-label="A Bezier curve over its control points. Drag any control point to bend the curve; the de Casteljau scaffold interpolates each control segment at the current t to find the curve point."
    >
      {gridLines}

      {showScaffold && hull.length >= 3 && (
        <polygon
          points={hull.map((p) => `${p.x},${p.y}`).join(" ")}
          fill={C.hull}
          stroke={C.control}
          strokeWidth={1}
          strokeDasharray="2 5"
        />
      )}

      <polyline
        points={points.map((p) => `${p.x},${p.y}`).join(" ")}
        fill="none"
        stroke={C.control}
        strokeWidth={1.6}
        strokeDasharray="5 5"
      />

      <path d={toPath(curve)} fill="none" stroke={C.curve} strokeWidth={2.8} strokeLinecap="round" strokeLinejoin="round" />

      {showScaffold &&
        levels.slice(1, -1).map((level, li) => {
          const color = LEVEL_COLORS[li % LEVEL_COLORS.length];
          return (
            <g key={`lvl${li}`}>
              {level.length > 1 && (
                <polyline
                  points={level.map((p) => `${p.x},${p.y}`).join(" ")}
                  fill="none"
                  stroke={color}
                  strokeWidth={1.6}
                  opacity={0.85}
                />
              )}
              {level.map((p, pi) => (
                <circle key={pi} cx={p.x} cy={p.y} r={4} fill={C.card} stroke={color} strokeWidth={2} />
              ))}
            </g>
          );
        })}

      {points.map((p, i) => (
        <g key={`cp${i}`}>
          <circle
            cx={p.x}
            cy={p.y}
            r={dragIndex === i ? 9 : 7.5}
            fill={C.card}
            stroke={C.point}
            strokeWidth={2.6}
            style={{ transition: reduce ? "none" : `r 120ms ${EASE}` }}
          />
          <text
            x={p.x + 11}
            y={p.y - 10}
            fontFamily={MONO}
            fontSize={12}
            fontWeight={700}
            fill={C.point}
          >
            P{i}
          </text>
        </g>
      ))}

      <circle cx={cursor.x} cy={cursor.y} r={5.5} fill={C.curve} stroke={C.card} strokeWidth={2} />
    </svg>
  );
}

function Card({ children, style }) {
  return (
    <div style={{ background: C.card, border: `1px solid ${C.border}`, borderRadius: 14, padding: 18, ...style }}>
      {children}
    </div>
  );
}

function Stat({ label, value, accent }) {
  return (
    <div
      style={{
        background: accent ? C.accentSoft : C.bg,
        borderRadius: 9,
        padding: "9px 12px",
        border: `1px solid ${accent ? C.accent + "33" : C.border}`,
      }}
    >
      <div
        style={{
          fontSize: 10,
          letterSpacing: "0.06em",
          textTransform: "uppercase",
          color: accent ? C.accent : C.muted,
          fontWeight: 700,
          marginBottom: 3,
        }}
      >
        {label}
      </div>
      <div
        style={{
          fontFamily: MONO,
          fontSize: 16,
          fontWeight: 700,
          color: accent ? C.accent : C.ink,
          lineHeight: 1.15,
          fontVariantNumeric: "tabular-nums",
        }}
      >
        {value}
      </div>
    </div>
  );
}

function btnStyle(active, disabled) {
  return {
    padding: "8px 14px",
    borderRadius: 9,
    border: `1.5px solid ${active ? C.accent : C.border}`,
    background: active ? C.accent : C.card,
    color: active ? "#fff" : C.muted,
    fontSize: 13,
    fontWeight: 600,
    cursor: disabled ? "not-allowed" : "pointer",
    opacity: disabled ? 0.5 : 1,
    fontFamily: "inherit",
    transition: `transform 140ms ${EASE}, background 160ms ${EASE}, border-color 160ms ${EASE}, color 160ms ${EASE}`,
  };
}

const DEGREE_LABEL = { 2: "Quadratic (3 points)", 3: "Cubic (4 points)" };

export default function App() {
  const reduce = usePrefersReducedMotion();
  const [points, setPoints] = useState(CUBIC.map((p) => ({ ...p })));
  const [t, setT] = useState(0.42);
  const [playing, setPlaying] = useState(false);
  const [showScaffold, setShowScaffold] = useState(true);
  const [dragIndex, setDragIndex] = useState(null);

  const degree = points.length - 1;

  const raf = useRef(0);
  const tRef = useRef(t);
  const dir = useRef(1);
  useEffect(() => {
    tRef.current = t;
  }, [t]);

  const handleDrag = useCallback((index, point) => {
    if (index === null) {
      setDragIndex(null);
      return;
    }
    if (!point) {
      setDragIndex(index);
      return;
    }
    setPoints((prev) => prev.map((p, i) => (i === index ? point : p)));
  }, []);

  useEffect(() => {
    if (reduce || !playing) {
      cancelAnimationFrame(raf.current);
      return;
    }
    let prev = 0;
    const loop = (ts) => {
      if (!prev) prev = ts;
      const dt = (ts - prev) / 1000;
      prev = ts;
      let next = tRef.current + dir.current * dt * 0.32;
      if (next >= 1) {
        next = 1;
        dir.current = -1;
      } else if (next <= 0) {
        next = 0;
        dir.current = 1;
      }
      tRef.current = next;
      setT(next);
      raf.current = requestAnimationFrame(loop);
    };
    raf.current = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf.current);
  }, [playing, reduce]);

  useEffect(() => () => cancelAnimationFrame(raf.current), []);

  const setDegree = (d) => {
    setPlaying(false);
    setPoints((d === 2 ? QUADRATIC : CUBIC).map((p) => ({ ...p })));
  };

  const addPoint = () => {
    setPlaying(false);
    setPoints((prev) => {
      if (prev.length >= 6) return prev;
      const a = prev[prev.length - 2];
      const b = prev[prev.length - 1];
      const inserted = { x: (a.x + b.x) / 2 + 24, y: (a.y + b.y) / 2 - 36 };
      return [...prev.slice(0, prev.length - 1), clampToView(inserted), b];
    });
  };

  const reset = () => {
    setPlaying(false);
    setPoints(CUBIC.map((p) => ({ ...p })));
    setT(0.42);
    tRef.current = 0.42;
    dir.current = 1;
  };

  const dcPoint = useMemo(() => deCasteljau(points, t).point, [points, t]);
  const bPoint = useMemo(() => bernstein(points, t), [points, t]);
  const matchGap = Math.hypot(dcPoint.x - bPoint.x, dcPoint.y - bPoint.y);
  const weights = useMemo(() => bernsteinWeights(degree, t), [degree, t]);

  const termColor = (i) => LEVEL_COLORS[i % LEVEL_COLORS.length];

  return (
    <div
      style={{
        fontFamily: "'Iowan Old Style', 'Palatino Linotype', Palatino, Georgia, serif",
        background: C.bg,
        minHeight: "100vh",
        padding: "30px 14px 56px",
        color: C.ink,
      }}
    >
      <style>{`
        @media (prefers-reduced-motion: reduce) {
          * { transition-duration: 1ms !important; animation-duration: 1ms !important; }
        }
        .bz-btn:active { transform: scale(0.97); }
        .bz-btn:focus-visible, input[type=range]:focus-visible {
          outline: 2px solid ${C.accent}; outline-offset: 2px;
        }
        input[type=range] { accent-color: ${C.accent}; }
      `}</style>

      <div style={{ maxWidth: 720, margin: "0 auto" }}>
        <header style={{ marginBottom: 22 }}>
          <div style={{ fontSize: 11, letterSpacing: "0.14em", textTransform: "uppercase", color: C.faint, marginBottom: 6 }}>
            Graphics / Curves and splines
          </div>
          <h1 style={{ fontSize: 30, fontWeight: 700, margin: 0, lineHeight: 1.1, letterSpacing: "-0.02em", textWrap: "balance" }}>
            Bezier Curves and de Casteljau
          </h1>
          <p style={{ color: C.muted, fontSize: 15, lineHeight: 1.6, margin: "10px 0 0", maxWidth: "64ch" }}>
            A pen tool gives you a handful of points and somehow a smooth arc threads between them. There is no
            spline magic underneath: pick a fraction t, slide along every control segment by that same fraction,
            then treat those new points as control points and slide again. Keep collapsing until one point is left.
            That single point is where the curve is at t, and sweeping t from start to finish draws the whole thing.
          </p>
        </header>

        <Card style={{ marginBottom: 18 }}>
          <div style={{ display: "flex", gap: 18, flexWrap: "wrap", alignItems: "flex-start", justifyContent: "center" }}>
            <Canvas
              points={points}
              t={t}
              showScaffold={showScaffold}
              dragIndex={dragIndex}
              onDragStart={handleDrag}
              reduce={reduce}
            />

            <div style={{ flex: "1 1 248px", minWidth: 232 }}>
              <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 14 }}>
                {!reduce && (
                  <button
                    type="button"
                    className="bz-btn"
                    onClick={() => setPlaying((p) => !p)}
                    style={btnStyle(playing)}
                    aria-pressed={playing}
                  >
                    {playing ? "Pause sweep" : "Animate t"}
                  </button>
                )}
                <button
                  type="button"
                  className="bz-btn"
                  onClick={() => setShowScaffold((s) => !s)}
                  style={btnStyle(showScaffold)}
                  aria-pressed={showScaffold}
                >
                  {showScaffold ? "Scaffold on" : "Scaffold off"}
                </button>
                <button type="button" className="bz-btn" onClick={reset} style={btnStyle(false)} aria-label="Reset to the default cubic curve">
                  Reset
                </button>
              </div>

              <label htmlFor="bz-t" style={{ fontSize: 13, color: C.muted, display: "block", marginBottom: 6 }}>
                Parameter t along the curve
              </label>
              <input
                id="bz-t"
                type="range"
                min={0}
                max={1}
                step={0.001}
                value={t}
                onChange={(e) => {
                  setPlaying(false);
                  setT(+e.target.value);
                }}
                style={{ width: "100%" }}
                aria-label="parameter t from 0 to 1"
              />

              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8, marginTop: 14 }}>
                <Stat label="t" value={t.toFixed(3)} accent />
                <Stat label="degree" value={degree} />
                <Stat label="control points" value={points.length} />
                <Stat label="construction levels" value={degree} />
              </div>

              <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginTop: 14 }}>
                <button type="button" className="bz-btn" onClick={() => setDegree(2)} style={btnStyle(degree === 2)} aria-pressed={degree === 2}>
                  Quadratic
                </button>
                <button type="button" className="bz-btn" onClick={() => setDegree(3)} style={btnStyle(degree === 3)} aria-pressed={degree === 3}>
                  Cubic
                </button>
                <button
                  type="button"
                  className="bz-btn"
                  onClick={addPoint}
                  disabled={points.length >= 6}
                  style={btnStyle(false, points.length >= 6)}
                  aria-label="Add a control point to raise the degree"
                >
                  {points.length >= 6 ? "Max degree" : "Add point"}
                </button>
              </div>
            </div>
          </div>
        </Card>

        <Card style={{ marginBottom: 18 }}>
          <div style={{ fontSize: 13, fontWeight: 700, marginBottom: 4 }}>Two views of the same point</div>
          <p style={{ fontSize: 13, color: C.muted, margin: "0 0 14px", lineHeight: 1.55, maxWidth: "64ch" }}>
            The geometric construction and the algebraic Bernstein form land on the identical point. de Casteljau
            collapses the points by repeated interpolation; the Bernstein form writes the result directly as one
            blend, where each control point is weighted by a polynomial in t. The weights always sum to one, which
            is why the curve can never escape the shaded hull of the points.
          </p>

          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))", gap: 8, marginBottom: 14 }}>
            <Stat label="de Casteljau" value={`(${dcPoint.x.toFixed(1)}, ${dcPoint.y.toFixed(1)})`} accent />
            <Stat label="Bernstein form" value={`(${bPoint.x.toFixed(1)}, ${bPoint.y.toFixed(1)})`} />
            <Stat label="gap between them" value={`${matchGap.toExponential(1)} px`} />
          </div>

          <div
            style={{
              fontFamily: MONO,
              fontSize: 13.5,
              lineHeight: 1.8,
              background: C.bg,
              border: `1px solid ${C.border}`,
              borderRadius: 10,
              padding: "12px 14px",
              color: C.ink,
              overflowX: "auto",
            }}
          >
            <div style={{ color: C.muted, marginBottom: 6 }}>B(t) = sum of P&nbsp;i weighted by a Bernstein polynomial</div>
            {weights.map((w, i) => (
              <span key={i} style={{ whiteSpace: "nowrap", marginRight: 6 }}>
                {i > 0 && <span style={{ color: C.faint }}>+ </span>}
                <span style={{ color: termColor(i), fontWeight: 700 }}>{w.toFixed(3)}</span>
                <span style={{ color: C.muted }}>&middot;P{i}</span>
              </span>
            ))}
            <div style={{ color: C.faint, marginTop: 8, fontSize: 12.5 }}>
              weights sum to {weights.reduce((s, w) => s + w, 0).toFixed(3)}
            </div>
          </div>
        </Card>

        <Card style={{ background: C.accentSoft, borderColor: `${C.accent}33` }}>
          <div style={{ fontSize: 13, fontWeight: 700, marginBottom: 10 }}>What the dots actually control</div>
          <p style={{ fontSize: 14, lineHeight: 1.65, color: "#5d4226", margin: "0 0 10px", maxWidth: "66ch" }}>
            The curve passes through the two endpoints, P0 and P{degree}, and only through those. The interior
            points are pulls, not destinations: the curve leans toward them and leaves each endpoint heading
            straight at its neighbour, which is why the dashed control polygon is tangent to the curve at the ends.
            Drag an interior point far out and the arc bulges that way without ever touching it.
          </p>
          <p style={{ fontSize: 14, lineHeight: 1.65, color: "#5d4226", margin: 0, maxWidth: "66ch" }}>
            Because every step is an average of points already inside the polygon, the result stays inside their
            convex hull, the shaded region. This is exactly the curve a font outline stores and a pen tool draws:
            a cubic per segment, four points each, stitched end to end into letterforms and icons.
          </p>
        </Card>

        <p style={{ textAlign: "center", fontSize: 11.5, color: C.faint, marginTop: 22, lineHeight: 1.6 }}>
          The plotted curve samples de Casteljau at 90 values of t. The readout point matches the Bernstein
          evaluation to floating-point precision, and stays within the convex hull of the {points.length} control
          points by construction.
        </p>
      </div>
    </div>
  );
}
