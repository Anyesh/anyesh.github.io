import { useState, useEffect, useRef, useMemo, useCallback } from "react";

export const meta = {
  title: "The Gradient and the Derivative",
  category: "Calculus",
  description:
    "Build the derivative from the limit of a secant, then stack two of them into a gradient. Drag a point on a curve to read its tangent, drag a point on a contour map to read the steepest-ascent vector, and watch the directional derivative peak when you aim along the gradient.",
  date: "2026-06-02",
  tags: ["calculus", "derivative", "gradient", "directional-derivative"],
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
  secant: "#2a5298",
  grad: "#c0561f",
  dir: "#2e7d51",
  grid: "#efebe4",
  axis: "#cfc7ba",
};

const EASE = "cubic-bezier(0.23, 1, 0.32, 1)";
const TAU = 2 * Math.PI;
const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));

// Each 1D function carries its closed-form derivative so the tangent line is
// drawn from exact slope, not a numerical estimate; the secant uses f directly.
const FUNCS = {
  cubic: {
    label: "x cubed minus x",
    expr: "f(x) = x^3 - x",
    f: (x) => x * x * x - x,
    df: (x) => 3 * x * x - 1,
    xmin: -2,
    xmax: 2,
    x0: 0.7,
  },
  sine: {
    label: "sine",
    expr: "f(x) = sin x",
    f: (x) => Math.sin(x),
    df: (x) => Math.cos(x),
    xmin: -TAU / 2,
    xmax: TAU / 2,
    x0: 0.9,
  },
  gauss: {
    label: "gaussian bump",
    expr: "f(x) = exp(-x^2)",
    f: (x) => Math.exp(-x * x),
    df: (x) => -2 * x * Math.exp(-x * x),
    xmin: -2.4,
    xmax: 2.4,
    x0: 0.8,
  },
  quad: {
    label: "x squared",
    expr: "f(x) = x^2",
    f: (x) => x * x,
    df: (x) => 2 * x,
    xmin: -2,
    xmax: 2,
    x0: 0.9,
  },
};

// Each scalar field carries the analytic gradient as a pair of partials, so the
// arrow and every readout come from exact closed forms.
const FIELDS = {
  bowl: {
    label: "Quadratic bowl",
    expr: "f(x, y) = x^2 + 2 y^2",
    blurb:
      "A single minimum at the origin. The gradient always points straight uphill, away from the center, and the level sets are nested ellipses it crosses at right angles.",
    f: (x, y) => x * x + 2 * y * y,
    grad: (x, y) => [2 * x, 4 * y],
    xmin: -2,
    xmax: 2,
    ymin: -2,
    ymax: 2,
    x0: -1.1,
    y0: 0.7,
    levels: [0.2, 0.8, 1.8, 3.2, 5, 7.2],
  },
  saddle: {
    label: "Saddle",
    expr: "f(x, y) = x^2 - y^2",
    blurb:
      "Up along one axis, down along the other. At the origin the gradient vanishes; everywhere else it still points toward the locally steepest climb, which can be sideways out of the pass.",
    f: (x, y) => x * x - y * y,
    grad: (x, y) => [2 * x, -2 * y],
    xmin: -2,
    xmax: 2,
    ymin: -2,
    ymax: 2,
    x0: 0.9,
    y0: 0.8,
    levels: [-3, -1.4, -0.4, 0.4, 1.4, 3],
  },
  bumps: {
    label: "Two bumps",
    expr: "f(x, y) = exp(-((x+1)^2 + y^2)) + 0.8 exp(-((x-1)^2 + y^2))",
    blurb:
      "Two gaussian hills of unequal height. Between them sits a low ridge; the gradient leans toward whichever peak pulls harder where you stand.",
    f: (x, y) =>
      Math.exp(-((x + 1) * (x + 1) + y * y)) +
      0.8 * Math.exp(-((x - 1) * (x - 1) + y * y)),
    grad: (x, y) => {
      const a = Math.exp(-((x + 1) * (x + 1) + y * y));
      const b = 0.8 * Math.exp(-((x - 1) * (x - 1) + y * y));
      const gx = a * -2 * (x + 1) + b * -2 * (x - 1);
      const gy = a * -2 * y + b * -2 * y;
      return [gx, gy];
    },
    xmin: -2.6,
    xmax: 2.6,
    ymin: -2,
    ymax: 2,
    x0: 0.2,
    y0: 0.55,
    levels: [0.04, 0.1, 0.2, 0.35, 0.55, 0.78],
  },
};

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

function Card({ children, style }) {
  return (
    <div
      style={{
        background: C.card,
        border: `1px solid ${C.border}`,
        borderRadius: 14,
        padding: 18,
        ...style,
      }}
    >
      {children}
    </div>
  );
}

function Stat({ label, value, tone }) {
  const color = tone === "grad" ? C.grad : tone === "dir" ? C.dir : tone === "secant" ? C.secant : C.ink;
  const bg = tone ? "transparent" : C.bg;
  return (
    <div
      style={{
        background: bg,
        borderRadius: 9,
        padding: "8px 11px",
        border: `1px solid ${tone ? color + "33" : C.border}`,
      }}
    >
      <div
        style={{
          fontSize: 10,
          letterSpacing: "0.05em",
          textTransform: "uppercase",
          color: tone ? color : C.muted,
          fontWeight: 700,
          marginBottom: 3,
        }}
      >
        {label}
      </div>
      <div
        style={{
          fontFamily: "'IBM Plex Mono', ui-monospace, SFMono-Regular, Menlo, monospace",
          fontSize: 16,
          fontWeight: 700,
          color: tone ? color : C.ink,
          lineHeight: 1.05,
          fontVariantNumeric: "tabular-nums",
        }}
      >
        {value}
      </div>
    </div>
  );
}

function pickBtn(active, onClick, children, key) {
  return (
    <button
      key={key}
      type="button"
      onClick={onClick}
      aria-pressed={active}
      style={{
        fontFamily: "inherit",
        fontSize: 12,
        padding: "5px 12px",
        borderRadius: 8,
        cursor: "pointer",
        border: `1px solid ${active ? C.accent : C.border}`,
        background: active ? C.accentSoft : "transparent",
        color: active ? C.accent : C.muted,
        fontWeight: active ? 700 : 500,
        transition: `background 150ms ${EASE}, border-color 150ms ${EASE}, color 150ms ${EASE}`,
      }}
    >
      {children}
    </button>
  );
}

const PLOT1 = 460;
const PAD = 34;

function CurvePlot({ func, x0, h, onDrag, reduce }) {
  const ref = useRef(null);
  const dragging = useRef(false);

  const span = func.xmax - func.xmin;
  const samples = [];
  for (let i = 0; i <= 240; i++) {
    const x = func.xmin + (span * i) / 240;
    samples.push([x, func.f(x)]);
  }
  let ymin = Infinity;
  let ymax = -Infinity;
  for (const [, y] of samples) {
    if (y < ymin) ymin = y;
    if (y > ymax) ymax = y;
  }
  const pad = (ymax - ymin) * 0.18 || 1;
  ymin -= pad;
  ymax += pad;
  const yspan = ymax - ymin;

  const sx = (x) => PAD + ((x - func.xmin) / span) * (PLOT1 - 2 * PAD);
  const sy = (y) => PLOT1 - PAD - ((y - ymin) / yspan) * (PLOT1 - 2 * PAD);
  const inv = (px) => clamp(func.xmin + ((px - PAD) / (PLOT1 - 2 * PAD)) * span, func.xmin, func.xmax);

  const curveD = samples
    .map(([x, y], i) => `${i === 0 ? "M" : "L"}${sx(x).toFixed(2)},${sy(y).toFixed(2)}`)
    .join("");

  const y0 = func.f(x0);
  const slope = func.df(x0);
  const x1 = clamp(x0 + h, func.xmin, func.xmax);
  const y1 = func.f(x1);
  const secSlope = x1 !== x0 ? (y1 - y0) / (x1 - x0) : slope;

  const lineAt = (m, b) => {
    const yL = m * func.xmin + b;
    const yR = m * func.xmax + b;
    return `M${sx(func.xmin).toFixed(2)},${sy(yL).toFixed(2)}L${sx(func.xmax).toFixed(2)},${sy(yR).toFixed(2)}`;
  };
  const tangentD = lineAt(slope, y0 - slope * x0);
  const secantD = lineAt(secSlope, y0 - secSlope * x0);

  const zeroInRange = ymin <= 0 && ymax >= 0;

  const onPointer = useCallback(
    (e) => {
      const rect = ref.current.getBoundingClientRect();
      const px = ((e.clientX - rect.left) / rect.width) * PLOT1;
      onDrag(inv(px));
    },
    [onDrag, inv]
  );

  return (
    <svg
      ref={ref}
      viewBox={`0 0 ${PLOT1} ${PLOT1}`}
      width="100%"
      height="100%"
      role="img"
      aria-label={`Plot of ${func.expr} with a draggable point at x = ${x0.toFixed(2)}, its tangent line, and a secant line spanning h = ${h.toFixed(3)}.`}
      style={{
        display: "block",
        background: C.bg,
        borderRadius: 12,
        border: `1px solid ${C.border}`,
        cursor: dragging.current ? "grabbing" : "grab",
        touchAction: "none",
      }}
      onPointerDown={(e) => {
        e.currentTarget.setPointerCapture(e.pointerId);
        dragging.current = true;
        onPointer(e);
      }}
      onPointerMove={(e) => {
        if (dragging.current) onPointer(e);
      }}
      onPointerUp={(e) => {
        dragging.current = false;
        e.currentTarget.releasePointerCapture(e.pointerId);
      }}
    >
      {zeroInRange && (
        <line x1={PAD} y1={sy(0)} x2={PLOT1 - PAD} y2={sy(0)} stroke={C.axis} strokeWidth={1} />
      )}
      {func.xmin <= 0 && func.xmax >= 0 && (
        <line x1={sx(0)} y1={PAD} x2={sx(0)} y2={PLOT1 - PAD} stroke={C.axis} strokeWidth={1} />
      )}
      <path d={curveD} fill="none" stroke={C.ink} strokeWidth={2.2} strokeLinejoin="round" />
      <path d={secantD} fill="none" stroke={C.secant} strokeWidth={1.6} strokeDasharray="5 4" opacity={0.9} />
      <path
        d={tangentD}
        fill="none"
        stroke={C.grad}
        strokeWidth={2.2}
        style={reduce ? undefined : { transition: `all 120ms ${EASE}` }}
      />
      <line x1={sx(x1)} y1={sy(y0)} x2={sx(x1)} y2={sy(y1)} stroke={C.secant} strokeWidth={1} strokeDasharray="2 3" opacity={0.7} />
      <line x1={sx(x0)} y1={sy(y0)} x2={sx(x1)} y2={sy(y0)} stroke={C.secant} strokeWidth={1} strokeDasharray="2 3" opacity={0.7} />
      <circle cx={sx(x1)} cy={sy(y1)} r={4} fill={C.card} stroke={C.secant} strokeWidth={1.8} />
      <circle cx={sx(x0)} cy={sy(y0)} r={6.5} fill={C.grad} stroke={C.card} strokeWidth={2} />
    </svg>
  );
}

// Heatmap of the scalar field, warm where the value is high so the gradient's
// uphill direction matches the eye's reading of bright toward dark.
function fieldColor(t) {
  const stops = [
    [42, 60, 78],
    [46, 125, 81],
    [228, 201, 130],
    [192, 86, 31],
    [120, 40, 14],
  ];
  const u = clamp(t, 0, 1) * (stops.length - 1);
  const i = Math.floor(u);
  const f = u - i;
  const a = stops[i];
  const b = stops[Math.min(i + 1, stops.length - 1)];
  return [
    Math.round(a[0] + (b[0] - a[0]) * f),
    Math.round(a[1] + (b[1] - a[1]) * f),
    Math.round(a[2] + (b[2] - a[2]) * f),
  ];
}

function FieldHeatmap({ field, reduce }) {
  const ref = useRef(null);
  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const res = 130;
    canvas.width = res;
    canvas.height = res;
    const ctx = canvas.getContext("2d");
    const img = ctx.createImageData(res, res);
    let lo = Infinity;
    let hi = -Infinity;
    const vals = new Float64Array(res * res);
    for (let j = 0; j < res; j++) {
      for (let i = 0; i < res; i++) {
        const x = field.xmin + (field.xmax - field.xmin) * (i / (res - 1));
        const y = field.ymax - (field.ymax - field.ymin) * (j / (res - 1));
        const v = field.f(x, y);
        vals[j * res + i] = v;
        if (v < lo) lo = v;
        if (v > hi) hi = v;
      }
    }
    const sp = hi - lo || 1;
    for (let p = 0; p < vals.length; p++) {
      const [r, g, b] = fieldColor((vals[p] - lo) / sp);
      const o = p * 4;
      img.data[o] = r;
      img.data[o + 1] = g;
      img.data[o + 2] = b;
      img.data[o + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
  }, [field]);
  return (
    <canvas
      ref={ref}
      aria-hidden="true"
      style={{
        position: "absolute",
        inset: 0,
        width: "100%",
        height: "100%",
        imageRendering: reduce ? "pixelated" : "auto",
        borderRadius: 12,
      }}
    />
  );
}

function contourPaths(field, levels, size, res = 76) {
  const grid = [];
  for (let j = 0; j <= res; j++) {
    const row = [];
    for (let i = 0; i <= res; i++) {
      const x = field.xmin + (field.xmax - field.xmin) * (i / res);
      const y = field.ymax - (field.ymax - field.ymin) * (j / res);
      row.push(field.f(x, y));
    }
    grid.push(row);
  }
  const toPx = (i, j) => [(i / res) * size, (j / res) * size];
  const lerp = (a, b, va, vb, lv) => (vb - va === 0 ? a : a + ((lv - va) * (b - a)) / (vb - va));
  const paths = [];
  for (const level of levels) {
    const segs = [];
    for (let j = 0; j < res; j++) {
      for (let i = 0; i < res; i++) {
        const tl = grid[j][i];
        const tr = grid[j][i + 1];
        const br = grid[j + 1][i + 1];
        const bl = grid[j + 1][i];
        let idx = 0;
        if (tl > level) idx |= 8;
        if (tr > level) idx |= 4;
        if (br > level) idx |= 2;
        if (bl > level) idx |= 1;
        if (idx === 0 || idx === 15) continue;
        const top = () => [lerp(i, i + 1, tl, tr, level), j];
        const right = () => [i + 1, lerp(j, j + 1, tr, br, level)];
        const bottom = () => [lerp(i, i + 1, bl, br, level), j + 1];
        const left = () => [i, lerp(j, j + 1, tl, bl, level)];
        const edges = {
          1: [left, bottom],
          2: [bottom, right],
          3: [left, right],
          4: [top, right],
          5: [top, left],
          6: [top, bottom],
          7: [top, left],
          8: [top, left],
          9: [top, bottom],
          10: [top, right],
          11: [top, right],
          12: [left, right],
          13: [bottom, right],
          14: [left, bottom],
        };
        const pair = edges[idx];
        if (!pair) continue;
        const [pa, pb] = pair;
        const [ai, aj] = pa();
        const [bi, bj] = pb();
        const [ax, ay] = toPx(ai, aj);
        const [bx, by] = toPx(bi, bj);
        segs.push(`M${ax.toFixed(1)},${ay.toFixed(1)}L${bx.toFixed(1)},${by.toFixed(1)}`);
      }
    }
    paths.push(segs.join(""));
  }
  return paths;
}

const PLOT2 = 440;

function FieldPlot({ field, pt, angle, onDrag, reduce }) {
  const ref = useRef(null);
  const dragging = useRef(false);

  const contours = useMemo(() => contourPaths(field, field.levels, PLOT2), [field]);

  const toPx = (x, y) => [
    ((x - field.xmin) / (field.xmax - field.xmin)) * PLOT2,
    ((field.ymax - y) / (field.ymax - field.ymin)) * PLOT2,
  ];
  const fromPx = (px, py) => [
    clamp(field.xmin + (px / PLOT2) * (field.xmax - field.xmin), field.xmin, field.xmax),
    clamp(field.ymax - (py / PLOT2) * (field.ymax - field.ymin), field.ymin, field.ymax),
  ];

  const [gx, gy] = field.grad(pt[0], pt[1]);
  const gmag = Math.hypot(gx, gy);
  const [px, py] = toPx(pt[0], pt[1]);

  // Pixel space flips y, so a world-space vector (vx, vy) draws as (vx, -vy);
  // scale gradient and direction arrows into a readable on-screen length.
  const worldToPxVec = (vx, vy, len) => {
    const m = Math.hypot(vx, vy) || 1;
    return [(vx / m) * len, (-vy / m) * len];
  };

  const gradLen = gmag > 1e-6 ? clamp(gmag * 26, 14, 96) : 0;
  const [gdx, gdy] = worldToPxVec(gx, gy, gradLen);

  const ux = Math.cos(angle);
  const uy = Math.sin(angle);
  const [ddx, ddy] = worldToPxVec(ux, uy, 64);

  // Tangent to the level set is perpendicular to the gradient; draw it both ways
  // through the point to show the gradient crosses the contour at a right angle.
  let tangD = "";
  if (gmag > 1e-6) {
    const [tx, ty] = worldToPxVec(-gy, gx, 52);
    tangD = `M${(px - tx).toFixed(1)},${(py - ty).toFixed(1)}L${(px + tx).toFixed(1)},${(py + ty).toFixed(1)}`;
  }

  const onPointer = useCallback(
    (e) => {
      const rect = ref.current.getBoundingClientRect();
      const lx = ((e.clientX - rect.left) / rect.width) * PLOT2;
      const ly = ((e.clientY - rect.top) / rect.height) * PLOT2;
      onDrag(fromPx(lx, ly));
    },
    [onDrag]
  );

  return (
    <div
      style={{
        position: "relative",
        width: "100%",
        maxWidth: PLOT2,
        margin: "0 auto",
        aspectRatio: "1 / 1",
        borderRadius: 12,
        overflow: "hidden",
        border: `1px solid ${C.border}`,
        cursor: "crosshair",
        touchAction: "none",
      }}
    >
      <FieldHeatmap field={field} reduce={reduce} />
      <svg
        ref={ref}
        viewBox={`0 0 ${PLOT2} ${PLOT2}`}
        width="100%"
        height="100%"
        role="img"
        aria-label={`Contour map of ${field.expr}. Drag to move the evaluation point, now at (${pt[0].toFixed(2)}, ${pt[1].toFixed(2)}). The terracotta arrow is the gradient, the green arrow is your chosen direction.`}
        style={{ position: "absolute", inset: 0, display: "block" }}
        onPointerDown={(e) => {
          e.currentTarget.setPointerCapture(e.pointerId);
          dragging.current = true;
          onPointer(e);
        }}
        onPointerMove={(e) => {
          if (dragging.current) onPointer(e);
        }}
        onPointerUp={(e) => {
          dragging.current = false;
          e.currentTarget.releasePointerCapture(e.pointerId);
        }}
      >
        <defs>
          <marker id="grad-head" markerWidth="7" markerHeight="7" refX="5.4" refY="3" orient="auto">
            <path d="M0,0 L6,3 L0,6 Z" fill={C.grad} />
          </marker>
          <marker id="dir-head" markerWidth="7" markerHeight="7" refX="5.4" refY="3" orient="auto">
            <path d="M0,0 L6,3 L0,6 Z" fill={C.dir} />
          </marker>
        </defs>
        {contours.map((d, i) => (
          <path key={i} d={d} fill="none" stroke="#ffffff" strokeOpacity={0.34} strokeWidth={1} />
        ))}
        {tangD && <path d={tangD} fill="none" stroke="#ffffff" strokeWidth={1.6} strokeDasharray="4 4" strokeOpacity={0.85} />}
        <line
          x1={px}
          y1={py}
          x2={px + ddx}
          y2={py + ddy}
          stroke={C.dir}
          strokeWidth={2.6}
          markerEnd="url(#dir-head)"
          style={reduce ? undefined : { transition: `all 90ms ${EASE}` }}
        />
        {gradLen > 0 && (
          <line
            x1={px}
            y1={py}
            x2={px + gdx}
            y2={py + gdy}
            stroke={C.grad}
            strokeWidth={3.2}
            markerEnd="url(#grad-head)"
            style={reduce ? undefined : { transition: `all 90ms ${EASE}` }}
          />
        )}
        <circle cx={px} cy={py} r={6.5} fill={C.card} stroke={C.grad} strokeWidth={2.4} />
      </svg>
    </div>
  );
}

function fmt(v, d = 3) {
  if (!isFinite(v)) return "n/a";
  if (Math.abs(v) >= 1000) return v.toExponential(2);
  return v.toFixed(d);
}

export default function App() {
  const reduce = usePrefersReducedMotion();

  const [funcKey, setFuncKey] = useState("cubic");
  const func = FUNCS[funcKey];
  const [x0, setX0] = useState(func.x0);
  const [h, setH] = useState(0.6);

  const [fieldKey, setFieldKey] = useState("bowl");
  const field = FIELDS[fieldKey];
  const [pt, setPt] = useState([field.x0, field.y0]);
  const [angleDeg, setAngleDeg] = useState(40);

  const pickFunc = (key) => {
    setFuncKey(key);
    setX0(FUNCS[key].x0);
  };
  const pickField = (key) => {
    setFieldKey(key);
    setPt([FIELDS[key].x0, FIELDS[key].y0]);
  };

  const slope = func.df(x0);
  const y0 = func.f(x0);
  const x1 = clamp(x0 + h, func.xmin, func.xmax);
  const secSlope = x1 !== x0 ? (func.f(x1) - y0) / (x1 - x0) : slope;
  const secErr = Math.abs(secSlope - slope);

  const [gx, gy] = field.grad(pt[0], pt[1]);
  const gmag = Math.hypot(gx, gy);
  const gangle = (Math.atan2(gy, gx) * 180) / Math.PI;
  const angle = (angleDeg * Math.PI) / 180;
  const ux = Math.cos(angle);
  const uy = Math.sin(angle);
  const dirDeriv = gx * ux + gy * uy;
  const alignFrac = gmag > 1e-6 ? dirDeriv / gmag : 0;

  const focusCss = `
    .gd-root :focus-visible { outline: 2px solid ${C.accent}; outline-offset: 2px; border-radius: 9px; }
    .gd-root input[type=range] { accent-color: ${C.accent}; }
    @media (prefers-reduced-motion: reduce) {
      .gd-root * { transition-duration: 0.001ms !important; animation-duration: 0.001ms !important; }
    }
  `;

  return (
    <div
      className="gd-root"
      style={{
        fontFamily: "'Iowan Old Style', 'Palatino Linotype', Palatino, Georgia, serif",
        background: C.bg,
        minHeight: "100vh",
        padding: "30px 14px 56px",
        color: C.ink,
      }}
    >
      <style>{focusCss}</style>
      <div style={{ maxWidth: 720, margin: "0 auto" }}>
        <header style={{ marginBottom: 22 }}>
          <div
            style={{
              fontSize: 11,
              letterSpacing: "0.14em",
              textTransform: "uppercase",
              color: C.faint,
              marginBottom: 6,
            }}
          >
            Calculus, computed exactly
          </div>
          <h1
            style={{
              fontSize: 31,
              fontWeight: 700,
              margin: 0,
              lineHeight: 1.08,
              letterSpacing: "-0.02em",
              textWrap: "balance",
            }}
          >
            The Gradient and the Derivative
          </h1>
          <p style={{ color: C.ink, fontSize: 15, lineHeight: 1.6, margin: "12px 0 0", maxWidth: "66ch" }}>
            A derivative is the slope of a curve at a single point: the limit a secant line approaches as the
            two points it joins slide together. A gradient is just a stack of those slopes, one per input
            axis, bundled into a vector. Drag the points below and read both straight off the math.
          </p>
        </header>

        <Card style={{ marginBottom: 18 }}>
          <div style={{ fontSize: 13, fontWeight: 700, marginBottom: 4 }}>
            Part 1: the derivative as the slope of the tangent
          </div>
          <p style={{ fontSize: 13.5, color: C.muted, lineHeight: 1.55, margin: "0 0 12px", maxWidth: "64ch" }}>
            The blue secant joins the point at x and the point at x + h. Its slope is the average rate of change
            over that gap. Shrink h and the secant pivots into the terracotta tangent: that limiting slope is
            f prime of x, the instantaneous rate of change.
          </p>

          <div style={{ display: "flex", gap: 6, flexWrap: "wrap", marginBottom: 12 }}>
            <span style={{ fontSize: 12, color: C.muted, alignSelf: "center", marginRight: 4 }}>Function:</span>
            {Object.entries(FUNCS).map(([k, fn]) => pickBtn(funcKey === k, () => pickFunc(k), fn.label, k))}
          </div>

          <div style={{ width: "100%", maxWidth: PLOT1, margin: "0 auto", aspectRatio: "1 / 1" }}>
            <CurvePlot func={func} x0={x0} h={h} onDrag={setX0} reduce={reduce} />
          </div>

          <div style={{ marginTop: 14 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap" }}>
              <label htmlFor="gd-x0" style={{ fontSize: 13, color: C.muted, minWidth: 96 }}>
                point x
              </label>
              <input
                id="gd-x0"
                type="range"
                min={func.xmin}
                max={func.xmax}
                step={(func.xmax - func.xmin) / 400}
                value={x0}
                onChange={(e) => setX0(+e.target.value)}
                style={{ flex: 1, minWidth: 140 }}
                aria-label="evaluation point x"
              />
              <span style={{ fontFamily: "ui-monospace, monospace", fontSize: 13, fontWeight: 700, color: C.grad, minWidth: 54, textAlign: "right" }}>
                {fmt(x0, 2)}
              </span>
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap", marginTop: 8 }}>
              <label htmlFor="gd-h" style={{ fontSize: 13, color: C.muted, minWidth: 96 }}>
                step h
              </label>
              <input
                id="gd-h"
                type="range"
                min={0.001}
                max={1.2}
                step={0.001}
                value={h}
                onChange={(e) => setH(+e.target.value)}
                style={{ flex: 1, minWidth: 140 }}
                aria-label="secant step h"
              />
              <span style={{ fontFamily: "ui-monospace, monospace", fontSize: 13, fontWeight: 700, color: C.secant, minWidth: 54, textAlign: "right" }}>
                {fmt(h, 3)}
              </span>
            </div>
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(120px, 1fr))", gap: 8, marginTop: 14 }}>
            <Stat label="f(x)" value={fmt(y0)} />
            <Stat label="secant slope" value={fmt(secSlope)} tone="secant" />
            <Stat label="f prime (x)" value={fmt(slope)} tone="grad" />
            <Stat label="gap to limit" value={fmt(secErr)} />
          </div>
          <p style={{ fontSize: 12.5, color: C.muted, margin: "12px 0 0", lineHeight: 1.55 }}>
            As you drag h toward zero the secant slope chases f prime of x and the gap collapses. That limit is
            the definition of the derivative; the closed form f prime is what the tangent is drawn from.
          </p>
        </Card>

        <Card style={{ marginBottom: 18 }}>
          <div style={{ fontSize: 13, fontWeight: 700, marginBottom: 4 }}>
            Part 2: the gradient as the steepest-ascent vector
          </div>
          <p style={{ fontSize: 13.5, color: C.muted, lineHeight: 1.55, margin: "0 0 12px", maxWidth: "64ch" }}>
            Now the input is a point in the plane and the output is a height. The gradient grad f is the vector
            of partial derivatives, the slope along x holding y fixed and the slope along y holding x fixed.
            It points in the direction of steepest increase and stands at a right angle to the contour through
            the point.
          </p>

          <div style={{ display: "flex", gap: 6, flexWrap: "wrap", marginBottom: 6 }}>
            <span style={{ fontSize: 12, color: C.muted, alignSelf: "center", marginRight: 4 }}>Field:</span>
            {Object.entries(FIELDS).map(([k, fl]) => pickBtn(fieldKey === k, () => pickField(k), fl.label, k))}
          </div>
          <p style={{ fontSize: 12.5, color: C.ink, lineHeight: 1.5, margin: "0 0 12px", maxWidth: "64ch" }}>
            {field.blurb}
          </p>

          <FieldPlot field={field} pt={pt} angle={angle} onDrag={setPt} reduce={reduce} />

          <div style={{ display: "flex", gap: 16, justifyContent: "center", flexWrap: "wrap", marginTop: 10, fontSize: 12 }}>
            <span style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
              <span style={{ width: 16, height: 3, background: C.grad, borderRadius: 2 }} />
              <span style={{ color: C.ink, fontWeight: 600 }}>gradient</span>
            </span>
            <span style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
              <span style={{ width: 16, height: 3, background: C.dir, borderRadius: 2 }} />
              <span style={{ color: C.ink, fontWeight: 600 }}>your direction u</span>
            </span>
            <span style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
              <span style={{ width: 16, height: 0, borderTop: `2px dashed ${C.faint}`, marginTop: 2 }} />
              <span style={{ color: C.ink, fontWeight: 600 }}>level set tangent</span>
            </span>
          </div>

          <div style={{ marginTop: 14 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap" }}>
              <label htmlFor="gd-angle" style={{ fontSize: 13, color: C.muted, minWidth: 96 }}>
                direction u
              </label>
              <input
                id="gd-angle"
                type="range"
                min={0}
                max={360}
                step={1}
                value={angleDeg}
                onChange={(e) => setAngleDeg(+e.target.value)}
                style={{ flex: 1, minWidth: 140 }}
                aria-label="direction angle in degrees for the directional derivative"
              />
              <span style={{ fontFamily: "ui-monospace, monospace", fontSize: 13, fontWeight: 700, color: C.dir, minWidth: 54, textAlign: "right" }}>
                {angleDeg}deg
              </span>
            </div>
            <button
              type="button"
              onClick={() => setAngleDeg(Math.round(((gangle % 360) + 360) % 360))}
              style={{
                marginTop: 10,
                fontFamily: "inherit",
                fontSize: 12.5,
                fontWeight: 600,
                padding: "7px 13px",
                borderRadius: 9,
                border: `1px solid ${C.grad}`,
                background: C.accentSoft,
                color: C.grad,
                cursor: "pointer",
                transition: `transform 140ms ${EASE}`,
              }}
              onMouseDown={(e) => (e.currentTarget.style.transform = "scale(0.97)")}
              onMouseUp={(e) => (e.currentTarget.style.transform = "scale(1)")}
              onMouseLeave={(e) => (e.currentTarget.style.transform = "scale(1)")}
              aria-label="Aim the direction along the gradient"
            >
              Aim u along the gradient
            </button>
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(120px, 1fr))", gap: 8, marginTop: 14 }}>
            <Stat label="f(x, y)" value={fmt(field.f(pt[0], pt[1]))} />
            <Stat label="df / dx" value={fmt(gx)} />
            <Stat label="df / dy" value={fmt(gy)} />
            <Stat label="|grad f|" value={fmt(gmag)} tone="grad" />
            <Stat label="grad angle" value={`${fmt(((gangle % 360) + 360) % 360, 0)}deg`} tone="grad" />
            <Stat label="dir. deriv. grad . u" value={fmt(dirDeriv)} tone="dir" />
          </div>

          <div
            style={{
              marginTop: 12,
              height: 8,
              borderRadius: 5,
              background: C.grid,
              position: "relative",
              overflow: "hidden",
            }}
            aria-hidden="true"
          >
            <div
              style={{
                position: "absolute",
                left: "50%",
                top: 0,
                bottom: 0,
                width: `${Math.abs(alignFrac) * 50}%`,
                transform: alignFrac >= 0 ? "translateX(0)" : "translateX(-100%)",
                background: alignFrac >= 0 ? C.dir : "#b23b5e",
                transition: reduce ? "none" : `width 90ms ${EASE}`,
              }}
            />
          </div>
          <p style={{ fontSize: 12.5, color: C.muted, margin: "8px 0 0", lineHeight: 1.55 }}>
            The directional derivative is grad f dotted with the unit direction u, equal to |grad f| times the
            cosine of the angle between them. It peaks at +|grad f| when u points along the gradient, hits zero
            when u runs along the contour (perpendicular to the gradient), and bottoms at minus|grad f| pointing
            straight downhill. The bar above tracks that cosine.
          </p>
        </Card>

        <Card style={{ background: C.accentSoft, borderColor: `${C.accent}33` }}>
          <div style={{ fontSize: 13, fontWeight: 700, marginBottom: 10, color: "#5d4226" }}>
            One idea, stacked
          </div>
          <p style={{ fontSize: 14, lineHeight: 1.65, color: "#5d4226", margin: "0 0 10px", maxWidth: "66ch" }}>
            A partial derivative is the Part 1 picture run on one axis: freeze every input but one, and you are
            back to a single curve whose slope is a plain derivative. The gradient collects those per-axis
            slopes into a vector. Because of that construction the gradient inherits two facts at once: its
            length is the steepest slope available at the point, and its direction is the way to climb fastest.
          </p>
          <p style={{ fontSize: 14, lineHeight: 1.65, color: "#5d4226", margin: 0, maxWidth: "66ch" }}>
            This is the engine under gradient-based learning. A model has thousands or millions of inputs
            instead of two, but the move is identical: compute the gradient of a loss, then step against it to
            go downhill. Every optimizer is a different rule for how far and how smoothly to take that step.
          </p>
        </Card>

        <p style={{ textAlign: "center", fontSize: 11.5, color: C.faint, marginTop: 22, lineHeight: 1.6 }}>
          Tangents and gradients use closed-form derivatives; the secant and the contour map evaluate the
          functions directly. Drag any point to recompute.
        </p>
      </div>
    </div>
  );
}
