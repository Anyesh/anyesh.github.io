import { useState, useEffect, useRef, useMemo, useCallback } from "react";

export const meta = {
  title: "Optimizer Race",
  category: "Machine Learning",
  description:
    "Drop four optimizers on the same loss surface from the same point and race them. Watch momentum carry through a ravine and Adam rescale each axis while plain SGD crawls or oscillates.",
  date: "2026-04-22",
  tags: ["optimization", "gradient-descent", "adam", "momentum"],
};

const C = {
  bg: "#f7f5f2",
  card: "#ffffff",
  border: "#e6e1d9",
  ink: "#1b1916",
  muted: "#857c70",
  accent: "#c0561f",
  accentSoft: "#f6ece5",
  grid: "#efebe4",
};

const EASE = "cubic-bezier(0.23, 1, 0.32, 1)";

const OPTS = [
  { key: "sgd", name: "SGD", color: "#1b1916", rule: "x -= lr * g" },
  { key: "momentum", name: "Momentum", color: "#2a5298", rule: "v = mu*v - lr*g; x += v" },
  { key: "rmsprop", name: "RMSProp", color: "#2e7d51", rule: "c = rho*c + (1-rho)*g^2; x -= lr*g/(sqrt(c)+eps)" },
  { key: "adam", name: "Adam", color: "#c0561f", rule: "m,v moments, bias-corrected; x -= lr*mhat/(sqrt(vhat)+eps)" },
];

const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));

// Each surface exposes f(x,y) and its closed-form gradient grad(x,y) -> [gx, gy],
// plus a viewport and the minimum used to report distance-to-optimum.
const SURFACES = {
  rosenbrock: {
    label: "Rosenbrock",
    blurb:
      "A curved banana valley. The floor of the valley bends, so the gradient points across the valley far more than along it. SGD zig-zags down the walls; momentum builds speed along the curve.",
    f: (x, y) => {
      const a = 1 - x;
      const b = y - x * x;
      return a * a + 20 * b * b;
    },
    grad: (x, y) => {
      const b = y - x * x;
      return [-2 * (1 - x) - 80 * x * b, 40 * b];
    },
    xmin: -2.0,
    xmax: 2.0,
    ymin: -1.0,
    ymax: 3.0,
    min: [1, 1],
    start: [-1.5, 2.4],
    lr: 0.0025,
  },
  ravine: {
    label: "Ravine",
    blurb:
      "An ill-conditioned bowl, two hundred times steeper across one axis than along the other. At the default rate, which sits right at SGD's stability edge for the steep axis, plain SGD oscillates and stalls while the others descend cleanly. One shared rate cannot serve both axes, which is exactly what per-dimension scaling fixes.",
    f: (x, y) => 0.5 * (x * x + 200 * y * y),
    grad: (x, y) => [x, 200 * y],
    xmin: -3.0,
    xmax: 3.0,
    ymin: -1.2,
    ymax: 1.2,
    min: [0, 0],
    start: [-2.6, 0.9],
    lr: 0.01,
  },
  saddle: {
    label: "Saddle",
    blurb:
      "Curves up along one axis and down along the other. The center is a stationary point where the gradient nearly vanishes, so plain SGD stalls there while the others keep enough motion to slide off and escape.",
    f: (x, y) => x * x - y * y,
    grad: (x, y) => [2 * x, -2 * y],
    xmin: -2.2,
    xmax: 2.2,
    ymin: -2.2,
    ymax: 2.2,
    min: null,
    start: [-1.7, 0.04],
    lr: 0.02,
  },
};

function makeStates(start) {
  return {
    sgd: { x: start[0], y: start[1] },
    momentum: { x: start[0], y: start[1], vx: 0, vy: 0 },
    rmsprop: { x: start[0], y: start[1], cx: 0, cy: 0 },
    adam: { x: start[0], y: start[1], mx: 0, my: 0, vx: 0, vy: 0, t: 0 },
  };
}

// All four read the identical analytic gradient at their own position, so the
// only thing that differs between trajectories is the update rule itself.
function stepStates(states, surface, hp) {
  const { lr, mu, rho, beta1, beta2 } = hp;
  const eps = 1e-8;
  const next = {};

  {
    const s = states.sgd;
    const [gx, gy] = surface.grad(s.x, s.y);
    next.sgd = { x: s.x - lr * gx, y: s.y - lr * gy };
  }
  {
    const s = states.momentum;
    const [gx, gy] = surface.grad(s.x, s.y);
    const vx = mu * s.vx - lr * gx;
    const vy = mu * s.vy - lr * gy;
    next.momentum = { x: s.x + vx, y: s.y + vy, vx, vy };
  }
  {
    const s = states.rmsprop;
    const [gx, gy] = surface.grad(s.x, s.y);
    const cx = rho * s.cx + (1 - rho) * gx * gx;
    const cy = rho * s.cy + (1 - rho) * gy * gy;
    next.rmsprop = {
      x: s.x - (lr * gx) / (Math.sqrt(cx) + eps),
      y: s.y - (lr * gy) / (Math.sqrt(cy) + eps),
      cx,
      cy,
    };
  }
  {
    const s = states.adam;
    const [gx, gy] = surface.grad(s.x, s.y);
    const t = s.t + 1;
    const mx = beta1 * s.mx + (1 - beta1) * gx;
    const my = beta1 * s.my + (1 - beta1) * gy;
    const vx = beta2 * s.vx + (1 - beta2) * gx * gx;
    const vy = beta2 * s.vy + (1 - beta2) * gy * gy;
    const bc1 = 1 - Math.pow(beta1, t);
    const bc2 = 1 - Math.pow(beta2, t);
    const mhx = mx / bc1;
    const mhy = my / bc1;
    const vhx = vx / bc2;
    const vhy = vy / bc2;
    next.adam = {
      x: s.x - (lr * mhx) / (Math.sqrt(vhx) + eps),
      y: s.y - (lr * mhy) / (Math.sqrt(vhy) + eps),
      mx,
      my,
      vx,
      vy,
      t,
    };
  }
  return next;
}

function useReducedMotion() {
  const [reduced, setReduced] = useState(false);
  useEffect(() => {
    const mq = window.matchMedia("(prefers-reduced-motion: reduce)");
    const update = () => setReduced(mq.matches);
    update();
    mq.addEventListener("change", update);
    return () => mq.removeEventListener("change", update);
  }, []);
  return reduced;
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

function btnStyle(variant, disabled) {
  const base = {
    fontFamily: "inherit",
    fontSize: 13,
    fontWeight: 600,
    borderRadius: 9,
    padding: "8px 14px",
    cursor: disabled ? "not-allowed" : "pointer",
    opacity: disabled ? 0.4 : 1,
    transition: `transform 140ms ${EASE}, background 160ms ease, border-color 160ms ease`,
  };
  if (variant === "primary")
    return { ...base, background: C.accent, color: "#fff", border: `1px solid ${C.accent}` };
  if (variant === "dark")
    return { ...base, background: C.ink, color: "#fff", border: `1px solid ${C.ink}` };
  return { ...base, background: "transparent", color: C.ink, border: `1px solid ${C.border}` };
}

function Btn({ children, onClick, variant = "ghost", disabled, ariaLabel }) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      aria-label={ariaLabel}
      style={btnStyle(variant, disabled)}
      onMouseDown={(e) => {
        if (!disabled) e.currentTarget.style.transform = "scale(0.97)";
      }}
      onMouseUp={(e) => (e.currentTarget.style.transform = "scale(1)")}
      onMouseLeave={(e) => (e.currentTarget.style.transform = "scale(1)")}
    >
      {children}
    </button>
  );
}

const PLOT = 460;

// Loss is heavy-tailed across these surfaces, so the heatmap maps a compressed
// log of the loss to a warm-to-cool ramp; raw loss would saturate to one color.
function lossColor(t) {
  const stops = [
    [247, 245, 242],
    [244, 226, 209],
    [232, 178, 132],
    [192, 86, 31],
    [120, 40, 14],
    [44, 22, 14],
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

function Heatmap({ surface, reduced }) {
  const ref = useRef(null);
  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const res = 120;
    canvas.width = res;
    canvas.height = res;
    const ctx = canvas.getContext("2d");
    const img = ctx.createImageData(res, res);
    let lo = Infinity;
    let hi = -Infinity;
    const vals = new Float64Array(res * res);
    for (let j = 0; j < res; j++) {
      for (let i = 0; i < res; i++) {
        const x = surface.xmin + (surface.xmax - surface.xmin) * (i / (res - 1));
        const y = surface.ymax - (surface.ymax - surface.ymin) * (j / (res - 1));
        const v = surface.f(x, y);
        const lg = Math.log1p(Math.max(v, 0)) * (v < 0 ? -1 : 1);
        vals[j * res + i] = lg;
        if (lg < lo) lo = lg;
        if (lg > hi) hi = lg;
      }
    }
    const span = hi - lo || 1;
    for (let p = 0; p < vals.length; p++) {
      const t = (vals[p] - lo) / span;
      const [r, g, b] = lossColor(t);
      const o = p * 4;
      img.data[o] = r;
      img.data[o + 1] = g;
      img.data[o + 2] = b;
      img.data[o + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
  }, [surface]);

  return (
    <canvas
      ref={ref}
      aria-hidden="true"
      style={{
        position: "absolute",
        inset: 0,
        width: "100%",
        height: "100%",
        imageRendering: reduced ? "pixelated" : "auto",
        borderRadius: 12,
      }}
    />
  );
}

// Marching-squares contour lines drawn over the heatmap to read the valley shape.
function contourPaths(surface, levels, res = 70) {
  const grid = [];
  for (let j = 0; j <= res; j++) {
    const row = [];
    for (let i = 0; i <= res; i++) {
      const x = surface.xmin + (surface.xmax - surface.xmin) * (i / res);
      const y = surface.ymax - (surface.ymax - surface.ymin) * (j / res);
      row.push(surface.f(x, y));
    }
    grid.push(row);
  }
  const toPx = (i, j) => [(i / res) * PLOT, (j / res) * PLOT];
  const lerp = (a, b, va, vb, level) => {
    const d = vb - va;
    return d === 0 ? a : a + ((level - va) * (b - a)) / d;
  };
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
        segs.push([toPx(ai, aj), toPx(bi, bj)]);
      }
    }
    const d = segs.map(([a, b]) => `M${a[0].toFixed(1)},${a[1].toFixed(1)}L${b[0].toFixed(1)},${b[1].toFixed(1)}`).join("");
    paths.push(d);
  }
  return paths;
}

function fmtLoss(v) {
  if (!isFinite(v)) return "diverged";
  if (Math.abs(v) >= 1000) return v.toExponential(2);
  if (Math.abs(v) < 0.001 && v !== 0) return v.toExponential(2);
  return v.toFixed(3);
}

export default function App() {
  const reduced = useReducedMotion();
  const [surfaceKey, setSurfaceKey] = useState("ravine");
  const [start, setStart] = useState(SURFACES.ravine.start);
  const [lr, setLr] = useState(SURFACES.ravine.lr);
  const [mu, setMu] = useState(0.9);
  const [beta1, setBeta1] = useState(0.9);
  const [rho, setRho] = useState(0.999);
  const [playing, setPlaying] = useState(false);
  const [step, setStep] = useState(0);
  const [trails, setTrails] = useState(() => {
    const init = {};
    for (const o of OPTS) init[o.key] = [SURFACES.ravine.start];
    return init;
  });
  const [states, setStates] = useState(() => makeStates(SURFACES.ravine.start));

  const surface = SURFACES[surfaceKey];
  const beta2 = 0.999;
  const hp = useMemo(() => ({ lr, mu, rho, beta1, beta2 }), [lr, mu, rho, beta1]);
  const rafRef = useRef(null);
  const accRef = useRef(0);

  const reset = useCallback(
    (s, sObj) => {
      const surf = sObj || surface;
      const startPt = s || surf.start;
      setPlaying(false);
      setStep(0);
      setStates(makeStates(startPt));
      const init = {};
      for (const o of OPTS) init[o.key] = [startPt];
      setTrails(init);
    },
    [surface],
  );

  const pickSurface = (key) => {
    const surf = SURFACES[key];
    setSurfaceKey(key);
    setStart(surf.start);
    setLr(surf.lr);
    reset(surf.start, surf);
  };

  const advance = useCallback(
    (n) => {
      setStates((prev) => {
        let cur = prev;
        const appended = {};
        for (const o of OPTS) appended[o.key] = [];
        for (let k = 0; k < n; k++) {
          cur = stepStates(cur, surface, hp);
          for (const o of OPTS) appended[o.key].push([cur[o.key].x, cur[o.key].y]);
        }
        setTrails((tprev) => {
          const nextT = {};
          for (const o of OPTS) {
            const merged = tprev[o.key].concat(appended[o.key]);
            nextT[o.key] = merged.length > 1200 ? merged.slice(merged.length - 1200) : merged;
          }
          return nextT;
        });
        return cur;
      });
      setStep((s) => s + n);
    },
    [surface, hp],
  );

  useEffect(() => {
    if (!playing) return;
    let stop = false;
    let last = performance.now();
    const tick = (now) => {
      if (stop) return;
      const dt = now - last;
      last = now;
      accRef.current += dt;
      const stepMs = reduced ? 90 : 28;
      let budget = 0;
      while (accRef.current >= stepMs && budget < 6) {
        advance(1);
        accRef.current -= stepMs;
        budget++;
      }
      if (step >= 600) {
        setPlaying(false);
        return;
      }
      rafRef.current = requestAnimationFrame(tick);
    };
    rafRef.current = requestAnimationFrame(tick);
    return () => {
      stop = true;
      cancelAnimationFrame(rafRef.current);
    };
  }, [playing, advance, reduced, step]);

  useEffect(() => {
    if (playing) return;
    reset(start, surface);
  }, [lr, mu, beta1, rho]);

  const contours = useMemo(() => {
    const levels = [];
    if (surfaceKey === "rosenbrock") {
      for (const v of [0.5, 3, 10, 30, 80, 180]) levels.push(v);
    } else if (surfaceKey === "ravine") {
      for (const v of [0.5, 3, 10, 26, 60, 120]) levels.push(v);
    } else {
      for (const v of [-3, -1.5, -0.4, 0.4, 1.5, 3]) levels.push(v);
    }
    return contourPaths(surface, levels);
  }, [surface, surfaceKey]);

  const toPlot = (x, y) => {
    const px = ((x - surface.xmin) / (surface.xmax - surface.xmin)) * PLOT;
    const py = ((surface.ymax - y) / (surface.ymax - surface.ymin)) * PLOT;
    return [px, py];
  };

  const fromPlot = (px, py) => {
    const x = surface.xmin + (px / PLOT) * (surface.xmax - surface.xmin);
    const y = surface.ymax - (py / PLOT) * (surface.ymax - surface.ymin);
    return [clamp(x, surface.xmin, surface.xmax), clamp(y, surface.ymin, surface.ymax)];
  };

  const handlePlot = (e) => {
    const rect = e.currentTarget.getBoundingClientRect();
    const px = ((e.clientX - rect.left) / rect.width) * PLOT;
    const py = ((e.clientY - rect.top) / rect.height) * PLOT;
    const pt = fromPlot(px, py);
    setStart(pt);
    reset(pt, surface);
  };

  const minPx = surface.min ? toPlot(surface.min[0], surface.min[1]) : null;

  const losses = OPTS.map((o) => {
    const s = states[o.key];
    return { key: o.key, loss: surface.f(s.x, s.y), x: s.x, y: s.y };
  });
  const finiteLosses = losses.filter((l) => isFinite(l.loss));
  const best = finiteLosses.length ? Math.min(...finiteLosses.map((l) => l.loss)) : null;

  const focusCss = `
    .or-root :focus-visible { outline: 2px solid ${C.accent}; outline-offset: 2px; border-radius: 9px; }
    .or-root input[type=range] { accent-color: ${C.accent}; }
    .or-plot { cursor: crosshair; }
    @media (prefers-reduced-motion: reduce) {
      .or-root * { transition-duration: 0.001ms !important; animation-duration: 0.001ms !important; }
    }
  `;

  const sliderRow = (id, label, value, min, max, stepv, onChange, fmt, hint) => (
    <div style={{ marginBottom: 10 }}>
      <div style={{ display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap" }}>
        <label htmlFor={id} style={{ fontSize: 13, color: C.muted, minWidth: 120 }}>
          {label}
        </label>
        <input
          id={id}
          type="range"
          min={min}
          max={max}
          step={stepv}
          value={value}
          onChange={(e) => onChange(+e.target.value)}
          style={{ flex: 1, minWidth: 120 }}
        />
        <span
          style={{
            fontFamily: "ui-monospace, monospace",
            fontSize: 13,
            fontWeight: 700,
            color: C.accent,
            minWidth: 60,
            textAlign: "right",
          }}
        >
          {fmt(value)}
        </span>
      </div>
      {hint && <div style={{ fontSize: 11, color: C.muted, margin: "2px 0 0 132px" }}>{hint}</div>}
    </div>
  );

  return (
    <div
      className="or-root"
      style={{
        fontFamily: "Georgia, 'Iowan Old Style', serif",
        background: C.bg,
        minHeight: "100vh",
        padding: "26px 14px 56px",
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
              color: C.muted,
              marginBottom: 6,
            }}
          >
            Optimization, computed live
          </div>
          <h1
            style={{
              fontSize: 30,
              fontWeight: 700,
              margin: 0,
              lineHeight: 1.1,
              letterSpacing: "-0.02em",
              textWrap: "balance",
            }}
          >
            Optimizer Race
          </h1>
          <p
            style={{
              color: C.muted,
              fontSize: 15,
              margin: "10px 0 0",
              lineHeight: 1.6,
              maxWidth: "62ch",
            }}
          >
            Four optimizers start at the same point and read the same analytic gradient, then step in
            lockstep down a real 2D loss surface. The only difference between the paths is the update
            rule. Click anywhere on the surface to drop a new start and watch them relaunch.
          </p>
        </header>

        <Card style={{ marginBottom: 16, padding: "16px 18px" }}>
          <div style={{ display: "flex", gap: 6, flexWrap: "wrap", marginBottom: 6 }}>
            <span style={{ fontSize: 12, color: C.muted, alignSelf: "center", marginRight: 4 }}>
              Surface:
            </span>
            {Object.entries(SURFACES).map(([key, s]) => (
              <button
                key={key}
                onClick={() => pickSurface(key)}
                aria-pressed={surfaceKey === key}
                style={{
                  fontFamily: "inherit",
                  fontSize: 12,
                  padding: "5px 12px",
                  borderRadius: 8,
                  cursor: "pointer",
                  border: `1px solid ${surfaceKey === key ? C.accent : C.border}`,
                  background: surfaceKey === key ? C.accentSoft : "transparent",
                  color: surfaceKey === key ? C.accent : C.muted,
                  fontWeight: surfaceKey === key ? 700 : 400,
                  transition: `all 150ms ${EASE}`,
                }}
              >
                {s.label}
              </button>
            ))}
          </div>
          <p style={{ fontSize: 13, lineHeight: 1.55, margin: "8px 0 14px", color: C.ink, maxWidth: "64ch" }}>
            {surface.blurb}
          </p>

          <div
            className="or-plot"
            onClick={handlePlot}
            role="img"
            aria-label={`Loss surface for ${surface.label} with four optimizer trajectories. Click to set a new start point.`}
            style={{
              position: "relative",
              width: "100%",
              maxWidth: PLOT,
              margin: "0 auto",
              aspectRatio: "1 / 1",
              borderRadius: 12,
              overflow: "hidden",
              border: `1px solid ${C.border}`,
            }}
          >
            <Heatmap surface={surface} reduced={reduced} />
            <svg
              viewBox={`0 0 ${PLOT} ${PLOT}`}
              width="100%"
              height="100%"
              style={{ position: "absolute", inset: 0, display: "block" }}
            >
              {contours.map((d, i) => (
                <path key={i} d={d} fill="none" stroke="#ffffff" strokeOpacity={0.28} strokeWidth={1} />
              ))}
              {minPx && (
                <g>
                  <circle cx={minPx[0]} cy={minPx[1]} r={6} fill="none" stroke="#fff" strokeWidth={1.6} />
                  <circle cx={minPx[0]} cy={minPx[1]} r={2} fill="#fff" />
                </g>
              )}
              {OPTS.map((o) => {
                const pts = trails[o.key];
                if (!pts || pts.length < 2) return null;
                const d = pts
                  .map(([x, y], i) => {
                    const [px, py] = toPlot(x, y);
                    if (!isFinite(px) || !isFinite(py)) return "";
                    return `${i === 0 ? "M" : "L"}${px.toFixed(1)},${py.toFixed(1)}`;
                  })
                  .join("");
                return (
                  <path
                    key={o.key}
                    d={d}
                    fill="none"
                    stroke={o.color}
                    strokeWidth={2}
                    strokeOpacity={0.92}
                    strokeLinejoin="round"
                    strokeLinecap="round"
                  />
                );
              })}
              {OPTS.map((o) => {
                const s = states[o.key];
                if (!isFinite(s.x) || !isFinite(s.y)) return null;
                const [px, py] = toPlot(s.x, s.y);
                if (px < -20 || px > PLOT + 20 || py < -20 || py > PLOT + 20) return null;
                return (
                  <g key={o.key}>
                    <circle cx={px} cy={py} r={5.5} fill={o.color} stroke="#fff" strokeWidth={1.8} />
                  </g>
                );
              })}
              {(() => {
                const [sx, sy] = toPlot(start[0], start[1]);
                return <circle cx={sx} cy={sy} r={3} fill="none" stroke="#fff" strokeWidth={1.4} strokeDasharray="2 2" />;
              })()}
            </svg>
          </div>

          <div style={{ display: "flex", gap: 14, justifyContent: "center", flexWrap: "wrap", marginTop: 12 }}>
            {OPTS.map((o) => (
              <span key={o.key} style={{ display: "inline-flex", alignItems: "center", gap: 6, fontSize: 12 }}>
                <span style={{ width: 14, height: 3, borderRadius: 2, background: o.color, display: "inline-block" }} />
                <span style={{ color: C.ink, fontWeight: 600 }}>{o.name}</span>
              </span>
            ))}
          </div>
        </Card>

        <Card style={{ marginBottom: 16 }}>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 8, alignItems: "center", marginBottom: 6 }}>
            <Btn
              onClick={() => setPlaying((p) => !p)}
              variant={playing ? "ghost" : "primary"}
              ariaLabel={playing ? "Pause the race" : "Play the race"}
            >
              {playing ? "Pause" : "Play"}
            </Btn>
            <Btn onClick={() => { setPlaying(false); advance(1); }} variant="dark" ariaLabel="Step one iteration">
              Step
            </Btn>
            <Btn onClick={() => { setPlaying(false); advance(10); }} variant="ghost" ariaLabel="Step ten iterations">
              Step x10
            </Btn>
            <Btn onClick={() => reset(start, surface)} variant="ghost" ariaLabel="Reset to the current start point">
              Reset
            </Btn>
            <span style={{ fontSize: 12, color: C.muted, marginLeft: "auto", fontFamily: "ui-monospace, monospace" }}>
              step {step}
            </span>
          </div>

          {sliderRow(
            "lr",
            "Learning rate",
            lr,
            surface.lr / 6,
            surface.lr * 4,
            surface.lr / 100,
            setLr,
            (v) => v.toPrecision(2),
            "Shared by all four. Raise it until SGD overshoots the ravine walls and diverges.",
          )}
          {sliderRow("mu", "Momentum mu", mu, 0, 0.98, 0.01, setMu, (v) => v.toFixed(2), "Friction for Momentum's velocity. Higher carries more speed along the valley.")}
          {sliderRow("beta1", "Adam beta1", beta1, 0.5, 0.98, 0.01, setBeta1, (v) => v.toFixed(2), "Decay of Adam's first moment, its smoothed gradient direction.")}
          {sliderRow("rho", "RMSProp rho", rho, 0.9, 0.9995, 0.0005, setRho, (v) => v.toFixed(4), "Decay of the squared-gradient cache that rescales each axis.")}
          <p style={{ fontSize: 11, color: C.muted, margin: "4px 0 0", lineHeight: 1.5 }}>
            Adam's second moment uses beta2 = 0.999, eps = 1e-8. Changing any slider relaunches the race
            from the current start so the comparison stays fair.
          </p>
        </Card>

        <Card style={{ marginBottom: 16 }}>
          <div style={{ fontSize: 13, fontWeight: 700, marginBottom: 12 }}>Live standings</div>
          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            {OPTS.map((o) => {
              const l = losses.find((x) => x.key === o.key);
              const isBest = best != null && isFinite(l.loss) && Math.abs(l.loss - best) < 1e-9;
              return (
                <div
                  key={o.key}
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: 10,
                    padding: "9px 12px",
                    borderRadius: 9,
                    background: isBest ? o.color + "12" : C.bg,
                    border: `1px solid ${isBest ? o.color + "55" : C.border}`,
                    transition: reduced ? "none" : `background 200ms ease, border-color 200ms ease`,
                  }}
                >
                  <span style={{ width: 12, height: 12, borderRadius: 3, background: o.color, flexShrink: 0 }} />
                  <span style={{ fontSize: 13, fontWeight: 700, minWidth: 92 }}>{o.name}</span>
                  <span
                    style={{
                      fontFamily: "ui-monospace, monospace",
                      fontSize: 13,
                      color: isFinite(l.loss) ? C.ink : "#b23b5e",
                      minWidth: 96,
                    }}
                  >
                    loss {fmtLoss(l.loss)}
                  </span>
                  <span style={{ fontFamily: "ui-monospace, monospace", fontSize: 11, color: C.muted, marginLeft: "auto" }}>
                    ({l.x.toFixed(2)}, {l.y.toFixed(2)})
                  </span>
                  {isBest && (
                    <span style={{ fontSize: 11, fontWeight: 700, color: o.color }}>lowest</span>
                  )}
                </div>
              );
            })}
          </div>
          <p style={{ fontSize: 12, color: C.muted, margin: "12px 0 0", lineHeight: 1.55 }}>
            On the ravine, let it run: Momentum, RMSProp, and Adam all reach a far lower loss than plain
            SGD in the same number of steps. At this rate SGD sits right at its stability edge for the
            steep axis, so it oscillates across the valley and its loss plateaus while the others slide
            down to the floor.
          </p>
        </Card>

        <Card>
          <div style={{ fontSize: 13, fontWeight: 700, marginBottom: 10 }}>The four update rules</div>
          <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
            {OPTS.map((o) => (
              <div key={o.key} style={{ display: "flex", gap: 10, alignItems: "flex-start" }}>
                <span
                  style={{
                    width: 10,
                    height: 10,
                    borderRadius: 3,
                    background: o.color,
                    marginTop: 4,
                    flexShrink: 0,
                  }}
                />
                <div>
                  <div style={{ fontSize: 13, fontWeight: 700 }}>{o.name}</div>
                  <code
                    style={{
                      fontFamily: "ui-monospace, monospace",
                      fontSize: 12,
                      color: C.ink,
                      background: C.bg,
                      borderRadius: 6,
                      padding: "2px 6px",
                      display: "inline-block",
                      marginTop: 2,
                    }}
                  >
                    {o.rule}
                  </code>
                </div>
              </div>
            ))}
          </div>
          <p style={{ fontSize: 14, lineHeight: 1.6, margin: "14px 0 0", color: C.ink, maxWidth: "64ch" }}>
            Plain SGD takes a fixed fraction of the raw gradient, so in a ravine it must keep the step
            small enough to not blow up on the steep axis, which leaves it crawling on the flat one.
            Momentum accumulates a velocity that averages out the side-to-side gradient and keeps the
            consistent down-valley component, so it powers through. RMSProp divides each axis by the
            running size of its own gradients, shrinking the step where gradients are large and growing
            it where they are small, which evens out the conditioning. Adam combines both: a momentum-like
            first moment for direction and an RMSProp-like second moment for per-axis scale, with a bias
            correction so the early steps are not damped toward zero.
          </p>
        </Card>
      </div>
    </div>
  );
}
