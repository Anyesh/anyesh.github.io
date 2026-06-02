import { useState, useEffect, useRef, useMemo, useCallback } from "react";

export const meta = {
  title: "Fourier: Drawing with Circles",
  category: "Signals",
  description:
    "A real discrete Fourier transform turns any closed path into a stack of rotating circles, then sums them back into the curve. Add circles one at a time and watch a crude loop sharpen into the exact shape.",
  date: "2026-02-27",
  tags: ["fourier", "dft", "signals", "epicycles"],
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
  circle: "#b9a98f",
  circleSoft: "rgba(155,147,138,0.35)",
  target: "#cdbfa8",
  trace: "#c0561f",
  good: "#3f7d52",
  grid: "#efebe4",
};

const EASE = "cubic-bezier(0.23, 1, 0.32, 1)";
const N = 256;
const TAU = 2 * Math.PI;

function resamplePath(raw, count) {
  if (raw.length < 2) return [];
  const closed = [...raw, raw[0]];
  const seg = [];
  let total = 0;
  for (let i = 0; i < closed.length - 1; i++) {
    const d = Math.hypot(closed[i + 1].x - closed[i].x, closed[i + 1].y - closed[i].y);
    seg.push(d);
    total += d;
  }
  if (total === 0) return [];
  const out = [];
  let segIdx = 0;
  let accumulated = 0;
  for (let k = 0; k < count; k++) {
    const target = (k / count) * total;
    while (segIdx < seg.length - 1 && accumulated + seg[segIdx] < target) {
      accumulated += seg[segIdx];
      segIdx++;
    }
    const segLen = seg[segIdx] || 1e-9;
    const f = (target - accumulated) / segLen;
    const a = closed[segIdx];
    const b = closed[segIdx + 1];
    out.push({ x: a.x + (b.x - a.x) * f, y: a.y + (b.y - a.y) * f });
  }
  return out;
}

function dft(points) {
  const n = points.length;
  const out = [];
  for (let k = 0; k < n; k++) {
    let re = 0;
    let im = 0;
    for (let i = 0; i < n; i++) {
      const phi = (TAU * k * i) / n;
      const cos = Math.cos(phi);
      const sin = Math.sin(phi);
      re += points[i].x * cos + points[i].y * sin;
      im += -points[i].x * sin + points[i].y * cos;
    }
    re /= n;
    im /= n;
    const freq = k <= n / 2 ? k : k - n;
    out.push({ freq, amp: Math.hypot(re, im), phase: Math.atan2(im, re) });
  }
  out.sort((a, b) => b.amp - a.amp);
  return out;
}

function epicycleChain(coeffs, count, t) {
  let x = 0;
  let y = 0;
  const tips = [{ x: 0, y: 0 }];
  const radii = [];
  const limit = Math.min(count, coeffs.length);
  for (let i = 0; i < limit; i++) {
    const c = coeffs[i];
    const angle = c.freq * TAU * t + c.phase;
    const px = x;
    const py = y;
    x += c.amp * Math.cos(angle);
    y += c.amp * Math.sin(angle);
    tips.push({ x, y });
    radii.push({ cx: px, cy: py, r: c.amp });
  }
  return { tip: { x, y }, tips, radii };
}

function makeStar(points, outer, inner) {
  const verts = [];
  for (let i = 0; i < points * 2; i++) {
    const r = i % 2 === 0 ? outer : inner;
    const a = (Math.PI * i) / points - Math.PI / 2;
    verts.push({ x: r * Math.cos(a), y: r * Math.sin(a) });
  }
  return densify(verts, 30);
}

function densify(verts, perEdge) {
  const out = [];
  for (let i = 0; i < verts.length; i++) {
    const a = verts[i];
    const b = verts[(i + 1) % verts.length];
    for (let j = 0; j < perEdge; j++) {
      const f = j / perEdge;
      out.push({ x: a.x + (b.x - a.x) * f, y: a.y + (b.y - a.y) * f });
    }
  }
  return out;
}

function makeSquare(s) {
  return densify(
    [
      { x: -s, y: -s },
      { x: s, y: -s },
      { x: s, y: s },
      { x: -s, y: s },
    ],
    64
  );
}

function makeHeart(scale) {
  const out = [];
  const steps = 240;
  for (let i = 0; i < steps; i++) {
    const t = (TAU * i) / steps;
    const x = 16 * Math.sin(t) ** 3;
    const y = -(13 * Math.cos(t) - 5 * Math.cos(2 * t) - 2 * Math.cos(3 * t) - Math.cos(4 * t));
    out.push({ x: x * scale, y: y * scale });
  }
  return out;
}

function makeLetterR(h) {
  const verts = [
    { x: -0.5, y: 1 },
    { x: 0.18, y: 1 },
    { x: 0.42, y: 0.86 },
    { x: 0.42, y: 0.52 },
    { x: 0.18, y: 0.38 },
    { x: 0.1, y: 0.38 },
    { x: 0.46, y: -1 },
    { x: 0.18, y: -1 },
    { x: -0.16, y: 0.18 },
    { x: -0.16, y: -1 },
    { x: -0.5, y: -1 },
  ];
  return densify(
    verts.map((v) => ({ x: v.x * h, y: -v.y * h })),
    26
  );
}

const PRESETS = {
  star: { label: "Star", build: () => makeStar(5, 118, 48) },
  square: { label: "Square", build: () => makeSquare(96) },
  heart: { label: "Heart", build: () => makeHeart(7.2) },
  letter: { label: "Letter R", build: () => makeLetterR(116) },
};

function recenter(points) {
  const cx = points.reduce((s, p) => s + p.x, 0) / points.length;
  const cy = points.reduce((s, p) => s + p.y, 0) / points.length;
  return points.map((p) => ({ x: p.x - cx, y: p.y - cy }));
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
          fontFamily: "'IBM Plex Mono', ui-monospace, SFMono-Regular, Menlo, monospace",
          fontSize: 17,
          fontWeight: 700,
          color: accent ? C.accent : C.ink,
          lineHeight: 1,
          fontVariantNumeric: "tabular-nums",
        }}
      >
        {value}
      </div>
    </div>
  );
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

const VIEW = 340;

function EpicycleCanvas({ coeffs, count, t, target, reduce }) {
  const ref = useRef(null);

  useEffect(() => {
    const cv = ref.current;
    if (!cv) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    cv.width = VIEW * dpr;
    cv.height = VIEW * dpr;
    const ctx = cv.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, VIEW, VIEW);

    const cx = VIEW / 2;
    const cy = VIEW / 2;
    const toPx = (p) => [cx + p.x, cy + p.y];

    ctx.strokeStyle = C.grid;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(cx, 10);
    ctx.lineTo(cx, VIEW - 10);
    ctx.moveTo(10, cy);
    ctx.lineTo(VIEW - 10, cy);
    ctx.stroke();

    if (target.length > 1) {
      ctx.strokeStyle = C.target;
      ctx.lineWidth = 1.5;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      target.forEach((p, i) => {
        const [px, py] = toPx(p);
        if (i === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      });
      ctx.closePath();
      ctx.stroke();
      ctx.setLineDash([]);
    }

    const samples = reduce ? 360 : 200;
    ctx.strokeStyle = C.trace;
    ctx.lineWidth = 2.4;
    ctx.lineJoin = "round";
    ctx.beginPath();
    const upto = reduce ? 1 : t;
    for (let i = 0; i <= samples; i++) {
      const tt = (i / samples) * upto;
      const { tip } = epicycleChain(coeffs, count, tt % 1);
      const [px, py] = toPx(tip);
      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    }
    ctx.stroke();

    const { tips, radii } = epicycleChain(coeffs, count, reduce ? 0 : t % 1);

    ctx.lineWidth = 1;
    for (const c of radii) {
      ctx.strokeStyle = C.circleSoft;
      ctx.beginPath();
      const [ox, oy] = toPx({ x: c.cx, y: c.cy });
      ctx.arc(ox, oy, c.r, 0, TAU);
      ctx.stroke();
    }

    ctx.strokeStyle = C.circle;
    ctx.lineWidth = 1.4;
    ctx.beginPath();
    tips.forEach((p, i) => {
      const [px, py] = toPx(p);
      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    });
    ctx.stroke();

    const last = tips[tips.length - 1];
    const [lx, ly] = toPx(last);
    ctx.fillStyle = C.accent;
    ctx.beginPath();
    ctx.arc(lx, ly, 3.6, 0, TAU);
    ctx.fill();
  }, [coeffs, count, t, target, reduce]);

  return (
    <canvas
      ref={ref}
      style={{
        width: VIEW,
        height: VIEW,
        maxWidth: "100%",
        borderRadius: 12,
        background: C.bg,
        border: `1px solid ${C.border}`,
        display: "block",
        touchAction: "none",
      }}
      role="img"
      aria-label="Rotating circles tracing a path reconstructed from Fourier coefficients"
    />
  );
}

function DrawPad({ onCommit, reduce }) {
  const ref = useRef(null);
  const drawing = useRef(false);
  const pts = useRef([]);
  const [hasInk, setHasInk] = useState(false);

  const redraw = useCallback(() => {
    const cv = ref.current;
    if (!cv) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    cv.width = VIEW * dpr;
    cv.height = VIEW * dpr;
    const ctx = cv.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, VIEW, VIEW);
    ctx.fillStyle = C.faint;
    ctx.font = "13px ui-monospace, monospace";
    ctx.textAlign = "center";
    if (pts.current.length < 2) {
      ctx.fillText("draw one continuous closed loop", VIEW / 2, VIEW / 2);
    }
    ctx.strokeStyle = C.trace;
    ctx.lineWidth = 2.4;
    ctx.lineJoin = "round";
    ctx.lineCap = "round";
    ctx.beginPath();
    pts.current.forEach((p, i) => {
      if (i === 0) ctx.moveTo(p.x, p.y);
      else ctx.lineTo(p.x, p.y);
    });
    ctx.stroke();
  }, []);

  useEffect(() => {
    redraw();
  }, [redraw]);

  const local = (e) => {
    const rect = ref.current.getBoundingClientRect();
    const scaleX = VIEW / rect.width;
    const scaleY = VIEW / rect.height;
    return { x: (e.clientX - rect.left) * scaleX, y: (e.clientY - rect.top) * scaleY };
  };

  const start = (e) => {
    e.currentTarget.setPointerCapture(e.pointerId);
    drawing.current = true;
    pts.current = [local(e)];
    setHasInk(false);
    redraw();
  };

  const move = (e) => {
    if (!drawing.current) return;
    const p = local(e);
    const last = pts.current[pts.current.length - 1];
    if (!last || Math.hypot(p.x - last.x, p.y - last.y) > 3) {
      pts.current.push(p);
      redraw();
      if (pts.current.length > 8) setHasInk(true);
    }
  };

  const end = () => {
    if (!drawing.current) return;
    drawing.current = false;
    if (pts.current.length > 8) {
      const centered = pts.current.map((p) => ({ x: p.x - VIEW / 2, y: p.y - VIEW / 2 }));
      onCommit(centered);
    }
  };

  const clear = () => {
    pts.current = [];
    setHasInk(false);
    redraw();
  };

  return (
    <div>
      <canvas
        ref={ref}
        onPointerDown={start}
        onPointerMove={move}
        onPointerUp={end}
        onPointerLeave={end}
        style={{
          width: VIEW,
          height: VIEW,
          maxWidth: "100%",
          borderRadius: 12,
          background: C.bg,
          border: `1px dashed ${C.accent}66`,
          display: "block",
          cursor: "crosshair",
          touchAction: "none",
        }}
        role="img"
        aria-label="Drawing surface. Draw a closed loop with the pointer to run the Fourier transform on it."
      />
      <div style={{ display: "flex", gap: 9, marginTop: 10, alignItems: "center" }}>
        <button type="button" className="fx-btn" onClick={clear} style={btnStyle(false)}>
          Clear
        </button>
        <span style={{ fontSize: 12.5, color: hasInk ? C.good : C.muted }}>
          {hasInk ? "Released: transform applied to the left." : "Press and drag to sketch a loop."}
        </span>
      </div>
    </div>
  );
}

function btnStyle(active) {
  return {
    padding: "8px 15px",
    borderRadius: 9,
    border: `1.5px solid ${active ? C.accent : C.border}`,
    background: active ? C.accent : C.card,
    color: active ? "#fff" : C.muted,
    fontSize: 13,
    fontWeight: 600,
    cursor: "pointer",
    fontFamily: "inherit",
    transition: `transform 140ms ${EASE}, background 160ms ${EASE}, border-color 160ms ${EASE}, color 160ms ${EASE}`,
  };
}

export default function App() {
  const reduce = usePrefersReducedMotion();
  const [preset, setPreset] = useState("star");
  const [drawn, setDrawn] = useState(null);
  const [count, setCount] = useState(8);
  const [speed, setSpeed] = useState(1);
  const [playing, setPlaying] = useState(true);
  const [t, setT] = useState(0);

  const targetPoints = useMemo(() => {
    const raw = drawn ? drawn : PRESETS[preset].build();
    const sampled = resamplePath(raw, N);
    return recenter(sampled.length ? sampled : raw);
  }, [preset, drawn]);

  const coeffs = useMemo(() => dft(targetPoints), [targetPoints]);

  const raf = useRef(0);
  const prev = useRef(0);
  const tRef = useRef(0);
  useEffect(() => {
    tRef.current = t;
  }, [t]);

  useEffect(() => {
    if (reduce || !playing) {
      cancelAnimationFrame(raf.current);
      return;
    }
    const loop = (ts) => {
      if (!prev.current) prev.current = ts;
      const dt = (ts - prev.current) / 1000;
      prev.current = ts;
      tRef.current = (tRef.current + dt * speed * 0.18) % 1;
      setT(tRef.current);
      raf.current = requestAnimationFrame(loop);
    };
    raf.current = requestAnimationFrame(loop);
    return () => {
      cancelAnimationFrame(raf.current);
      prev.current = 0;
    };
  }, [playing, speed, reduce]);

  const maxCircles = coeffs.length;
  const clampedCount = Math.min(count, maxCircles);

  const reconError = useMemo(() => {
    let worst = 0;
    const step = Math.max(1, Math.floor(targetPoints.length / 48));
    for (let i = 0; i < targetPoints.length; i += step) {
      const tt = i / targetPoints.length;
      const { tip } = epicycleChain(coeffs, clampedCount, tt);
      worst = Math.max(worst, Math.hypot(tip.x - targetPoints[i].x, tip.y - targetPoints[i].y));
    }
    return worst;
  }, [coeffs, clampedCount, targetPoints]);

  const coverage = useMemo(() => {
    const totalEnergy = coeffs.reduce((s, c) => s + c.amp * c.amp, 0) || 1;
    let kept = 0;
    for (let i = 0; i < clampedCount; i++) kept += coeffs[i].amp * coeffs[i].amp;
    return kept / totalEnergy;
  }, [coeffs, clampedCount]);

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
        .fx-btn:active { transform: scale(0.97); }
        .fx-btn:focus-visible, input[type=range]:focus-visible {
          outline: 2px solid ${C.accent}; outline-offset: 2px;
        }
        input[type=range] { accent-color: ${C.accent}; }
      `}</style>

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
            Signals / Fourier analysis
          </div>
          <h1 style={{ fontSize: 30, fontWeight: 700, margin: 0, lineHeight: 1.1, letterSpacing: "-0.02em", textWrap: "balance" }}>
            Drawing with Circles
          </h1>
          <p style={{ color: C.muted, fontSize: 15, lineHeight: 1.6, margin: "10px 0 0", maxWidth: "64ch" }}>
            Treat a closed path as a list of complex numbers, one per point: the x coordinate is the real part, the
            y coordinate is the imaginary part. The discrete Fourier transform rewrites that list as a sum of pure
            rotations, each at a fixed frequency. Stack those rotations tip to tail and the last tip retraces the
            original drawing.
          </p>
        </header>

        <Card style={{ marginBottom: 18 }}>
          <div style={{ display: "flex", gap: 18, flexWrap: "wrap", alignItems: "flex-start", justifyContent: "center" }}>
            <EpicycleCanvas coeffs={coeffs} count={clampedCount} t={t} target={targetPoints} reduce={reduce} />

            <div style={{ flex: "1 1 248px", minWidth: 232 }}>
              <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 16 }}>
                {!reduce && (
                  <button
                    type="button"
                    className="fx-btn"
                    onClick={() => setPlaying((p) => !p)}
                    style={btnStyle(true)}
                    aria-label={playing ? "Pause rotation" : "Play rotation"}
                  >
                    {playing ? "Pause" : "Play"}
                  </button>
                )}
                <button
                  type="button"
                  className="fx-btn"
                  onClick={() => {
                    setT(0);
                    tRef.current = 0;
                  }}
                  style={btnStyle(false)}
                  aria-label="Restart the trace from the beginning"
                >
                  Restart
                </button>
              </div>

              <label htmlFor="fx-count" style={{ fontSize: 13, color: C.muted, display: "block", marginBottom: 6 }}>
                Circles in the reconstruction
              </label>
              <input
                id="fx-count"
                type="range"
                min={1}
                max={Math.min(120, maxCircles)}
                value={clampedCount}
                onChange={(e) => setCount(+e.target.value)}
                style={{ width: "100%" }}
                aria-label="number of epicycles included"
              />
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8, marginTop: 14 }}>
                <Stat label="circles" value={clampedCount} accent />
                <Stat label="energy kept" value={`${(coverage * 100).toFixed(1)}%`} />
                <Stat label="max gap (px)" value={reconError.toFixed(reconError < 1 ? 2 : 1)} />
                <Stat label="path points" value={targetPoints.length} />
              </div>

              {reduce ? (
                <p style={{ fontSize: 12.5, color: C.muted, marginTop: 14, lineHeight: 1.55 }}>
                  Reduced motion is on, so the circles do not spin. The trace shows the full reconstructed loop at
                  the current circle count. Move the slider to watch detail return.
                </p>
              ) : (
                <div style={{ marginTop: 14 }}>
                  <label htmlFor="fx-speed" style={{ fontSize: 13, color: C.muted, display: "block", marginBottom: 6 }}>
                    Rotation speed
                  </label>
                  <input
                    id="fx-speed"
                    type="range"
                    min={0.25}
                    max={3}
                    step={0.25}
                    value={speed}
                    onChange={(e) => setSpeed(+e.target.value)}
                    style={{ width: "100%" }}
                    aria-label="rotation speed"
                  />
                </div>
              )}
            </div>
          </div>
        </Card>

        <Card style={{ marginBottom: 18 }}>
          <div style={{ fontSize: 13, fontWeight: 700, marginBottom: 4 }}>Pick a shape, or draw your own</div>
          <p style={{ fontSize: 13, color: C.muted, margin: "0 0 14px", lineHeight: 1.55, maxWidth: "62ch" }}>
            Each shape is sampled into {N} points around its perimeter, then run through the same transform. A
            drawn loop is resampled to the same {N} points so the math is identical.
          </p>
          <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 18 }}>
            {Object.entries(PRESETS).map(([k, p]) => (
              <button
                key={k}
                type="button"
                className="fx-btn"
                onClick={() => {
                  setDrawn(null);
                  setPreset(k);
                }}
                style={btnStyle(!drawn && preset === k)}
                aria-pressed={!drawn && preset === k}
              >
                {p.label}
              </button>
            ))}
            <span
              style={{
                alignSelf: "center",
                fontSize: 12.5,
                color: drawn ? C.good : C.faint,
                fontWeight: drawn ? 700 : 400,
              }}
            >
              {drawn ? "Showing your drawing" : "or sketch one below"}
            </span>
          </div>
          <div style={{ display: "flex", justifyContent: "center" }}>
            <DrawPad onCommit={setDrawn} reduce={reduce} />
          </div>
        </Card>

        <Card style={{ background: C.accentSoft, borderColor: `${C.accent}33` }}>
          <div style={{ fontSize: 13, fontWeight: 700, marginBottom: 10 }}>What the slider is doing</div>
          <p style={{ fontSize: 14, lineHeight: 1.65, color: "#5d4226", margin: "0 0 10px", maxWidth: "66ch" }}>
            Every circle is one frequency. Its radius is that frequency&rsquo;s amplitude (how much it contributes),
            and where it starts in its spin is its phase. The transform hands back one circle per path point; sorting
            them by amplitude puts the biggest, slowest contributions first.
          </p>
          <p style={{ fontSize: 14, lineHeight: 1.65, color: "#5d4226", margin: 0, maxWidth: "66ch" }}>
            With one circle you get a plain loop. Add the next few and the rough shape appears. The small fast
            circles that come later only carve out corners and fine wiggles, which is why &ldquo;energy kept&rdquo;
            climbs fast at first and then crawls: a handful of circles already captures most of the curve, and the
            long tail just sharpens edges. Drag the count to the maximum and the gap to the target collapses to
            sub-pixel.
          </p>
        </Card>

        <p style={{ textAlign: "center", fontSize: 11.5, color: C.faint, marginTop: 22, lineHeight: 1.6 }}>
          Discrete Fourier transform over {N} complex samples. Reconstruction sums {clampedCount} rotating
          vectors; with all {maxCircles} the path returns exactly.
        </p>
      </div>
    </div>
  );
}
