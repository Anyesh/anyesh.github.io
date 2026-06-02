import { useState, useEffect, useRef, useCallback } from "react";

export const meta = {
  title: "Boids: Flocking From Local Rules",
  category: "Emergence",
  description:
    "No leader, no plan. Each boid watches only its nearby neighbors and obeys three steering rules. Dial separation, alignment, and cohesion and watch a flock cohere or fall apart.",
  date: "2026-04-18",
  tags: ["boids", "flocking", "emergence", "agent-based"],
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
  field: "#1f1d1a",
  fieldGrid: "#2c2a26",
  boid: "#e8b48c",
  boidHi: "#f3d9b5",
  focus: "#5a8f7b",
  focusSoft: "#dceae3",
  good: "#3f7d52",
};

const EASE = "cubic-bezier(0.23, 1, 0.32, 1)";

function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const WORLD_W = 640;
const WORLD_H = 420;
const SEP_RATIO = 0.45;

function makeFlock(count, seed) {
  const rng = mulberry32(seed);
  const boids = new Array(count);
  for (let i = 0; i < count; i++) {
    const angle = rng() * Math.PI * 2;
    const speed = 1.4 + rng() * 1.2;
    boids[i] = {
      x: rng() * WORLD_W,
      y: rng() * WORLD_H,
      vx: Math.cos(angle) * speed,
      vy: Math.sin(angle) * speed,
    };
  }
  return boids;
}

function limit(vx, vy, max) {
  const m2 = vx * vx + vy * vy;
  if (m2 > max * max && m2 > 1e-9) {
    const s = max / Math.sqrt(m2);
    return [vx * s, vy * s];
  }
  return [vx, vy];
}

function steerToward(targetVx, targetVy, vx, vy, maxSpeed, maxForce) {
  const m = Math.hypot(targetVx, targetVy);
  if (m < 1e-9) return [0, 0];
  const dvx = (targetVx / m) * maxSpeed - vx;
  const dvy = (targetVy / m) * maxSpeed - vy;
  return limit(dvx, dvy, maxForce);
}

const MAX_FORCE = 0.06;
const EDGE_MARGIN = 70;
const EDGE_TURN = 0.22;

function stepFlock(boids, p, predator) {
  const n = boids.length;
  const percep2 = p.perception * p.perception;
  const sepR = p.perception * SEP_RATIO;
  const sep2 = sepR * sepR;

  const cell = p.perception;
  const cols = Math.max(1, Math.ceil(WORLD_W / cell));
  const rows = Math.max(1, Math.ceil(WORLD_H / cell));
  const grid = new Array(cols * rows);
  for (let i = 0; i < n; i++) {
    const cx = Math.min(cols - 1, Math.max(0, Math.floor(boids[i].x / cell)));
    const cy = Math.min(rows - 1, Math.max(0, Math.floor(boids[i].y / cell)));
    const idx = cy * cols + cx;
    if (!grid[idx]) grid[idx] = [];
    grid[idx].push(i);
  }

  const ax = new Float32Array(n);
  const ay = new Float32Array(n);

  for (let i = 0; i < n; i++) {
    const b = boids[i];
    const cx = Math.min(cols - 1, Math.max(0, Math.floor(b.x / cell)));
    const cy = Math.min(rows - 1, Math.max(0, Math.floor(b.y / cell)));

    let sepX = 0, sepY = 0;
    let aliX = 0, aliY = 0;
    let cohX = 0, cohY = 0;
    let nNeighbors = 0;
    let nSep = 0;

    for (let gy = cy - 1; gy <= cy + 1; gy++) {
      if (gy < 0 || gy >= rows) continue;
      for (let gx = cx - 1; gx <= cx + 1; gx++) {
        if (gx < 0 || gx >= cols) continue;
        const bucket = grid[gy * cols + gx];
        if (!bucket) continue;
        for (let k = 0; k < bucket.length; k++) {
          const j = bucket[k];
          if (j === i) continue;
          const o = boids[j];
          const dx = o.x - b.x;
          const dy = o.y - b.y;
          const d2 = dx * dx + dy * dy;
          if (d2 > percep2 || d2 < 1e-9) continue;
          nNeighbors++;
          aliX += o.vx;
          aliY += o.vy;
          cohX += o.x;
          cohY += o.y;
          if (d2 < sep2) {
            const d = Math.sqrt(d2);
            sepX -= dx / d;
            sepY -= dy / d;
            nSep++;
          }
        }
      }
    }

    if (nNeighbors > 0) {
      const [sax, say] = steerToward(aliX, aliY, b.vx, b.vy, p.maxSpeed, MAX_FORCE);
      ax[i] += sax * p.alignW;
      ay[i] += say * p.alignW;

      const toCenterX = cohX / nNeighbors - b.x;
      const toCenterY = cohY / nNeighbors - b.y;
      const [scx, scy] = steerToward(toCenterX, toCenterY, b.vx, b.vy, p.maxSpeed, MAX_FORCE);
      ax[i] += scx * p.cohesionW;
      ay[i] += scy * p.cohesionW;
    }

    if (nSep > 0) {
      const [ssx, ssy] = steerToward(sepX, sepY, b.vx, b.vy, p.maxSpeed, MAX_FORCE);
      ax[i] += ssx * p.separationW;
      ay[i] += ssy * p.separationW;
    }

    if (predator) {
      const dx = b.x - predator.x;
      const dy = b.y - predator.y;
      const d2 = dx * dx + dy * dy;
      const fleeR = 110;
      if (d2 < fleeR * fleeR && d2 > 1e-6) {
        const d = Math.sqrt(d2);
        const [fx, fy] = steerToward(dx / d, dy / d, b.vx, b.vy, p.maxSpeed, MAX_FORCE * 3);
        const urgency = 1 - d / fleeR;
        ax[i] += fx * 2.2 * urgency;
        ay[i] += fy * 2.2 * urgency;
      }
    }
  }

  for (let i = 0; i < n; i++) {
    const b = boids[i];
    b.vx += ax[i];
    b.vy += ay[i];

    if (b.x < EDGE_MARGIN) b.vx += EDGE_TURN;
    else if (b.x > WORLD_W - EDGE_MARGIN) b.vx -= EDGE_TURN;
    if (b.y < EDGE_MARGIN) b.vy += EDGE_TURN;
    else if (b.y > WORLD_H - EDGE_MARGIN) b.vy -= EDGE_TURN;

    [b.vx, b.vy] = limit(b.vx, b.vy, p.maxSpeed);
    const sp = Math.hypot(b.vx, b.vy);
    if (sp < p.maxSpeed * 0.35 && sp > 1e-6) {
      const boost = (p.maxSpeed * 0.35) / sp;
      b.vx *= boost;
      b.vy *= boost;
    }

    b.x += b.vx;
    b.y += b.vy;

    if (b.x < 0) b.x += WORLD_W;
    else if (b.x >= WORLD_W) b.x -= WORLD_W;
    if (b.y < 0) b.y += WORLD_H;
    else if (b.y >= WORLD_H) b.y -= WORLD_H;
  }
}

function orderParameter(boids) {
  let sx = 0, sy = 0, total = 0;
  for (let i = 0; i < boids.length; i++) {
    const sp = Math.hypot(boids[i].vx, boids[i].vy);
    if (sp < 1e-9) continue;
    sx += boids[i].vx / sp;
    sy += boids[i].vy / sp;
    total++;
  }
  if (total === 0) return 0;
  return Math.hypot(sx, sy) / total;
}

function averageSpeed(boids) {
  let s = 0;
  for (let i = 0; i < boids.length; i++) s += Math.hypot(boids[i].vx, boids[i].vy);
  return boids.length ? s / boids.length : 0;
}

function Card({ children, style }) {
  return (
    <div style={{ background: C.card, border: `1px solid ${C.border}`, borderRadius: 14, padding: 18, ...style }}>
      {children}
    </div>
  );
}

function Eyebrow({ children }) {
  return (
    <div style={{ fontSize: 10, letterSpacing: "0.14em", textTransform: "uppercase", color: C.muted, fontWeight: 700 }}>
      {children}
    </div>
  );
}

function Stat({ label, value, accent, focus }) {
  const fg = focus ? C.focus : accent ? C.accent : C.ink;
  const lab = focus ? C.focus : accent ? C.accent : C.muted;
  const bg = focus ? C.focusSoft : accent ? C.accentSoft : C.bg;
  const bd = focus ? C.focus + "33" : accent ? C.accent + "33" : C.border;
  return (
    <div style={{ background: bg, borderRadius: 9, padding: "9px 12px", border: `1px solid ${bd}` }}>
      <div style={{ fontSize: 10, letterSpacing: "0.06em", textTransform: "uppercase", color: lab, fontWeight: 700, marginBottom: 3 }}>{label}</div>
      <div style={{ fontFamily: "'IBM Plex Mono', ui-monospace, Menlo, monospace", fontSize: 17, fontWeight: 700, color: fg, lineHeight: 1 }}>{value}</div>
    </div>
  );
}

function Slider({ id, label, value, min, max, step, onChange, display, hint }) {
  return (
    <div>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", gap: 8 }}>
        <label htmlFor={id} style={{ fontSize: 13, color: C.ink, fontWeight: 600 }}>{label}</label>
        <span style={{ fontFamily: "'IBM Plex Mono', ui-monospace, Menlo, monospace", fontSize: 13, fontWeight: 700, color: C.accent }}>{display}</span>
      </div>
      <input
        id={id}
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(+e.target.value)}
        aria-label={label}
        style={{ width: "100%", marginTop: 5 }}
      />
      {hint && <div style={{ fontSize: 11.5, color: C.muted, marginTop: 2, lineHeight: 1.4 }}>{hint}</div>}
    </div>
  );
}

function btnStyle(active, disabled) {
  return {
    padding: "8px 16px",
    borderRadius: 9,
    border: `1.5px solid ${active ? C.accent : C.border}`,
    background: active ? C.accent : C.card,
    color: active ? "#fff" : C.muted,
    fontSize: 13,
    fontWeight: 600,
    cursor: disabled ? "not-allowed" : "pointer",
    opacity: disabled ? 0.5 : 1,
    fontFamily: "inherit",
    transition: `transform 140ms ${EASE}, background 160ms ${EASE}, border-color 160ms ${EASE}`,
  };
}

const COUNTS = [60, 120, 200, 300, 400];

export default function App() {
  const canvasRef = useRef(null);
  const boidsRef = useRef([]);
  const predatorRef = useRef(null);
  const rafRef = useRef(0);
  const paramsRef = useRef(null);

  const [count, setCount] = useState(200);
  const [seed, setSeed] = useState(1337);
  const [playing, setPlaying] = useState(true);
  const [separationW, setSeparationW] = useState(1.5);
  const [alignW, setAlignW] = useState(1.0);
  const [cohesionW, setCohesionW] = useState(1.0);
  const [perception, setPerception] = useState(48);
  const [maxSpeed, setMaxSpeed] = useState(3.0);
  const [showNeighborhood, setShowNeighborhood] = useState(true);
  const [order, setOrder] = useState(0);
  const [avgSpeed, setAvgSpeed] = useState(0);

  useEffect(() => {
    paramsRef.current = { separationW, alignW, cohesionW, perception, maxSpeed };
  }, [separationW, alignW, cohesionW, perception, maxSpeed]);

  const reseed = useCallback(() => {
    boidsRef.current = makeFlock(count, seed);
    predatorRef.current = null;
    setOrder(orderParameter(boidsRef.current));
    setAvgSpeed(averageSpeed(boidsRef.current));
  }, [count, seed]);

  useEffect(() => { reseed(); }, [reseed]);

  const draw = useCallback(() => {
    const cv = canvasRef.current;
    if (!cv) return;
    const ctx = cv.getContext("2d");
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    if (cv.width !== WORLD_W * dpr) {
      cv.width = WORLD_W * dpr;
      cv.height = WORLD_H * dpr;
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);

    ctx.fillStyle = C.field;
    ctx.fillRect(0, 0, WORLD_W, WORLD_H);

    ctx.strokeStyle = C.fieldGrid;
    ctx.lineWidth = 1;
    const gridStep = 64;
    ctx.beginPath();
    for (let gx = gridStep; gx < WORLD_W; gx += gridStep) {
      ctx.moveTo(gx, 0); ctx.lineTo(gx, WORLD_H);
    }
    for (let gy = gridStep; gy < WORLD_H; gy += gridStep) {
      ctx.moveTo(0, gy); ctx.lineTo(WORLD_W, gy);
    }
    ctx.stroke();

    const boids = boidsRef.current;
    const p = paramsRef.current;

    if (showNeighborhood && boids.length > 0) {
      const hi = boids[0];
      ctx.fillStyle = "rgba(90,143,123,0.13)";
      ctx.strokeStyle = "rgba(90,143,123,0.55)";
      ctx.lineWidth = 1.25;
      ctx.beginPath();
      ctx.arc(hi.x, hi.y, p.perception, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();

      ctx.strokeStyle = "rgba(90,143,123,0.32)";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.arc(hi.x, hi.y, p.perception * SEP_RATIO, 0, Math.PI * 2);
      ctx.stroke();

      const percep2 = p.perception * p.perception;
      ctx.strokeStyle = "rgba(90,143,123,0.4)";
      ctx.lineWidth = 0.75;
      ctx.beginPath();
      for (let j = 1; j < boids.length; j++) {
        const dx = boids[j].x - hi.x;
        const dy = boids[j].y - hi.y;
        if (dx * dx + dy * dy <= percep2) {
          ctx.moveTo(hi.x, hi.y);
          ctx.lineTo(boids[j].x, boids[j].y);
        }
      }
      ctx.stroke();
    }

    for (let i = 0; i < boids.length; i++) {
      const b = boids[i];
      const ang = Math.atan2(b.vy, b.vx);
      const highlighted = showNeighborhood && i === 0;
      ctx.save();
      ctx.translate(b.x, b.y);
      ctx.rotate(ang);
      ctx.fillStyle = highlighted ? C.focus : C.boid;
      const s = highlighted ? 6.5 : 5;
      ctx.beginPath();
      ctx.moveTo(s, 0);
      ctx.lineTo(-s * 0.72, s * 0.6);
      ctx.lineTo(-s * 0.72, -s * 0.6);
      ctx.closePath();
      ctx.fill();
      ctx.restore();
    }

    const pred = predatorRef.current;
    if (pred) {
      ctx.fillStyle = C.accent;
      ctx.strokeStyle = "rgba(192,86,31,0.4)";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.arc(pred.x, pred.y, 110, 0, Math.PI * 2);
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(pred.x, pred.y, 7, 0, Math.PI * 2);
      ctx.fill();
    }
  }, [showNeighborhood]);

  useEffect(() => {
    const reduce = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;

    if (!playing) {
      draw();
      return;
    }

    let frame = 0;
    const loop = () => {
      const boids = boidsRef.current;
      if (boids.length) {
        stepFlock(boids, paramsRef.current, predatorRef.current);
        frame++;
        if (frame % 6 === 0) {
          setOrder(orderParameter(boids));
          setAvgSpeed(averageSpeed(boids));
        }
      }
      draw();
      rafRef.current = requestAnimationFrame(loop);
    };

    if (reduce) {
      const id = setInterval(() => {
        const boids = boidsRef.current;
        if (boids.length) {
          stepFlock(boids, paramsRef.current, predatorRef.current);
          setOrder(orderParameter(boids));
          setAvgSpeed(averageSpeed(boids));
        }
        draw();
      }, 250);
      return () => clearInterval(id);
    }

    rafRef.current = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(rafRef.current);
  }, [playing, draw]);

  const onCanvasClick = useCallback((e) => {
    const cv = canvasRef.current;
    if (!cv) return;
    const rect = cv.getBoundingClientRect();
    const x = ((e.clientX - rect.left) / rect.width) * WORLD_W;
    const y = ((e.clientY - rect.top) / rect.height) * WORLD_H;
    if (predatorRef.current) {
      predatorRef.current = null;
    } else {
      predatorRef.current = { x, y };
    }
    if (!playing) draw();
  }, [playing, draw]);

  const onCanvasMove = useCallback((e) => {
    if (!predatorRef.current) return;
    const cv = canvasRef.current;
    if (!cv) return;
    const rect = cv.getBoundingClientRect();
    predatorRef.current = {
      x: ((e.clientX - rect.left) / rect.width) * WORLD_W,
      y: ((e.clientY - rect.top) / rect.height) * WORLD_H,
    };
    if (!playing) draw();
  }, [playing, draw]);

  const orderLabel = order > 0.85 ? "aligned" : order > 0.45 ? "forming" : "disordered";
  const orderColor = order > 0.85 ? C.good : order > 0.45 ? C.accent : C.muted;

  return (
    <div style={{ fontFamily: "'Iowan Old Style', 'Palatino Linotype', Palatino, Georgia, serif", background: C.bg, minHeight: "100vh", padding: "30px 14px 56px", color: C.ink }}>
      <style>{`
        @media (prefers-reduced-motion: reduce) {
          .bd-btn { transition-duration: 1ms !important; }
        }
        .bd-btn:active { transform: scale(0.97); }
        .bd-btn:focus-visible, input[type=range]:focus-visible, input[type=checkbox]:focus-visible, canvas:focus-visible {
          outline: 2px solid ${C.accent}; outline-offset: 2px;
        }
        input[type=range] { accent-color: ${C.accent}; }
        input[type=checkbox] { accent-color: ${C.accent}; }
      `}</style>

      <div style={{ maxWidth: 720, margin: "0 auto" }}>
        <header style={{ marginBottom: 22 }}>
          <Eyebrow>Emergence · Agent-Based Models</Eyebrow>
          <h1 style={{ fontSize: 30, fontWeight: 700, margin: "6px 0 0", lineHeight: 1.1, textWrap: "balance" }}>
            Boids: Flocking From Local Rules
          </h1>
          <p style={{ color: C.muted, fontSize: 15, margin: "10px 0 0", lineHeight: 1.6, maxWidth: "62ch" }}>
            A flock looks choreographed, yet nothing choreographs it. In Craig Reynolds' 1986 model every boid sees
            only its near neighbors and follows three steering rules. The global pattern below is genuinely emergent:
            it falls out of those local interactions alone, with no leader steering and no path laid down in advance.
          </p>
        </header>

        <Card style={{ marginBottom: 18, padding: 0, overflow: "hidden" }}>
          <canvas
            ref={canvasRef}
            onClick={onCanvasClick}
            onMouseMove={onCanvasMove}
            tabIndex={0}
            role="img"
            aria-label="Animated flock of boids steering by local rules. Click to drop or move a predator they flee."
            style={{ width: "100%", height: "auto", aspectRatio: `${WORLD_W} / ${WORLD_H}`, display: "block", cursor: "crosshair" }}
          />
          <div style={{ display: "flex", gap: 10, flexWrap: "wrap", alignItems: "center", padding: "12px 14px", borderTop: `1px solid ${C.border}` }}>
            <button className="bd-btn" onClick={() => setPlaying((v) => !v)} style={btnStyle(true)} aria-pressed={playing}>
              {playing ? "Pause" : "Play"}
            </button>
            <button className="bd-btn" onClick={reseed} style={btnStyle(false)}>
              Reset flock
            </button>
            <button className="bd-btn" onClick={() => setSeed((s) => (s * 1664525 + 1013904223) >>> 0)} style={btnStyle(false)}>
              New seed
            </button>
            <div style={{ display: "flex", alignItems: "center", gap: 6, marginLeft: "auto", flexWrap: "wrap" }}>
              <span style={{ fontSize: 12, color: C.muted }}>Boids</span>
              {COUNTS.map((c) => (
                <button key={c} className="bd-btn" onClick={() => setCount(c)} style={{ ...btnStyle(count === c), padding: "5px 10px", fontSize: 12 }} aria-pressed={count === c}>
                  {c}
                </button>
              ))}
            </div>
          </div>
        </Card>

        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(120px, 1fr))", gap: 10, marginBottom: 18 }}>
          <Stat label="boids" value={count} />
          <Stat label="order param" value={order.toFixed(3)} accent />
          <Stat label="alignment" value={orderLabel} focus={order > 0.85} />
          <Stat label="avg speed" value={avgSpeed.toFixed(2)} />
        </div>

        <Card style={{ marginBottom: 18 }}>
          <Eyebrow>The three steering rules</Eyebrow>
          <div style={{ display: "flex", flexDirection: "column", gap: 18, marginTop: 14 }}>
            <Slider
              id="sep"
              label="Separation"
              value={separationW}
              min={0}
              max={3}
              step={0.1}
              onChange={setSeparationW}
              display={separationW.toFixed(1)}
              hint="Steer away from neighbors that crowd too close. Crank it up and the flock repels itself into open space."
            />
            <Slider
              id="ali"
              label="Alignment"
              value={alignW}
              min={0}
              max={3}
              step={0.1}
              onChange={setAlignW}
              display={alignW.toFixed(1)}
              hint="Steer toward the average heading of neighbors. This is what lifts the order parameter toward 1."
            />
            <Slider
              id="coh"
              label="Cohesion"
              value={cohesionW}
              min={0}
              max={3}
              step={0.1}
              onChange={setCohesionW}
              display={cohesionW.toFixed(1)}
              hint="Steer toward the average position of neighbors. Drop it to zero and the flock loses its center and disperses."
            />
            <Slider
              id="per"
              label="Perception radius"
              value={perception}
              min={20}
              max={110}
              step={2}
              onChange={setPerception}
              display={`${perception} px`}
              hint="How far each boid can sense. The separation zone is a fraction of this. Wider perception couples more boids into one decision."
            />
            <Slider
              id="spd"
              label="Max speed"
              value={maxSpeed}
              min={1}
              max={6}
              step={0.1}
              onChange={setMaxSpeed}
              display={maxSpeed.toFixed(1)}
              hint="Caps how fast a boid travels. Steering forces are clamped too, so turns stay smooth rather than instant."
            />
          </div>
          <label style={{ display: "flex", alignItems: "center", gap: 9, marginTop: 18, fontSize: 13, color: C.ink, cursor: "pointer" }}>
            <input
              type="checkbox"
              checked={showNeighborhood}
              onChange={(e) => setShowNeighborhood(e.target.checked)}
              style={{ width: 16, height: 16 }}
            />
            Highlight one boid and draw its perception radius, separation zone, and the neighbors it currently sees.
          </label>
        </Card>

        <Card style={{ marginBottom: 18, background: C.accentSoft, borderColor: `${C.accent}33` }}>
          <Eyebrow>Reading the order parameter</Eyebrow>
          <p style={{ fontSize: 13.5, color: C.ink, opacity: 0.86, margin: "10px 0 0", lineHeight: 1.6 }}>
            The order parameter is the length of the mean unit velocity:{" "}
            <span style={{ fontFamily: "'IBM Plex Mono', ui-monospace, Menlo, monospace", fontSize: 12.5 }}>
              phi = |&Sigma; v&#770;<sub>i</sub>| / N
            </span>
            . It is near 1 when every boid points the same way and near 0 when headings are random. Right now it reads{" "}
            <b style={{ color: orderColor }}>{order.toFixed(3)} ({orderLabel})</b>. Start from a fresh seed with alignment
            at zero and watch it sit low, then raise alignment and cohesion and watch it climb as a single flock takes shape.
          </p>
        </Card>

        <Card>
          <Eyebrow>Why a flock appears</Eyebrow>
          <p style={{ fontSize: 13.5, color: C.ink, opacity: 0.86, margin: "10px 0 0", lineHeight: 1.6 }}>
            Each boid runs the same local computation every frame: gather the neighbors inside its perception radius, then
            blend three steering vectors. <b>Separation</b> pushes away from anyone inside the tighter inner zone so bodies
            do not pile up. <b>Alignment</b> nudges its heading toward the average heading of those neighbors. <b>Cohesion</b>{" "}
            pulls it toward their average position. Each vector is normalized to the max speed, the correction is clamped to a
            small max force, and the result integrates into velocity and then position.
          </p>
          <p style={{ fontSize: 13.5, color: C.ink, opacity: 0.86, margin: "12px 0 0", lineHeight: 1.6 }}>
            No boid can see the whole flock and none is in charge. The coherent group is a fixed point of thousands of these
            myopic decisions interacting: alignment lets a heading spread neighbor to neighbor until a region agrees,
            cohesion keeps the region from drifting apart, and separation keeps it from collapsing to a point. Click the
            field to drop a predator; the boids that sense it inside its radius flee, the disturbance ripples outward through
            purely local steering, and the flock re-forms once it passes. Click again to remove it.
          </p>
        </Card>

        <p style={{ fontSize: 11.5, color: C.faint, marginTop: 20, textAlign: "center", lineHeight: 1.5 }}>
          Seeded with a mulberry32 PRNG so Reset reproduces the same start. After Craig W. Reynolds, "Flocks, Herds, and Schools" (1987).
        </p>
      </div>
    </div>
  );
}
