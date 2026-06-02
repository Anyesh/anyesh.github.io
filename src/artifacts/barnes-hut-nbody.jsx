import { useState, useEffect, useRef, useCallback } from "react";

export const meta = {
  title: "Barnes-Hut: Gravity in n log n",
  category: "Physics",
  description:
    "Hundreds of bodies pull on each other under real gravity. A quadtree rebuilt every frame lets distant clumps act as one mass, turning the brute-force n squared force sum into an n log n walk. Tune the opening angle and watch accuracy trade against speed.",
  date: "2026-04-13",
  tags: ["n-body", "barnes-hut", "quadtree", "gravity"],
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
  good: "#3f7d52",
  goodSoft: "#e3efe6",
  tree: "#9aa7bd",
  treeSoft: "#eef0f4",
  warm: "#d98a3d",
  cool: "#5d7392",
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

function randnFrom(rng) {
  let u = 0, v = 0;
  while (u === 0) u = rng();
  while (v === 0) v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

const WORLD = 1000;
const G = 4.0;

class Bodies {
  constructor(n) {
    this.n = n;
    this.x = new Float64Array(n);
    this.y = new Float64Array(n);
    this.vx = new Float64Array(n);
    this.vy = new Float64Array(n);
    this.ax = new Float64Array(n);
    this.ay = new Float64Array(n);
    this.m = new Float64Array(n);
  }
  set(i, x, y, vx, vy, m) {
    this.x[i] = x;
    this.y[i] = y;
    this.vx[i] = vx;
    this.vy[i] = vy;
    this.m[i] = m;
  }
}

function makeDisk(n, seed) {
  const rng = mulberry32(seed);
  const cx = WORLD / 2;
  const cy = WORLD / 2;
  const bodies = new Bodies(n);
  // Central body anchors the disk so the orbits have something to swing around.
  bodies.set(0, cx, cy, 0, 0, 900);
  for (let i = 1; i < n; i++) {
    const r = 40 + Math.sqrt(rng()) * 360;
    const ang = rng() * 2 * Math.PI;
    const x = cx + r * Math.cos(ang);
    const y = cy + r * Math.sin(ang);
    const mass = 1 + rng() * 2;
    // Circular orbit speed for the enclosed central mass, with a touch of scatter.
    const vc = Math.sqrt((G * 900) / r);
    const jitter = 0.06;
    const vx = -Math.sin(ang) * vc * (1 + randnFrom(rng) * jitter);
    const vy = Math.cos(ang) * vc * (1 + randnFrom(rng) * jitter);
    bodies.set(i, x, y, vx, vy, mass);
  }
  return bodies;
}

function makeCollision(n, seed) {
  const rng = mulberry32(seed);
  const bodies = new Bodies(n);
  const half = Math.floor(n / 2);
  const clumps = [
    { cx: WORLD * 0.32, cy: WORLD * 0.42, vx: 22, vy: 6 },
    { cx: WORLD * 0.68, cy: WORLD * 0.58, vx: -22, vy: -6 },
  ];
  for (let i = 0; i < n; i++) {
    const c = clumps[i < half ? 0 : 1];
    const r = Math.sqrt(rng()) * 150;
    const ang = rng() * 2 * Math.PI;
    const x = c.cx + r * Math.cos(ang);
    const y = c.cy + r * Math.sin(ang);
    const mass = 1 + rng() * 2;
    // Rotation inside each clump plus the clump's bulk drift toward the other.
    const vc = Math.sqrt((G * 200) / (r + 30)) * 0.8;
    const vx = c.vx - Math.sin(ang) * vc;
    const vy = c.vy + Math.cos(ang) * vc;
    bodies.set(i, x, y, vx, vy, mass);
  }
  return bodies;
}

// Flat-array quadtree. A node is internal when it has children; mass and the
// center-of-mass (comx, comy) are accumulated as bodies are inserted so the
// force walk can treat a whole subtree as one point mass.
class QuadTree {
  constructor(cap) {
    this.cap = cap;
    this.cx = new Float64Array(cap);
    this.cy = new Float64Array(cap);
    this.half = new Float64Array(cap);
    this.mass = new Float64Array(cap);
    this.comx = new Float64Array(cap);
    this.comy = new Float64Array(cap);
    this.body = new Int32Array(cap);
    this.child = new Int32Array(cap * 4);
    this.nodes = 0;
  }
  reset(cx, cy, half) {
    this.nodes = 1;
    this.cx[0] = cx;
    this.cy[0] = cy;
    this.half[0] = half;
    this.mass[0] = 0;
    this.comx[0] = 0;
    this.comy[0] = 0;
    this.body[0] = -1;
    this.child[0] = this.child[1] = this.child[2] = this.child[3] = -1;
  }
  _alloc(cx, cy, half) {
    const i = this.nodes++;
    if (i >= this.cap) this._grow();
    this.cx[i] = cx;
    this.cy[i] = cy;
    this.half[i] = half;
    this.mass[i] = 0;
    this.comx[i] = 0;
    this.comy[i] = 0;
    this.body[i] = -1;
    this.child[i * 4] = this.child[i * 4 + 1] = this.child[i * 4 + 2] = this.child[i * 4 + 3] = -1;
    return i;
  }
  _grow() {
    const cap = this.cap * 2;
    const grow1 = (arr) => {
      const next = new arr.constructor(cap);
      next.set(arr);
      return next;
    };
    this.cx = grow1(this.cx);
    this.cy = grow1(this.cy);
    this.half = grow1(this.half);
    this.mass = grow1(this.mass);
    this.comx = grow1(this.comx);
    this.comy = grow1(this.comy);
    this.body = grow1(this.body);
    const nc = new Int32Array(cap * 4);
    nc.set(this.child);
    this.child = nc;
    this.cap = cap;
  }
  _isLeaf(node) {
    const b = node * 4;
    return this.child[b] === -1 && this.child[b + 1] === -1 && this.child[b + 2] === -1 && this.child[b + 3] === -1;
  }
  _quadrant(node, x, y) {
    const right = x >= this.cx[node] ? 1 : 0;
    const bottom = y >= this.cy[node] ? 1 : 0;
    return bottom * 2 + right;
  }
  _childNode(node, q) {
    const h = this.half[node] / 2;
    const right = q & 1;
    const bottom = (q >> 1) & 1;
    const ncx = this.cx[node] + (right ? h : -h);
    const ncy = this.cy[node] + (bottom ? h : -h);
    return this._alloc(ncx, ncy, h);
  }
  insert(bodies, idx) {
    let node = 0;
    const x = bodies.x[idx];
    const y = bodies.y[idx];
    const mi = bodies.m[idx];
    while (true) {
      this.mass[node] += mi;
      this.comx[node] += x * mi;
      this.comy[node] += y * mi;
      if (this._isLeaf(node) && this.body[node] === -1) {
        this.body[node] = idx;
        return;
      }
      if (this._isLeaf(node)) {
        const existing = this.body[node];
        this.body[node] = -1;
        const eq = this._quadrant(node, bodies.x[existing], bodies.y[existing]);
        const ec = this._childNode(node, eq);
        this.child[node * 4 + eq] = ec;
        this.mass[ec] += bodies.m[existing];
        this.comx[ec] += bodies.x[existing] * bodies.m[existing];
        this.comy[ec] += bodies.y[existing] * bodies.m[existing];
        this.body[ec] = existing;
      }
      const q = this._quadrant(node, x, y);
      let c = this.child[node * 4 + q];
      if (c === -1) {
        c = this._childNode(node, q);
        this.child[node * 4 + q] = c;
      }
      node = c;
    }
  }
  finalize() {
    for (let i = 0; i < this.nodes; i++) {
      if (this.mass[i] > 0) {
        this.comx[i] /= this.mass[i];
        this.comy[i] /= this.mass[i];
      }
    }
  }
}

function buildTree(tree, bodies) {
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (let i = 0; i < bodies.n; i++) {
    if (bodies.x[i] < minX) minX = bodies.x[i];
    if (bodies.y[i] < minY) minY = bodies.y[i];
    if (bodies.x[i] > maxX) maxX = bodies.x[i];
    if (bodies.y[i] > maxY) maxY = bodies.y[i];
  }
  const cx = (minX + maxX) / 2;
  const cy = (minY + maxY) / 2;
  const half = Math.max(maxX - minX, maxY - minY) / 2 + 1;
  tree.reset(cx, cy, half);
  for (let i = 0; i < bodies.n; i++) tree.insert(bodies, i);
  tree.finalize();
}

// Walks the tree for one body. A node whose width s over distance d is below
// theta is accepted as a single mass; otherwise we recurse. eps2 is the squared
// softening length: it is added inside the r^2 term so two bodies at near-zero
// separation feel a finite force instead of a singularity that blows the
// integrator up. Returns the force-evaluation count for the O(n log n) readout.
function accelOnBody(tree, bodies, idx, theta2, eps2, out) {
  const bx = bodies.x[idx];
  const by = bodies.y[idx];
  let ax = 0, ay = 0;
  let evals = 0;
  const stack = out.stack;
  let sp = 0;
  stack[sp++] = 0;
  while (sp > 0) {
    const node = stack[--sp];
    const m = tree.mass[node];
    if (m === 0) continue;
    const dx = tree.comx[node] - bx;
    const dy = tree.comy[node] - by;
    const dist2 = dx * dx + dy * dy;
    const leaf = tree._isLeaf(node);
    const s = tree.half[node] * 2;
    if (leaf || s * s < theta2 * dist2) {
      if (leaf && tree.body[node] === idx) continue;
      evals++;
      const inv = 1 / Math.sqrt(dist2 + eps2);
      const inv3 = inv * inv * inv;
      const f = G * m * inv3;
      ax += f * dx;
      ay += f * dy;
    } else {
      const base = node * 4;
      for (let q = 0; q < 4; q++) {
        const c = tree.child[base + q];
        if (c !== -1) {
          if (sp >= stack.length) {
            const grown = new Int32Array(stack.length * 2);
            grown.set(stack);
            out.stack = grown;
            return accelOnBody(tree, bodies, idx, theta2, eps2, out);
          }
          stack[sp++] = c;
        }
      }
    }
  }
  out.ax = ax;
  out.ay = ay;
  return evals;
}

function computeForces(tree, bodies, theta, eps2, work) {
  const theta2 = theta * theta;
  let evals = 0;
  for (let i = 0; i < bodies.n; i++) {
    evals += accelOnBody(tree, bodies, i, theta2, eps2, work);
    bodies.ax[i] = work.ax;
    bodies.ay[i] = work.ay;
  }
  return evals;
}

// Velocity-Verlet: half-kick the velocity, drift the position a full step,
// recompute the force, then half-kick again. Symplectic, so the orbital energy
// stays bounded over long runs rather than drifting the way naive Euler does.
function stepVerlet(tree, bodies, theta, eps2, dt, work) {
  const n = bodies.n;
  for (let i = 0; i < n; i++) {
    bodies.vx[i] += 0.5 * bodies.ax[i] * dt;
    bodies.vy[i] += 0.5 * bodies.ay[i] * dt;
    bodies.x[i] += bodies.vx[i] * dt;
    bodies.y[i] += bodies.vy[i] * dt;
  }
  buildTree(tree, bodies);
  const evals = computeForces(tree, bodies, theta, eps2, work);
  for (let i = 0; i < n; i++) {
    bodies.vx[i] += 0.5 * bodies.ax[i] * dt;
    bodies.vy[i] += 0.5 * bodies.ay[i] * dt;
  }
  return evals;
}

function totalEnergy(tree, bodies, eps2) {
  let ke = 0;
  for (let i = 0; i < bodies.n; i++) {
    ke += 0.5 * bodies.m[i] * (bodies.vx[i] ** 2 + bodies.vy[i] ** 2);
  }
  let pe = 0;
  const n = bodies.n;
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const dx = bodies.x[j] - bodies.x[i];
      const dy = bodies.y[j] - bodies.y[i];
      const r = Math.sqrt(dx * dx + dy * dy + eps2);
      pe -= (G * bodies.m[i] * bodies.m[j]) / r;
    }
  }
  return ke + pe;
}

function collectCells(tree, maxCells) {
  const cells = [];
  for (let i = 0; i < tree.nodes && cells.length < maxCells; i++) {
    if (tree.mass[i] === 0) continue;
    cells.push([tree.cx[i] - tree.half[i], tree.cy[i] - tree.half[i], tree.half[i] * 2]);
  }
  return cells;
}

const SCENES = {
  disk: { label: "Rotating disk", make: makeDisk },
  collision: { label: "Two clusters", make: makeCollision },
};

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

function Stat({ label, value, sub, accent }) {
  return (
    <div style={{ background: accent ? C.accentSoft : C.bg, borderRadius: 9, padding: "9px 12px", border: `1px solid ${accent ? C.accent + "33" : C.border}` }}>
      <div style={{ fontSize: 10, letterSpacing: "0.06em", textTransform: "uppercase", color: accent ? C.accent : C.muted, fontWeight: 700, marginBottom: 3 }}>{label}</div>
      <div style={{ fontFamily: "'IBM Plex Mono', ui-monospace, Menlo, monospace", fontSize: 17, fontWeight: 700, color: accent ? C.accent : C.ink, lineHeight: 1.1 }}>{value}</div>
      {sub && <div style={{ fontSize: 10.5, color: C.muted, marginTop: 2 }}>{sub}</div>}
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

function Slider({ id, label, value, min, max, step, onChange, format }) {
  return (
    <div>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 5 }}>
        <label htmlFor={id} style={{ fontSize: 12.5, color: C.muted }}>{label}</label>
        <span style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: 13, fontWeight: 700, color: C.accent }}>{format(value)}</span>
      </div>
      <input id={id} type="range" min={min} max={max} step={step} value={value}
        onChange={(e) => onChange(+e.target.value)} style={{ width: "100%" }}
        aria-label={label} />
    </div>
  );
}

const VIEW = 360;
const COUNTS = [100, 200, 400, 600, 800];

export default function App() {
  const [scene, setScene] = useState("disk");
  const [seed, setSeed] = useState(1);
  const [count, setCount] = useState(400);
  const [theta, setTheta] = useState(0.7);
  const [dt, setDt] = useState(0.35);
  const [softening, setSoftening] = useState(6);
  const [showTree, setShowTree] = useState(true);
  const [playing, setPlaying] = useState(true);

  const [readout, setReadout] = useState({ evals: 0, nodes: 0, fps: 0, energy: 0 });

  const canvasRef = useRef(null);
  const bodiesRef = useRef(null);
  const treeRef = useRef(null);
  const workRef = useRef(null);
  const rafRef = useRef(0);
  const energy0Ref = useRef(0);
  const paramsRef = useRef({ theta, dt, softening, showTree });

  useEffect(() => { paramsRef.current = { theta, dt, softening, showTree }; }, [theta, dt, softening, showTree]);

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    const bodies = bodiesRef.current;
    const tree = treeRef.current;
    if (!canvas || !bodies || !tree) return;
    const ctx = canvas.getContext("2d");
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    if (canvas.width !== VIEW * dpr) {
      canvas.width = VIEW * dpr;
      canvas.height = VIEW * dpr;
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, VIEW, VIEW);
    ctx.fillStyle = "#fbfaf8";
    ctx.fillRect(0, 0, VIEW, VIEW);

    const s = VIEW / WORLD;

    if (paramsRef.current.showTree) {
      ctx.strokeStyle = C.tree;
      ctx.globalAlpha = 0.4;
      ctx.lineWidth = 0.5;
      const cells = collectCells(tree, 8000);
      ctx.beginPath();
      for (const [x, y, w] of cells) ctx.rect(x * s, y * s, w * s, w * s);
      ctx.stroke();
      ctx.globalAlpha = 1;
    }

    for (let i = 0; i < bodies.n; i++) {
      const px = bodies.x[i] * s;
      const py = bodies.y[i] * s;
      const m = bodies.m[i];
      const heavy = m > 50;
      const r = heavy ? 4 : 0.7 + Math.min(m, 4) * 0.45;
      const speed = Math.sqrt(bodies.vx[i] ** 2 + bodies.vy[i] ** 2);
      ctx.beginPath();
      ctx.arc(px, py, r, 0, 2 * Math.PI);
      ctx.fillStyle = heavy ? C.accent : speed > 14 ? C.warm : C.cool;
      ctx.globalAlpha = heavy ? 1 : 0.85;
      ctx.fill();
    }
    ctx.globalAlpha = 1;
  }, []);

  const reseed = useCallback((nextSeed) => {
    const s = nextSeed === undefined ? seed : nextSeed;
    const bodies = SCENES[scene].make(count, s);
    bodiesRef.current = bodies;
    const tree = new QuadTree(Math.max(64, count * 4));
    treeRef.current = tree;
    workRef.current = { ax: 0, ay: 0, stack: new Int32Array(512) };
    buildTree(tree, bodies);
    const eps2 = softening * softening;
    computeForces(tree, bodies, theta, eps2, workRef.current);
    energy0Ref.current = totalEnergy(tree, bodies, eps2);
    setReadout({ evals: 0, nodes: tree.nodes, fps: 0, energy: 0 });
    draw();
    // theta, dt and softening are read live from paramsRef inside the loop, so
    // they must not retrigger a reseed (that would reset the bodies on a drag).
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [scene, count, seed, draw]);

  useEffect(() => {
    reseed();
    return () => cancelAnimationFrame(rafRef.current);
  }, [reseed]);

  useEffect(() => {
    if (!playing) {
      cancelAnimationFrame(rafRef.current);
      return;
    }
    let frames = 0;
    let fpsClock = performance.now();
    let sinceEnergy = 0;
    let lastEnergy = 0;

    const loop = () => {
      const bodies = bodiesRef.current;
      const tree = treeRef.current;
      const work = workRef.current;
      const p = paramsRef.current;
      if (bodies && tree && work) {
        const eps2 = p.softening * p.softening;
        const evals = stepVerlet(tree, bodies, p.theta, eps2, p.dt, work);
        draw();
        frames++;
        sinceEnergy++;
        const now = performance.now();
        if (now - fpsClock >= 500) {
          const fps = (frames * 1000) / (now - fpsClock);
          if (sinceEnergy > 40) {
            const e = totalEnergy(tree, bodies, eps2);
            lastEnergy = Math.abs((e - energy0Ref.current) / (Math.abs(energy0Ref.current) + 1e-9)) * 100;
            sinceEnergy = 0;
          }
          setReadout({ evals, nodes: tree.nodes, fps: Math.round(fps), energy: lastEnergy });
          frames = 0;
          fpsClock = now;
        }
      }
      rafRef.current = requestAnimationFrame(loop);
    };
    rafRef.current = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(rafRef.current);
  }, [playing, draw]);

  const stepOnce = useCallback(() => {
    const bodies = bodiesRef.current;
    const tree = treeRef.current;
    const work = workRef.current;
    const p = paramsRef.current;
    if (!bodies || !tree || !work) return;
    const eps2 = p.softening * p.softening;
    const evals = stepVerlet(tree, bodies, p.theta, eps2, p.dt, work);
    draw();
    const e = totalEnergy(tree, bodies, eps2);
    const energy = Math.abs((e - energy0Ref.current) / (Math.abs(energy0Ref.current) + 1e-9)) * 100;
    setReadout({ evals, nodes: tree.nodes, fps: 0, energy });
  }, [draw]);

  const n2 = count * (count - 1);
  const speedup = readout.evals > 0 ? n2 / readout.evals : 0;

  return (
    <div style={{ fontFamily: "'Iowan Old Style', 'Palatino Linotype', Palatino, Georgia, serif", background: C.bg, minHeight: "100vh", padding: "30px 14px 56px", color: C.ink }}>
      <style>{`
        @media (prefers-reduced-motion: reduce) {
          * { transition-duration: 1ms !important; animation-duration: 1ms !important; }
        }
        .bh-btn:active:not(:disabled) { transform: scale(0.97); }
        .bh-btn:focus-visible, input[type=range]:focus-visible, input[type=checkbox]:focus-visible {
          outline: 2px solid ${C.accent}; outline-offset: 2px;
        }
        input[type=range] { accent-color: ${C.accent}; }
        input[type=checkbox] { accent-color: ${C.accent}; }
      `}</style>

      <div style={{ maxWidth: 720, margin: "0 auto" }}>
        <header style={{ marginBottom: 22 }}>
          <Eyebrow>Physics · Gravitational Dynamics</Eyebrow>
          <h1 style={{ fontSize: 30, fontWeight: 700, margin: "6px 0 0", lineHeight: 1.1, textWrap: "balance" }}>
            Barnes-Hut: Gravity in n log n
          </h1>
          <p style={{ color: C.ink, opacity: 0.86, fontSize: 15, margin: "10px 0 0", lineHeight: 1.6, maxWidth: "64ch" }}>
            Every body pulls on every other one. Summing all those pairs costs n squared work per frame, which
            stalls past a few hundred bodies. Barnes-Hut rebuilds a quadtree each frame and lets a distant clump of
            stars act as a single point mass, so the force on one body costs a walk of order log n, and hundreds of
            bodies run in real time. The control is the opening angle theta, which decides how distant is distant
            enough.
          </p>
        </header>

        <Card style={{ marginBottom: 18, padding: 16 }}>
          <div style={{ display: "flex", gap: 18, flexWrap: "wrap", alignItems: "flex-start" }}>
            <div style={{ flex: "0 0 auto", margin: "0 auto" }}>
              <canvas
                ref={canvasRef}
                style={{ width: VIEW, height: VIEW, maxWidth: "100%", borderRadius: 12, background: "#fbfaf8", border: `1px solid ${C.border}`, display: "block", touchAction: "none" }}
                role="img"
                aria-label="Live N-body gravity simulation with quadtree cell overlay"
              />
            </div>

            <div style={{ flex: "1 1 240px", minWidth: 220 }}>
              <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 14 }}>
                <button className="bh-btn" onClick={() => setPlaying((p) => !p)} style={btnStyle(true)} aria-pressed={playing}>
                  {playing ? "Pause" : "Play"}
                </button>
                <button className="bh-btn" onClick={() => { setPlaying(false); stepOnce(); }} style={btnStyle(false)}>
                  Step
                </button>
                <button className="bh-btn" onClick={() => reseed()} style={btnStyle(false)}>
                  Reset
                </button>
                <button className="bh-btn" onClick={() => setSeed((v) => v + 1)} style={btnStyle(false)}>
                  Reseed
                </button>
              </div>

              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8, marginBottom: 14 }}>
                <Stat label="bodies" value={count} />
                <Stat label="tree nodes" value={readout.nodes} />
                <Stat label="force evals / frame" value={readout.evals.toLocaleString()} accent sub={`brute force ${n2.toLocaleString()}`} />
                <Stat label="speedup vs n squared" value={speedup ? `${speedup.toFixed(1)}x` : "-"} />
                <Stat label="fps" value={readout.fps || "-"} />
                <Stat label="energy drift" value={`${readout.energy.toFixed(2)}%`} sub="vs start" />
              </div>

              <label style={{ display: "flex", alignItems: "center", gap: 9, fontSize: 13, color: C.ink, cursor: "pointer" }}>
                <input type="checkbox" checked={showTree} onChange={(e) => setShowTree(e.target.checked)} style={{ width: 16, height: 16 }} />
                Show quadtree cells
              </label>
            </div>
          </div>
        </Card>

        <Card style={{ marginBottom: 18 }}>
          <Eyebrow>Initial conditions</Eyebrow>
          <div style={{ display: "flex", gap: 8, flexWrap: "wrap", margin: "12px 0 16px", alignItems: "center" }}>
            {Object.entries(SCENES).map(([k, v]) => (
              <button key={k} className="bh-btn" onClick={() => setScene(k)} style={btnStyle(scene === k)} aria-pressed={scene === k}>
                {v.label}
              </button>
            ))}
            <div style={{ width: 1, alignSelf: "stretch", background: C.border, margin: "0 4px" }} aria-hidden="true" />
            {COUNTS.map((c) => (
              <button key={c} className="bh-btn" onClick={() => setCount(c)} style={{ ...btnStyle(count === c), padding: "8px 12px" }} aria-pressed={count === c} aria-label={`${c} bodies`}>
                {c}
              </button>
            ))}
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(200px, 1fr))", gap: 16 }}>
            <Slider id="theta" label="Opening angle theta" value={theta} min={0} max={1.5} step={0.05}
              onChange={setTheta} format={(v) => v.toFixed(2)} />
            <Slider id="dt" label="Timestep dt" value={dt} min={0.05} max={0.8} step={0.05}
              onChange={setDt} format={(v) => v.toFixed(2)} />
            <Slider id="soft" label="Softening epsilon" value={softening} min={1} max={20} step={1}
              onChange={setSoftening} format={(v) => v.toFixed(0)} />
          </div>
        </Card>

        <Card style={{ marginBottom: 18, background: C.accentSoft, borderColor: `${C.accent}33` }}>
          <Eyebrow>The opening-angle criterion</Eyebrow>
          <div style={{
            background: C.card, borderRadius: 9, padding: "11px 14px", margin: "12px 0",
            fontFamily: "'IBM Plex Mono', ui-monospace, Menlo, monospace",
            fontSize: 13, color: C.ink, border: `1px solid ${C.accent}22`, lineHeight: 1.8,
          }}>
            if s / d &lt; theta : treat the node as one mass at its center of gravity<br />
            else : open the node and walk its four children
          </div>
          <p style={{ fontSize: 13.5, color: C.ink, opacity: 0.86, margin: "0 0 8px", lineHeight: 1.6 }}>
            For each body the walk starts at the root cell. <b>s</b> is the cell width and <b>d</b> is the distance
            from the body to the cell's center of mass. A small s over d means the cell sits far away and tight on
            the sky, so the bodies inside it are well approximated by their combined mass at one point. A large ratio
            means the cell looms large, and lumping it would be inaccurate, so the walk descends into the four
            children and asks again. Because each descent quarters the area, a body reaches the approximation after
            about log n steps instead of touching all n bodies.
          </p>
          <p style={{ fontSize: 13.5, color: C.ink, opacity: 0.86, margin: 0, lineHeight: 1.6 }}>
            <b>theta = 0</b> never accepts an approximation, so the walk opens every cell down to single bodies and
            you get exact brute-force gravity at n squared cost: watch the force-eval count climb to match the brute
            figure. Raising theta accepts coarser groupings, cutting the eval count and lifting the frame rate while
            adding a small force error. Galaxy simulations usually sit near theta = 0.5 to 1.0, which is the trade
            this slider lets you feel directly.
          </p>
        </Card>

        <Card style={{ marginBottom: 18 }}>
          <Eyebrow>Real gravity, stable orbits</Eyebrow>
          <div style={{
            background: C.bg, borderRadius: 9, padding: "11px 14px", margin: "12px 0",
            fontFamily: "'IBM Plex Mono', ui-monospace, Menlo, monospace",
            fontSize: 13, color: C.ink, border: `1px solid ${C.border}`, lineHeight: 1.8,
          }}>
            F = G m1 m2 / (r^2 + epsilon^2)
          </div>
          <p style={{ fontSize: 13.5, color: C.ink, opacity: 0.86, margin: "0 0 8px", lineHeight: 1.6 }}>
            The force law is Newtonian with a softening length epsilon added inside the denominator. Without it, two
            bodies that pass close share a near-zero separation, the 1 over r squared term explodes, and the
            integrator flings them to infinity in a single step. Softening caps the closest-approach force so the run
            stays well behaved. Larger epsilon makes the gravity gentler and the disk puffier.
          </p>
          <p style={{ fontSize: 13.5, color: C.ink, opacity: 0.86, margin: 0, lineHeight: 1.6 }}>
            Integration is velocity-Verlet, a symplectic scheme that half-kicks the velocity, drifts the position,
            recomputes the force, then half-kicks again. Symplectic integrators keep the total energy bounded over
            long runs rather than letting it bleed away the way plain Euler does, which is why the orbits hold their
            shape instead of spiraling in or flying apart. The energy-drift readout tracks how far the total energy
            has wandered from its starting value. Shrink the timestep to tighten it.
          </p>
        </Card>

        <div style={{ fontSize: 12, color: C.muted, textAlign: "center", lineHeight: 1.6 }}>
          Quadtree rebuilt every frame · Barnes-Hut force walk with adjustable theta · velocity-Verlet integration ·
          seeded with mulberry32 so every reseed is reproducible
        </div>
      </div>
    </div>
  );
}
