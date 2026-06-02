import { useState, useEffect, useRef, useMemo, useCallback } from "react";
import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceLine, Legend } from "recharts";

export const meta = {
  title: "Diffusion: Noise and Denoise",
  category: "Machine Learning",
  description:
    "Destroy a shape into pure noise, then rebuild it. Watch a point cloud dissolve under the forward process, then a score-based sampler walk it back to the data one small step at a time.",
  date: "2026-02-15",
  tags: ["diffusion", "generative-models", "ddpm", "score"],
};

const C = {
  bg: "#f7f5f2",
  card: "#ffffff",
  border: "#e6e1d9",
  ink: "#1b1916",
  muted: "#857c70",
  accent: "#c0561f",
  accentSoft: "#f6ece5",
  data: "#c0561f",
  noise: "#7d8aa0",
  grid: "#efebe4",
  good: "#3f7d52",
};

const EASE = "cubic-bezier(0.23, 1, 0.32, 1)";

function randn() {
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

const N_COMPONENTS = 6;
const CLUSTER_STD = 0.16;
const RING_RADIUS = 2.1;

const MEANS = Array.from({ length: N_COMPONENTS }, (_, k) => {
  const a = (2 * Math.PI * k) / N_COMPONENTS - Math.PI / 2;
  return [RING_RADIUS * Math.cos(a), RING_RADIUS * Math.sin(a)];
});

function sampleX0() {
  const k = Math.floor(Math.random() * N_COMPONENTS);
  return [MEANS[k][0] + randn() * CLUSTER_STD, MEANS[k][1] + randn() * CLUSTER_STD];
}

const DATA_POINTS = Array.from({ length: 320 }, sampleX0);

function linearSchedule(T) {
  const b1 = 1e-4, bT = 0.02;
  const betas = Array.from({ length: T }, (_, i) => b1 + (bT - b1) * (i / (T - 1)));
  return finishSchedule(betas);
}

function cosineSchedule(T) {
  const s = 0.008;
  const f = (t) => Math.cos(((t / T + s) / (1 + s)) * (Math.PI / 2)) ** 2;
  const f0 = f(0);
  const abar = Array.from({ length: T }, (_, i) => f(i + 1) / f0);
  const betas = [];
  let prev = 1;
  for (let i = 0; i < T; i++) {
    const b = 1 - abar[i] / prev;
    betas.push(Math.min(Math.max(b, 1e-5), 0.999));
    prev = abar[i];
  }
  return finishSchedule(betas);
}

function finishSchedule(betas) {
  const T = betas.length;
  const alphas = betas.map((b) => 1 - b);
  const abar = new Array(T);
  let acc = 1;
  for (let i = 0; i < T; i++) {
    acc *= alphas[i];
    abar[i] = acc;
  }
  return { betas, alphas, abar, T };
}

function buildSchedule(kind, T) {
  return kind === "cosine" ? cosineSchedule(T) : linearSchedule(T);
}

function gmmScore(x, y, abart) {
  const sa = Math.sqrt(abart);
  const vt = abart * CLUSTER_STD * CLUSTER_STD + (1 - abart);
  let maxLog = -Infinity;
  const logs = new Array(N_COMPONENTS);
  for (let k = 0; k < N_COMPONENTS; k++) {
    const mx = sa * MEANS[k][0];
    const my = sa * MEANS[k][1];
    const d2 = (x - mx) ** 2 + (y - my) ** 2;
    const lg = -d2 / (2 * vt);
    logs[k] = lg;
    if (lg > maxLog) maxLog = lg;
  }
  let z = 0;
  const r = new Array(N_COMPONENTS);
  for (let k = 0; k < N_COMPONENTS; k++) {
    r[k] = Math.exp(logs[k] - maxLog);
    z += r[k];
  }
  let sx = 0, sy = 0;
  for (let k = 0; k < N_COMPONENTS; k++) {
    const w = r[k] / z;
    sx += (w * (sa * MEANS[k][0] - x)) / vt;
    sy += (w * (sa * MEANS[k][1] - y)) / vt;
  }
  return [sx, sy];
}

function forwardCloud(points, abart) {
  const sa = Math.sqrt(abart);
  const sn = Math.sqrt(1 - abart);
  return points.map(([x, y]) => [sa * x + sn * randn(), sa * y + sn * randn()]);
}

function ddpmStep(x, y, t, sched, stochastic) {
  const { betas, alphas, abar } = sched;
  const abart = abar[t];
  const at = alphas[t];
  const bt = betas[t];
  const [sx, sy] = gmmScore(x, y, abart);
  const epsX = -Math.sqrt(1 - abart) * sx;
  const epsY = -Math.sqrt(1 - abart) * sy;
  const coef = bt / Math.sqrt(1 - abart);
  const inv = 1 / Math.sqrt(at);
  let mx = inv * (x - coef * epsX);
  let my = inv * (y - coef * epsY);
  if (stochastic && t > 0) {
    const abarPrev = abar[t - 1];
    const sigma = Math.sqrt(((1 - abarPrev) / (1 - abart)) * bt);
    mx += sigma * randn();
    my += sigma * randn();
  }
  return [mx, my];
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

function Formula({ children }) {
  return (
    <div style={{
      background: C.bg, borderRadius: 9, padding: "9px 13px",
      fontFamily: "'IBM Plex Mono', ui-monospace, SFMono-Regular, Menlo, monospace",
      fontSize: 12.5, color: C.ink, border: `1px solid ${C.border}`, lineHeight: 1.7, overflowX: "auto",
    }}>
      {children}
    </div>
  );
}

function btnStyle(active) {
  return {
    padding: "8px 16px", borderRadius: 9,
    border: `1.5px solid ${active ? C.accent : C.border}`,
    background: active ? C.accent : C.card,
    color: active ? "#fff" : C.muted,
    fontSize: 13, fontWeight: 600, cursor: "pointer", fontFamily: "inherit",
    transition: `transform 140ms ${EASE}, background 160ms ${EASE}, border-color 160ms ${EASE}`,
  };
}

function PointCanvas({ points, extent, color, trails, width = 320, height = 320 }) {
  const ref = useRef(null);
  useEffect(() => {
    const cv = ref.current;
    if (!cv) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    cv.width = width * dpr;
    cv.height = height * dpr;
    const ctx = cv.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);

    const toPx = (x, y) => [
      width / 2 + (x / extent) * (width / 2) * 0.9,
      height / 2 - (y / extent) * (height / 2) * 0.9,
    ];

    ctx.strokeStyle = C.grid;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(width / 2, 8); ctx.lineTo(width / 2, height - 8);
    ctx.moveTo(8, height / 2); ctx.lineTo(width - 8, height / 2);
    ctx.stroke();

    if (trails) {
      ctx.lineWidth = 1;
      for (const tr of trails) {
        if (tr.length < 2) continue;
        ctx.strokeStyle = "rgba(192,86,31,0.16)";
        ctx.beginPath();
        for (let i = 0; i < tr.length; i++) {
          const [px, py] = toPx(tr[i][0], tr[i][1]);
          if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
        }
        ctx.stroke();
      }
    }

    ctx.fillStyle = color;
    for (const [x, y] of points) {
      const [px, py] = toPx(x, y);
      ctx.beginPath();
      ctx.arc(px, py, trails ? 3 : 2.1, 0, 2 * Math.PI);
      ctx.fill();
    }
  }, [points, extent, color, trails, width, height]);

  return (
    <canvas
      ref={ref}
      style={{ width, height, maxWidth: "100%", borderRadius: 10, background: C.bg, border: `1px solid ${C.border}`, display: "block" }}
      role="img"
      aria-label="Two dimensional scatter of sample points"
    />
  );
}

function ScheduleChart({ sched, t }) {
  const data = useMemo(() => {
    const T = sched.T;
    const stride = Math.max(1, Math.floor(T / 80));
    const out = [];
    for (let i = 0; i < T; i += stride) {
      out.push({ t: i, beta: sched.betas[i], abar: sched.abar[i], snr: sched.abar[i] / (1 - sched.abar[i] + 1e-9) });
    }
    if (out[out.length - 1].t !== T - 1) {
      const i = T - 1;
      out.push({ t: i, beta: sched.betas[i], abar: sched.abar[i], snr: sched.abar[i] / (1 - sched.abar[i] + 1e-9) });
    }
    return out;
  }, [sched]);

  return (
    <ResponsiveContainer width="100%" height={170}>
      <LineChart data={data} margin={{ left: -8, right: 8, top: 6, bottom: 2 }}>
        <XAxis dataKey="t" tick={{ fontSize: 10, fill: C.muted }} stroke={C.border}
          label={{ value: "step t", position: "insideBottom", offset: -1, fontSize: 10, fill: C.muted }} />
        <YAxis domain={[0, 1]} tick={{ fontSize: 10, fill: C.muted }} stroke={C.border} width={34} />
        <Tooltip
          contentStyle={{ fontSize: 11, borderRadius: 8, border: `1px solid ${C.border}` }}
          formatter={(v, n) => [Number(v).toFixed(4), n === "abar" ? "alpha_bar_t" : "beta_t"]}
          labelFormatter={(l) => `t = ${l}`}
        />
        <Legend wrapperStyle={{ fontSize: 11 }} formatter={(v) => (v === "abar" ? "alpha_bar_t (signal kept)" : "beta_t (noise added)")} />
        <ReferenceLine x={t} stroke={C.accent} strokeWidth={1.5} strokeDasharray="4 3" />
        <Line type="monotone" dataKey="abar" stroke={C.data} dot={false} strokeWidth={2} isAnimationActive={false} />
        <Line type="monotone" dataKey="beta" stroke={C.noise} dot={false} strokeWidth={2} isAnimationActive={false} />
      </LineChart>
    </ResponsiveContainer>
  );
}

const N_REVERSE = 64;

export default function App() {
  const [scheduleKind, setScheduleKind] = useState("cosine");
  const [T, setT] = useState(200);
  const sched = useMemo(() => buildSchedule(scheduleKind, T), [scheduleKind, T]);

  const [forwardT, setForwardT] = useState(0);
  const noisedCloud = useMemo(() => forwardCloud(DATA_POINTS, sched.abar[forwardT]), [sched, forwardT]);

  const [stochastic, setStochastic] = useState(true);
  const [revState, setRevState] = useState(null);
  const [revT, setRevT] = useState(T - 1);
  const [playing, setPlaying] = useState(false);
  const raf = useRef(0);
  const lastTick = useRef(0);

  const initReverse = useCallback(() => {
    const pts = Array.from({ length: N_REVERSE }, () => [randn(), randn()]);
    setRevState({ points: pts, trails: pts.map((p) => [p]) });
    setRevT(T - 1);
    setPlaying(false);
  }, [T]);

  useEffect(() => { initReverse(); }, [initReverse, scheduleKind]);

  const revTRef = useRef(revT);
  useEffect(() => { revTRef.current = revT; }, [revT]);

  const stepOnce = useCallback(() => {
    const stepT = revTRef.current;
    if (stepT < 0) return;
    setRevState((prev) => {
      if (!prev) return prev;
      const newPoints = prev.points.map(([x, y]) => ddpmStep(x, y, stepT, sched, stochastic));
      const newTrails = prev.trails.map((tr, i) => {
        const next = [...tr, newPoints[i]];
        return next.length > 28 ? next.slice(next.length - 28) : next;
      });
      return { points: newPoints, trails: newTrails };
    });
    setRevT((v) => v - 1);
  }, [sched, stochastic]);

  useEffect(() => {
    if (!playing) return;
    const reduce = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const interval = reduce ? 0 : Math.max(16, Math.min(120, 4000 / T));
    const loop = (ts) => {
      if (revTRef.current < 0) { setPlaying(false); return; }
      if (ts - lastTick.current >= interval) {
        lastTick.current = ts;
        stepOnce();
      }
      raf.current = requestAnimationFrame(loop);
    };
    raf.current = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf.current);
  }, [playing, stepOnce, T]);

  const fAbar = sched.abar[forwardT];
  const fNoise = Math.sqrt(1 - fAbar);
  const revAbar = revT >= 0 ? sched.abar[revT] : 1;
  const revNoise = revT >= 0 ? Math.sqrt(1 - revAbar) : 0;
  const done = revT < 0;

  const extentForward = RING_RADIUS + 0.6;

  return (
    <div style={{ fontFamily: "'Iowan Old Style', 'Palatino Linotype', Palatino, Georgia, serif", background: C.bg, minHeight: "100vh", padding: "30px 14px 56px", color: C.ink }}>
      <style>{`
        @media (prefers-reduced-motion: reduce) {
          * { transition-duration: 1ms !important; animation-duration: 1ms !important; }
        }
        .ddx-btn:active { transform: scale(0.97); }
        .ddx-btn:focus-visible, .ddx-tab:focus-visible, input[type=range]:focus-visible {
          outline: 2px solid ${C.accent}; outline-offset: 2px;
        }
        input[type=range] { accent-color: ${C.accent}; }
      `}</style>

      <div style={{ maxWidth: 720, margin: "0 auto" }}>
        <header style={{ marginBottom: 22 }}>
          <Eyebrow>Machine Learning · Generative Models</Eyebrow>
          <h1 style={{ fontSize: 30, fontWeight: 700, margin: "6px 0 0", lineHeight: 1.1, textWrap: "balance" }}>
            Diffusion: Noise and Denoise
          </h1>
          <p style={{ color: C.muted, fontSize: 15, margin: "10px 0 0", lineHeight: 1.6, maxWidth: "62ch" }}>
            A diffusion model learns to reverse a process that gradually destroys data. Here the data is a 2D
            ring of six clusters. The forward process dissolves it into pure noise; the reverse process rebuilds
            it. Because the target is a Gaussian mixture, its score is known in closed form, so you can watch the
            exact sampler walk noise back into structure.
          </p>
        </header>

        <Card style={{ marginBottom: 18, background: C.accentSoft, borderColor: `${C.accent}33` }}>
          <Eyebrow>The shared schedule</Eyebrow>
          <div style={{ display: "flex", gap: 10, flexWrap: "wrap", alignItems: "center", margin: "12px 0 14px" }}>
            <span style={{ fontSize: 13, color: C.muted }}>Noise schedule</span>
            {["cosine", "linear"].map((k) => (
              <button key={k} className="ddx-btn" onClick={() => setScheduleKind(k)} style={btnStyle(scheduleKind === k)}
                aria-pressed={scheduleKind === k}>
                {k}
              </button>
            ))}
            <span style={{ fontSize: 13, color: C.muted, marginLeft: 8 }}>Steps T</span>
            <input type="range" min={40} max={400} step={20} value={T}
              onChange={(e) => setT(+e.target.value)} aria-label="number of diffusion steps"
              style={{ flex: "1 1 160px", minWidth: 120 }} />
            <span style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: 14, fontWeight: 700, color: C.accent, minWidth: 42 }}>{T}</span>
          </div>
          <ScheduleChart sched={sched} t={Math.min(forwardT, T - 1)} />
          <p style={{ fontSize: 13, color: C.ink, opacity: 0.82, margin: "10px 0 0", lineHeight: 1.6 }}>
            <b>alpha_bar_t</b> is the fraction of the original signal that survives after t steps; it falls from 1
            (all data) to nearly 0 (all noise). <b>beta_t</b> is how much fresh noise each step injects. The cosine
            schedule holds onto signal longer in the middle, which tends to give the reverse sampler more useful
            gradient to work with.
          </p>
        </Card>

        <Card style={{ marginBottom: 18 }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", gap: 10, flexWrap: "wrap" }}>
            <div>
              <Eyebrow>Forward process · q(x_t | x_0)</Eyebrow>
              <h2 style={{ fontSize: 19, fontWeight: 700, margin: "5px 0 0" }}>Dissolving structure into noise</h2>
            </div>
          </div>
          <Formula style={{ margin: "12px 0" }}>
            x_t = &radic;(alpha_bar_t) &middot; x_0 &nbsp;+&nbsp; &radic;(1 &minus; alpha_bar_t) &middot; &epsilon, &nbsp;&nbsp; &epsilon ~ N(0, I)
          </Formula>
          <div style={{ display: "flex", gap: 18, flexWrap: "wrap", alignItems: "center", justifyContent: "center" }}>
            <PointCanvas points={noisedCloud} extent={extentForward} color={fAbar > 0.5 ? C.data : C.noise} />
            <div style={{ flex: "1 1 220px", minWidth: 200 }}>
              <label htmlFor="fwd-t" style={{ fontSize: 13, color: C.muted, display: "block", marginBottom: 6 }}>
                Drag through diffusion time
              </label>
              <input id="fwd-t" type="range" min={0} max={T - 1} value={Math.min(forwardT, T - 1)}
                onChange={(e) => setForwardT(+e.target.value)} style={{ width: "100%" }}
                aria-label="forward process time step" />
              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8, marginTop: 14 }}>
                <Stat label="time t" value={Math.min(forwardT, T - 1)} />
                <Stat label="alpha_bar_t" value={fAbar.toFixed(3)} />
                <Stat label="signal kept" value={`${(Math.sqrt(fAbar) * 100).toFixed(0)}%`} accent />
                <Stat label="noise level" value={fNoise.toFixed(3)} />
              </div>
              <p style={{ fontSize: 13, color: C.ink, opacity: 0.82, marginTop: 14, lineHeight: 1.6 }}>
                Every point is scaled toward the origin by &radic;(alpha_bar_t) and has Gaussian noise of scale
                &radic;(1&minus;alpha_bar_t) added. Early on the six clusters are still visible. As t grows the
                structure washes out until the cloud is an isotropic Gaussian blob that carries no memory of where
                it started. That endpoint is the same for any dataset, which is exactly why we can start sampling
                from plain noise.
              </p>
            </div>
          </div>
        </Card>

        <Card>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", gap: 10, flexWrap: "wrap" }}>
            <div>
              <Eyebrow>Reverse process · sampling</Eyebrow>
              <h2 style={{ fontSize: 19, fontWeight: 700, margin: "5px 0 0" }}>Walking noise back to data</h2>
            </div>
            <span style={{ fontSize: 12, color: done ? C.good : C.muted, fontWeight: 700 }}>
              {done ? "reached x_0" : `t = ${revT}`}
            </span>
          </div>
          <Formula style={{ margin: "12px 0" }}>
            x_(t&minus;1) = (1/&radic;alpha_t)&middot;(x_t &minus; (beta_t/&radic;(1&minus;alpha_bar_t))&middot;&epsilon&#770;) + sigma_t&middot;z<br />
            &epsilon&#770; = &minus;&radic;(1&minus;alpha_bar_t)&middot;score, &nbsp;&nbsp; score = &nabla; log q(x_t)
          </Formula>

          <div style={{ display: "flex", gap: 18, flexWrap: "wrap", alignItems: "center", justifyContent: "center" }}>
            <PointCanvas points={revState ? revState.points : []} trails={revState ? revState.trails : []}
              extent={extentForward} color={C.data} />
            <div style={{ flex: "1 1 220px", minWidth: 200 }}>
              <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 12 }}>
                <button className="ddx-btn" onClick={() => setPlaying((p) => !p)} disabled={done}
                  style={{ ...btnStyle(true), opacity: done ? 0.45 : 1, cursor: done ? "not-allowed" : "pointer" }}>
                  {playing ? "Pause" : done ? "Done" : "Play"}
                </button>
                <button className="ddx-btn" onClick={() => { setPlaying(false); stepOnce(); }} disabled={done}
                  style={{ ...btnStyle(false), opacity: done ? 0.45 : 1, cursor: done ? "not-allowed" : "pointer" }}>
                  Step
                </button>
                <button className="ddx-btn" onClick={initReverse} style={btnStyle(false)}>
                  Reset
                </button>
              </div>

              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
                <Stat label="time t" value={done ? 0 : revT} />
                <Stat label="alpha_bar_t" value={revAbar.toFixed(3)} />
                <Stat label="noise level" value={revNoise.toFixed(3)} accent />
                <Stat label="trajectories" value={N_REVERSE} />
              </div>

              <label style={{ display: "flex", alignItems: "center", gap: 9, marginTop: 14, fontSize: 13, color: C.ink, cursor: "pointer" }}>
                <input type="checkbox" checked={stochastic} onChange={(e) => setStochastic(e.target.checked)}
                  style={{ accentColor: C.accent, width: 16, height: 16 }} />
                Add sampling noise sigma_t&middot;z (DDPM). Off = deterministic mean path.
              </label>
            </div>
          </div>

          <p style={{ fontSize: 13, color: C.ink, opacity: 0.82, marginTop: 16, lineHeight: 1.6, maxWidth: "66ch" }}>
            Each trajectory starts as one draw from pure noise and runs the update above from t = T&minus;1 down to 0.
            The <b>score</b> points toward where data is denser; the network's job in a trained model is to estimate
            it, and equivalently to predict the noise &epsilon&#770; that was added. We move only a little per step
            because a single giant jump would aim at the average of all six clusters and land in the empty middle.
            Small steps let each sample commit to one cluster, which is why the trails fan out and settle onto the ring.
          </p>

          <div style={{ marginTop: 14, padding: "11px 14px", background: C.bg, borderRadius: 10, border: `1px solid ${C.border}`, fontSize: 12.5, color: C.muted, lineHeight: 1.6 }}>
            <b style={{ color: C.ink }}>Honest simplification:</b> a real model learns the score from data with a
            neural network. Here the target is a known Gaussian mixture, so its score is computed in closed form. The
            sampling math (DDPM ancestral update, noise schedules, the score-to-&epsilon relation) is exactly what a
            trained model uses; only the learned denoiser is replaced by its analytic optimum.
          </div>
        </Card>
      </div>
    </div>
  );
}

function Stat({ label, value, accent }) {
  return (
    <div style={{ background: accent ? C.accentSoft : C.bg, borderRadius: 9, padding: "9px 12px", border: `1px solid ${accent ? C.accent + "33" : C.border}` }}>
      <div style={{ fontSize: 10, letterSpacing: "0.06em", textTransform: "uppercase", color: accent ? C.accent : C.muted, fontWeight: 700, marginBottom: 3 }}>{label}</div>
      <div style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: 17, fontWeight: 700, color: accent ? C.accent : C.ink, lineHeight: 1 }}>{value}</div>
    </div>
  );
}
