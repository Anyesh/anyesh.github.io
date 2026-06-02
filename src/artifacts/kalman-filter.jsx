import { useState, useEffect, useRef, useMemo, useCallback } from "react";

export const meta = {
  title: "Kalman Filter: Tracking Through Noise",
  category: "Signals",
  description:
    "Track a moving target through noisy measurements. Watch the predict step grow uncertainty and the update step shrink it, and drag the noise sliders to see the covariance ellipse breathe.",
  date: "2026-03-29",
  tags: ["kalman-filter", "estimation", "covariance", "tracking"],
};

const C = {
  bg: "#f7f5f2",
  card: "#ffffff",
  border: "#e6e1d9",
  ink: "#1b1916",
  muted: "#857c70",
  accent: "#c0561f",
  accentSoft: "#f6ece5",
  truth: "#3f7d52",
  meas: "#7d8aa0",
  est: "#c0561f",
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

const I4 = () => [
  [1, 0, 0, 0],
  [0, 1, 0, 0],
  [0, 0, 1, 0],
  [0, 0, 0, 1],
];

function matMul(A, B) {
  const n = A.length, m = B[0].length, k = B.length;
  const out = Array.from({ length: n }, () => new Array(m).fill(0));
  for (let i = 0; i < n; i++)
    for (let j = 0; j < m; j++) {
      let s = 0;
      for (let p = 0; p < k; p++) s += A[i][p] * B[p][j];
      out[i][j] = s;
    }
  return out;
}
const transpose = (A) => A[0].map((_, j) => A.map((row) => row[j]));
const matVec = (A, v) => A.map((row) => row.reduce((s, x, j) => s + x * v[j], 0));
const matAdd = (A, B) => A.map((row, i) => row.map((x, j) => x + B[i][j]));
const matSub = (A, B) => A.map((row, i) => row.map((x, j) => x - B[i][j]));
const vecAdd = (a, b) => a.map((x, i) => x + b[i]);
const vecSub = (a, b) => a.map((x, i) => x - b[i]);

// H selects position, so the innovation covariance S is 2x2 and a closed-form inverse is enough.
function inv2(M) {
  const [a, b] = M[0];
  const [c, d] = M[1];
  const det = a * d - b * c;
  return [
    [d / det, -b / det],
    [-c / det, a / det],
  ];
}

function buildF(dt) {
  return [
    [1, 0, dt, 0],
    [0, 1, 0, dt],
    [0, 0, 1, 0],
    [0, 0, 0, 1],
  ];
}

// Continuous white-noise-acceleration model discretized: Q = q * [[dt^4/4, dt^3/2],[dt^3/2, dt^2]] per axis.
function buildQ(dt, q) {
  const dt2 = dt * dt, dt3 = dt2 * dt, dt4 = dt3 * dt;
  const a = dt4 / 4, b = dt3 / 2, c = dt2;
  return [
    [a, 0, b, 0],
    [0, a, 0, b],
    [b, 0, c, 0],
    [0, b, 0, c],
  ].map((row) => row.map((x) => x * q));
}

const H = [
  [1, 0, 0, 0],
  [0, 1, 0, 0],
];
const Ht = transpose(H);

function truthAt(t) {
  return [40 * Math.cos(0.05 * t), 30 * Math.sin(0.085 * t)];
}

const DT = 1;
const WORLD = 56;

function predictStep(x, P, F, Q) {
  const xp = matVec(F, x);
  const Pp = matAdd(matMul(matMul(F, P), transpose(F)), Q);
  return { x: xp, P: Pp };
}

function updateStep(x, P, z, R) {
  const S = matAdd(matMul(matMul(H, P), Ht), R);
  const K = matMul(matMul(P, Ht), inv2(S));
  const innov = vecSub(z, matVec(H, x));
  const xn = vecAdd(x, matVec(K, innov));
  const KH = matMul(K, H);
  const Pn = matMul(matSub(I4(), KH), P);
  return { x: xn, P: Pn, K, innov };
}

// Eigen-decomposition of the symmetric 2x2 position block, used to draw the real 1-sigma ellipse.
function covEllipse(P) {
  const a = P[0][0], b = P[0][1], c = P[1][1];
  const tr = a + c;
  const diff = Math.sqrt(((a - c) / 2) ** 2 + b * b);
  const l1 = tr / 2 + diff;
  const l2 = tr / 2 - diff;
  const angle = 0.5 * Math.atan2(2 * b, a - c);
  return {
    rx: Math.sqrt(Math.max(l1, 1e-9)),
    ry: Math.sqrt(Math.max(l2, 1e-9)),
    angleDeg: (angle * 180) / Math.PI,
  };
}

function initialState() {
  const [x0, y0] = truthAt(0);
  return {
    x: [x0, y0, 0, 0],
    P: I4().map((row, i) => row.map((v) => v * (i < 2 ? 60 : 30))),
  };
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

function Formula({ children, style }) {
  return (
    <div style={{
      background: C.bg, borderRadius: 9, padding: "9px 13px",
      fontFamily: "'IBM Plex Mono', ui-monospace, SFMono-Regular, Menlo, monospace",
      fontSize: 12.5, color: C.ink, border: `1px solid ${C.border}`, lineHeight: 1.7, overflowX: "auto", ...style,
    }}>
      {children}
    </div>
  );
}

function Stat({ label, value, accent }) {
  return (
    <div style={{ background: accent ? C.accentSoft : C.bg, borderRadius: 9, padding: "9px 12px", border: `1px solid ${accent ? C.accent + "33" : C.border}` }}>
      <div style={{ fontSize: 10, letterSpacing: "0.06em", textTransform: "uppercase", color: accent ? C.accent : C.muted, fontWeight: 700, marginBottom: 3 }}>{label}</div>
      <div style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: 16, fontWeight: 700, color: accent ? C.accent : C.ink, lineHeight: 1.1 }}>{value}</div>
    </div>
  );
}

function btnStyle(active, disabled) {
  return {
    padding: "8px 16px", borderRadius: 9,
    border: `1.5px solid ${active ? C.accent : C.border}`,
    background: active ? C.accent : C.card,
    color: active ? "#fff" : C.muted,
    fontSize: 13, fontWeight: 600, cursor: disabled ? "not-allowed" : "pointer",
    opacity: disabled ? 0.45 : 1, fontFamily: "inherit",
    transition: `transform 140ms ${EASE}, background 160ms ${EASE}, border-color 160ms ${EASE}`,
  };
}

function Swatch({ color, label, dashed }) {
  return (
    <span style={{ display: "inline-flex", alignItems: "center", gap: 6, fontSize: 12, color: C.ink }}>
      <span style={{
        width: 18, height: 0, borderTop: `${dashed ? "2px dashed" : "3px solid"} ${color}`,
        display: "inline-block",
      }} />
      {label}
    </span>
  );
}

const VIEW = 360;

function TrackCanvas({ truthPath, measurements, estPath, current, ellipse, predictGap }) {
  const ref = useRef(null);

  const toPx = useCallback((x, y) => [
    VIEW / 2 + (x / WORLD) * (VIEW / 2) * 0.92,
    VIEW / 2 - (y / WORLD) * (VIEW / 2) * 0.92,
  ], []);
  const scalePx = (VIEW / 2) * 0.92 / WORLD;

  useEffect(() => {
    const cv = ref.current;
    if (!cv) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    cv.width = VIEW * dpr;
    cv.height = VIEW * dpr;
    const ctx = cv.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, VIEW, VIEW);

    ctx.strokeStyle = C.grid;
    ctx.lineWidth = 1;
    for (let g = -40; g <= 40; g += 20) {
      const [px] = toPx(g, 0);
      const [, py] = toPx(0, g);
      ctx.beginPath(); ctx.moveTo(px, 8); ctx.lineTo(px, VIEW - 8); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(8, py); ctx.lineTo(VIEW - 8, py); ctx.stroke();
    }

    if (truthPath.length > 1) {
      ctx.strokeStyle = C.truth;
      ctx.lineWidth = 2;
      ctx.setLineDash([5, 4]);
      ctx.beginPath();
      truthPath.forEach(([x, y], i) => {
        const [px, py] = toPx(x, y);
        if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
      });
      ctx.stroke();
      ctx.setLineDash([]);
    }

    if (estPath.length > 1) {
      ctx.strokeStyle = C.est;
      ctx.lineWidth = 2.4;
      ctx.beginPath();
      estPath.forEach(([x, y], i) => {
        const [px, py] = toPx(x, y);
        if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
      });
      ctx.stroke();
    }

    ctx.fillStyle = C.meas;
    for (const m of measurements) {
      if (!m) continue;
      const [px, py] = toPx(m[0], m[1]);
      ctx.beginPath();
      ctx.moveTo(px - 3, py - 3); ctx.lineTo(px + 3, py + 3);
      ctx.moveTo(px + 3, py - 3); ctx.lineTo(px - 3, py + 3);
      ctx.strokeStyle = C.meas; ctx.lineWidth = 1.4; ctx.stroke();
    }

    if (ellipse && current) {
      const [cx, cy] = toPx(current[0], current[1]);
      ctx.save();
      ctx.translate(cx, cy);
      ctx.rotate((-ellipse.angleDeg * Math.PI) / 180);
      ctx.beginPath();
      ctx.ellipse(0, 0, Math.max(ellipse.rx * scalePx, 1), Math.max(ellipse.ry * scalePx, 1), 0, 0, 2 * Math.PI);
      ctx.fillStyle = predictGap ? "rgba(192,86,31,0.10)" : "rgba(192,86,31,0.16)";
      ctx.fill();
      ctx.strokeStyle = predictGap ? "rgba(192,86,31,0.55)" : C.accent;
      ctx.lineWidth = predictGap ? 2 : 1.5;
      ctx.setLineDash(predictGap ? [4, 3] : []);
      ctx.stroke();
      ctx.restore();
    }

    if (current) {
      const [cx, cy] = toPx(current[0], current[1]);
      ctx.fillStyle = C.est;
      ctx.beginPath();
      ctx.arc(cx, cy, 4, 0, 2 * Math.PI);
      ctx.fill();
      ctx.strokeStyle = "#fff"; ctx.lineWidth = 1.5; ctx.stroke();
    }
  }, [truthPath, measurements, estPath, current, ellipse, predictGap, toPx, scalePx]);

  return (
    <canvas
      ref={ref}
      style={{ width: VIEW, height: VIEW, maxWidth: "100%", borderRadius: 10, background: C.bg, border: `1px solid ${C.border}`, display: "block" }}
      role="img"
      aria-label="Tracking view: true path, noisy measurements, Kalman estimate, and the covariance ellipse around the current estimate"
    />
  );
}

const MAX_TRAIL = 220;

export default function App() {
  const [qVar, setQVar] = useState(0.04);
  const [rVar, setRVar] = useState(6);
  const [dropMeas, setDropMeas] = useState(false);
  const [playing, setPlaying] = useState(false);

  const [sim, setSim] = useState(() => {
    const init = initialState();
    return {
      t: 0,
      x: init.x,
      P: init.P,
      truthPath: [truthAt(0)],
      estPath: [[init.x[0], init.x[1]]],
      measurements: [],
      lastMeas: null,
      lastUpdated: true,
      sumRawSq: 0,
      sumEstSq: 0,
      nScored: 0,
    };
  });

  const reset = useCallback(() => {
    setPlaying(false);
    const init = initialState();
    setSim({
      t: 0,
      x: init.x,
      P: init.P,
      truthPath: [truthAt(0)],
      estPath: [[init.x[0], init.x[1]]],
      measurements: [],
      lastMeas: null,
      lastUpdated: true,
      sumRawSq: 0,
      sumEstSq: 0,
      nScored: 0,
    });
  }, []);

  const qRef = useRef(qVar);
  const rRef = useRef(rVar);
  const dropRef = useRef(dropMeas);
  useEffect(() => { qRef.current = qVar; }, [qVar]);
  useEffect(() => { rRef.current = rVar; }, [rVar]);
  useEffect(() => { dropRef.current = dropMeas; }, [dropMeas]);

  const stepOnce = useCallback(() => {
    setSim((prev) => {
      const t = prev.t + DT;
      const F = buildF(DT);
      const Q = buildQ(DT, qRef.current);
      const R = [
        [rRef.current, 0],
        [0, rRef.current],
      ];

      const pred = predictStep(prev.x, prev.P, F, Q);

      const [tx, ty] = truthAt(t);
      let x = pred.x;
      let P = pred.P;
      let meas = null;
      let updated = false;

      if (!dropRef.current) {
        const sd = Math.sqrt(rRef.current);
        meas = [tx + randn() * sd, ty + randn() * sd];
        const upd = updateStep(pred.x, pred.P, meas, R);
        x = upd.x;
        P = upd.P;
        updated = true;
      }

      const truthPath = [...prev.truthPath, [tx, ty]].slice(-MAX_TRAIL);
      const estPath = [...prev.estPath, [x[0], x[1]]].slice(-MAX_TRAIL);
      const measurements = [...prev.measurements, meas].slice(-MAX_TRAIL);

      let { sumRawSq, sumEstSq, nScored } = prev;
      if (meas) {
        sumRawSq += (meas[0] - tx) ** 2 + (meas[1] - ty) ** 2;
        sumEstSq += (x[0] - tx) ** 2 + (x[1] - ty) ** 2;
        nScored += 1;
      }

      return {
        t, x, P, truthPath, estPath, measurements,
        lastMeas: meas, lastUpdated: updated,
        sumRawSq, sumEstSq, nScored,
      };
    });
  }, []);

  const raf = useRef(0);
  const lastTick = useRef(0);
  useEffect(() => {
    if (!playing) return;
    const reduce = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const interval = reduce ? 240 : 90;
    const loop = (ts) => {
      if (ts - lastTick.current >= interval) {
        lastTick.current = ts;
        stepOnce();
      }
      raf.current = requestAnimationFrame(loop);
    };
    raf.current = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf.current);
  }, [playing, stepOnce]);

  const ellipse = useMemo(() => covEllipse(sim.P), [sim.P]);
  const traceP = sim.P[0][0] + sim.P[1][1] + sim.P[2][2] + sim.P[3][3];
  const posTrace = sim.P[0][0] + sim.P[1][1];
  const current = [sim.x[0], sim.x[1]];

  const rawRmse = sim.nScored ? Math.sqrt(sim.sumRawSq / sim.nScored) : null;
  const estRmse = sim.nScored ? Math.sqrt(sim.sumEstSq / sim.nScored) : null;

  return (
    <div style={{ fontFamily: "'Iowan Old Style', 'Palatino Linotype', Palatino, Georgia, serif", background: C.bg, minHeight: "100vh", padding: "30px 14px 56px", color: C.ink }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600;700&display=swap');
        @media (prefers-reduced-motion: reduce) {
          .kf-btn { transition-duration: 1ms !important; }
        }
        .kf-btn:active { transform: scale(0.97); }
        .kf-btn:focus-visible, input[type=range]:focus-visible, input[type=checkbox]:focus-visible {
          outline: 2px solid ${C.accent}; outline-offset: 2px;
        }
        input[type=range] { accent-color: ${C.accent}; }
        input[type=checkbox] { accent-color: ${C.accent}; }
      `}</style>

      <div style={{ maxWidth: 720, margin: "0 auto" }}>
        <header style={{ marginBottom: 22 }}>
          <Eyebrow>Signals · State Estimation</Eyebrow>
          <h1 style={{ fontSize: 30, fontWeight: 700, margin: "6px 0 0", lineHeight: 1.1, textWrap: "balance" }}>
            Kalman Filter: Tracking Through Noise
          </h1>
          <p style={{ color: C.muted, fontSize: 15, margin: "10px 0 0", lineHeight: 1.6, maxWidth: "62ch" }}>
            A target moves along the dashed green path. A sensor reports only its position, and every reading is
            corrupted by noise. The filter keeps a belief about position and velocity, then alternates two moves:
            predict where the target should be, and correct that guess against each new reading. The result, in
            terracotta, stays close to the truth even though it never sees it directly.
          </p>
        </header>

        <Card style={{ marginBottom: 18 }}>
          <div style={{ display: "flex", gap: 18, flexWrap: "wrap", alignItems: "flex-start", justifyContent: "center" }}>
            <TrackCanvas
              truthPath={sim.truthPath}
              measurements={sim.measurements}
              estPath={sim.estPath}
              current={current}
              ellipse={ellipse}
              predictGap={!sim.lastUpdated}
            />
            <div style={{ flex: "1 1 240px", minWidth: 220 }}>
              <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 14 }}>
                <button className="kf-btn" onClick={() => setPlaying((p) => !p)} style={btnStyle(true)} aria-pressed={playing}>
                  {playing ? "Pause" : "Play"}
                </button>
                <button className="kf-btn" onClick={() => { setPlaying(false); stepOnce(); }} style={btnStyle(false)}>
                  Step
                </button>
                <button className="kf-btn" onClick={reset} style={btnStyle(false)}>
                  Reset
                </button>
              </div>

              <label style={{ display: "flex", alignItems: "center", gap: 9, marginBottom: 14, fontSize: 13, color: C.ink, cursor: "pointer" }}>
                <input type="checkbox" checked={dropMeas} onChange={(e) => setDropMeas(e.target.checked)}
                  style={{ width: 16, height: 16 }} />
                Drop measurements (predict only)
              </label>

              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
                <Stat label="step t" value={sim.t} />
                <Stat label="mode" value={sim.lastUpdated ? "update" : "predict"} accent={!sim.lastUpdated} />
                <Stat label="estimate x" value={`${sim.x[0].toFixed(1)}, ${sim.x[1].toFixed(1)}`} />
                <Stat label="velocity" value={`${sim.x[2].toFixed(2)}, ${sim.x[3].toFixed(2)}`} />
                <Stat label="measurement" value={sim.lastMeas ? `${sim.lastMeas[0].toFixed(1)}, ${sim.lastMeas[1].toFixed(1)}` : "dropped"} />
                <Stat label="trace(P)" value={traceP.toFixed(2)} accent />
              </div>

              <div style={{ display: "flex", gap: 14, flexWrap: "wrap", marginTop: 14 }}>
                <Swatch color={C.truth} label="true path" dashed />
                <Swatch color={C.meas} label="measurements" />
                <Swatch color={C.est} label="estimate" />
              </div>
            </div>
          </div>

          <p style={{ fontSize: 13, color: C.ink, opacity: 0.82, marginTop: 16, lineHeight: 1.6, maxWidth: "66ch" }}>
            The shaded terracotta ellipse is the filter's 1-sigma position uncertainty, drawn from the
            eigenvectors of the 2x2 position block of the covariance P. Each update pulls it tighter. Turn on
            "drop measurements" and the ellipse swells with every predict-only step, because nothing is left to
            correct the growing drift.
          </p>
        </Card>

        <Card style={{ marginBottom: 18, background: C.accentSoft, borderColor: `${C.accent}33` }}>
          <Eyebrow>Tuning the two noises</Eyebrow>
          <div style={{ display: "grid", gridTemplateColumns: "1fr", gap: 16, marginTop: 12 }}>
            <div>
              <label htmlFor="kf-r" style={{ display: "flex", justifyContent: "space-between", fontSize: 13, color: C.ink, marginBottom: 6 }}>
                <span>Measurement noise R (sensor variance)</span>
                <span style={{ fontFamily: "'IBM Plex Mono', monospace", fontWeight: 700, color: C.accent }}>{rVar.toFixed(1)}</span>
              </label>
              <input id="kf-r" type="range" min={0.5} max={40} step={0.5} value={rVar}
                onChange={(e) => setRVar(+e.target.value)} style={{ width: "100%" }}
                aria-label="measurement noise variance R" />
            </div>
            <div>
              <label htmlFor="kf-q" style={{ display: "flex", justifyContent: "space-between", fontSize: 13, color: C.ink, marginBottom: 6 }}>
                <span>Process noise Q (model slack)</span>
                <span style={{ fontFamily: "'IBM Plex Mono', monospace", fontWeight: 700, color: C.accent }}>{qVar.toFixed(3)}</span>
              </label>
              <input id="kf-q" type="range" min={0.001} max={0.5} step={0.001} value={qVar}
                onChange={(e) => setQVar(+e.target.value)} style={{ width: "100%" }}
                aria-label="process noise variance Q" />
            </div>
          </div>
          <p style={{ fontSize: 13, color: C.ink, opacity: 0.85, marginTop: 14, lineHeight: 1.6 }}>
            R says how much you distrust the sensor; Q says how much you distrust the constant-velocity assumption.
            Raise R and the filter leans on its own prediction, so the estimate is smooth but slow to react when the
            target turns. Raise Q and the filter trusts each fresh reading, so it tracks turns quickly but jitters
            with the noise. The Kalman gain K is exactly the lever between them: it lands wherever the ratio of
            these two uncertainties puts it.
          </p>
        </Card>

        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))", gap: 16, marginBottom: 18 }}>
          <Card>
            <Eyebrow>Predict</Eyebrow>
            <h2 style={{ fontSize: 17, fontWeight: 700, margin: "5px 0 10px" }}>Where should it be next?</h2>
            <Formula>
              x = F x<br />
              P = F P F&#7488; + Q
            </Formula>
            <p style={{ fontSize: 13, color: C.ink, opacity: 0.82, marginTop: 12, lineHeight: 1.6 }}>
              F moves position by velocity times dt and leaves velocity alone. Projecting P through F and adding Q
              always grows the uncertainty: prediction alone can only lose information, which is why the ellipse
              breathes outward during a measurement gap.
            </p>
          </Card>
          <Card>
            <Eyebrow>Update</Eyebrow>
            <h2 style={{ fontSize: 17, fontWeight: 700, margin: "5px 0 10px" }}>Correct against the reading</h2>
            <Formula>
              K = P H&#7488; (H P H&#7488; + R)&#8315;&#185;<br />
              x = x + K (z &minus; H x)<br />
              P = (I &minus; K H) P
            </Formula>
            <p style={{ fontSize: 13, color: C.ink, opacity: 0.82, marginTop: 12, lineHeight: 1.6 }}>
              H pulls position out of the state. The gain K weighs the surprise z minus H x by how much the filter
              trusts the reading versus its own belief. Folding that correction in always shrinks P, which is the
              ellipse snapping tighter on every update.
            </p>
          </Card>
        </div>

        <Card>
          <Eyebrow>Is it actually helping?</Eyebrow>
          <h2 style={{ fontSize: 17, fontWeight: 700, margin: "5px 0 12px" }}>Estimate error versus raw error</h2>
          {sim.nScored > 0 ? (
            <>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))", gap: 8 }}>
                <Stat label="raw RMSE" value={rawRmse.toFixed(2)} />
                <Stat label="filter RMSE" value={estRmse.toFixed(2)} accent />
                <Stat label="error cut" value={rawRmse > 0 ? `${(((rawRmse - estRmse) / rawRmse) * 100).toFixed(0)}%` : "—"} />
                <Stat label="pos trace(P)" value={posTrace.toFixed(2)} />
              </div>
              <p style={{ fontSize: 13, color: C.ink, opacity: 0.82, marginTop: 14, lineHeight: 1.6, maxWidth: "66ch" }}>
                RMSE is the root-mean-square distance to the true position, averaged over every step with a reading.
                The filter's number stays below the raw sensor's because each estimate blends the new reading with an
                informed prediction instead of trusting the reading alone.
              </p>
            </>
          ) : (
            <p style={{ fontSize: 13, color: C.muted, lineHeight: 1.6 }}>
              Press Play or Step to gather readings. With measurements dropped, there is nothing to score against.
            </p>
          )}
        </Card>

        <div style={{ marginTop: 18, padding: "11px 14px", background: C.card, borderRadius: 10, border: `1px solid ${C.border}`, fontSize: 12.5, color: C.muted, lineHeight: 1.6 }}>
          <b style={{ color: C.ink }}>What is real here:</b> the state is [x, y, vx, vy] with matrices F, Q, H, R as
          shown. Every predict and update runs the full matrix algebra in plain JavaScript, the innovation
          covariance is inverted in closed form because H selects position, and the ellipse comes from the actual
          eigen-decomposition of P. Only the target trajectory and the Gaussian sensor noise are synthetic.
        </div>
      </div>
    </div>
  );
}
