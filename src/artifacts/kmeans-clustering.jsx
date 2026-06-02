import { useState, useEffect, useRef, useMemo, useCallback } from "react";

export const meta = {
  title: "K-Means: Finding Clusters by Hand",
  category: "Machine Learning",
  description:
    "Drop points on a plane and watch Lloyd's algorithm chase the centers. Every assignment, every mean, and every drop in inertia is computed for real, including the local minima that make k-means restart-sensitive.",
  date: "2026-06-02",
  tags: ["k-means", "clustering", "unsupervised", "lloyd"],
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
  good: "#3f7d52",
  goodSoft: "#e7f0ea",
};

const EASE = "cubic-bezier(0.23, 1, 0.32, 1)";

// Cluster colors are chosen to stay distinct from the terracotta accent (reserved
// for centroids and active controls) and to keep ~4.5:1 ink contrast on white labels.
const CLUSTER_COLORS = [
  "#2f6f9f",
  "#c0561f",
  "#3f7d52",
  "#9a6cb0",
  "#b58a1e",
  "#b0506a",
];

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

const EXTENT = 10;

function seededBlobs(seed, k) {
  const rng = mulberry32(seed);
  const nBlobs = Math.max(3, k);
  const means = [];
  for (let b = 0; b < nBlobs; b++) {
    means.push([
      (rng() - 0.5) * 2 * (EXTENT - 2.4),
      (rng() - 0.5) * 2 * (EXTENT - 2.4),
    ]);
  }
  const pts = [];
  const perBlob = Math.round(150 / nBlobs);
  for (let b = 0; b < nBlobs; b++) {
    const spread = 0.9 + rng() * 0.9;
    for (let i = 0; i < perBlob; i++) {
      pts.push([
        means[b][0] + (rng() - 0.5) * 2 * spread + (rng() - 0.5) * spread,
        means[b][1] + (rng() - 0.5) * 2 * spread + (rng() - 0.5) * spread,
      ]);
    }
  }
  return pts.map(([x, y]) => [clamp(x, -EXTENT, EXTENT), clamp(y, -EXTENT, EXTENT)]);
}

function clamp(v, lo, hi) {
  return v < lo ? lo : v > hi ? hi : v;
}

function dist2(ax, ay, bx, by) {
  const dx = ax - bx;
  const dy = ay - by;
  return dx * dx + dy * dy;
}

function assign(points, centroids) {
  const labels = new Array(points.length);
  for (let i = 0; i < points.length; i++) {
    let best = 0;
    let bestD = Infinity;
    for (let c = 0; c < centroids.length; c++) {
      const d = dist2(points[i][0], points[i][1], centroids[c][0], centroids[c][1]);
      if (d < bestD) {
        bestD = d;
        best = c;
      }
    }
    labels[i] = best;
  }
  return labels;
}

function updateCentroids(points, labels, k, prev) {
  const sx = new Array(k).fill(0);
  const sy = new Array(k).fill(0);
  const n = new Array(k).fill(0);
  for (let i = 0; i < points.length; i++) {
    const c = labels[i];
    sx[c] += points[i][0];
    sy[c] += points[i][1];
    n[c]++;
  }
  const next = [];
  for (let c = 0; c < k; c++) {
    if (n[c] === 0) {
      // An empty cluster has no mean to move to, so we reseed it onto the point that
      // is currently worst served by its own centroid. That point most needs a closer
      // center, which gives the dead cluster a real job next round instead of leaving
      // it frozen forever.
      let far = 0;
      let farD = -1;
      for (let i = 0; i < points.length; i++) {
        const d = dist2(points[i][0], points[i][1], prev[labels[i]][0], prev[labels[i]][1]);
        if (d > farD) {
          farD = d;
          far = i;
        }
      }
      next.push([points[far][0], points[far][1]]);
    } else {
      next.push([sx[c] / n[c], sy[c] / n[c]]);
    }
  }
  return next;
}

function inertiaOf(points, labels, centroids) {
  let s = 0;
  for (let i = 0; i < points.length; i++) {
    s += dist2(points[i][0], points[i][1], centroids[labels[i]][0], centroids[labels[i]][1]);
  }
  return s;
}

function initCentroids(points, k, seed) {
  const rng = mulberry32(seed);
  const chosen = [];
  const used = new Set();
  let guard = 0;
  while (chosen.length < k && guard < 5000) {
    guard++;
    const idx = Math.floor(rng() * points.length);
    if (used.has(idx)) continue;
    used.add(idx);
    chosen.push([points[idx][0], points[idx][1]]);
  }
  while (chosen.length < k) {
    chosen.push([(rng() - 0.5) * EXTENT, (rng() - 0.5) * EXTENT]);
  }
  return chosen;
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
      fontSize: 12.5, color: C.ink, border: `1px solid ${C.border}`, lineHeight: 1.7, overflowX: "auto",
      ...style,
    }}>
      {children}
    </div>
  );
}

function Stat({ label, value, accent, good }) {
  const tone = good ? C.good : accent ? C.accent : C.ink;
  const tint = good ? C.goodSoft : accent ? C.accentSoft : C.bg;
  const border = good ? C.good + "33" : accent ? C.accent + "33" : C.border;
  return (
    <div style={{ background: tint, borderRadius: 9, padding: "9px 12px", border: `1px solid ${border}` }}>
      <div style={{ fontSize: 10, letterSpacing: "0.06em", textTransform: "uppercase", color: good ? C.good : accent ? C.accent : C.muted, fontWeight: 700, marginBottom: 3 }}>{label}</div>
      <div style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: 17, fontWeight: 700, color: tone, lineHeight: 1 }}>{value}</div>
    </div>
  );
}

function btnStyle(active, disabled) {
  return {
    padding: "8px 16px", borderRadius: 9,
    border: `1.5px solid ${active ? C.accent : C.border}`,
    background: active ? C.accent : C.card,
    color: active ? "#fff" : C.muted,
    fontSize: 13, fontWeight: 600,
    cursor: disabled ? "not-allowed" : "pointer",
    opacity: disabled ? 0.45 : 1,
    fontFamily: "inherit",
    transition: `transform 140ms ${EASE}, background 160ms ${EASE}, border-color 160ms ${EASE}, color 160ms ${EASE}`,
  };
}

const SIZE = 360;
const PAD = 14;

function toPx(x, y) {
  const inner = SIZE - 2 * PAD;
  return [
    PAD + ((x + EXTENT) / (2 * EXTENT)) * inner,
    PAD + ((EXTENT - y) / (2 * EXTENT)) * inner,
  ];
}

function toData(px, py) {
  const inner = SIZE - 2 * PAD;
  return [
    ((px - PAD) / inner) * 2 * EXTENT - EXTENT,
    EXTENT - ((py - PAD) / inner) * 2 * EXTENT,
  ];
}

function hexToRgb(hex) {
  const h = hex.replace("#", "");
  return [parseInt(h.slice(0, 2), 16), parseInt(h.slice(2, 4), 16), parseInt(h.slice(4, 6), 16)];
}

function ClusterCanvas({ points, labels, centroids, trails, onPlace, onDragPoint }) {
  const ref = useRef(null);
  const drag = useRef(null);

  useEffect(() => {
    const cv = ref.current;
    if (!cv) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    cv.width = SIZE * dpr;
    cv.height = SIZE * dpr;
    const ctx = cv.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, SIZE, SIZE);

    const k = centroids.length;
    if (k > 0) {
      // Voronoi partition by sampling the plane on a coarse grid and coloring each cell
      // by its nearest centroid. A pixel-exact partition is unnecessary here; the eye
      // reads the boundaries fine from a tinted grid, and sampling keeps this cheap
      // enough to repaint on every drag.
      const step = 6;
      const cpx = centroids.map(([cx, cy]) => toPx(cx, cy));
      for (let py = PAD; py < SIZE - PAD; py += step) {
        for (let px = PAD; px < SIZE - PAD; px += step) {
          let best = 0;
          let bestD = Infinity;
          for (let c = 0; c < k; c++) {
            const d = (px - cpx[c][0]) ** 2 + (py - cpx[c][1]) ** 2;
            if (d < bestD) {
              bestD = d;
              best = c;
            }
          }
          const [r, g, b] = hexToRgb(CLUSTER_COLORS[best % CLUSTER_COLORS.length]);
          ctx.fillStyle = `rgba(${r},${g},${b},0.09)`;
          ctx.fillRect(px, py, step, step);
        }
      }
    }

    ctx.strokeStyle = C.grid;
    ctx.lineWidth = 1;
    const [ox, oy] = toPx(0, 0);
    ctx.beginPath();
    ctx.moveTo(ox, PAD);
    ctx.lineTo(ox, SIZE - PAD);
    ctx.moveTo(PAD, oy);
    ctx.lineTo(SIZE - PAD, oy);
    ctx.stroke();

    if (trails) {
      ctx.lineWidth = 1.5;
      for (let c = 0; c < trails.length; c++) {
        const tr = trails[c];
        if (!tr || tr.length < 2) continue;
        const col = CLUSTER_COLORS[c % CLUSTER_COLORS.length];
        const [r, g, b] = hexToRgb(col);
        ctx.strokeStyle = `rgba(${r},${g},${b},0.55)`;
        ctx.beginPath();
        for (let i = 0; i < tr.length; i++) {
          const [px, py] = toPx(tr[i][0], tr[i][1]);
          if (i === 0) ctx.moveTo(px, py);
          else ctx.lineTo(px, py);
        }
        ctx.stroke();
        for (let i = 0; i < tr.length - 1; i++) {
          const [px, py] = toPx(tr[i][0], tr[i][1]);
          ctx.fillStyle = `rgba(${r},${g},${b},0.4)`;
          ctx.beginPath();
          ctx.arc(px, py, 2, 0, 2 * Math.PI);
          ctx.fill();
        }
      }
    }

    for (let i = 0; i < points.length; i++) {
      const [px, py] = toPx(points[i][0], points[i][1]);
      const lab = labels ? labels[i] : null;
      ctx.fillStyle = lab == null ? "#b8aea2" : CLUSTER_COLORS[lab % CLUSTER_COLORS.length];
      ctx.beginPath();
      ctx.arc(px, py, 3.4, 0, 2 * Math.PI);
      ctx.fill();
    }

    for (let c = 0; c < centroids.length; c++) {
      const [px, py] = toPx(centroids[c][0], centroids[c][1]);
      const col = CLUSTER_COLORS[c % CLUSTER_COLORS.length];
      ctx.fillStyle = "#fff";
      ctx.beginPath();
      ctx.arc(px, py, 9, 0, 2 * Math.PI);
      ctx.fill();
      ctx.fillStyle = col;
      ctx.beginPath();
      ctx.arc(px, py, 7, 0, 2 * Math.PI);
      ctx.fill();
      ctx.strokeStyle = "#fff";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(px - 3.4, py);
      ctx.lineTo(px + 3.4, py);
      ctx.moveTo(px, py - 3.4);
      ctx.lineTo(px, py + 3.4);
      ctx.stroke();
    }
  }, [points, labels, centroids, trails]);

  const localPos = useCallback((e) => {
    const rect = ref.current.getBoundingClientRect();
    const scaleX = SIZE / rect.width;
    const scaleY = SIZE / rect.height;
    return [(e.clientX - rect.left) * scaleX, (e.clientY - rect.top) * scaleY];
  }, []);

  const onPointerDown = useCallback((e) => {
    if (!onDragPoint && !onPlace) return;
    const [px, py] = localPos(e);
    let nearest = -1;
    let nearestD = 14 * 14;
    for (let i = 0; i < points.length; i++) {
      const [qx, qy] = toPx(points[i][0], points[i][1]);
      const d = (px - qx) ** 2 + (py - qy) ** 2;
      if (d < nearestD) {
        nearestD = d;
        nearest = i;
      }
    }
    if (nearest >= 0 && onDragPoint) {
      drag.current = nearest;
      ref.current.setPointerCapture(e.pointerId);
    } else if (onPlace) {
      const [dx, dy] = toData(px, py);
      onPlace([clamp(dx, -EXTENT, EXTENT), clamp(dy, -EXTENT, EXTENT)]);
    }
  }, [localPos, points, onDragPoint, onPlace]);

  const onPointerMove = useCallback((e) => {
    if (drag.current == null || !onDragPoint) return;
    const [px, py] = localPos(e);
    const [dx, dy] = toData(px, py);
    onDragPoint(drag.current, [clamp(dx, -EXTENT, EXTENT), clamp(dy, -EXTENT, EXTENT)]);
  }, [localPos, onDragPoint]);

  const onPointerUp = useCallback((e) => {
    if (drag.current != null) {
      try { ref.current.releasePointerCapture(e.pointerId); } catch (err) { void err; }
    }
    drag.current = null;
  }, []);

  return (
    <canvas
      ref={ref}
      onPointerDown={onPointerDown}
      onPointerMove={onPointerMove}
      onPointerUp={onPointerUp}
      style={{
        width: SIZE, height: SIZE, maxWidth: "100%", aspectRatio: "1 / 1",
        borderRadius: 12, background: C.bg, border: `1px solid ${C.border}`,
        display: "block", touchAction: "none", cursor: onDragPoint ? "grab" : "crosshair",
      }}
      role="img"
      aria-label="Two dimensional scatter of points colored by their current cluster assignment, with centroid markers and their movement trails"
    />
  );
}

const MAX_TRAIL = 12;

export default function App() {
  const [k, setK] = useState(4);
  const [dataSeed, setDataSeed] = useState(7);
  const [initSeed, setInitSeed] = useState(1);
  const [placeMode, setPlaceMode] = useState("drag");

  const [points, setPoints] = useState(() => seededBlobs(7, 4));
  const [centroids, setCentroids] = useState(() => initCentroids(seededBlobs(7, 4), 4, 1));
  const [labels, setLabels] = useState(null);
  const [trails, setTrails] = useState(() => initCentroids(seededBlobs(7, 4), 4, 1).map((c) => [c]));
  const [iteration, setIteration] = useState(0);
  const [inertia, setInertia] = useState(null);
  const [changed, setChanged] = useState(null);
  const [converged, setConverged] = useState(false);
  const [playing, setPlaying] = useState(false);

  const raf = useRef(0);
  const lastTick = useRef(0);
  const stateRef = useRef({ centroids, labels, converged });
  useEffect(() => {
    stateRef.current = { centroids, labels, converged };
  }, [centroids, labels, converged]);

  const resetRun = useCallback((pts, kk, iSeed) => {
    const c0 = initCentroids(pts, kk, iSeed);
    setCentroids(c0);
    setTrails(c0.map((c) => [c]));
    setLabels(null);
    setIteration(0);
    setInertia(null);
    setChanged(null);
    setConverged(false);
    setPlaying(false);
  }, []);

  const regenerate = useCallback((seed) => {
    const pts = seededBlobs(seed, k);
    setPoints(pts);
    resetRun(pts, k, initSeed);
  }, [k, initSeed, resetRun]);

  const step = useCallback(() => {
    setPoints((pts) => {
      const { centroids: cur, labels: curLabels, converged: done } = stateRef.current;
      if (done) return pts;

      if (curLabels == null) {
        const lab = assign(pts, cur);
        setLabels(lab);
        setInertia(inertiaOf(pts, lab, cur));
        setChanged(pts.length);
        setIteration(1);
        return pts;
      }

      const nextCentroids = updateCentroids(pts, curLabels, cur.length, cur);
      const newLabels = assign(pts, nextCentroids);
      let nChanged = 0;
      for (let i = 0; i < newLabels.length; i++) if (newLabels[i] !== curLabels[i]) nChanged++;

      setCentroids(nextCentroids);
      setTrails((prev) =>
        nextCentroids.map((c, i) => {
          const tr = (prev[i] || []).concat([[c[0], c[1]]]);
          return tr.length > MAX_TRAIL ? tr.slice(tr.length - MAX_TRAIL) : tr;
        })
      );
      setLabels(newLabels);
      setInertia(inertiaOf(pts, newLabels, nextCentroids));
      setChanged(nChanged);
      setIteration((it) => it + 1);
      if (nChanged === 0) {
        setConverged(true);
        setPlaying(false);
      }
      return pts;
    });
  }, []);

  useEffect(() => {
    if (!playing) return;
    const reduce = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const interval = reduce ? 700 : 520;
    const loop = (ts) => {
      if (stateRef.current.converged) {
        setPlaying(false);
        return;
      }
      if (ts - lastTick.current >= interval) {
        lastTick.current = ts;
        step();
      }
      raf.current = requestAnimationFrame(loop);
    };
    raf.current = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf.current);
  }, [playing, step]);

  const dragHandler = useCallback((idx, pos) => {
    setPoints((prev) => {
      const next = prev.slice();
      next[idx] = pos;
      return next;
    });
    setLabels(null);
    setIteration(0);
    setInertia(null);
    setChanged(null);
    setConverged(false);
    setPlaying(false);
  }, []);

  const placeHandler = useCallback((pos) => {
    setPoints((prev) => [...prev, pos]);
    setLabels(null);
    setIteration(0);
    setInertia(null);
    setChanged(null);
    setConverged(false);
    setPlaying(false);
  }, []);

  const newInit = useCallback(() => {
    const s = initSeed + 1;
    setInitSeed(s);
    resetRun(points, k, s);
  }, [initSeed, points, k, resetRun]);

  const changeK = useCallback((nk) => {
    setK(nk);
    resetRun(points, nk, initSeed);
  }, [points, initSeed, resetRun]);

  const statusLabel = converged
    ? "converged"
    : iteration === 0
      ? "ready"
      : `iteration ${iteration}`;

  const counts = useMemo(() => {
    if (!labels) return null;
    const n = new Array(k).fill(0);
    for (const l of labels) n[l]++;
    return n;
  }, [labels, k]);

  return (
    <div style={{ fontFamily: "'Iowan Old Style', 'Palatino Linotype', Palatino, Georgia, serif", background: C.bg, minHeight: "100vh", padding: "30px 14px 56px", color: C.ink }}>
      <style>{`
        @media (prefers-reduced-motion: reduce) {
          .km-anim { transition-duration: 1ms !important; }
        }
        .km-btn:active:not(:disabled) { transform: scale(0.97); }
        .km-btn:focus-visible, .km-tab:focus-visible, input[type=range]:focus-visible, canvas:focus-visible {
          outline: 2px solid ${C.accent}; outline-offset: 2px;
        }
        input[type=range] { accent-color: ${C.accent}; }
      `}</style>

      <div style={{ maxWidth: 720, margin: "0 auto" }}>
        <header style={{ marginBottom: 22 }}>
          <Eyebrow>Machine Learning · Unsupervised</Eyebrow>
          <h1 style={{ fontSize: 30, fontWeight: 700, margin: "6px 0 0", lineHeight: 1.1, textWrap: "balance" }}>
            K-Means: Finding Clusters by Hand
          </h1>
          <p style={{ color: C.ink, opacity: 0.86, fontSize: 15, margin: "10px 0 0", lineHeight: 1.6, maxWidth: "64ch", textWrap: "pretty" }}>
            K-means groups points into <b>k</b> clusters by repeating two moves until nothing changes. First it assigns
            every point to its nearest center. Then it slides each center to the average of the points that picked it.
            That loop, called Lloyd's algorithm, is running for real below: every assignment, every mean, and every drop
            in the within-cluster sum of squares is computed from the points you see, not scripted.
          </p>
        </header>

        <Card style={{ marginBottom: 18, background: C.accentSoft, borderColor: `${C.accent}33` }}>
          <Eyebrow>The two steps</Eyebrow>
          <Formula style={{ margin: "12px 0 0" }}>
            <div><b>assign:</b> &nbsp; label(i) = argmin&#8202;<sub>c</sub> &nbsp;&#8741; x&#8202;<sub>i</sub> &minus; &mu;&#8202;<sub>c</sub> &#8741;&sup2;</div>
            <div style={{ marginTop: 4 }}><b>update:</b> &nbsp; &mu;&#8202;<sub>c</sub> = mean of all points with label c</div>
            <div style={{ marginTop: 4 }}><b>inertia:</b> &nbsp; J = &Sigma;&#8202;<sub>i</sub> &nbsp;&#8741; x&#8202;<sub>i</sub> &minus; &mu;&#8202;<sub>label(i)</sub> &#8741;&sup2;</div>
          </Formula>
          <p style={{ fontSize: 13, color: C.ink, opacity: 0.82, margin: "12px 0 0", lineHeight: 1.6 }}>
            Each move can only lower inertia or leave it unchanged, so the algorithm always stops. It stops when an
            assignment step reassigns zero points. That guarantee is exactly what makes k-means fast and also what
            makes it fragile: it settles into the nearest valley from where the centers happened to start, which is
            not always the deepest one.
          </p>
        </Card>

        <Card style={{ marginBottom: 18 }}>
          <div style={{ display: "flex", gap: 18, flexWrap: "wrap", alignItems: "flex-start", justifyContent: "center" }}>
            <ClusterCanvas
              points={points}
              labels={labels}
              centroids={centroids}
              trails={trails}
              onDragPoint={placeMode === "drag" ? dragHandler : undefined}
              onPlace={placeMode === "add" ? placeHandler : undefined}
            />

            <div style={{ flex: "1 1 240px", minWidth: 220 }}>
              <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 10 }}>
                <Eyebrow>Run the algorithm</Eyebrow>
                <span style={{ fontSize: 12, fontWeight: 700, color: converged ? C.good : C.muted }}>{statusLabel}</span>
              </div>

              <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 12 }}>
                <button className="km-btn km-anim" onClick={() => setPlaying((p) => !p)} disabled={converged}
                  style={btnStyle(true, converged)} aria-label={playing ? "Pause auto run" : "Play auto run"}>
                  {playing ? "Pause" : converged ? "Done" : "Play"}
                </button>
                <button className="km-btn km-anim" onClick={() => { setPlaying(false); step(); }} disabled={converged}
                  style={btnStyle(false, converged)} aria-label="Run one iteration">
                  {labels == null ? "Assign" : "Next step"}
                </button>
                <button className="km-btn km-anim" onClick={() => resetRun(points, k, initSeed)}
                  style={btnStyle(false, false)} aria-label="Reset run with same initial centroids">
                  Reset
                </button>
              </div>

              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8, marginBottom: 12 }}>
                <Stat label="iteration" value={iteration} />
                <Stat label="reassigned" value={changed == null ? "—" : changed} accent={changed != null && changed > 0} good={changed === 0 && iteration > 0} />
                <Stat label="inertia (SSE)" value={inertia == null ? "—" : inertia.toFixed(1)} accent />
                <Stat label="status" value={converged ? "stable" : iteration === 0 ? "init" : "moving"} good={converged} />
              </div>

              <div style={{ fontSize: 12.5, color: C.ink, opacity: 0.82, lineHeight: 1.55 }}>
                {converged
                  ? "Zero points changed cluster, so the centers are at the mean of their members and will not move again. This is a local minimum of the inertia."
                  : labels == null
                    ? "Press Assign for the first assignment step, or Play to run the whole loop. Watch inertia fall and the reassignment count head toward zero."
                    : "Each Next step does one update then one assignment. The faint trails show how far each center has traveled so far."}
              </div>
            </div>
          </div>

          <div style={{ display: "flex", flexWrap: "wrap", gap: 6, marginTop: 14, justifyContent: "center" }}>
            {Array.from({ length: k }, (_, c) => (
              <span key={c} style={{
                display: "inline-flex", alignItems: "center", gap: 6,
                fontSize: 11.5, color: C.ink, opacity: 0.85,
                background: C.bg, border: `1px solid ${C.border}`, borderRadius: 999, padding: "3px 10px",
              }}>
                <span style={{ width: 10, height: 10, borderRadius: 999, background: CLUSTER_COLORS[c % CLUSTER_COLORS.length] }} />
                cluster {c + 1}
                {counts && <span style={{ color: C.muted, fontFamily: "'IBM Plex Mono', monospace" }}>· {counts[c]}</span>}
              </span>
            ))}
          </div>
        </Card>

        <Card style={{ marginBottom: 18 }}>
          <Eyebrow>Set up the problem</Eyebrow>

          <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap", margin: "14px 0 4px" }}>
            <span style={{ fontSize: 13, color: C.muted, minWidth: 96 }}>Clusters k</span>
            {[2, 3, 4, 5, 6].map((nk) => (
              <button key={nk} className="km-btn km-anim" onClick={() => changeK(nk)} style={btnStyle(k === nk, false)}
                aria-pressed={k === nk} aria-label={`Set k to ${nk}`}>
                {nk}
              </button>
            ))}
          </div>

          <div style={{ display: "flex", alignItems: "center", gap: 10, flexWrap: "wrap", margin: "16px 0 4px" }}>
            <span style={{ fontSize: 13, color: C.muted, minWidth: 96 }}>Point tool</span>
            {[
              ["drag", "Drag points"],
              ["add", "Click to add"],
            ].map(([mode, lbl]) => (
              <button key={mode} className="km-btn km-anim" onClick={() => setPlaceMode(mode)} style={btnStyle(placeMode === mode, false)}
                aria-pressed={placeMode === mode}>
                {lbl}
              </button>
            ))}
          </div>

          <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginTop: 16 }}>
            <button className="km-btn km-anim" onClick={() => { const s = dataSeed + 1; setDataSeed(s); regenerate(s); }}
              style={btnStyle(false, false)} aria-label="Regenerate a fresh scatter of seeded clusters">
              Shuffle data
            </button>
            <button className="km-btn km-anim" onClick={newInit}
              style={btnStyle(false, false)} aria-label="Reseed the initial centroids without changing the data">
              New random init
            </button>
          </div>

          <p style={{ fontSize: 13, color: C.ink, opacity: 0.82, margin: "16px 0 0", lineHeight: 1.6 }}>
            <b>Shuffle data</b> draws a fresh set of seeded blobs. <b>New random init</b> keeps the same points but picks
            different starting centers, which is the honest way to see local minima: run the algorithm to convergence,
            note the inertia, then reseed and run again. With awkward starts, two centers can end up splitting one true
            blob while a single center is stretched across two, and the final inertia comes out clearly higher. Real
            implementations run several inits and keep the lowest-inertia result for this reason.
          </p>
        </Card>

        <Card style={{ background: C.bg, borderColor: C.border }}>
          <Eyebrow>Why empty clusters get reseeded</Eyebrow>
          <p style={{ fontSize: 13, color: C.ink, opacity: 0.84, margin: "10px 0 0", lineHeight: 1.6 }}>
            If a center ends an assignment step with no points, its mean is undefined and it would sit dead for the rest
            of the run. When that happens here, the empty center jumps onto the single point that is currently farthest
            from its own center, the point most in need of a closer home. That keeps all k clusters alive and tends to
            shave inertia, since the worst-served point gets a dedicated center on the next pass.
          </p>
        </Card>

        <p style={{ fontSize: 12, color: C.muted, marginTop: 18, lineHeight: 1.6, textAlign: "center" }}>
          Distances are squared Euclidean on the plane shown. Inertia is the within-cluster sum of squares; it never
          rises across a step.
        </p>
      </div>
    </div>
  );
}
