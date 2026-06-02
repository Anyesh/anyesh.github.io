import { useState, useEffect, useRef, useMemo, useCallback } from "react";

export const meta = {
  title: "PCA vs t-SNE: What Each One Lies About",
  category: "Machine Learning",
  description:
    "Two ways to flatten high-dimensional data into a picture you can actually look at. PCA keeps the global shape but cannot unbend a curved manifold; t-SNE draws gorgeous clusters whose sizes and spacing mean nothing. Run both on the same data and watch where each one cheats.",
  date: "2026-06-02",
  tags: ["pca", "t-sne", "dimensionality-reduction", "embeddings"],
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
};

const EASE = "cubic-bezier(0.23, 1, 0.32, 1)";

const CLUSTER_COLORS = ["#c0561f", "#2f6f9e", "#3f7d52", "#9a6b1f", "#7d4a8c", "#b23b5e"];

function mulberry32(seed) {
  let a = seed >>> 0;
  return () => {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function makeRandn(rng) {
  return () => {
    let u = 0, v = 0;
    while (u === 0) u = rng();
    while (v === 0) v = rng();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  };
}

const DATASETS = {
  blobs: {
    label: "Separated blobs",
    dim: 12,
    blurb:
      "Three compact Gaussian clusters placed far apart in 12 dimensions. The honest case: both methods should keep the three groups apart.",
  },
  scurve: {
    label: "S-curve manifold",
    dim: 10,
    blurb:
      "Points sampled along a curved 1D path (an S laid out in 3D) then embedded in 10D with small noise. The data is intrinsically a line that has been bent. PCA can only rotate and flatten, so the two ends of the S collapse on top of each other; t-SNE follows neighbors along the curve and unrolls it.",
  },
  uneven: {
    label: "Unequal clusters",
    dim: 12,
    blurb:
      "Three clusters of very different spread: one tight, one medium, one diffuse, plus unequal point counts. PCA preserves their true relative sizes; t-SNE inflates the tight one and shrinks the diffuse one until all three look about the same. Reading size off a t-SNE plot is reading noise.",
  },
};

function generateBlobs(rng, dim) {
  const randn = makeRandn(rng);
  const k = 3;
  const perCluster = 50;
  const centers = [];
  for (let c = 0; c < k; c++) {
    const center = new Array(dim).fill(0);
    for (let d = 0; d < dim; d++) center[d] = (rng() * 2 - 1) * 0.4;
    center[c % dim] += 9;
    centers.push(center);
  }
  const X = [];
  const labels = [];
  for (let c = 0; c < k; c++) {
    for (let i = 0; i < perCluster; i++) {
      const row = new Array(dim);
      for (let d = 0; d < dim; d++) row[d] = centers[c][d] + randn() * 0.7;
      X.push(row);
      labels.push(c);
    }
  }
  return { X, labels, nClusters: k };
}

function generateScurve(rng, dim) {
  const randn = makeRandn(rng);
  const n = 150;
  const X = [];
  const labels = [];
  for (let i = 0; i < n; i++) {
    const t = (i / (n - 1)) * 3 * Math.PI - 1.5 * Math.PI;
    const sx = Math.sin(t);
    const sy = 2 * (rng() - 0.5);
    const sz = Math.sign(t) * (Math.cos(t) - 1);
    const row = new Array(dim).fill(0);
    row[0] = sx * 3;
    row[1] = sy;
    row[2] = sz * 3;
    for (let d = 3; d < dim; d++) row[d] = randn() * 0.12;
    X.push(row);
    // colour by position along the curve so a continuous gradient is visible
    labels.push(Math.min(2, Math.floor(((t + 1.5 * Math.PI) / (3 * Math.PI)) * 3)));
  }
  return { X, labels, nClusters: 3 };
}

function generateUneven(rng, dim) {
  const randn = makeRandn(rng);
  const specs = [
    { center: 0, spread: 0.35, count: 70 },
    { center: 1, spread: 1.4, count: 55 },
    { center: 2, spread: 3.0, count: 40 },
  ];
  const X = [];
  const labels = [];
  specs.forEach((spec, c) => {
    const center = new Array(dim).fill(0);
    center[c % dim] += 11 + c * 2;
    for (let i = 0; i < spec.count; i++) {
      const row = new Array(dim);
      for (let d = 0; d < dim; d++) row[d] = center[d] + randn() * spec.spread;
      X.push(row);
      labels.push(c);
    }
  });
  return { X, labels, nClusters: 3 };
}

function generateDataset(kind, seed) {
  const rng = mulberry32(seed);
  const dim = DATASETS[kind].dim;
  if (kind === "scurve") return generateScurve(rng, dim);
  if (kind === "uneven") return generateUneven(rng, dim);
  return generateBlobs(rng, dim);
}

function centerColumns(X) {
  const n = X.length;
  const d = X[0].length;
  const mean = new Array(d).fill(0);
  for (let i = 0; i < n; i++) for (let j = 0; j < d; j++) mean[j] += X[i][j];
  for (let j = 0; j < d; j++) mean[j] /= n;
  return X.map((row) => row.map((v, j) => v - mean[j]));
}

function covariance(Xc) {
  const n = Xc.length;
  const d = Xc[0].length;
  const cov = Array.from({ length: d }, () => new Array(d).fill(0));
  for (let i = 0; i < n; i++) {
    const row = Xc[i];
    for (let a = 0; a < d; a++) {
      const ra = row[a];
      for (let b = a; b < d; b++) {
        cov[a][b] += ra * row[b];
      }
    }
  }
  const denom = n - 1;
  for (let a = 0; a < d; a++) {
    for (let b = a; b < d; b++) {
      cov[a][b] /= denom;
      cov[b][a] = cov[a][b];
    }
  }
  return cov;
}

// Jacobi eigenvalue rotation for a real symmetric matrix. Returns eigenvalues
// and the matrix V whose columns are the orthonormal eigenvectors.
function jacobiEigen(Ain, maxSweeps = 100) {
  const n = Ain.length;
  const A = Ain.map((row) => row.slice());
  const V = Array.from({ length: n }, (_, i) =>
    Array.from({ length: n }, (_, j) => (i === j ? 1 : 0))
  );

  const offDiagNorm = () => {
    let s = 0;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) s += A[p][q] * A[p][q];
    return Math.sqrt(s);
  };

  for (let sweep = 0; sweep < maxSweeps; sweep++) {
    if (offDiagNorm() < 1e-12) break;
    for (let p = 0; p < n; p++) {
      for (let q = p + 1; q < n; q++) {
        const apq = A[p][q];
        if (Math.abs(apq) < 1e-300) continue;
        const app = A[p][p];
        const aqq = A[q][q];
        const phi = 0.5 * Math.atan2(2 * apq, aqq - app);
        const c = Math.cos(phi);
        const s = Math.sin(phi);
        for (let k = 0; k < n; k++) {
          const akp = A[k][p];
          const akq = A[k][q];
          A[k][p] = c * akp - s * akq;
          A[k][q] = s * akp + c * akq;
        }
        for (let k = 0; k < n; k++) {
          const apk = A[p][k];
          const aqk = A[q][k];
          A[p][k] = c * apk - s * aqk;
          A[q][k] = s * apk + c * aqk;
        }
        for (let k = 0; k < n; k++) {
          const vkp = V[k][p];
          const vkq = V[k][q];
          V[k][p] = c * vkp - s * vkq;
          V[k][q] = s * vkp + c * vkq;
        }
      }
    }
  }

  const eigenvalues = A.map((row, i) => row[i]);
  const order = eigenvalues.map((v, i) => i).sort((a, b) => eigenvalues[b] - eigenvalues[a]);
  const sortedVals = order.map((i) => eigenvalues[i]);
  const vectors = order.map((idx) => V.map((row) => row[idx]));
  return { values: sortedVals, vectors };
}

function runPCA(X) {
  const Xc = centerColumns(X);
  const cov = covariance(Xc);
  const { values, vectors } = jacobiEigen(cov);
  const pc1 = vectors[0];
  const pc2 = vectors[1];
  const coords = Xc.map((row) => {
    let p1 = 0, p2 = 0;
    for (let j = 0; j < row.length; j++) {
      p1 += row[j] * pc1[j];
      p2 += row[j] * pc2[j];
    }
    return [p1, p2];
  });
  const totalVar = values.reduce((a, v) => a + Math.max(v, 0), 0) || 1;
  return {
    coords,
    explained: [Math.max(values[0], 0) / totalVar, Math.max(values[1], 0) / totalVar],
    eigvals: values,
  };
}

function pairwiseSqDist(X) {
  const n = X.length;
  const D = Array.from({ length: n }, () => new Float64Array(n));
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      let s = 0;
      const xi = X[i];
      const xj = X[j];
      for (let d = 0; d < xi.length; d++) {
        const diff = xi[d] - xj[d];
        s += diff * diff;
      }
      D[i][j] = s;
      D[j][i] = s;
    }
  }
  return D;
}

// Per-point Gaussian bandwidth chosen by binary search so that the perplexity
// (effective neighbour count) of the conditional distribution matches target.
function conditionalP(D, perplexity) {
  const n = D.length;
  const P = Array.from({ length: n }, () => new Float64Array(n));
  const logU = Math.log(perplexity);
  for (let i = 0; i < n; i++) {
    let betaMin = -Infinity;
    let betaMax = Infinity;
    let beta = 1;
    const Di = D[i];
    let row = new Float64Array(n);
    for (let iter = 0; iter < 50; iter++) {
      let sum = 0;
      for (let j = 0; j < n; j++) {
        if (j === i) { row[j] = 0; continue; }
        const v = Math.exp(-Di[j] * beta);
        row[j] = v;
        sum += v;
      }
      if (sum < 1e-12) sum = 1e-12;
      let H = 0;
      for (let j = 0; j < n; j++) {
        if (j === i) continue;
        const p = row[j] / sum;
        if (p > 1e-12) H += -p * Math.log(p);
      }
      const diff = H - logU;
      if (Math.abs(diff) < 1e-5) {
        for (let j = 0; j < n; j++) P[i][j] = row[j] / sum;
        break;
      }
      if (diff > 0) {
        betaMin = beta;
        beta = betaMax === Infinity ? beta * 2 : (beta + betaMax) / 2;
      } else {
        betaMax = beta;
        beta = betaMin === -Infinity ? beta / 2 : (beta + betaMin) / 2;
      }
      if (iter === 49) for (let j = 0; j < n; j++) P[i][j] = row[j] / sum;
    }
  }
  return P;
}

function symmetrizeP(P) {
  const n = P.length;
  const sym = Array.from({ length: n }, () => new Float64Array(n));
  let total = 0;
  for (let i = 0; i < n; i++) {
    for (let j = 0; j < n; j++) {
      sym[i][j] = P[i][j] + P[j][i];
      total += sym[i][j];
    }
  }
  if (total < 1e-12) total = 1e-12;
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) sym[i][j] = Math.max(sym[i][j] / total, 1e-12);
  return sym;
}

function makeTSNE(X, perplexity, seed) {
  const n = X.length;
  const D = pairwiseSqDist(X);
  const P = symmetrizeP(conditionalP(D, perplexity));
  const rng = mulberry32(seed ^ 0x9e3779b9);
  const randn = makeRandn(rng);
  const Y = Array.from({ length: n }, () => [randn() * 1e-2, randn() * 1e-2]);
  const gains = Array.from({ length: n }, () => [1, 1]);
  const incs = Array.from({ length: n }, () => [0, 0]);

  const EARLY_EXAGGERATION = 4;
  const EARLY_ITERS = 100;
  const baseLR = Math.max(50, n / 4);

  let iter = 0;

  function computeKL() {
    let sumQ = 0;
    const num = Array.from({ length: n }, () => new Float64Array(n));
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        const dx = Y[i][0] - Y[j][0];
        const dy = Y[i][1] - Y[j][1];
        const q = 1 / (1 + dx * dx + dy * dy);
        num[i][j] = q;
        num[j][i] = q;
        sumQ += 2 * q;
      }
    }
    if (sumQ < 1e-12) sumQ = 1e-12;
    let kl = 0;
    for (let i = 0; i < n; i++) {
      for (let j = 0; j < n; j++) {
        if (i === j) continue;
        const q = Math.max(num[i][j] / sumQ, 1e-12);
        kl += P[i][j] * Math.log(P[i][j] / q);
      }
    }
    return kl;
  }

  function step(momentum, lr) {
    const exaggeration = iter < EARLY_ITERS ? EARLY_EXAGGERATION : 1;
    const num = Array.from({ length: n }, () => new Float64Array(n));
    let sumQ = 0;
    for (let i = 0; i < n; i++) {
      for (let j = i + 1; j < n; j++) {
        const dx = Y[i][0] - Y[j][0];
        const dy = Y[i][1] - Y[j][1];
        const q = 1 / (1 + dx * dx + dy * dy);
        num[i][j] = q;
        num[j][i] = q;
        sumQ += 2 * q;
      }
    }
    if (sumQ < 1e-12) sumQ = 1e-12;

    const grad = Array.from({ length: n }, () => [0, 0]);
    for (let i = 0; i < n; i++) {
      let gx = 0, gy = 0;
      for (let j = 0; j < n; j++) {
        if (i === j) continue;
        const qUnnorm = num[i][j];
        const q = Math.max(qUnnorm / sumQ, 1e-12);
        const mult = (exaggeration * P[i][j] - q) * qUnnorm;
        gx += mult * (Y[i][0] - Y[j][0]);
        gy += mult * (Y[i][1] - Y[j][1]);
      }
      grad[i][0] = 4 * gx;
      grad[i][1] = 4 * gy;
    }

    for (let i = 0; i < n; i++) {
      for (let d = 0; d < 2; d++) {
        const g = grad[i][d];
        const sameSign = (g > 0) === (incs[i][d] > 0);
        gains[i][d] = sameSign ? gains[i][d] * 0.8 : gains[i][d] + 0.2;
        if (gains[i][d] < 0.01) gains[i][d] = 0.01;
        incs[i][d] = momentum * incs[i][d] - lr * gains[i][d] * g;
        Y[i][d] += incs[i][d];
      }
    }

    let mx = 0, my = 0;
    for (let i = 0; i < n; i++) { mx += Y[i][0]; my += Y[i][1]; }
    mx /= n; my /= n;
    for (let i = 0; i < n; i++) { Y[i][0] -= mx; Y[i][1] -= my; }

    iter++;
  }

  return {
    n,
    P,
    D,
    get iter() { return iter; },
    coords: () => Y.map((p) => [p[0], p[1]]),
    step: () => {
      const momentum = iter < 250 ? 0.5 : 0.8;
      step(momentum, baseLR);
    },
    kl: computeKL,
  };
}

// Trustworthiness: fraction of low-D neighbours that were also high-D
// neighbours, penalised by how far the intruders sat in the original space.
function trustworthiness(D, coords, k = 8) {
  const n = D.length;
  const lowD = pairwiseSqDist(coords);
  const rankHigh = Array.from({ length: n }, () => new Int32Array(n));
  for (let i = 0; i < n; i++) {
    const order = Array.from({ length: n }, (_, j) => j)
      .filter((j) => j !== i)
      .sort((a, b) => D[i][a] - D[i][b]);
    order.forEach((j, r) => { rankHigh[i][j] = r + 1; });
  }
  let sum = 0;
  for (let i = 0; i < n; i++) {
    const order = Array.from({ length: n }, (_, j) => j)
      .filter((j) => j !== i)
      .sort((a, b) => lowD[i][a] - lowD[i][b]);
    const lowNeighbors = order.slice(0, k);
    for (const j of lowNeighbors) {
      const r = rankHigh[i][j];
      if (r > k) sum += r - k;
    }
  }
  const norm = (2 / (n * k * (2 * n - 3 * k - 1))) || 0;
  return 1 - norm * sum;
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

function Stat({ label, value, accent }) {
  return (
    <div style={{ background: accent ? C.accentSoft : C.bg, borderRadius: 9, padding: "9px 12px", border: `1px solid ${accent ? C.accent + "33" : C.border}` }}>
      <div style={{ fontSize: 10, letterSpacing: "0.06em", textTransform: "uppercase", color: accent ? C.accent : C.muted, fontWeight: 700, marginBottom: 3 }}>{label}</div>
      <div style={{ fontFamily: "'IBM Plex Mono', ui-monospace, SFMono-Regular, Menlo, monospace", fontSize: 17, fontWeight: 700, color: accent ? C.accent : C.ink, lineHeight: 1 }}>{value}</div>
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
    transition: `transform 140ms ${EASE}, background 160ms ${EASE}, border-color 160ms ${EASE}`,
  };
}

function ScatterCanvas({ coords, labels, title, subtitle, width = 300, height = 300 }) {
  const ref = useRef(null);
  useEffect(() => {
    const cv = ref.current;
    if (!cv || !coords || coords.length === 0) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    cv.width = width * dpr;
    cv.height = height * dpr;
    const ctx = cv.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);

    let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
    for (const [x, y] of coords) {
      if (x < minX) minX = x;
      if (x > maxX) maxX = x;
      if (y < minY) minY = y;
      if (y > maxY) maxY = y;
    }
    const spanX = maxX - minX || 1;
    const spanY = maxY - minY || 1;
    const span = Math.max(spanX, spanY);
    const pad = 18;
    const cx = (minX + maxX) / 2;
    const cy = (minY + maxY) / 2;
    const scale = (Math.min(width, height) - 2 * pad) / span;
    const toPx = (x, y) => [
      width / 2 + (x - cx) * scale,
      height / 2 - (y - cy) * scale,
    ];

    ctx.strokeStyle = C.grid;
    ctx.lineWidth = 1;
    ctx.strokeRect(0.5, 0.5, width - 1, height - 1);

    for (let i = 0; i < coords.length; i++) {
      const [px, py] = toPx(coords[i][0], coords[i][1]);
      ctx.fillStyle = CLUSTER_COLORS[labels[i] % CLUSTER_COLORS.length];
      ctx.globalAlpha = 0.78;
      ctx.beginPath();
      ctx.arc(px, py, 2.6, 0, 2 * Math.PI);
      ctx.fill();
    }
    ctx.globalAlpha = 1;
  }, [coords, labels, width, height]);

  return (
    <div style={{ flex: "1 1 280px", minWidth: 0 }}>
      <div style={{ fontSize: 13, fontWeight: 700, color: C.ink }}>{title}</div>
      <div style={{ fontSize: 11.5, color: C.muted, margin: "2px 0 8px", lineHeight: 1.4, minHeight: 30 }}>{subtitle}</div>
      <canvas
        ref={ref}
        style={{ width, height, maxWidth: "100%", borderRadius: 10, background: C.bg, border: `1px solid ${C.border}`, display: "block" }}
        role="img"
        aria-label={`${title} scatter plot, points coloured by their true cluster`}
      />
    </div>
  );
}

const MAX_ITERS = 600;

export default function App() {
  const [datasetKind, setDatasetKind] = useState("scurve");
  const [seed, setSeed] = useState(7);
  const [perplexity, setPerplexity] = useState(30);

  const dataset = useMemo(() => generateDataset(datasetKind, seed), [datasetKind, seed]);
  const pca = useMemo(() => runPCA(dataset.X), [dataset]);

  const engineRef = useRef(null);
  const [tsneCoords, setTsneCoords] = useState(null);
  const [tsneIter, setTsneIter] = useState(0);
  const [kl, setKl] = useState(null);
  const [playing, setPlaying] = useState(false);
  const [trust, setTrust] = useState({ pca: null, tsne: null });
  const raf = useRef(0);
  const lastTick = useRef(0);

  const resetTSNE = useCallback(() => {
    setPlaying(false);
    const engine = makeTSNE(dataset.X, perplexity, seed);
    engineRef.current = engine;
    setTsneCoords(engine.coords());
    setTsneIter(0);
    setKl(engine.kl());
    setTrust({
      pca: trustworthiness(engine.D, pca.coords),
      tsne: null,
    });
  }, [dataset, perplexity, seed, pca]);

  useEffect(() => { resetTSNE(); }, [resetTSNE]);

  const stepOnce = useCallback(() => {
    const engine = engineRef.current;
    if (!engine || engine.iter >= MAX_ITERS) return false;
    engine.step();
    setTsneCoords(engine.coords());
    setTsneIter(engine.iter);
    if (engine.iter % 10 === 0 || engine.iter >= MAX_ITERS) {
      setKl(engine.kl());
    }
    return engine.iter < MAX_ITERS;
  }, []);

  useEffect(() => {
    if (!playing) return;
    const reduce = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const interval = reduce ? 60 : 0;
    const loop = (ts) => {
      if (ts - lastTick.current >= interval) {
        lastTick.current = ts;
        const more = stepOnce();
        if (!more) {
          setPlaying(false);
          const engine = engineRef.current;
          if (engine) {
            setKl(engine.kl());
            setTrust((prev) => ({ ...prev, tsne: trustworthiness(engine.D, engine.coords()) }));
          }
          return;
        }
      }
      raf.current = requestAnimationFrame(loop);
    };
    raf.current = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf.current);
  }, [playing, stepOnce]);

  useEffect(() => {
    if (tsneIter >= 200 && engineRef.current) {
      setTrust((prev) => (prev.tsne === null ? { ...prev, tsne: trustworthiness(engineRef.current.D, engineRef.current.coords()) } : prev));
    }
  }, [tsneIter]);

  const done = tsneIter >= MAX_ITERS;
  const ds = DATASETS[datasetKind];

  const pcaSubtitle = `PC1 + PC2 hold ${((pca.explained[0] + pca.explained[1]) * 100).toFixed(0)}% of total variance. Distances here are real projected distances.`;
  const tsneSubtitle = done
    ? "Converged. Tight local groups, but cluster size and the gaps between groups carry no meaning."
    : tsneIter === 0
      ? "Press Run. The embedding starts as a tiny random cloud and is pulled into shape by the KL gradient."
      : `Iteration ${tsneIter}${tsneIter < 100 ? " (early exaggeration on)" : ""}. KL is being minimised step by step.`;

  return (
    <div style={{ fontFamily: "'Iowan Old Style', 'Palatino Linotype', Palatino, Georgia, serif", background: C.bg, minHeight: "100vh", padding: "30px 14px 56px", color: C.ink }}>
      <style>{`
        @media (prefers-reduced-motion: reduce) {
          * { transition-duration: 1ms !important; animation-duration: 1ms !important; }
        }
        .pt-btn:active { transform: scale(0.97); }
        .pt-btn:focus-visible, .pt-tab:focus-visible, input[type=range]:focus-visible {
          outline: 2px solid ${C.accent}; outline-offset: 2px;
        }
        input[type=range] { accent-color: ${C.accent}; }
      `}</style>

      <div style={{ maxWidth: 720, margin: "0 auto" }}>
        <header style={{ marginBottom: 22 }}>
          <Eyebrow>Machine Learning · Dimensionality Reduction</Eyebrow>
          <h1 style={{ fontSize: 30, fontWeight: 700, margin: "6px 0 0", lineHeight: 1.1, textWrap: "balance" }}>
            PCA vs t-SNE: What Each One Lies About
          </h1>
          <p style={{ color: C.ink, opacity: 0.82, fontSize: 15, margin: "10px 0 0", lineHeight: 1.6, maxWidth: "64ch" }}>
            High-dimensional data has to be squashed to two dimensions before you can see it, and every method
            distorts something on the way down. PCA finds the directions of greatest variance and projects onto
            them: linear, global, honest about distance, but unable to bend. t-SNE matches the neighbourhood
            structure instead, drawing crisp clusters that look definitive while quietly throwing away cluster size
            and the distances between groups. Both run for real below, on the same data, so you can see exactly
            where each one cheats.
          </p>
        </header>

        <Card style={{ marginBottom: 18, background: C.accentSoft, borderColor: `${C.accent}33` }}>
          <Eyebrow>The dataset</Eyebrow>
          <div style={{ display: "flex", gap: 8, flexWrap: "wrap", margin: "12px 0 12px" }}>
            {Object.entries(DATASETS).map(([k, v]) => (
              <button key={k} className="pt-btn pt-tab" onClick={() => setDatasetKind(k)}
                style={btnStyle(datasetKind === k)} aria-pressed={datasetKind === k}>
                {v.label}
              </button>
            ))}
          </div>
          <p style={{ fontSize: 13, color: C.ink, opacity: 0.85, margin: 0, lineHeight: 1.6 }}>
            {ds.blurb}
          </p>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(110px, 1fr))", gap: 8, marginTop: 14 }}>
            <Stat label="points" value={dataset.X.length} />
            <Stat label="dimensions" value={ds.dim} />
            <Stat label="true clusters" value={dataset.nClusters} />
            <Stat label="seed" value={seed} accent />
          </div>
        </Card>

        <Card style={{ marginBottom: 18 }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", gap: 10, flexWrap: "wrap", marginBottom: 4 }}>
            <div>
              <Eyebrow>Side by side</Eyebrow>
              <h2 style={{ fontSize: 19, fontWeight: 700, margin: "5px 0 0" }}>The same points, two projections</h2>
            </div>
            <span style={{ fontSize: 12, color: done ? C.good : C.muted, fontWeight: 700 }}>
              {done ? "t-SNE converged" : tsneIter > 0 ? `iter ${tsneIter}` : "t-SNE idle"}
            </span>
          </div>

          <div style={{ display: "flex", gap: 18, flexWrap: "wrap", marginTop: 12 }}>
            <ScatterCanvas coords={pca.coords} labels={dataset.labels} title="PCA" subtitle={pcaSubtitle} />
            <ScatterCanvas coords={tsneCoords} labels={dataset.labels} title="t-SNE" subtitle={tsneSubtitle} />
          </div>

          <div style={{ display: "flex", gap: 8, flexWrap: "wrap", alignItems: "center", marginTop: 16 }}>
            <button className="pt-btn" onClick={() => setPlaying((p) => !p)} disabled={done}
              style={btnStyle(true, done)}>
              {playing ? "Pause" : done ? "Done" : tsneIter > 0 ? "Resume" : "Run t-SNE"}
            </button>
            <button className="pt-btn" onClick={() => { setPlaying(false); stepOnce(); }} disabled={done}
              style={btnStyle(false, done)}>
              Step
            </button>
            <button className="pt-btn" onClick={resetTSNE} style={btnStyle(false)}>
              Reset embedding
            </button>
            <button className="pt-btn" onClick={() => setSeed((s) => ((Math.imul(s, 1664525) + 1013904223) >>> 0) % 100000)} style={btnStyle(false)}>
              Reseed data + init
            </button>
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(130px, 1fr))", gap: 8, marginTop: 16 }}>
            <Stat label="PC1 variance" value={`${(pca.explained[0] * 100).toFixed(1)}%`} />
            <Stat label="PC2 variance" value={`${(pca.explained[1] * 100).toFixed(1)}%`} />
            <Stat label="t-SNE iter" value={tsneIter} accent />
            <Stat label="KL divergence" value={kl === null ? "—" : kl.toFixed(4)} accent />
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8, marginTop: 8 }}>
            <Stat label="PCA trustworthiness" value={trust.pca === null ? "—" : trust.pca.toFixed(3)} />
            <Stat label="t-SNE trustworthiness" value={trust.tsne === null ? "run to ≥200" : trust.tsne.toFixed(3)} />
          </div>
          <p style={{ fontSize: 12, color: C.muted, margin: "10px 0 0", lineHeight: 1.5 }}>
            Trustworthiness (k = 8 neighbours, higher is better) asks: of the points that look like near
            neighbours in the 2D picture, how many were genuinely close in the original space. It rewards t-SNE for
            preserving local structure, which is exactly what t-SNE optimises for.
          </p>
        </Card>

        <Card style={{ marginBottom: 18 }}>
          <Eyebrow>The perplexity knob</Eyebrow>
          <h2 style={{ fontSize: 19, fontWeight: 700, margin: "5px 0 12px" }}>t-SNE has a free parameter that reshapes the answer</h2>
          <div style={{ display: "flex", gap: 12, alignItems: "center", flexWrap: "wrap" }}>
            <label htmlFor="perp" style={{ fontSize: 13, color: C.muted, minWidth: 90 }}>Perplexity</label>
            <input id="perp" type="range" min={5} max={80} step={1} value={perplexity}
              onChange={(e) => setPerplexity(+e.target.value)} aria-label="t-SNE perplexity"
              style={{ flex: "1 1 200px", minWidth: 140 }} />
            <span style={{ fontFamily: "'IBM Plex Mono', monospace", fontSize: 16, fontWeight: 700, color: C.accent, minWidth: 30 }}>{perplexity}</span>
          </div>
          <p style={{ fontSize: 13, color: C.ink, opacity: 0.85, margin: "12px 0 0", lineHeight: 1.6 }}>
            Perplexity sets roughly how many neighbours each point tries to stay close to. It picks the Gaussian
            bandwidth per point through a binary search so the effective neighbour count matches this target.
            Changing it rebuilds the embedding from scratch (watch the plot above reset). Low perplexity fractures
            the data into many tiny islands; high perplexity blurs real clusters together. There is no single
            correct value, which is the point: the same data yields different pictures depending on a knob you set
            by hand. PCA has no such knob.
          </p>
        </Card>

        <Card style={{ marginBottom: 18 }}>
          <Eyebrow>The seed</Eyebrow>
          <h2 style={{ fontSize: 19, fontWeight: 700, margin: "5px 0 10px" }}>t-SNE is random; PCA is not</h2>
          <p style={{ fontSize: 13, color: C.ink, opacity: 0.85, margin: 0, lineHeight: 1.6 }}>
            Press <b>Reset embedding</b> a few times. The PCA panel never moves: it is a deterministic eigen
            decomposition of the covariance matrix, so the projection is fixed by the data alone. The t-SNE panel
            lands differently each time because it starts from a random initial cloud and descends a non-convex
            objective. Cluster shapes, orientations, and which side each group ends up on are artifacts of that
            random start, not facts about the data. <b>Reseed data + init</b> draws a fresh dataset too, so you can
            confirm the structure is stable while the t-SNE rendering of it is not.
          </p>
        </Card>

        <Card>
          <Eyebrow>How to read each plot honestly</Eyebrow>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(260px, 1fr))", gap: 14, marginTop: 12 }}>
            <div>
              <div style={{ fontSize: 14, fontWeight: 700, color: C.accent, marginBottom: 6 }}>PCA</div>
              <p style={{ fontSize: 13, color: C.ink, opacity: 0.85, margin: 0, lineHeight: 1.6 }}>
                The 2D axes are the top two eigenvectors of the covariance matrix, the directions along which the
                data varies most. Distances and the overall layout are trustworthy projections of the real
                geometry. The cost: anything that lives along a low-variance direction gets squashed, and a curved
                manifold cannot be unbent by a linear map, so the S-curve folds onto itself and clusters that
                separate only along a thin direction can hide.
              </p>
            </div>
            <div>
              <div style={{ fontSize: 14, fontWeight: 700, color: "#2f6f9e", marginBottom: 6 }}>t-SNE</div>
              <p style={{ fontSize: 13, color: C.ink, opacity: 0.85, margin: 0, lineHeight: 1.6 }}>
                It minimises the KL divergence between high-D affinities p and low-D Student-t affinities q, so
                near neighbours stay near and everything else is free to move. That gives clean, well-separated
                clusters. The caveats are firm: cluster size is meaningless (dense and diffuse groups are inflated
                to similar blobs), the distance between two clusters is not proportional to their real separation,
                and the result depends on perplexity and the random seed. Read membership, never geometry.
              </p>
            </div>
          </div>
          <div style={{ marginTop: 14, padding: "11px 14px", background: C.bg, borderRadius: 10, border: `1px solid ${C.border}`, fontSize: 12.5, color: C.muted, lineHeight: 1.6 }}>
            <b style={{ color: C.ink }}>What is real here:</b> PCA is the exact top-2 eigenvectors of the data
            covariance via Jacobi rotation, with the genuine explained-variance ratios. t-SNE is the real
            algorithm: per-point sigma found by binary search to the target perplexity, symmetrised affinities,
            Student-t low-D kernel, KL gradient with momentum, adaptive gains, and early exaggeration. Point counts
            are kept near 150 so the O(n squared) gradient runs at interactive speed in the browser.
          </div>
        </Card>
      </div>
    </div>
  );
}
