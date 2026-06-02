import { useState, useEffect, useRef, useMemo, useCallback } from "react";

export const meta = {
  title: "Decision Boundary Zoo",
  category: "Machine Learning",
  description:
    "A straight line cannot separate two moons. Train four classifiers on the same tangled data and watch the linear model fail while kNN, an RBF SVM, and a small neural net curve around it.",
  date: "2026-05-20",
  tags: ["classification", "decision-boundary", "svm", "neural-networks", "knn"],
};

const C = {
  bg: "#f7f5f2",
  card: "#ffffff",
  border: "#e6e1d9",
  ink: "#1b1916",
  muted: "#857c70",
  accent: "#c0561f",
  accentSoft: "#f6ece5",
  classA: "#2a5298",
  classB: "#c0561f",
  classAFill: [42, 82, 152],
  classBFill: [192, 86, 31],
  paper: [247, 245, 242],
};

const EASE = "cubic-bezier(0.23, 1, 0.32, 1)";

const sigmoid = (x) => 1 / (1 + Math.exp(-x));

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

function makeGaussian(rng) {
  let spare = null;
  return () => {
    if (spare !== null) {
      const v = spare;
      spare = null;
      return v;
    }
    let u = 0;
    let v = 0;
    while (u === 0) u = rng();
    while (v === 0) v = rng();
    const mag = Math.sqrt(-2 * Math.log(u));
    spare = mag * Math.sin(2 * Math.PI * v);
    return mag * Math.cos(2 * Math.PI * v);
  };
}

// All datasets live in roughly [-3, 3] on both axes so one shared viewport and
// one grid resolution serve every classifier without rescaling.
const DOMAIN = { min: -3.2, max: 3.2 };

const DATASETS = {
  blobs: {
    label: "Blobs",
    blurb:
      "Two Gaussian clusters with a clear gap between them. A single straight line separates the classes, so even logistic regression nails it. This is the easy case that every model should handle.",
    generate: (seed, n, noise) => {
      const rng = mulberry32(seed);
      const gauss = makeGaussian(rng);
      const pts = [];
      const spread = 0.55 + noise * 1.1;
      const centers = [
        [-1.4, -1.0],
        [1.4, 1.0],
      ];
      for (let i = 0; i < n; i++) {
        const cls = i % 2;
        const [cx, cy] = centers[cls];
        pts.push({ x: cx + gauss() * spread, y: cy + gauss() * spread, label: cls });
      }
      return pts;
    },
  },
  moons: {
    label: "Moons",
    blurb:
      "Two interleaving half circles. No straight line can split them cleanly, but a curve threading between the arcs can. A linear model gets the gist and stalls in the high eighties, while the nonlinear models climb past it.",
    generate: (seed, n, noise) => {
      const rng = mulberry32(seed);
      const gauss = makeGaussian(rng);
      const pts = [];
      const half = Math.floor(n / 2);
      for (let i = 0; i < n; i++) {
        const cls = i < half ? 0 : 1;
        const t = rng() * Math.PI;
        let px;
        let py;
        if (cls === 0) {
          px = Math.cos(t) * 1.7 - 0.85;
          py = Math.sin(t) * 1.7 - 0.55;
        } else {
          px = 0.85 - Math.cos(t) * 1.7;
          py = 0.55 - Math.sin(t) * 1.7;
        }
        const j = 0.22 + noise * 0.7;
        pts.push({ x: px + gauss() * j, y: py + gauss() * j, label: cls });
      }
      return pts;
    },
  },
  circles: {
    label: "Circles",
    blurb:
      "One class sits inside a ring of the other. The boundary is a closed loop, the worst case for a straight line: logistic regression collapses to guessing. A kernel or a hidden layer can wrap the loop.",
    generate: (seed, n, noise) => {
      const rng = mulberry32(seed);
      const gauss = makeGaussian(rng);
      const pts = [];
      const half = Math.floor(n / 2);
      for (let i = 0; i < n; i++) {
        const cls = i < half ? 0 : 1;
        const r = cls === 0 ? 0.8 : 2.1;
        const t = rng() * 2 * Math.PI;
        const j = 0.12 + noise * 0.55;
        pts.push({ x: Math.cos(t) * r + gauss() * j, y: Math.sin(t) * r + gauss() * j, label: cls });
      }
      return pts;
    },
  },
  xor: {
    label: "XOR",
    blurb:
      "Four quadrants in a checkerboard: top-right and bottom-left are one class, the other diagonal is the second. Any single line gets exactly half right, so a linear model is stuck at chance.",
    generate: (seed, n, noise) => {
      const rng = mulberry32(seed);
      const gauss = makeGaussian(rng);
      const pts = [];
      for (let i = 0; i < n; i++) {
        const qx = rng() < 0.5 ? -1 : 1;
        const qy = rng() < 0.5 ? -1 : 1;
        const cls = qx * qy > 0 ? 0 : 1;
        const cx = qx * 1.4;
        const cy = qy * 1.4;
        const spread = 0.5 + noise * 0.7;
        pts.push({ x: cx + gauss() * spread, y: cy + gauss() * spread, label: cls });
      }
      return pts;
    },
  },
};

// Logistic regression by full-batch gradient descent on the logistic loss with
// an L2 penalty. Linear in the two raw features, so the boundary is a line.
function trainLogistic(points, opts) {
  const epochs = opts.epochs ?? 300;
  const lr = opts.lr ?? 0.3;
  const l2 = opts.l2 ?? 0.001;
  let w0 = 0;
  let w1 = 0;
  let b = 0;
  const n = points.length || 1;
  for (let e = 0; e < epochs; e++) {
    let g0 = 0;
    let g1 = 0;
    let gb = 0;
    for (const p of points) {
      const z = w0 * p.x + w1 * p.y + b;
      const err = sigmoid(z) - p.label;
      g0 += err * p.x;
      g1 += err * p.y;
      gb += err;
    }
    w0 -= lr * (g0 / n + l2 * w0);
    w1 -= lr * (g1 / n + l2 * w1);
    b -= lr * (gb / n);
  }
  return {
    score: (x, y) => sigmoid(w0 * x + w1 * y + b),
  };
}

// k-Nearest-Neighbors. No training: the score at a query point is the fraction
// of its k closest training points that belong to class 1.
function trainKNN(points, opts) {
  const k = Math.max(1, Math.min(opts.k ?? 5, points.length));
  return {
    score: (x, y) => {
      if (points.length === 0) return 0.5;
      const nearest = [];
      for (const p of points) {
        const d = (p.x - x) * (p.x - x) + (p.y - y) * (p.y - y);
        if (nearest.length < k) {
          nearest.push({ d, label: p.label });
          nearest.sort((a, b) => a.d - b.d);
        } else if (d < nearest[k - 1].d) {
          nearest[k - 1] = { d, label: p.label };
          nearest.sort((a, b) => a.d - b.d);
        }
      }
      let ones = 0;
      for (const m of nearest) ones += m.label;
      return ones / nearest.length;
    },
  };
}

// RBF random-feature SVM. We approximate the Gaussian kernel with a fixed bank
// of random Fourier features (Rahimi and Recht), then train a linear SVM in that
// feature space with Pegasos (subgradient descent on the hinge loss). This is a
// genuinely nonlinear classifier: the explicit feature map carries the curvature.
function trainKernelSVM(points, opts) {
  const gamma = opts.gamma ?? 1.0;
  const C2 = opts.C ?? 1.0;
  const D = 80;
  const seed = opts.seed ?? 1;
  const rng = mulberry32((seed * 2654435761) >>> 0);
  const gauss = makeGaussian(rng);
  const omega = new Float64Array(D * 2);
  const phase = new Float64Array(D);
  const scale = Math.sqrt(2 * gamma);
  for (let j = 0; j < D; j++) {
    omega[j * 2] = gauss() * scale;
    omega[j * 2 + 1] = gauss() * scale;
    phase[j] = rng() * 2 * Math.PI;
  }
  const norm = Math.sqrt(2 / D);
  const feat = (x, y) => {
    const f = new Float64Array(D);
    for (let j = 0; j < D; j++) {
      f[j] = norm * Math.cos(omega[j * 2] * x + omega[j * 2 + 1] * y + phase[j]);
    }
    return f;
  };
  const lambda = 1 / (C2 * Math.max(points.length, 1));
  const w = new Float64Array(D);
  const iters = opts.iters ?? 4000;
  const n = points.length;
  if (n === 0) return { score: () => 0.5 };
  const cache = points.map((p) => ({ f: feat(p.x, p.y), y: p.label === 1 ? 1 : -1 }));
  for (let t = 1; t <= iters; t++) {
    const i = Math.floor(rng() * n);
    const { f, y } = cache[i];
    let dot = 0;
    for (let j = 0; j < D; j++) dot += w[j] * f[j];
    const eta = 1 / (lambda * t);
    const shrink = 1 - eta * lambda;
    if (y * dot < 1) {
      for (let j = 0; j < D; j++) w[j] = w[j] * shrink + eta * y * f[j];
    } else {
      for (let j = 0; j < D; j++) w[j] = w[j] * shrink;
    }
  }
  return {
    score: (x, y) => {
      const f = feat(x, y);
      let dot = 0;
      for (let j = 0; j < D; j++) dot += w[j] * f[j];
      return sigmoid(dot * 2.2);
    },
  };
}

// A small fully connected net, 2 -> H -> H -> 1, tanh hidden layers and a sigmoid
// output, trained with real backprop on the binary cross-entropy loss. Inputs
// are standardized so the same learning rate works across datasets.
function trainMLP(points, opts) {
  const H = opts.hidden ?? 8;
  const epochs = opts.epochs ?? 250;
  const lr = opts.lr ?? 0.08;
  const seed = opts.seed ?? 1;
  const rng = mulberry32((seed * 40503 + 17) >>> 0);
  const rand = (s) => (rng() * 2 - 1) * s;

  let mx = 0;
  let my = 0;
  for (const p of points) {
    mx += p.x;
    my += p.y;
  }
  const n = Math.max(points.length, 1);
  mx /= n;
  my /= n;
  let sx = 0;
  let sy = 0;
  for (const p of points) {
    sx += (p.x - mx) * (p.x - mx);
    sy += (p.y - my) * (p.y - my);
  }
  sx = Math.sqrt(sx / n) || 1;
  sy = Math.sqrt(sy / n) || 1;
  const std = (x, y) => [(x - mx) / sx, (y - my) / sy];

  const s1 = Math.sqrt(2 / 2);
  const s2 = Math.sqrt(2 / H);
  const W1 = Array.from({ length: H }, () => [rand(s1), rand(s1)]);
  const b1 = new Float64Array(H);
  const W2 = Array.from({ length: H }, () => Array.from({ length: H }, () => rand(s2)));
  const b2 = new Float64Array(H);
  const W3 = Array.from({ length: H }, () => rand(s2));
  let b3 = 0;

  const data = points.map((p) => {
    const [x, y] = std(p.x, p.y);
    return { x, y, label: p.label };
  });

  const h1 = new Float64Array(H);
  const h2 = new Float64Array(H);
  const losses = [];

  if (data.length > 0) {
    for (let e = 0; e < epochs; e++) {
      let lossSum = 0;
      const gW1 = Array.from({ length: H }, () => [0, 0]);
      const gb1 = new Float64Array(H);
      const gW2 = Array.from({ length: H }, () => new Float64Array(H));
      const gb2 = new Float64Array(H);
      const gW3 = new Float64Array(H);
      let gb3 = 0;

      for (const d of data) {
        for (let i = 0; i < H; i++) {
          h1[i] = Math.tanh(W1[i][0] * d.x + W1[i][1] * d.y + b1[i]);
        }
        for (let i = 0; i < H; i++) {
          let z = b2[i];
          for (let j = 0; j < H; j++) z += W2[i][j] * h1[j];
          h2[i] = Math.tanh(z);
        }
        let zo = b3;
        for (let i = 0; i < H; i++) zo += W3[i] * h2[i];
        const yhat = sigmoid(zo);
        const eps = 1e-9;
        lossSum -= d.label * Math.log(yhat + eps) + (1 - d.label) * Math.log(1 - yhat + eps);

        const dzo = yhat - d.label;
        gb3 += dzo;
        const dh2 = new Float64Array(H);
        for (let i = 0; i < H; i++) {
          gW3[i] += dzo * h2[i];
          dh2[i] = dzo * W3[i];
        }
        const dz2 = new Float64Array(H);
        for (let i = 0; i < H; i++) dz2[i] = dh2[i] * (1 - h2[i] * h2[i]);
        const dh1 = new Float64Array(H);
        for (let i = 0; i < H; i++) {
          gb2[i] += dz2[i];
          for (let j = 0; j < H; j++) {
            gW2[i][j] += dz2[i] * h1[j];
            dh1[j] += dz2[i] * W2[i][j];
          }
        }
        for (let i = 0; i < H; i++) {
          const dz1 = dh1[i] * (1 - h1[i] * h1[i]);
          gb1[i] += dz1;
          gW1[i][0] += dz1 * d.x;
          gW1[i][1] += dz1 * d.y;
        }
      }

      const m = data.length;
      for (let i = 0; i < H; i++) {
        W1[i][0] -= (lr * gW1[i][0]) / m;
        W1[i][1] -= (lr * gW1[i][1]) / m;
        b1[i] -= (lr * gb1[i]) / m;
        for (let j = 0; j < H; j++) W2[i][j] -= (lr * gW2[i][j]) / m;
        b2[i] -= (lr * gb2[i]) / m;
        W3[i] -= (lr * gW3[i]) / m;
      }
      b3 -= (lr * gb3) / m;
      if (e % Math.max(1, Math.floor(epochs / 60)) === 0 || e === epochs - 1) {
        losses.push(lossSum / m);
      }
    }
  }

  return {
    losses,
    score: (x, y) => {
      const [sxx, syy] = std(x, y);
      for (let i = 0; i < H; i++) h1[i] = Math.tanh(W1[i][0] * sxx + W1[i][1] * syy + b1[i]);
      for (let i = 0; i < H; i++) {
        let z = b2[i];
        for (let j = 0; j < H; j++) z += W2[i][j] * h1[j];
        h2[i] = Math.tanh(z);
      }
      let zo = b3;
      for (let i = 0; i < H; i++) zo += W3[i] * h2[i];
      return sigmoid(zo);
    },
  };
}

const MODELS = {
  logistic: {
    label: "Logistic regression",
    bias: "Linear. The boundary is always a single straight line, so it can only solve problems where one class sits cleanly on one side.",
    train: trainLogistic,
  },
  knn: {
    label: "k-Nearest-Neighbors",
    bias: "Local and lazy. It stores every point and votes among the k closest, so the boundary hugs the data. Small k gives a jagged boundary that overfits; larger k smooths it.",
    train: trainKNN,
  },
  svm: {
    label: "RBF kernel SVM",
    bias: "Nonlinear through a kernel. It maps the inputs into a high-dimensional random feature space and finds a max-margin line there, which curves back into a smooth closed boundary. Gamma sets how tightly it curves.",
    train: trainKernelSVM,
  },
  mlp: {
    label: "Neural net (2-H-H-1)",
    bias: "Nonlinear and learned. Stacked tanh layers bend the input space until the classes separate. More hidden units and epochs carve a tighter boundary, eventually contorting around noise.",
    train: trainMLP,
  },
};

function accuracy(model, points) {
  if (points.length === 0) return 0;
  let correct = 0;
  for (const p of points) {
    const pred = model.score(p.x, p.y) >= 0.5 ? 1 : 0;
    if (pred === p.label) correct++;
  }
  return correct / points.length;
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

function Pills({ options, value, onChange, label }) {
  return (
    <div style={{ display: "flex", gap: 6, flexWrap: "wrap", alignItems: "center" }}>
      {label && <span style={{ fontSize: 12, color: C.muted, marginRight: 2 }}>{label}</span>}
      {options.map((o) => {
        const active = o.key === value;
        return (
          <button
            key={o.key}
            onClick={() => onChange(o.key)}
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
              fontWeight: active ? 700 : 400,
              transition: `transform 140ms ${EASE}, background 150ms ease, border-color 150ms ease, color 150ms ease`,
            }}
            onMouseDown={(e) => (e.currentTarget.style.transform = "scale(0.97)")}
            onMouseUp={(e) => (e.currentTarget.style.transform = "scale(1)")}
            onMouseLeave={(e) => (e.currentTarget.style.transform = "scale(1)")}
          >
            {o.label}
          </button>
        );
      })}
    </div>
  );
}

const GRID_RES = 96;

function toCanvas(x, y, size) {
  const px = ((x - DOMAIN.min) / (DOMAIN.max - DOMAIN.min)) * size;
  const py = ((DOMAIN.max - y) / (DOMAIN.max - DOMAIN.min)) * size;
  return [px, py];
}

function fromCanvasNorm(nx, ny) {
  const x = DOMAIN.min + nx * (DOMAIN.max - DOMAIN.min);
  const y = DOMAIN.max - ny * (DOMAIN.max - DOMAIN.min);
  return [x, y];
}

function BoundaryPlot({ model, points, reduced, onAddPoint, activeClass, multi }) {
  const canvasRef = useRef(null);
  const size = 360;

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !model) return;
    canvas.width = GRID_RES;
    canvas.height = GRID_RES;
    const ctx = canvas.getContext("2d");
    const img = ctx.createImageData(GRID_RES, GRID_RES);
    for (let j = 0; j < GRID_RES; j++) {
      for (let i = 0; i < GRID_RES; i++) {
        const [x, y] = fromCanvasNorm((i + 0.5) / GRID_RES, (j + 0.5) / GRID_RES);
        const s = model.score(x, y);
        // s is P(class 1); shade toward the class color, intensity by confidence.
        const conf = Math.abs(s - 0.5) * 2;
        const fill = s >= 0.5 ? C.classBFill : C.classAFill;
        const alpha = 0.14 + conf * 0.42;
        const o = (j * GRID_RES + i) * 4;
        img.data[o] = Math.round(C.paper[0] * (1 - alpha) + fill[0] * alpha);
        img.data[o + 1] = Math.round(C.paper[1] * (1 - alpha) + fill[1] * alpha);
        img.data[o + 2] = Math.round(C.paper[2] * (1 - alpha) + fill[2] * alpha);
        img.data[o + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
  }, [model]);

  const handleClick = (e) => {
    if (!onAddPoint) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const nx = (e.clientX - rect.left) / rect.width;
    const ny = (e.clientY - rect.top) / rect.height;
    const [x, y] = fromCanvasNorm(nx, ny);
    onAddPoint(x, y);
  };

  const r = multi ? 3.1 : 4.2;

  return (
    <div
      onClick={handleClick}
      role="img"
      aria-label={`Decision regions shaded by predicted class with training points overlaid.${
        onAddPoint ? " Click to add a point of the active class." : ""
      }`}
      style={{
        position: "relative",
        width: "100%",
        aspectRatio: "1 / 1",
        borderRadius: 12,
        overflow: "hidden",
        border: `1px solid ${C.border}`,
        cursor: onAddPoint ? "crosshair" : "default",
      }}
    >
      <canvas
        ref={canvasRef}
        aria-hidden="true"
        style={{
          position: "absolute",
          inset: 0,
          width: "100%",
          height: "100%",
          imageRendering: "auto",
        }}
      />
      <svg
        viewBox={`0 0 ${size} ${size}`}
        width="100%"
        height="100%"
        style={{ position: "absolute", inset: 0, display: "block" }}
      >
        <line x1={size / 2} y1={0} x2={size / 2} y2={size} stroke={C.ink} strokeOpacity={0.08} strokeWidth={1} />
        <line x1={0} y1={size / 2} x2={size} y2={size / 2} stroke={C.ink} strokeOpacity={0.08} strokeWidth={1} />
        {points.map((p, i) => {
          const [px, py] = toCanvas(p.x, p.y, size);
          return (
            <circle
              key={i}
              cx={px}
              cy={py}
              r={r}
              fill={p.label === 1 ? C.classB : C.classA}
              stroke="#fff"
              strokeWidth={multi ? 1 : 1.4}
              style={{ transition: reduced ? "none" : `r 160ms ${EASE}` }}
            />
          );
        })}
      </svg>
      {onAddPoint && (
        <div
          style={{
            position: "absolute",
            bottom: 8,
            right: 8,
            fontSize: 11,
            color: C.muted,
            background: "rgba(255,255,255,0.82)",
            borderRadius: 6,
            padding: "3px 7px",
            border: `1px solid ${C.border}`,
            display: "flex",
            alignItems: "center",
            gap: 5,
          }}
        >
          <span
            style={{
              width: 9,
              height: 9,
              borderRadius: "50%",
              background: activeClass === 1 ? C.classB : C.classA,
              display: "inline-block",
            }}
          />
          click adds class {activeClass}
        </div>
      )}
    </div>
  );
}

function LossCurve({ losses, reduced }) {
  const w = 300;
  const h = 70;
  const pad = 4;
  if (!losses || losses.length < 2) return null;
  const max = Math.max(...losses);
  const min = Math.min(...losses);
  const span = max - min || 1;
  const pts = losses.map((v, i) => {
    const px = pad + (i / (losses.length - 1)) * (w - 2 * pad);
    const py = pad + (1 - (v - min) / span) * (h - 2 * pad);
    return `${px.toFixed(1)},${py.toFixed(1)}`;
  });
  return (
    <svg viewBox={`0 0 ${w} ${h}`} width="100%" role="img" aria-label={`Training loss curve, final loss ${losses[losses.length - 1].toFixed(3)}`}>
      <polyline
        points={pts.join(" ")}
        fill="none"
        stroke={C.accent}
        strokeWidth={2}
        strokeLinejoin="round"
        strokeLinecap="round"
        style={{ transition: reduced ? "none" : `all 200ms ${EASE}` }}
      />
    </svg>
  );
}

const DEFAULT_HP = {
  epochs: 300,
  lr: 0.3,
  k: 5,
  gamma: 1.2,
  C: 1.0,
  hidden: 10,
  mlpEpochs: 400,
  mlpLr: 0.15,
};

const HP_DEFS = {
  logistic: [
    { key: "epochs", label: "Epochs", min: 20, max: 800, step: 20, fmt: (v) => String(v), hint: "Gradient descent steps over the full dataset." },
    { key: "lr", label: "Learning rate", min: 0.02, max: 1.0, step: 0.02, fmt: (v) => v.toFixed(2), hint: "Step size for each descent update." },
  ],
  knn: [
    { key: "k", label: "Neighbors k", min: 1, max: 35, step: 1, fmt: (v) => String(v), hint: "Votes among the k closest points. Small k overfits, large k smooths." },
  ],
  svm: [
    { key: "gamma", label: "Gamma", min: 0.1, max: 4.0, step: 0.1, fmt: (v) => v.toFixed(1), hint: "RBF width. High gamma curves tightly around points; low gamma is nearly linear." },
    { key: "C", label: "C (slack)", min: 0.2, max: 8, step: 0.2, fmt: (v) => v.toFixed(1), hint: "Penalty for misclassification. High C fits hard, low C tolerates errors." },
  ],
  mlp: [
    { key: "hidden", label: "Hidden units", min: 2, max: 24, step: 1, fmt: (v) => String(v), hint: "Width of each hidden layer. More units can carve a tighter boundary." },
    { key: "mlpEpochs", label: "Epochs", min: 30, max: 800, step: 10, fmt: (v) => String(v), hint: "Full backprop passes over the data." },
    { key: "mlpLr", label: "Learning rate", min: 0.01, max: 0.4, step: 0.01, fmt: (v) => v.toFixed(2), hint: "Step size for the weight updates." },
  ],
};

function buildOpts(modelKey, hp, seed) {
  if (modelKey === "logistic") return { epochs: hp.epochs, lr: hp.lr };
  if (modelKey === "knn") return { k: hp.k };
  if (modelKey === "svm") return { gamma: hp.gamma, C: hp.C, seed };
  return { hidden: hp.hidden, epochs: hp.mlpEpochs, lr: hp.mlpLr, seed };
}

export default function App() {
  const reduced = useReducedMotion();
  const [datasetKey, setDatasetKey] = useState("moons");
  const [modelKey, setModelKey] = useState("logistic");
  const [seed, setSeed] = useState(7);
  const [noise, setNoise] = useState(0.25);
  const [nPoints, setNPoints] = useState(120);
  const [hp, setHp] = useState(DEFAULT_HP);
  const [activeClass, setActiveClass] = useState(0);
  const [compare, setCompare] = useState(false);
  const [extraPoints, setExtraPoints] = useState([]);

  const basePoints = useMemo(
    () => DATASETS[datasetKey].generate(seed, nPoints, noise),
    [datasetKey, seed, nPoints, noise],
  );
  const points = useMemo(() => basePoints.concat(extraPoints), [basePoints, extraPoints]);

  useEffect(() => {
    setExtraPoints([]);
  }, [datasetKey, seed, noise, nPoints]);

  const model = useMemo(
    () => MODELS[modelKey].train(points, buildOpts(modelKey, hp, seed)),
    [modelKey, points, hp, seed],
  );
  const acc = useMemo(() => accuracy(model, points), [model, points]);

  const compareModels = useMemo(() => {
    if (!compare) return null;
    return Object.keys(MODELS).map((key) => {
      const m = MODELS[key].train(points, buildOpts(key, hp, seed));
      return { key, label: MODELS[key].label, model: m, acc: accuracy(m, points) };
    });
  }, [compare, points, hp, seed]);

  const addPoint = useCallback(
    (x, y) => setExtraPoints((prev) => [...prev, { x, y, label: activeClass }]),
    [activeClass],
  );

  const reseed = () => setSeed((s) => (s * 1103515245 + 12345) % 2147483647);
  const resetHp = () => {
    setHp(DEFAULT_HP);
    setNoise(0.25);
    setExtraPoints([]);
  };

  const setHpVal = (key, val) => setHp((prev) => ({ ...prev, [key]: val }));

  const dataset = DATASETS[datasetKey];
  const modelDef = MODELS[modelKey];

  const focusCss = `
    .dbz-root :focus-visible { outline: 2px solid ${C.accent}; outline-offset: 2px; border-radius: 9px; }
    .dbz-root input[type=range] { accent-color: ${C.accent}; }
    @media (prefers-reduced-motion: reduce) {
      .dbz-root * { transition-duration: 0.001ms !important; animation-duration: 0.001ms !important; }
    }
  `;

  const accLabel = (a) => `${(a * 100).toFixed(1)}%`;
  const accColor = (a) => (a >= 0.9 ? "#2e7d51" : a <= 0.6 ? C.classB : C.ink);

  const sliderRow = (def) => {
    const key = def.key;
    const value = hp[key];
    return (
      <div key={key} style={{ marginBottom: 10 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap" }}>
          <label htmlFor={`hp-${key}`} style={{ fontSize: 13, color: C.muted, minWidth: 110 }}>
            {def.label}
          </label>
          <input
            id={`hp-${key}`}
            type="range"
            min={def.min}
            max={def.max}
            step={def.step}
            value={value}
            onChange={(e) => setHpVal(key, +e.target.value)}
            style={{ flex: 1, minWidth: 120 }}
          />
          <span
            style={{
              fontFamily: "ui-monospace, monospace",
              fontSize: 13,
              fontWeight: 700,
              color: C.accent,
              minWidth: 48,
              textAlign: "right",
            }}
          >
            {def.fmt(value)}
          </span>
        </div>
        {def.hint && (
          <div style={{ fontSize: 11, color: C.muted, margin: "2px 0 0", paddingLeft: 122, lineHeight: 1.4 }}>
            {def.hint}
          </div>
        )}
      </div>
    );
  };

  return (
    <div
      className="dbz-root"
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
          <div style={{ fontSize: 11, letterSpacing: "0.14em", textTransform: "uppercase", color: C.muted, marginBottom: 6 }}>
            Classification, computed live
          </div>
          <h1 style={{ fontSize: 30, fontWeight: 700, margin: 0, lineHeight: 1.1, letterSpacing: "-0.02em", textWrap: "balance" }}>
            Decision Boundary Zoo
          </h1>
          <p style={{ color: C.muted, fontSize: 15, margin: "10px 0 0", lineHeight: 1.6, maxWidth: "62ch" }}>
            A classifier carves the plane into regions, one per class; the seam between them is the decision
            boundary. Pick a dataset and a model, and the boundary you see is traced by classifying a fine grid of
            points after the model trains. Watch where a straight line is enough and where it bends or gives up.
          </p>
        </header>

        <Card style={{ marginBottom: 16, padding: "16px 18px" }}>
          <div style={{ marginBottom: 12 }}>
            <Pills
              label="Dataset:"
              value={datasetKey}
              onChange={setDatasetKey}
              options={Object.entries(DATASETS).map(([k, d]) => ({ key: k, label: d.label }))}
            />
          </div>
          <div style={{ marginBottom: 12 }}>
            <Pills
              label="Model:"
              value={modelKey}
              onChange={setModelKey}
              options={Object.entries(MODELS).map(([k, m]) => ({ key: k, label: m.label }))}
            />
          </div>
          <p style={{ fontSize: 13, lineHeight: 1.55, margin: "0 0 14px", color: C.ink, maxWidth: "64ch" }}>
            {dataset.blurb}
          </p>

          {!compare ? (
            <>
              <BoundaryPlot
                model={model}
                points={points}
                reduced={reduced}
                onAddPoint={addPoint}
                activeClass={activeClass}
              />
              <div
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  alignItems: "center",
                  flexWrap: "wrap",
                  gap: 10,
                  marginTop: 12,
                }}
              >
                <div style={{ display: "flex", gap: 14, flexWrap: "wrap" }}>
                  <span style={{ display: "inline-flex", alignItems: "center", gap: 6, fontSize: 12 }}>
                    <span style={{ width: 11, height: 11, borderRadius: "50%", background: C.classA }} />
                    Class 0
                  </span>
                  <span style={{ display: "inline-flex", alignItems: "center", gap: 6, fontSize: 12 }}>
                    <span style={{ width: 11, height: 11, borderRadius: "50%", background: C.classB }} />
                    Class 1
                  </span>
                </div>
                <div style={{ fontSize: 13, color: C.muted }}>
                  Train accuracy{" "}
                  <b style={{ fontFamily: "ui-monospace, monospace", color: accColor(acc) }}>{accLabel(acc)}</b>{" "}
                  on {points.length} points
                </div>
              </div>
            </>
          ) : (
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))", gap: 10 }}>
              {compareModels.map((cm) => (
                <div key={cm.key}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 6 }}>
                    <span style={{ fontSize: 12, fontWeight: 700 }}>{cm.label}</span>
                    <span
                      style={{
                        fontFamily: "ui-monospace, monospace",
                        fontSize: 12,
                        fontWeight: 700,
                        color: accColor(cm.acc),
                      }}
                    >
                      {accLabel(cm.acc)}
                    </span>
                  </div>
                  <BoundaryPlot model={cm.model} points={points} reduced={reduced} multi />
                </div>
              ))}
            </div>
          )}
        </Card>

        <Card style={{ marginBottom: 16 }}>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 8, alignItems: "center", marginBottom: 14 }}>
            <Btn onClick={() => setCompare((c) => !c)} variant={compare ? "primary" : "ghost"} ariaLabel="Toggle four-model comparison grid">
              {compare ? "Single view" : "Compare all four"}
            </Btn>
            <Btn onClick={reseed} variant="dark" ariaLabel="Resample the dataset with a new seed">
              Reseed data
            </Btn>
            <Btn onClick={resetHp} variant="ghost" ariaLabel="Reset hyperparameters and added points">
              Reset
            </Btn>
            {!compare && (
              <div style={{ marginLeft: "auto" }}>
                <Pills
                  label="Add as:"
                  value={activeClass}
                  onChange={setActiveClass}
                  options={[
                    { key: 0, label: "Class 0" },
                    { key: 1, label: "Class 1" },
                  ]}
                />
              </div>
            )}
          </div>

          <div style={{ marginBottom: 4 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap", marginBottom: 10 }}>
              <label htmlFor="noise" style={{ fontSize: 13, color: C.muted, minWidth: 110 }}>
                Dataset noise
              </label>
              <input
                id="noise"
                type="range"
                min={0}
                max={1}
                step={0.02}
                value={noise}
                onChange={(e) => setNoise(+e.target.value)}
                style={{ flex: 1, minWidth: 120 }}
              />
              <span style={{ fontFamily: "ui-monospace, monospace", fontSize: 13, fontWeight: 700, color: C.accent, minWidth: 48, textAlign: "right" }}>
                {noise.toFixed(2)}
              </span>
            </div>
            <div style={{ display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap", marginBottom: 10 }}>
              <label htmlFor="npts" style={{ fontSize: 13, color: C.muted, minWidth: 110 }}>
                Sample size
              </label>
              <input
                id="npts"
                type="range"
                min={40}
                max={300}
                step={10}
                value={nPoints}
                onChange={(e) => setNPoints(+e.target.value)}
                style={{ flex: 1, minWidth: 120 }}
              />
              <span style={{ fontFamily: "ui-monospace, monospace", fontSize: 13, fontWeight: 700, color: C.accent, minWidth: 48, textAlign: "right" }}>
                {nPoints}
              </span>
            </div>
          </div>

          {!compare && (
            <div style={{ borderTop: `1px solid ${C.border}`, marginTop: 6, paddingTop: 12 }}>
              <div style={{ fontSize: 12, fontWeight: 700, color: C.ink, marginBottom: 10 }}>
                {modelDef.label} hyperparameters
              </div>
              {HP_DEFS[modelKey].map((def) => sliderRow(def))}
              {modelKey === "mlp" && model.losses && model.losses.length > 1 && (
                <div style={{ marginTop: 8 }}>
                  <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 2 }}>
                    <span style={{ fontSize: 12, color: C.muted }}>Training loss (cross-entropy)</span>
                    <span style={{ fontFamily: "ui-monospace, monospace", fontSize: 12, color: C.accent, fontWeight: 700 }}>
                      {model.losses[model.losses.length - 1].toFixed(3)}
                    </span>
                  </div>
                  <LossCurve losses={model.losses} reduced={reduced} />
                </div>
              )}
            </div>
          )}
          {compare && (
            <p style={{ fontSize: 12, color: C.muted, margin: "4px 0 0", lineHeight: 1.5 }}>
              The comparison grid trains all four models on the same points with the current hyperparameters.
              Switch back to single view to adjust per-model settings or add your own points.
            </p>
          )}
        </Card>

        <Card style={{ marginBottom: 16, background: C.accentSoft, border: `1px solid ${C.accent}33` }}>
          <div style={{ fontSize: 13, fontWeight: 700, marginBottom: 6, color: C.accent }}>
            {modelDef.label}: how it draws the line
          </div>
          <p style={{ fontSize: 14, lineHeight: 1.6, margin: 0, color: C.ink, maxWidth: "64ch" }}>
            {modelDef.bias}
          </p>
        </Card>

        <Card>
          <div style={{ fontSize: 13, fontWeight: 700, marginBottom: 8 }}>Linear versus nonlinear separability</div>
          <p style={{ fontSize: 14, lineHeight: 1.6, margin: "0 0 12px", color: C.ink, maxWidth: "64ch" }}>
            A dataset is linearly separable when a single straight line can put each class on its own side. The blobs are;
            the moons, circles, and XOR are not. Logistic regression can only draw that one straight line, so on the
            nonlinear sets it stalls: near chance on circles and XOR, stuck in the high eighties on moons no matter how
            long you train it. The shaded regions make the failure visible. The line simply cannot bend.
          </p>
          <p style={{ fontSize: 14, lineHeight: 1.6, margin: 0, color: C.ink, maxWidth: "64ch" }}>
            The other three buy nonlinearity in different currencies. kNN pays with memory and locality: it keeps every
            point and lets the nearest neighbors vote, so the boundary follows the data and frays into islands when k is
            small. The kernel SVM pays with dimensions: it lifts the two features into a high-dimensional random feature
            space where a straight max-margin cut becomes a smooth curve back in the plane, and gamma sets how tightly
            that curve hugs the points. The neural net pays with learned weights: stacked tanh layers bend the input
            space until the classes come apart, and as you add hidden units and epochs the boundary tightens, eventually
            wrapping around individual noisy points, which is overfitting you can watch happen.
          </p>
        </Card>
      </div>
    </div>
  );
}
