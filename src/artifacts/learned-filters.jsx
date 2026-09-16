import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  createNet,
  createSampleSource,
  trainChunk,
  evaluate,
  filterAt,
  describeFilter,
  convolveImage,
  orientationTuning,
  mulberry32,
  SOBEL_X,
  SOBEL_Y,
} from "./data/learned-filters-core.js";

export const meta = {
  title: "Nobody Wrote These Filters",
  category: "Machine Learning",
  description:
    "When someone says a model learned something, what actually changed? A vision model shows you directly. Its first layer is a set of small grids of numbers that sit in the same space as pixels, so you can print them as pictures. Train one here on real photographs and watch sixteen grids of static turn into edge detectors nobody wrote, then slide one over a photo the model has never seen.",
  date: "2026-09-17",
  tags: ["cnn", "weights", "training", "computer-vision", "representation"],
};

const MONO = "'IBM Plex Mono', ui-monospace, SFMono-Regular, Menlo, monospace";
const SERIF = "Georgia, 'Iowan Old Style', serif";
const EASE = "cubic-bezier(0.23, 1, 0.32, 1)";

const C = {
  bg: "#f7f5f2",
  card: "#ffffff",
  border: "#e4dfd8",
  ink: "#1c1a17",
  muted: "#857c72",
  faint: "#efeae3",
  accent: "#c0561f",
  accentSoft: "#f6ece5",
  blue: "#1e6fa0",
  blueSoft: "#e8f0f5",
  green: "#3f7d52",
  greenSoft: "#e7f0e9",
};

const CLASSES = ["airplane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"];
const SHEET_COLS = 100;
const TILE = 32;
const PER = TILE * TILE;
const N_TRAIN = 8000;
const N_TEST = 1000;
const TRAIN_BUDGET_MS = 11;
const EVAL_PER_FRAME = 20;
const ORIENTED_THRESHOLD = 0.6;
const BASELINE_NETS = 12;
const DATA_ROOT = "/data/learned-filters";

const PHOTOS = [
  { id: "zebra", label: "Zebras", file: "zebra.jpg" },
  { id: "facade", label: "Facade", file: "facade.jpg" },
];

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
    <div style={{ background: C.card, border: `1px solid ${C.border}`, borderRadius: 14, padding: 20, ...style }}>
      {children}
    </div>
  );
}

function SectionTitle({ children }) {
  return (
    <h2 style={{ fontSize: 19, fontWeight: 700, margin: "34px 0 8px", lineHeight: 1.25, textWrap: "balance" }}>
      {children}
    </h2>
  );
}

function Prose({ children, style }) {
  return (
    <p style={{ fontSize: 14, lineHeight: 1.65, color: C.ink, margin: "0 0 14px", maxWidth: "68ch", ...style }}>
      {children}
    </p>
  );
}

function Caption({ children }) {
  return (
    <p style={{ fontSize: 12.5, lineHeight: 1.6, color: C.muted, margin: "12px 0 0", maxWidth: "68ch" }}>{children}</p>
  );
}

function Label({ children }) {
  return (
    <div style={{ fontSize: 10.5, fontFamily: MONO, color: C.muted, letterSpacing: "0.07em", marginBottom: 7 }}>
      {children}
    </div>
  );
}

function StatBox({ label, value, sub, color = C.ink, bg = C.faint }) {
  return (
    <div style={{ background: bg, borderRadius: 10, padding: "10px 14px", flex: "1 1 130px" }}>
      <div
        style={{
          fontSize: 10.5,
          color,
          fontWeight: 700,
          letterSpacing: "0.05em",
          textTransform: "uppercase",
          marginBottom: 4,
        }}
      >
        {label}
      </div>
      <div style={{ fontSize: 18, fontWeight: 700, color, lineHeight: 1.15, fontFamily: MONO }}>{value}</div>
      {sub && <div style={{ fontSize: 10.5, color, opacity: 0.75, marginTop: 3 }}>{sub}</div>}
    </div>
  );
}

function Button({ children, onClick, primary, disabled }) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      className="lf-press"
      style={{
        fontFamily: MONO,
        fontSize: 12.5,
        fontWeight: 600,
        padding: "9px 16px",
        borderRadius: 9,
        cursor: disabled ? "not-allowed" : "pointer",
        border: `1px solid ${primary ? C.accent : C.border}`,
        background: primary ? C.accent : C.card,
        color: primary ? "#fff" : C.ink,
        opacity: disabled ? 0.45 : 1,
      }}
    >
      {children}
    </button>
  );
}

function Chip({ active, onClick, children }) {
  return (
    <button
      type="button"
      onClick={onClick}
      className="lf-press"
      aria-pressed={active}
      style={{
        fontFamily: MONO,
        fontSize: 11.5,
        padding: "6px 12px",
        borderRadius: 999,
        cursor: "pointer",
        border: `1px solid ${active ? C.accent : C.border}`,
        background: active ? C.accentSoft : C.card,
        color: active ? C.accent : C.muted,
        fontWeight: active ? 700 : 500,
      }}
    >
      {children}
    </button>
  );
}

function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => reject(new Error(`could not load ${src}`));
    img.src = src;
  });
}

function readPixels(img, maxWidth) {
  const scale = maxWidth && img.width > maxWidth ? maxWidth / img.width : 1;
  const W = Math.round(img.width * scale);
  const H = Math.round(img.height * scale);
  const canvas = document.createElement("canvas");
  canvas.width = W;
  canvas.height = H;
  const ctx = canvas.getContext("2d", { willReadFrequently: true });
  ctx.drawImage(img, 0, 0, W, H);
  return { W, H, raw: ctx.getImageData(0, 0, W, H).data };
}

// The network is fed one standardised channel of brightness, so the photograph
// is put through exactly the same treatment before any learned filter touches
// it: grey, zero mean, unit standard deviation.
function preparePhoto(img) {
  const { W, H, raw } = readPixels(img, 512);
  const gray = new Float32Array(W * H);
  let mean = 0;
  for (let i = 0; i < W * H; i++) {
    const v = (0.299 * raw[i * 4] + 0.587 * raw[i * 4 + 1] + 0.114 * raw[i * 4 + 2]) / 255;
    gray[i] = v;
    mean += v;
  }
  mean /= W * H;
  let varsum = 0;
  for (let i = 0; i < W * H; i++) varsum += (gray[i] - mean) * (gray[i] - mean);
  const inv = 1 / (Math.sqrt(varsum / (W * H)) + 1e-6);
  for (let i = 0; i < W * H; i++) gray[i] = (gray[i] - mean) * inv;
  return { W, H, raw, gray };
}

function sheetToBytes(img) {
  const { W, raw } = readPixels(img);
  const count = Math.floor(W / TILE) * Math.floor(img.height / TILE);
  const bytes = new Uint8Array(count * PER);
  for (let i = 0; i < count; i++) {
    const row = Math.floor(i / SHEET_COLS) * TILE;
    const col = (i % SHEET_COLS) * TILE;
    for (let y = 0; y < TILE; y++) {
      for (let x = 0; x < TILE; x++) {
        bytes[i * PER + y * TILE + x] = raw[((row + y) * W + col + x) * 4];
      }
    }
  }
  return bytes;
}

function drawFilterTile(canvas, weights, k) {
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const img = ctx.createImageData(k, k);
  let lo = Infinity;
  let hi = -Infinity;
  for (let i = 0; i < weights.length; i++) {
    if (weights[i] < lo) lo = weights[i];
    if (weights[i] > hi) hi = weights[i];
  }
  const span = Math.max(hi - lo, 1e-6);
  for (let i = 0; i < k * k; i++) {
    const v = Math.round(((weights[i] - lo) / span) * 255);
    img.data[i * 4] = v;
    img.data[i * 4 + 1] = v;
    img.data[i * 4 + 2] = v;
    img.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
}

// A handful of extreme pixels would otherwise set the scale and wash the rest
// of the map out, so the ramp saturates at the 98th percentile of magnitude.
function saturationPoint(data, quantile = 0.98, bins = 512) {
  let peak = 1e-6;
  for (let i = 0; i < data.length; i++) peak = Math.max(peak, Math.abs(data[i]));
  const hist = new Int32Array(bins);
  for (let i = 0; i < data.length; i++) {
    hist[Math.min(bins - 1, Math.floor((Math.abs(data[i]) / peak) * bins))] += 1;
  }
  const target = data.length * quantile;
  let seen = 0;
  for (let b = 0; b < bins; b++) {
    seen += hist[b];
    if (seen >= target) return Math.max(((b + 1) / bins) * peak, peak * 0.05);
  }
  return peak;
}

// Signed responses use a diverging ramp: the kernel's positive side in
// terracotta, its negative side in blue, so polarity stays readable instead of
// collapsing into an edge-magnitude image.
function drawResponse(canvas, response) {
  if (!canvas) return;
  const { data, H, W } = response;
  canvas.width = W;
  canvas.height = H;
  const ctx = canvas.getContext("2d");
  const img = ctx.createImageData(W, H);
  const scale = saturationPoint(data);
  const pos = [192, 86, 31];
  const neg = [30, 111, 160];
  const base = [247, 245, 242];
  for (let i = 0; i < data.length; i++) {
    const t = Math.pow(Math.min(Math.abs(data[i]) / scale, 1), 0.65);
    const target = data[i] >= 0 ? pos : neg;
    for (let c = 0; c < 3; c++) img.data[i * 4 + c] = Math.round(base[c] + (target[c] - base[c]) * t);
    img.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
}

function drawTile(canvas, bytes, index) {
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const img = ctx.createImageData(TILE, TILE);
  for (let i = 0; i < PER; i++) {
    const v = bytes[index * PER + i];
    img.data[i * 4] = v;
    img.data[i * 4 + 1] = v;
    img.data[i * 4 + 2] = v;
    img.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
}

function PixelCanvas({ size, onNode, onClick, selected, label, style }) {
  const assign = useCallback((node) => onNode(node), [onNode]);
  const content = (
    <canvas
      ref={assign}
      width={size}
      height={size}
      style={{ width: "100%", display: "block", imageRendering: "pixelated", aspectRatio: "1 / 1" }}
    />
  );
  if (!onClick) {
    return (
      <div style={{ border: `2px solid ${C.border}`, borderRadius: 8, overflow: "hidden", background: C.faint, lineHeight: 0, ...style }}>
        {content}
      </div>
    );
  }
  return (
    <button
      type="button"
      onClick={onClick}
      aria-label={label}
      aria-pressed={selected}
      style={{
        padding: 0,
        border: `2px solid ${selected ? C.accent : C.border}`,
        borderRadius: 8,
        overflow: "hidden",
        cursor: "pointer",
        background: C.faint,
        lineHeight: 0,
        transition: `border-color 160ms ${EASE}`,
        ...style,
      }}
    >
      {content}
    </button>
  );
}

function Sparkline({ series, color, height = 42, floor = 0 }) {
  if (series.length < 2) return <div style={{ height }} />;
  const hi = Math.max(...series, floor + 0.05);
  const lo = Math.min(...series, floor);
  const span = Math.max(hi - lo, 1e-6);
  const points = series
    .map((v, i) => `${((i / (series.length - 1)) * 100).toFixed(2)},${(height - ((v - lo) / span) * height).toFixed(2)}`)
    .join(" ");
  return (
    <svg viewBox={`0 0 100 ${height}`} preserveAspectRatio="none" style={{ width: "100%", height, display: "block" }}>
      <polyline points={points} fill="none" stroke={color} strokeWidth={1.5} vectorEffect="non-scaling-stroke" />
    </svg>
  );
}

// Orientation wraps at 180 degrees, so the first sample is repeated at the end
// and the curve closes on itself instead of appearing to jump.
function TuningChart({ now: rawNow, before: rawBefore }) {
  const now = [...rawNow, rawNow[0]];
  const before = [...rawBefore, rawBefore[0]];
  const W = 320;
  const H = 140;
  const pad = { left: 6, right: 6, top: 8, bottom: 20 };
  const peakNow = Math.max(...now, 1e-6);
  const peakBefore = Math.max(...before, 1e-6);
  const path = (series, peak) =>
    series
      .map((v, i) => {
        const x = pad.left + (i / (series.length - 1)) * (W - pad.left - pad.right);
        const y = H - pad.bottom - (v / peak) * (H - pad.top - pad.bottom);
        return `${x.toFixed(1)},${y.toFixed(1)}`;
      })
      .join(" ");
  const ticks = [0, 6, 12, 18, 24];
  return (
    <div>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: "100%", height: "auto", display: "block" }} role="img"
        aria-label="Orientation tuning of the selected filter, before and after training">
        <line x1={pad.left} y1={H - pad.bottom} x2={W - pad.right} y2={H - pad.bottom} stroke={C.border} strokeWidth={1} />
        {ticks.map((t) => {
          const x = pad.left + (t / (now.length - 1)) * (W - pad.left - pad.right);
          return (
            <text
              key={t}
              x={x}
              y={H - 6}
              fontSize={8}
              fill={C.muted}
              fontFamily={MONO}
              textAnchor={t === 0 ? "start" : t === now.length - 1 ? "end" : "middle"}
            >
              {Math.round((t / (now.length - 1)) * 180)}&deg;
            </text>
          );
        })}
        <polyline points={path(before, peakBefore)} fill="none" stroke={C.muted} strokeWidth={1.4} strokeDasharray="3 3" />
        <polyline points={path(now, peakNow)} fill="none" stroke={C.accent} strokeWidth={2} />
      </svg>
      <div style={{ display: "flex", gap: 14, flexWrap: "wrap", fontFamily: MONO, fontSize: 10, color: C.muted, marginTop: 4 }}>
        <span style={{ color: C.accent }}>solid: now, peak {peakNow.toFixed(2)}</span>
        <span>dashed: at initialisation, peak {peakBefore.toFixed(2)}</span>
      </div>
    </div>
  );
}

function measureRandomBaseline(cfg) {
  const scores = [];
  for (let seed = 1000; seed < 1000 + BASELINE_NETS; seed++) {
    const probe = createNet({ ...cfg, seed });
    for (let f = 0; f < probe.cfg.f1; f++) scores.push(describeFilter(probe, f).orientationScore);
  }
  const mean = scores.reduce((a, b) => a + b, 0) / scores.length;
  return { count: scores.length, mean, over: scores.filter((v) => v > ORIENTED_THRESHOLD).length };
}

export default function App() {
  const reduce = usePrefersReducedMotion();
  const [source, setSource] = useState(null);
  const [tileBytes, setTileBytes] = useState(null);
  const [photos, setPhotos] = useState(null);
  const [loadError, setLoadError] = useState(null);
  const [photoId, setPhotoId] = useState("zebra");
  const [customPhoto, setCustomPhoto] = useState(null);
  const [pinned, setPinned] = useState(null);
  const [running, setRunning] = useState(false);
  const [stats, setStats] = useState({ samples: 0, loss: null, acc: null });
  const [accHistory, setAccHistory] = useState([]);
  const [snapshotId, setSnapshotId] = useState(0);
  const [sobelAxis, setSobelAxis] = useState("x");

  const netRef = useRef(null);
  const trainRef = useRef(null);
  const tileRefs = useRef([]);
  const initRefs = useRef([]);
  const sampleRefs = useRef([]);
  const rafRef = useRef(0);
  const learnedCanvas = useRef(null);
  const sobelCanvas = useRef(null);
  const photoCanvas = useRef(null);

  if (!netRef.current) netRef.current = createNet();
  const net = netRef.current;
  const { k1, f1 } = net.cfg;

  const baseline = useMemo(() => measureRandomBaseline(net.cfg), [net.cfg]);

  useEffect(() => {
    let cancelled = false;
    Promise.all([
      loadImage(`${DATA_ROOT}/cifar-gray.webp`),
      fetch(`${DATA_ROOT}/cifar-labels.txt`).then((r) => r.text()),
      ...PHOTOS.map((p) => loadImage(`${DATA_ROOT}/${p.file}`)),
    ])
      .then(([sheet, labelText, ...photoImgs]) => {
        if (cancelled) return;
        const bytes = sheetToBytes(sheet);
        const text = labelText.trim();
        const labels = new Uint8Array(text.length);
        for (let i = 0; i < labels.length; i++) labels[i] = text.charCodeAt(i) - 48;
        setTileBytes(bytes);
        setSource(createSampleSource(bytes, labels, PER));
        const map = {};
        PHOTOS.forEach((p, i) => {
          map[p.id] = preparePhoto(photoImgs[i]);
        });
        setPhotos(map);
      })
      .catch((err) => {
        if (!cancelled) setLoadError(err.message);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const freshTrainState = useCallback(
    () => ({
      order: Array.from({ length: N_TRAIN }, (_, i) => i),
      cursor: N_TRAIN,
      rand: mulberry32(101),
      evalCursor: N_TRAIN,
      evalCorrect: 0,
      lossEma: null,
    }),
    []
  );

  useEffect(() => {
    if (source) trainRef.current = freshTrainState();
  }, [source, freshTrainState]);

  const drawAllTiles = useCallback(() => {
    for (let i = 0; i < f1; i++) drawFilterTile(tileRefs.current[i], filterAt(net, i), k1);
  }, [net, f1, k1]);

  useEffect(() => {
    drawAllTiles();
    const size = k1 * k1;
    for (let i = 0; i < f1; i++) {
      drawFilterTile(initRefs.current[i], net.initialFilters.subarray(i * size, (i + 1) * size), k1);
    }
  }, [drawAllTiles, net, f1, k1, snapshotId]);

  useEffect(() => {
    if (!tileBytes) return;
    for (let i = 0; i < sampleRefs.current.length; i++) drawTile(sampleRefs.current[i], tileBytes, i);
  }, [tileBytes]);

  useEffect(() => {
    if (!running || !source) return;
    let lastStats = 0;
    let lastSnapshot = performance.now();
    const loop = () => {
      const ts = trainRef.current;
      const start = performance.now();
      let loss = ts.lossEma;
      while (performance.now() - start < TRAIN_BUDGET_MS) {
        const r = trainChunk(net, source, ts.order, ts.cursor, 8, ts.rand);
        ts.cursor = r.cursor;
        loss = loss === null ? r.loss : loss * 0.94 + r.loss * 0.06;
      }
      ts.lossEma = loss;

      const evalEnd = Math.min(ts.evalCursor + EVAL_PER_FRAME, N_TRAIN + N_TEST);
      const chunk = evaluate(net, source, ts.evalCursor, evalEnd);
      ts.evalCorrect += chunk.acc * (evalEnd - ts.evalCursor);
      ts.evalCursor = evalEnd;
      let published = null;
      if (ts.evalCursor >= N_TRAIN + N_TEST) {
        published = ts.evalCorrect / N_TEST;
        ts.evalCursor = N_TRAIN;
        ts.evalCorrect = 0;
      }

      drawAllTiles();
      const now = performance.now();
      if (published !== null || now - lastStats > 180) {
        lastStats = now;
        setStats((prev) => ({
          samples: net.samplesSeen,
          loss,
          acc: published === null ? prev.acc : published,
        }));
      }
      if (published !== null) setAccHistory((h) => [...h.slice(-119), published]);
      // The inspector re-convolves a full-size photograph, so it refreshes on a
      // slow cadence rather than every frame.
      if (now - lastSnapshot > 1400) {
        lastSnapshot = now;
        setSnapshotId((v) => v + 1);
      }
      rafRef.current = requestAnimationFrame(loop);
    };
    rafRef.current = requestAnimationFrame(loop);
    return () => {
      cancelAnimationFrame(rafRef.current);
      setSnapshotId((v) => v + 1);
    };
  }, [running, source, net, drawAllTiles]);

  const reset = useCallback(() => {
    setRunning(false);
    netRef.current = createNet();
    trainRef.current = freshTrainState();
    setStats({ samples: 0, loss: null, acc: null });
    setAccHistory([]);
    setPinned(null);
    setSnapshotId((v) => v + 1);
  }, [freshTrainState]);

  const activePhoto = customPhoto || (photos ? photos[photoId] : null);

  const filterReport = useMemo(() => {
    const items = Array.from({ length: f1 }, (_, i) => describeFilter(net, i));
    let best = 0;
    items.forEach((d, i) => {
      if (d.orientationScore > items[best].orientationScore) best = i;
    });
    return {
      items,
      best,
      mean: items.reduce((sum, d) => sum + d.orientationScore, 0) / items.length,
      over: items.filter((d) => d.orientationScore > ORIENTED_THRESHOLD).length,
    };
  }, [net, f1, snapshotId]);

  // Until the reader picks a tile, the inspector follows whichever filter is
  // currently the most edge-like, so the first thing they see is the clearest
  // example rather than whatever happens to sit in slot one.
  const selected = pinned === null ? filterReport.best : pinned;

  const learnedResponse = useMemo(() => {
    if (!activePhoto) return null;
    return convolveImage(activePhoto.gray, activePhoto.H, activePhoto.W, 1, filterAt(net, selected), k1);
  }, [activePhoto, net, selected, k1, snapshotId]);

  const tuning = useMemo(() => {
    const size = k1 * k1;
    const now = orientationTuning(filterAt(net, selected), k1);
    const before = orientationTuning(net.initialFilters.subarray(selected * size, (selected + 1) * size), k1);
    return { now: Array.from(now), before: Array.from(before) };
  }, [net, selected, k1, snapshotId]);

  const sobelResponse = useMemo(() => {
    if (!activePhoto) return null;
    return convolveImage(activePhoto.gray, activePhoto.H, activePhoto.W, 1, sobelAxis === "x" ? SOBEL_X : SOBEL_Y, 3);
  }, [activePhoto, sobelAxis]);

  const selectedWeights = useMemo(
    () => Array.from(filterAt(net, selected)),
    [net, selected, snapshotId]
  );

  useEffect(() => {
    if (learnedResponse) drawResponse(learnedCanvas.current, learnedResponse);
  }, [learnedResponse]);

  useEffect(() => {
    if (sobelResponse) drawResponse(sobelCanvas.current, sobelResponse);
  }, [sobelResponse]);

  useEffect(() => {
    const canvas = photoCanvas.current;
    if (!canvas || !activePhoto) return;
    canvas.width = activePhoto.W;
    canvas.height = activePhoto.H;
    const img = canvas.getContext("2d").createImageData(activePhoto.W, activePhoto.H);
    img.data.set(activePhoto.raw);
    canvas.getContext("2d").putImageData(img, 0, 0);
  }, [activePhoto]);

  const onUpload = useCallback((event) => {
    const file = event.target.files && event.target.files[0];
    if (!file) return;
    const url = URL.createObjectURL(file);
    loadImage(url)
      .then((img) => {
        setCustomPhoto(preparePhoto(img));
        URL.revokeObjectURL(url);
      })
      .catch(() => URL.revokeObjectURL(url));
  }, []);

  const pickPhoto = useCallback((id) => {
    setCustomPhoto(null);
    setPhotoId(id);
  }, []);

  const passes = stats.samples / N_TRAIN;
  const selectedReport = filterReport.items[selected];
  const trainingNumbers = N_TRAIN * PER;

  const css = `
    .lf-root :focus-visible { outline: 2px solid ${C.accent}; outline-offset: 2px; border-radius: 6px; }
    .lf-press { transition: transform 140ms ${EASE}; }
    .lf-press:active { transform: scale(0.97); }
    .lf-grid { display: grid; grid-template-columns: repeat(8, 1fr); gap: 7px; }
    .lf-samples { display: grid; grid-template-columns: repeat(10, 1fr); gap: 6px; }
    .lf-cols { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
    @media (max-width: 560px) {
      .lf-grid { grid-template-columns: repeat(4, 1fr); }
      .lf-samples { grid-template-columns: repeat(5, 1fr); }
      .lf-cols { grid-template-columns: 1fr; }
    }
    @media (prefers-reduced-motion: reduce) { .lf-root * { transition-duration: 0.001ms !important; } }
  `;

  return (
    <div
      className="lf-root"
      style={{ fontFamily: SERIF, background: C.bg, minHeight: "100vh", padding: "26px 14px 56px", color: C.ink }}
    >
      <style>{css}</style>
      <div style={{ maxWidth: 720, margin: "0 auto" }}>
        <div
          style={{
            fontSize: 10,
            letterSpacing: "0.14em",
            textTransform: "uppercase",
            color: C.muted,
            marginBottom: 6,
            fontFamily: MONO,
          }}
        >
          Machine Learning · Representation
        </div>
        <h1 style={{ fontSize: 25, fontWeight: 700, margin: 0, lineHeight: 1.2, textWrap: "balance" }}>
          Nobody Wrote These Filters
        </h1>

        <Prose style={{ marginTop: 14, fontSize: 15 }}>
          In 1968 Irwin Sobel and Gary Feldman picked nine numbers by hand, laid them out in a three by three grid,
          and that grid could find the edges in any photograph. It was good engineering and it held up for decades.
          Train a network on photos today and something very like it turns up in the first layer on its own, except
          nobody picked the numbers and nobody ever mentioned edges.
        </Prose>
        <Prose>
          That is the part people wave their hands at when they say a model learned something. Here you can watch it
          instead. The network below has {net.paramCount.toLocaleString()} weights, every one of them a floating point
          number, every one currently random. Sixteen of its grids are drawn on this page, and right now they are
          static.
        </Prose>

        {loadError && (
          <Card style={{ marginTop: 16, borderColor: C.accent }}>
            <Prose style={{ margin: 0 }}>The photographs could not load ({loadError}).</Prose>
          </Card>
        )}

        <SectionTitle>A filter is a small grid of numbers</SectionTitle>
        <Prose>
          A convolution slides a small grid across an image. At each position it multiplies every grid number by the
          pixel underneath and adds up the results, and that single sum becomes one pixel of the output. Sobel&apos;s
          grid has a negative column down one side and a positive column down the other, so it cancels to nothing on
          flat areas and spikes wherever brightness changes. That is the whole of an edge detector.
        </Prose>
        <Card>
          <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 14 }}>
            {PHOTOS.map((p) => (
              <Chip key={p.id} active={!customPhoto && photoId === p.id} onClick={() => pickPhoto(p.id)}>
                {p.label}
              </Chip>
            ))}
            <Chip active={sobelAxis === "x"} onClick={() => setSobelAxis("x")}>
              Vertical edges
            </Chip>
            <Chip active={sobelAxis === "y"} onClick={() => setSobelAxis("y")}>
              Horizontal edges
            </Chip>
          </div>
          <div className="lf-cols">
            <div>
              <Label>PHOTOGRAPH</Label>
              <canvas ref={photoCanvas} style={{ width: "100%", display: "block", borderRadius: 8 }} />
            </div>
            <div>
              <Label>SOBEL RESPONSE</Label>
              <canvas ref={sobelCanvas} style={{ width: "100%", display: "block", borderRadius: 8 }} />
            </div>
          </div>
          <Caption>
            Terracotta where the sum comes out positive, blue where it comes out negative, pale where the grid found
            nothing. The nine hand-picked numbers are{" "}
            <code style={{ fontFamily: MONO }}>{Array.from(sobelAxis === "x" ? SOBEL_X : SOBEL_Y).join(" ")}</code>. For
            a slower walk through the multiply and add, see <a href="/a/convolution-kernels">Convolution: Kernels on an Image</a>.
          </Caption>
        </Card>

        <SectionTitle>Train it and watch the static resolve</SectionTitle>
        <Prose>
          The network gets {N_TRAIN.toLocaleString()} photographs at 32 by 32 pixels, each labelled with one of ten
          things: cat, ship, truck, frog and so on. It sees brightness only, no colour, and each photo is
          standardised so that overall lightness carries no information, which leaves nothing but structure to work
          with. Its only instruction is to name the thing in the photo. Nothing in the loss function mentions edges,
          or orientation, or the word filter.
        </Prose>
        <Card>
          <Label>A FEW OF THE PHOTOGRAPHS IT TRAINS ON</Label>
          <div className="lf-samples" style={{ marginBottom: 18 }}>
            {Array.from({ length: 10 }, (_, i) => (
              <div key={i}>
                <PixelCanvas
                  size={TILE}
                  onNode={(node) => {
                    sampleRefs.current[i] = node;
                  }}
                />
                <div style={{ fontFamily: MONO, fontSize: 9, color: C.muted, marginTop: 4, textAlign: "center" }}>
                  {source ? CLASSES[source.labels[i]] : ""}
                </div>
              </div>
            ))}
          </div>

          <div style={{ display: "flex", gap: 10, flexWrap: "wrap", alignItems: "center", marginBottom: 16 }}>
            <Button primary onClick={() => setRunning((r) => !r)} disabled={!source}>
              {running ? "Pause" : stats.samples ? "Keep training" : "Train"}
            </Button>
            <Button onClick={reset} disabled={!source || !stats.samples}>
              Back to random
            </Button>
            {!source && !loadError && (
              <span style={{ fontSize: 12, color: C.muted, fontFamily: MONO }}>loading photographs…</span>
            )}
          </div>

          <Label>THE FIRST LAYER, NOW</Label>
          <div className="lf-grid">
            {Array.from({ length: f1 }, (_, i) => (
              <PixelCanvas
                key={i}
                size={k1}
                selected={selected === i}
                onClick={() => setPinned(i)}
                label={`Select filter ${i + 1}`}
                onNode={(node) => {
                  tileRefs.current[i] = node;
                }}
              />
            ))}
          </div>

          <div style={{ marginTop: 18 }}>
            <Label>THE SAME SIXTEEN AT INITIALISATION</Label>
            <div className="lf-grid" style={{ opacity: 0.85 }}>
              {Array.from({ length: f1 }, (_, i) => (
                <PixelCanvas
                  key={i}
                  size={k1}
                  onNode={(node) => {
                    initRefs.current[i] = node;
                  }}
                />
              ))}
            </div>
          </div>

          <div style={{ display: "flex", gap: 10, flexWrap: "wrap", marginTop: 18 }}>
            <StatBox
              label="Photos seen"
              value={stats.samples.toLocaleString()}
              sub={`${passes.toFixed(1)} passes over the set`}
            />
            <StatBox
              label="Held-out accuracy"
              value={stats.acc === null ? "—" : `${(stats.acc * 100).toFixed(1)}%`}
              sub={`${N_TEST.toLocaleString()} unseen photos, chance is 10%`}
              bg={stats.acc && stats.acc > 0.25 ? C.greenSoft : C.faint}
              color={stats.acc && stats.acc > 0.25 ? C.green : C.ink}
            />
            <StatBox
              label="Filters that look like edges"
              value={`${filterReport.over} of ${f1}`}
              sub={`${baseline.over} of ${baseline.count} random ones did`}
              bg={filterReport.over > baseline.over ? C.blueSoft : C.faint}
              color={filterReport.over > baseline.over ? C.blue : C.ink}
            />
          </div>

          {accHistory.length > 1 && (
            <div style={{ marginTop: 16 }}>
              <Label>HELD-OUT ACCURACY OVER TRAINING</Label>
              <Sparkline series={accHistory} color={C.green} />
            </div>
          )}

          <Caption>
            Every number here is computed in this tab: the forward pass, the gradients, the Adam updates, and the
            accuracy on {N_TEST.toLocaleString()} photographs the network never trains on. It settles somewhere
            around forty percent, which is four times chance and nowhere near a real vision model. The filters are
            the point, not the score.
          </Caption>
        </Card>

        <SectionTitle>Look at one of them closely</SectionTitle>
        <Prose>
          This starts on whichever filter currently matches an edge most closely, and clicking any tile above pins it
          here instead. You get its actual forty nine numbers, the angle it answers to, and what happens when it slides
          over a photograph it has never been anywhere near. The match score is a correlation against an ideal edge, so
          it is a measurement rather than an impression, and out of {baseline.count} freshly initialised filters,{" "}
          {baseline.over === 0 ? "not one" : `only ${baseline.over}`} cleared {ORIENTED_THRESHOLD}.
        </Prose>
        <Card>
          <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 14, alignItems: "center" }}>
            {PHOTOS.map((p) => (
              <Chip key={p.id} active={!customPhoto && photoId === p.id} onClick={() => pickPhoto(p.id)}>
                {p.label}
              </Chip>
            ))}
            <label
              className="lf-press"
              style={{
                fontFamily: MONO,
                fontSize: 11.5,
                padding: "6px 12px",
                borderRadius: 999,
                cursor: "pointer",
                border: `1px solid ${customPhoto ? C.accent : C.border}`,
                background: customPhoto ? C.accentSoft : C.card,
                color: customPhoto ? C.accent : C.muted,
              }}
            >
              Your own photo
              <input type="file" accept="image/*" onChange={onUpload} style={{ display: "none" }} />
            </label>
          </div>

          <div className="lf-cols">
            <div>
              <Label>FILTER {selected + 1}, ALL {k1 * k1} WEIGHTS</Label>
              <div style={{ fontFamily: MONO, fontSize: 9.5, color: C.ink, lineHeight: 1.55, whiteSpace: "pre" }}>
                {Array.from({ length: k1 }, (_, y) =>
                  selectedWeights
                    .slice(y * k1, (y + 1) * k1)
                    .map((v) => (v < 0 ? "" : " ") + v.toFixed(2))
                    .join(" ")
                ).join("\n")}
              </div>
              <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginTop: 12 }}>
                <StatBox
                  label="Best matching edge"
                  value={`${Math.round((selectedReport.angle * 180) / Math.PI)}°`}
                  sub={`match ${selectedReport.orientationScore.toFixed(2)}, random averages ${baseline.mean.toFixed(2)}`}
                  bg={selectedReport.orientationScore > ORIENTED_THRESHOLD ? C.blueSoft : C.faint}
                  color={selectedReport.orientationScore > ORIENTED_THRESHOLD ? C.blue : C.ink}
                />
              </div>
            </div>
            <div>
              <Label>WHERE IT FIRES IN THE PHOTOGRAPH</Label>
              <canvas ref={learnedCanvas} style={{ width: "100%", display: "block", borderRadius: 8 }} />
            </div>
          </div>
          <div style={{ marginTop: 18 }}>
            <Label>HOW HARD IT ANSWERS TO STRIPES AT EACH ANGLE</Label>
            <TuningChart now={tuning.now} before={tuning.before} />
          </div>
          <Caption>
            This is the measurement a visual neuroscientist would run on a cell: show the filter a striped patch at
            every angle in turn and record how hard it answers. A trained filter usually gives one hump, which means
            it has picked an orientation and largely ignores the others. The dashed line is the same filter before
            training, when it answered weakly to everything and strongly to nothing. Each line is drawn against its
            own peak, and the peaks are printed underneath, because the trained filter also answers several times
            harder.
          </Caption>
        </Card>

        <SectionTitle>So what does it mean that a model learned something</SectionTitle>
        <Prose>
          The whole network holds {net.paramCount.toLocaleString()} numbers and the photographs it trains on are{" "}
          {trainingNumbers.toLocaleString()} numbers. It could not memorise them if it tried, and that shortfall is
          the mechanism. The only way to push the loss down with that little room is to find what the photographs
          have in common, and edges at various angles are the first thing worth finding. Scale by itself does not
          create that pressure. The gap between what a model would have to remember and what it has room for does.
        </Prose>
        <Prose>
          This is also the honest answer for grammar, or arithmetic, or anything else a language model appears to
          know. No rule was written down anywhere. A pile of numbers was nudged, one gradient step at a time, until it
          encoded something general enough to work on inputs nobody had shown it. Vision just hands you the receipt as
          a picture, because a first-layer filter lives in pixel space and can be printed. A transformer&apos;s weight
          matrix maps one abstract vector space to another, so printing it raw gives you static, and finding what it
          learned takes other instruments: the geometry of an <a href="/a/word-embeddings">embedding table</a>, the
          pattern an <a href="/a/attention-explainer">attention head</a> settles into, or the plain fact that a working
          model is <a href="/a/inside-an-llm">a file of numbers you can corrupt</a>.
        </Prose>

        <footer style={{ marginTop: 30, paddingTop: 16, borderTop: `1px solid ${C.border}` }}>
          <p style={{ fontSize: 12.5, color: C.muted, lineHeight: 1.7, margin: 0, maxWidth: "68ch" }}>
            The photographs are {(N_TRAIN + N_TEST).toLocaleString()} images from CIFAR-10 (Krizhevsky, 2009),
            converted to grey, 800 per class for training and 100 per class held back. The zebras and the facade are
            CC0 photographs from Wikimedia Commons. The network is two convolutional layers and a linear head trained
            with Adam on cross entropy, and its first layer starts at a smaller scale than the usual initialisation so
            the structure outgrows the random draw inside the seconds a browser tab can spare. From a standard
            initialisation the same oriented filters appear, several times slower.
          </p>
        </footer>
      </div>
    </div>
  );
}
