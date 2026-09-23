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
  const lum = new Float32Array(W * H);
  let mean = 0;
  for (let i = 0; i < W * H; i++) {
    const l = 0.299 * raw[i * 4] + 0.587 * raw[i * 4 + 1] + 0.114 * raw[i * 4 + 2];
    lum[i] = l;
    const v = l / 255;
    gray[i] = v;
    mean += v;
  }
  mean /= W * H;
  let varsum = 0;
  for (let i = 0; i < W * H; i++) varsum += (gray[i] - mean) * (gray[i] - mean);
  const inv = 1 / (Math.sqrt(varsum / (W * H)) + 1e-6);
  for (let i = 0; i < W * H; i++) gray[i] = (gray[i] - mean) * inv;
  return { W, H, raw, gray, lum };
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

// Zero maps to mid grey and each tile is scaled by its own largest weight, so
// that dark reads as negative and light as positive in every tile, the
// hand-written Sobel grid included.
function weightShade(weight, peak) {
  return Math.round((0.5 + (0.5 * weight) / peak) * 255);
}

function drawFilterTile(canvas, weights, k) {
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const img = ctx.createImageData(k, k);
  let peak = 1e-6;
  for (let i = 0; i < k * k; i++) peak = Math.max(peak, Math.abs(weights[i]));
  for (let i = 0; i < k * k; i++) {
    const v = weightShade(weights[i], peak);
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
const RAMP_POS = [192, 86, 31];
const RAMP_NEG = [30, 111, 160];
const RAMP_BASE = [247, 245, 242];

function rampColor(value, scale) {
  const t = Math.pow(Math.min(Math.abs(value) / scale, 1), 0.65);
  const target = value >= 0 ? RAMP_POS : RAMP_NEG;
  return RAMP_BASE.map((b, c) => Math.round(b + (target[c] - b) * t));
}

function drawResponse(canvas, response, scale) {
  if (!canvas) return;
  const { data, H, W } = response;
  canvas.width = W;
  canvas.height = H;
  const ctx = canvas.getContext("2d");
  const img = ctx.createImageData(W, H);
  for (let i = 0; i < data.length; i++) {
    const rgb = rampColor(data[i], scale);
    for (let c = 0; c < 3; c++) img.data[i * 4 + c] = rgb[c];
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

// The tick runs along the edge the filter looks for. The angle is that of the
// stripe pattern the filter answers to, and the edge line sits at right angles
// to it: angle 0 varies left to right, which is a vertical edge.
function EdgeTick({ angle }) {
  const dx = -Math.sin(angle) * 34;
  const dy = Math.cos(angle) * 34;
  return (
    <svg
      viewBox="0 0 100 100"
      aria-hidden="true"
      style={{ position: "absolute", inset: 0, width: "100%", height: "100%", pointerEvents: "none" }}
    >
      <line x1={50 - dx} y1={50 - dy} x2={50 + dx} y2={50 + dy} stroke="#fff" strokeWidth={10} strokeLinecap="round" />
      <line x1={50 - dx} y1={50 - dy} x2={50 + dx} y2={50 + dy} stroke={C.blue} strokeWidth={5.5} strokeLinecap="round" />
    </svg>
  );
}

function edgeWords(angle) {
  const deg = ((angle * 180) / Math.PI) % 180;
  if (deg < 22.5 || deg >= 157.5) return "vertical";
  if (deg >= 67.5 && deg < 112.5) return "horizontal";
  return "diagonal";
}

function PixelCanvas({ size, onNode, onClick, selected, label, style, tickAngle }) {
  const assign = useCallback((node) => onNode(node), [onNode]);
  const content = (
    <>
      <canvas
        ref={assign}
        width={size}
        height={size}
        style={{ width: "100%", display: "block", imageRendering: "pixelated", aspectRatio: "1 / 1" }}
      />
      {tickAngle !== undefined && <EdgeTick angle={tickAngle} />}
    </>
  );
  const frame = {
    position: "relative",
    border: `2px solid ${selected ? C.accent : C.border}`,
    borderRadius: 8,
    overflow: "hidden",
    background: C.faint,
    lineHeight: 0,
    transition: `border-color 160ms ${EASE}`,
    ...style,
  };
  if (!onClick) return <div style={frame}>{content}</div>;
  return (
    <button
      type="button"
      onClick={onClick}
      aria-label={label}
      aria-pressed={selected}
      style={{ ...frame, padding: 0, cursor: "pointer" }}
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

const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));

function PhotoCanvas({ photo }) {
  const ref = useRef(null);
  useEffect(() => {
    const canvas = ref.current;
    if (!canvas || !photo) return;
    canvas.width = photo.W;
    canvas.height = photo.H;
    const ctx = canvas.getContext("2d");
    const img = ctx.createImageData(photo.W, photo.H);
    img.data.set(photo.raw);
    ctx.putImageData(img, 0, 0);
  }, [photo]);
  return <canvas ref={ref} style={{ width: "100%", display: "block", borderRadius: 8 }} />;
}

function ResponseCanvas({ response, scale }) {
  const ref = useRef(null);
  useEffect(() => {
    if (response) drawResponse(ref.current, response, scale);
  }, [response, scale]);
  return <canvas ref={ref} style={{ width: "100%", display: "block", borderRadius: 8 }} />;
}

function PhotoPicker({ photoId, uploaded, onPick, onUpload }) {
  return (
    <>
      {PHOTOS.map((p) => (
        <Chip key={p.id} active={!uploaded && photoId === p.id} onClick={() => onPick(p.id)}>
          {p.label}
        </Chip>
      ))}
      {onUpload && (
        <label
          className="lf-press"
          style={{
            fontFamily: MONO,
            fontSize: 11.5,
            padding: "6px 12px",
            borderRadius: 999,
            cursor: "pointer",
            border: `1px solid ${uploaded ? C.accent : C.border}`,
            background: uploaded ? C.accentSoft : C.card,
            color: uploaded ? C.accent : C.muted,
          }}
        >
          Your own photo
          <input type="file" accept="image/*" onChange={onUpload} style={{ display: "none" }} />
        </label>
      )}
    </>
  );
}

function ProbeMarker({ x, y }) {
  return (
    <div
      aria-hidden="true"
      style={{
        position: "absolute",
        left: `${x * 100}%`,
        top: `${y * 100}%`,
        width: 14,
        height: 14,
        transform: "translate(-50%, -50%)",
        border: "2px solid #fff",
        boxShadow: `0 0 0 1.5px ${C.ink}`,
        borderRadius: 3,
        pointerEvents: "none",
      }}
    />
  );
}

function ShadedGrid({ values, shade, format }) {
  return (
    <div style={{ display: "grid", gridTemplateColumns: "repeat(3, var(--lf-cell))", gap: 2 }}>
      {values.map((v, i) => {
        const g = shade(v);
        return (
          <div
            key={i}
            style={{
              height: "var(--lf-cell)",
              borderRadius: 3,
              background: `rgb(${g},${g},${g})`,
              color: g > 120 ? C.ink : "#fff",
              fontFamily: MONO,
              fontSize: 9.5,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
            }}
          >
            {format(v)}
          </div>
        );
      })}
    </div>
  );
}

function SubLabel({ children }) {
  return <div style={{ fontFamily: MONO, fontSize: 9, color: C.muted, letterSpacing: "0.06em", marginBottom: 5 }}>{children}</div>;
}

function Operator({ children }) {
  return <div style={{ fontFamily: MONO, fontSize: 18, color: C.muted }}>{children}</div>;
}

function windowMeaning(axis, sum, flat) {
  if (flat) return "about the same brightness all round, so the sum comes out near zero and nothing shows";
  if (axis === "x") return sum > 0 ? "brighter on the right than on the left, so the sum is positive" : "brighter on the left than on the right, so the sum is negative";
  return sum > 0 ? "brighter below than above, so the sum is positive" : "brighter above than below, so the sum is negative";
}

function SobelDemo({ photo, axis }) {
  const kernel = axis === "x" ? SOBEL_X : SOBEL_Y;
  const [probe, setProbe] = useState(null);

  useEffect(() => {
    setProbe(null);
  }, [photo]);

  const response = useMemo(
    () => (photo ? convolveImage(photo.gray, photo.H, photo.W, 1, kernel, 3) : null),
    [photo, kernel]
  );
  const scale = useMemo(() => (response ? saturationPoint(response.data) : 1), [response]);
  const home = useMemo(() => {
    if (!response) return null;
    let best = 0;
    for (let i = 1; i < response.data.length; i++) {
      if (Math.abs(response.data[i]) > Math.abs(response.data[best])) best = i;
    }
    return { x: best % response.W, y: Math.floor(best / response.W) };
  }, [response]);

  if (!photo || !response) return <Caption>Loading photographs…</Caption>;

  const at = probe || home;
  const place = (event) => {
    const rect = event.currentTarget.getBoundingClientRect();
    setProbe({
      x: clamp(Math.round(((event.clientX - rect.left) / rect.width) * photo.W) - 1, 0, response.W - 1),
      y: clamp(Math.round(((event.clientY - rect.top) / rect.height) * photo.H) - 1, 0, response.H - 1),
    });
  };
  const onKey = (event) => {
    const step = event.shiftKey ? 24 : 6;
    const move = { ArrowLeft: [-step, 0], ArrowRight: [step, 0], ArrowUp: [0, -step], ArrowDown: [0, step] }[event.key];
    if (!move) return;
    event.preventDefault();
    setProbe({ x: clamp(at.x + move[0], 0, response.W - 1), y: clamp(at.y + move[1], 0, response.H - 1) });
  };

  const patch = [];
  for (let ky = 0; ky < 3; ky++) {
    for (let kx = 0; kx < 3; kx++) patch.push(photo.lum[(at.y + ky) * photo.W + at.x + kx]);
  }
  const sum = patch.reduce((total, v, i) => total + v * kernel[i], 0);
  const standardised = response.data[at.y * response.W + at.x];
  const flat = Math.abs(standardised) < scale * 0.08;
  const [r, g, b] = rampColor(standardised, scale);
  const pointerProps = { onPointerMove: place, onPointerDown: place };
  const surface = { position: "relative", touchAction: "pan-y", cursor: "crosshair", lineHeight: 0 };

  return (
    <div>
      <div className="lf-cols">
        <div>
          <Label>PHOTOGRAPH, MOVE OVER IT</Label>
          <div
            {...pointerProps}
            onKeyDown={onKey}
            tabIndex={0}
            role="group"
            aria-label="Photograph. Move the pointer, tap, or use the arrow keys to move the window."
            style={surface}
          >
            <PhotoCanvas photo={photo} />
            <ProbeMarker x={(at.x + 1.5) / photo.W} y={(at.y + 1.5) / photo.H} />
          </div>
        </div>
        <div>
          <Label>SOBEL RESPONSE</Label>
          <div {...pointerProps} style={surface}>
            <ResponseCanvas response={response} scale={scale} />
            <ProbeMarker x={(at.x + 0.5) / response.W} y={(at.y + 0.5) / response.H} />
          </div>
        </div>
      </div>

      <div style={{ background: C.faint, borderRadius: 10, padding: 14, marginTop: 14 }}>
        <Label>THE 3 BY 3 WINDOW UNDER THE MARKER</Label>
        <div style={{ display: "flex", flexWrap: "wrap", alignItems: "center", gap: 8 }}>
          <div>
            <SubLabel>PIXELS, 0 DARK TO 255 WHITE</SubLabel>
            <ShadedGrid values={patch} shade={(v) => Math.round(v)} format={(v) => Math.round(v)} />
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <Operator>×</Operator>
            <div>
              <SubLabel>SOBEL NUMBERS</SubLabel>
              <ShadedGrid values={Array.from(kernel)} shade={(v) => weightShade(v, 2)} format={(v) => v} />
            </div>
          </div>
          <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <Operator>=</Operator>
            <div>
              <SubLabel>ADD IT ALL UP</SubLabel>
              <div style={{ display: "flex", alignItems: "center", gap: 8, height: "calc(3 * var(--lf-cell) + 4px)" }}>
                <div
                  style={{
                    width: "var(--lf-cell)",
                    height: "var(--lf-cell)",
                    borderRadius: 3,
                    background: `rgb(${r},${g},${b})`,
                    border: `1px solid ${C.border}`,
                  }}
                />
                <span style={{ fontFamily: MONO, fontSize: 14, fontWeight: 700 }}>
                  {sum > 0 ? "+" : ""}
                  {Math.round(sum)}
                </span>
              </div>
            </div>
          </div>
        </div>
        <Caption>
          Here the window is {windowMeaning(axis, sum, flat)}. The swatch is that one output pixel, and the marker on
          the response is the same spot. The Sobel numbers are shaded by the rule every filter tile below uses: dark
          for negative, light for positive, grey for zero.
        </Caption>
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

  const learnedScale = useMemo(
    () => (learnedResponse ? saturationPoint(learnedResponse.data) : 1),
    [learnedResponse]
  );

  const tuning = useMemo(() => {
    const size = k1 * k1;
    const now = orientationTuning(filterAt(net, selected), k1);
    const before = orientationTuning(net.initialFilters.subarray(selected * size, (selected + 1) * size), k1);
    return { now: Array.from(now), before: Array.from(before) };
  }, [net, selected, k1, snapshotId]);

  const selectedWeights = useMemo(
    () => Array.from(filterAt(net, selected)),
    [net, selected, snapshotId]
  );

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
  const selectedIsEdge = selectedReport.orientationScore > ORIENTED_THRESHOLD;
  const selectedWords = selectedIsEdge ? `${edgeWords(selectedReport.angle)} edge` : "no clear edge";
  const trainingNumbers = N_TRAIN * PER;

  const css = `
    .lf-root { --lf-cell: 30px; }
    .lf-root :focus-visible { outline: 2px solid ${C.accent}; outline-offset: 2px; border-radius: 6px; }
    .lf-press { transition: transform 140ms ${EASE}; }
    .lf-press:active { transform: scale(0.97); }
    .lf-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 7px; }
    .lf-samples { display: grid; grid-template-columns: repeat(10, 1fr); gap: 6px; }
    .lf-cols { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; }
    @media (max-width: 560px) {
      .lf-root { --lf-cell: 20px; }
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
          When people say a model learned something, what changed inside it? For most models the answer is a shrug,
          but a vision model&apos;s first layer is small enough to look at. It is a set of tiny grids of numbers, and
          a grid of numbers can be drawn as a picture. Right now the sixteen below are static. Train the network on
          real photographs and watch them turn into edge detectors that nobody wrote.
        </Prose>
        <Prose>
          The network is a pile of adjustable numbers called weights, {net.paramCount.toLocaleString()} of them,
          all currently random. Training shows it a photo, checks whether its guess at what the photo contains was
          right, and nudges every weight a tiny step toward a better guess. The first layer is the first thing a photo
          passes through, and its weights are the grids you are about to see.
        </Prose>

        {loadError && (
          <Card style={{ marginTop: 16, borderColor: C.accent }}>
            <Prose style={{ margin: 0 }}>The photographs could not load ({loadError}).</Prose>
          </Card>
        )}

        <SectionTitle>A filter is a small grid of numbers</SectionTitle>
        <Prose>
          Edge detection has existed since 1968, when Irwin Sobel and Gary Feldman picked nine numbers by hand and
          arranged them in a three by three grid. A grid like that is called a filter. To use it you slide it across
          the photograph, and at each position you multiply every grid number by the pixel underneath it and add the
          results into a single number, which becomes one pixel of the output. It is a weighted sum over a small patch,
          run at every position. Sobel&apos;s grid has a negative column down one side and a positive column down the
          other, so on flat areas the sum cancels to nothing and wherever brightness changes it spikes. Move over the
          photograph to see one position at a time.
        </Prose>
        <Card>
          <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 14 }}>
            <PhotoPicker photoId={photoId} uploaded={!!customPhoto} onPick={pickPhoto} />
            <Chip active={sobelAxis === "x"} onClick={() => setSobelAxis("x")}>
              Vertical edges
            </Chip>
            <Chip active={sobelAxis === "y"} onClick={() => setSobelAxis("y")}>
              Horizontal edges
            </Chip>
          </div>
          <SobelDemo photo={activePhoto} axis={sobelAxis} />
          <Caption>
            Terracotta where the sum comes out positive, blue where it comes out negative, pale where the grid found
            nothing. For a slower walk through the multiply and add, see{" "}
            <a href="/a/convolution-kernels">Convolution: Kernels on an Image</a>.
          </Caption>
        </Card>

        <SectionTitle>Train it and watch the static resolve</SectionTitle>
        <Prose>
          The network gets {N_TRAIN.toLocaleString()} photographs at 32 by 32 pixels, each labelled with one of ten
          things: cat, ship, truck, frog and so on. It sees brightness only, no colour. Its one job is to guess the
          label, and after each guess it adjusts its weights to be a little less wrong. That measure of wrongness is
          called the loss, and nothing in it mentions edges, or orientation, or the word filter.
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

          <div style={{ display: "flex", gap: 10, flexWrap: "wrap", marginBottom: 20 }}>
            <StatBox
              label="Filters that look like edges"
              value={`${filterReport.over} of ${f1}`}
              sub={`${baseline.over} of ${baseline.count} random ones did`}
              bg={filterReport.over > baseline.over ? C.blueSoft : C.faint}
              color={filterReport.over > baseline.over ? C.blue : C.ink}
            />
            <StatBox
              label="Accuracy on unseen photos"
              value={stats.acc === null ? "—" : `${(stats.acc * 100).toFixed(1)}%`}
              sub={`${N_TEST.toLocaleString()} photos held back, chance is 10%`}
              bg={stats.acc && stats.acc > 0.25 ? C.greenSoft : C.faint}
              color={stats.acc && stats.acc > 0.25 ? C.green : C.ink}
            />
            <StatBox
              label="Photos seen"
              value={stats.samples.toLocaleString()}
              sub={`${passes.toFixed(1)} passes over the set`}
            />
          </div>

          <div className="lf-cols">
            <div>
              <Label>BEFORE TRAINING: RANDOM NUMBERS</Label>
              <div className="lf-grid" style={{ opacity: 0.85 }}>
                {Array.from({ length: f1 }, (_, i) => (
                  <PixelCanvas
                    key={i}
                    size={k1}
                    selected={selected === i}
                    onNode={(node) => {
                      initRefs.current[i] = node;
                    }}
                  />
                ))}
              </div>
            </div>
            <div>
              <Label>NOW: THE SAME SIXTEEN FILTERS</Label>
              <div className="lf-grid">
                {Array.from({ length: f1 }, (_, i) => {
                  const d = filterReport.items[i];
                  return (
                    <PixelCanvas
                      key={i}
                      size={k1}
                      selected={selected === i}
                      onClick={() => setPinned(i)}
                      label={`Select filter ${i + 1}`}
                      tickAngle={d.orientationScore > ORIENTED_THRESHOLD ? d.angle : undefined}
                      onNode={(node) => {
                        tileRefs.current[i] = node;
                      }}
                    />
                  );
                })}
              </div>
            </div>
          </div>
          <Caption>
            Each tile is one filter&apos;s {k1 * k1} numbers drawn as a picture, shaded the same way as the Sobel
            numbers above: dark for negative, light for positive, grey for zero. A blue tick means the tile now
            matches an oriented edge, and the tick lies along that edge. Click a tile to inspect it.
          </Caption>

          <div style={{ marginTop: 22, paddingTop: 18, borderTop: `1px solid ${C.border}` }}>
            <Label>
              FILTER {selected + 1} ON A PHOTOGRAPH IT NEVER TRAINED ON: {selectedWords.toUpperCase()}
            </Label>
            <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 12 }}>
              <PhotoPicker photoId={photoId} uploaded={!!customPhoto} onPick={pickPhoto} onUpload={onUpload} />
            </div>
            {activePhoto && (
              <div className="lf-cols">
                <div>
                  <Label>PHOTOGRAPH</Label>
                  <PhotoCanvas photo={activePhoto} />
                </div>
                <div>
                  <Label>WHERE FILTER {selected + 1} FIRES</Label>
                  {learnedResponse && <ResponseCanvas response={learnedResponse} scale={learnedScale} />}
                </div>
              </div>
            )}
            <Caption>
              Terracotta where the photograph matches the filter&apos;s light side, blue where it matches its dark
              side, pale where it matches neither. This is the same slide, multiply and add as the Sobel demo, only the
              nine numbers have been replaced by these forty nine.
            </Caption>
          </div>

          {accHistory.length > 1 && (
            <div style={{ marginTop: 18 }}>
              <Label>ACCURACY ON UNSEEN PHOTOS OVER TRAINING</Label>
              <Sparkline series={accHistory} color={C.green} />
            </div>
          )}

          <Caption>
            Accuracy settles somewhere around forty percent, four times better than guessing and nowhere near a real
            vision model. The filters are the point, not the score.
          </Caption>
        </Card>

        <SectionTitle>Measure how edge-like it is</SectionTitle>
        <Prose>
          Clicking a tile above picks the filter shown here. To call a tile an edge detector without eyeballing it,
          each filter is compared with ideal edges at 36 angles, and the closest match becomes its score: 1 is a
          perfect edge and 0 is no resemblance. Out of {baseline.count} freshly initialised random filters,{" "}
          {baseline.over === 0 ? "not one" : `only ${baseline.over}`} cleared {ORIENTED_THRESHOLD}.
        </Prose>
        <Card>
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
                  value={selectedIsEdge ? edgeWords(selectedReport.angle) : "none yet"}
                  sub={`match ${selectedReport.orientationScore.toFixed(2)}, random averages ${baseline.mean.toFixed(2)}`}
                  bg={selectedIsEdge ? C.blueSoft : C.faint}
                  color={selectedIsEdge ? C.blue : C.ink}
                />
              </div>
            </div>
            <div>
              <Label>HOW HARD IT ANSWERS TO STRIPES AT EACH ANGLE</Label>
              <TuningChart now={tuning.now} before={tuning.before} />
            </div>
          </div>
          <Caption>
            Neuroscientists run this same test on cells in the visual cortex: show a striped patch at every angle in
            turn and record how hard the cell answers. A trained filter usually gives one hump, which means it has
            picked an orientation and mostly ignores the rest. The dashed line is the same filter before training,
            when it answered weakly to everything and strongly to nothing. Each line is drawn against its own peak so
            the shapes can be compared, and the peaks are printed underneath because the trained filter also answers
            several times harder.
          </Caption>
        </Card>

        <SectionTitle>So what does it mean that a model learned something</SectionTitle>
        <Prose>
          It means the numbers moved. Nobody told the network what an edge is; the only feedback was whether its guess
          about the photo was right, and the weights drifted, one small nudge at a time, until the first layer held
          something useful. Edges are useful because they are what photographs are made of: outlines, textures, the
          boundary of a shadow. This network has {net.paramCount.toLocaleString()} weights against{" "}
          {trainingNumbers.toLocaleString()} pixel values in its training photos, so it cannot store the photos and has
          to keep what they share. A much bigger network has room to store far more, yet its first layer still learns
          oriented edge filters much like those found in the cells of an animal&apos;s visual cortex, because that
          structure comes from photographs, not from the size of the budget.
        </Prose>
        <Prose>
          The same goes for grammar, or arithmetic, or anything else a language model appears to know. No rule was
          written down anywhere. Numbers were nudged, one gradient step at a time (a tiny change to every weight, in the
          direction that shrinks the loss), until they encoded something general enough to work on inputs nobody had
          shown the model. Vision hands you the receipt as a picture, because a first-layer filter lives in pixel space
          and can be printed. The weights of a language model, a design called a transformer, map one abstract space
          to another, so printing them raw gives you static, and finding what they learned takes other instruments:
          the geometry of an <a href="/a/word-embeddings">embedding table</a> (how words are placed as points in
          space), the pattern an <a href="/a/attention-explainer">attention head</a> settles into (which earlier words
          a word looks at), or the plain fact that a working model is{" "}
          <a href="/a/inside-an-llm">a file of numbers you can corrupt</a>.
        </Prose>

        <footer style={{ marginTop: 30, paddingTop: 16, borderTop: `1px solid ${C.border}` }}>
          <p style={{ fontSize: 12.5, color: C.muted, lineHeight: 1.7, margin: 0, maxWidth: "68ch" }}>
            The photographs are {(N_TRAIN + N_TEST).toLocaleString()} images from CIFAR-10 (Krizhevsky, 2009),
            converted to grey, 800 per class for training and 100 per class held back. Each is rescaled to the same
            average brightness and contrast before the network sees it, so only structure is left to learn from. The
            zebras and the facade are CC0 photographs from Wikimedia Commons. Everything runs in this tab: the forward
            pass (photo in, guess out), the gradients (how much each weight should change) and Adam (the rule that
            applies the change), with accuracy measured on the held-back photos. The network is two convolutional
            layers and a linear head trained on cross entropy, a loss that punishes confident wrong guesses hardest,
            and its first layer starts at a smaller scale than the usual initialisation so the structure outgrows the
            random draw inside the seconds a browser tab can spare. From a standard initialisation the same oriented
            filters appear, several times slower.
          </p>
        </footer>
      </div>
    </div>
  );
}
