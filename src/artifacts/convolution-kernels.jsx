import { useState, useEffect, useRef, useMemo, useCallback } from "react";

export const meta = {
  title: "Convolution: Kernels on an Image",
  category: "Computer Vision",
  description:
    "Edge detection, blur, and sharpen are all the same nine-number trick. Slide a 3x3 kernel across an image and watch the multiply-and-sum behind each output pixel, the operation a CNN learns on its own.",
  date: "2026-03-17",
  tags: ["convolution", "cnn", "kernels", "computer-vision"],
};

const C = {
  bg: "#f7f5f2",
  card: "#ffffff",
  border: "#e6e1d9",
  ink: "#1b1916",
  muted: "#857c70",
  accent: "#c0561f",
  accentSoft: "#f6ece5",
  good: "#3f7d52",
  cool: "#3a5a78",
  grid: "#efebe4",
};

const EASE = "cubic-bezier(0.23, 1, 0.32, 1)";

const PRESETS = {
  identity: {
    label: "Identity",
    k: [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
    note: "Copies each pixel to itself. The output equals the input, the simplest sanity check.",
  },
  box: {
    label: "Box blur",
    k: [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
    note: "Every neighbor counts equally. Summing to 9 it must be normalized (divide by 9) to keep brightness, then it is a plain local average.",
  },
  gaussian: {
    label: "Gaussian blur",
    k: [[1, 2, 1], [2, 4, 2], [1, 2, 1]],
    note: "A weighted average that trusts the center most and falls off toward the corners. Sums to 16, so normalize to preserve brightness. Smoother than box blur.",
  },
  sharpen: {
    label: "Sharpen",
    k: [[0, -1, 0], [-1, 5, -1], [0, -1, 0]],
    note: "Boosts the center and subtracts its neighbors, amplifying local contrast. Weights sum to 1, so flat regions stay put while edges get crisper.",
  },
  sobelX: {
    label: "Sobel X",
    k: [[1, 0, -1], [2, 0, -2], [1, 0, -1]],
    note: "Left weights positive, right weights negative. The response is large where brightness changes horizontally, so it lights up vertical edges. Sums to 0.",
  },
  sobelY: {
    label: "Sobel Y",
    k: [[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
    note: "Top minus bottom. It responds to vertical brightness change and lights up horizontal edges. Sums to 0.",
  },
  laplacian: {
    label: "Edge (Laplacian)",
    k: [[0, -1, 0], [-1, 4, -1], [0, -1, 0]],
    note: "The center against all four neighbors at once. It responds to edges in any direction and is zero across flat regions. Sums to 0.",
  },
  emboss: {
    label: "Emboss",
    k: [[-2, -1, 0], [-1, 1, 1], [0, 1, 2]],
    note: "An asymmetric edge filter along the diagonal that turns slopes into a lit-from-one-side relief. Sums to 1.",
  },
};

const PRESET_ORDER = ["identity", "box", "gaussian", "sharpen", "sobelX", "sobelY", "laplacian", "emboss"];

const GRID = 24;

function buildImage(n) {
  const img = new Float32Array(n * n);
  const cx = n * 0.42;
  const cy = n * 0.4;
  const r = n * 0.26;
  for (let y = 0; y < n; y++) {
    for (let x = 0; x < n; x++) {
      let v = 0.16;
      const dist = Math.hypot(x - cx, y - cy);
      if (dist < r) v = 0.9;
      const edge = r - dist;
      if (edge > 0 && edge < 1.5) v = 0.16 + (0.9 - 0.16) * (edge / 1.5);
      if (x - y > n * 0.28) v = Math.max(v, 0.62);
      if (y > n * 0.78) v = 0.34;
      img[y * n + x] = Math.min(1, Math.max(0, v));
    }
  }
  return img;
}

function clampIndex(i, n) {
  if (i < 0) return 0;
  if (i >= n) return n - 1;
  return i;
}

function convolvePixel(img, n, kernel, divisor, x, y) {
  let acc = 0;
  for (let ky = 0; ky < 3; ky++) {
    for (let kx = 0; kx < 3; kx++) {
      const sy = clampIndex(y + ky - 1, n);
      const sx = clampIndex(x + kx - 1, n);
      acc += img[sy * n + sx] * kernel[ky][kx];
    }
  }
  return acc / divisor;
}

function convolve(img, n, kernel, divisor) {
  const out = new Float32Array(n * n);
  for (let y = 0; y < n; y++) {
    for (let x = 0; x < n; x++) {
      out[y * n + x] = convolvePixel(img, n, kernel, divisor, x, y);
    }
  }
  return out;
}

function kernelSum(kernel) {
  let s = 0;
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) s += kernel[i][j];
  return s;
}

function PixelGrid({ img, n, cell, focus, highlight, signed, label, onPick }) {
  const ref = useRef(null);
  useEffect(() => {
    const cv = ref.current;
    if (!cv) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const size = n * cell;
    cv.width = size * dpr;
    cv.height = size * dpr;
    const ctx = cv.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, size, size);

    for (let y = 0; y < n; y++) {
      for (let x = 0; x < n; x++) {
        const raw = img[y * n + x];
        let g;
        if (signed) {
          const mag = Math.min(1, Math.abs(raw));
          g = raw >= 0
            ? `rgb(${Math.round(247 - mag * 60)},${Math.round(245 - mag * 200)},${Math.round(242 - mag * 240)})`
            : `rgb(${Math.round(247 - mag * 150)},${Math.round(245 - mag * 130)},${Math.round(242 - mag * 40)})`;
        } else {
          const v = Math.round(Math.min(1, Math.max(0, raw)) * 255);
          g = `rgb(${v},${v},${v})`;
        }
        ctx.fillStyle = g;
        ctx.fillRect(x * cell, y * cell, cell, cell);
      }
    }

    if (highlight && focus) {
      const { x, y } = focus;
      ctx.strokeStyle = C.accent;
      ctx.lineWidth = 1.5;
      ctx.strokeRect((x - 1) * cell + 0.75, (y - 1) * cell + 0.75, cell * 3 - 1.5, cell * 3 - 1.5);
      ctx.fillStyle = "rgba(192,86,31,0.16)";
      ctx.fillRect((x - 1) * cell, (y - 1) * cell, cell * 3, cell * 3);
      ctx.strokeStyle = C.ink;
      ctx.lineWidth = 1.5;
      ctx.strokeRect(x * cell + 0.75, y * cell + 0.75, cell - 1.5, cell - 1.5);
    } else if (focus) {
      const { x, y } = focus;
      ctx.strokeStyle = C.accent;
      ctx.lineWidth = 1.5;
      ctx.strokeRect(x * cell + 0.75, y * cell + 0.75, cell - 1.5, cell - 1.5);
    }
  }, [img, n, cell, focus, highlight, signed]);

  const handle = useCallback((e) => {
    if (!onPick) return;
    const rect = e.currentTarget.getBoundingClientRect();
    const px = e.clientX - rect.left;
    const py = e.clientY - rect.top;
    const x = clampIndex(Math.floor((px / rect.width) * n), n);
    const y = clampIndex(Math.floor((py / rect.height) * n), n);
    onPick(x, y);
  }, [onPick, n]);

  const size = n * cell;
  return (
    <canvas
      ref={ref}
      onMouseMove={onPick ? handle : undefined}
      style={{
        width: size, height: size, maxWidth: "100%", aspectRatio: "1 / 1",
        borderRadius: 8, background: C.bg, border: `1px solid ${C.border}`,
        display: "block", cursor: onPick ? "crosshair" : "default",
        imageRendering: "pixelated",
      }}
      role="img"
      aria-label={label}
    />
  );
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

function btnStyle(active) {
  return {
    padding: "7px 13px", borderRadius: 9,
    border: `1.5px solid ${active ? C.accent : C.border}`,
    background: active ? C.accent : C.card,
    color: active ? "#fff" : C.muted,
    fontSize: 12.5, fontWeight: 600, cursor: "pointer", fontFamily: "inherit",
    transition: `transform 140ms ${EASE}, background 160ms ${EASE}, border-color 160ms ${EASE}, color 160ms ${EASE}`,
  };
}

const MONO = "'IBM Plex Mono', ui-monospace, SFMono-Regular, Menlo, monospace";

function KernelEditor({ kernel, setCell, focusK }) {
  return (
    <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 1fr)", gap: 6 }}>
      {kernel.map((row, i) =>
        row.map((val, j) => {
          const lit = focusK && focusK.i === i && focusK.j === j;
          return (
            <div key={`${i}-${j}`} style={{ position: "relative" }}>
              <input
                type="number"
                step="1"
                value={val}
                aria-label={`kernel weight row ${i + 1} column ${j + 1}`}
                onChange={(e) => setCell(i, j, e.target.value)}
                style={{
                  width: "100%", boxSizing: "border-box", textAlign: "center",
                  fontFamily: MONO, fontSize: 16, fontWeight: 700,
                  padding: "12px 4px", borderRadius: 8,
                  border: `1.5px solid ${lit ? C.accent : C.border}`,
                  background: lit ? C.accentSoft : C.bg,
                  color: val < 0 ? C.cool : C.ink,
                  transition: `background 140ms ${EASE}, border-color 140ms ${EASE}`,
                  MozAppearance: "textfield",
                }}
              />
            </div>
          );
        })
      )}
    </div>
  );
}

function Stepper({ onUp, onDown, children }) {
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
      <button className="ckx-btn" onClick={onDown} aria-label="decrement" style={{ ...btnStyle(false), padding: "5px 11px", fontSize: 15 }}>&minus;</button>
      <span style={{ minWidth: 30, textAlign: "center", fontFamily: MONO, fontSize: 14, fontWeight: 700, color: C.ink }}>{children}</span>
      <button className="ckx-btn" onClick={onUp} aria-label="increment" style={{ ...btnStyle(false), padding: "5px 11px", fontSize: 15 }}>+</button>
    </div>
  );
}

export default function App() {
  const [presetKey, setPresetKey] = useState("sobelX");
  const [kernel, setKernel] = useState(() => PRESETS.sobelX.k.map((r) => [...r]));
  const [normalize, setNormalize] = useState(false);
  const [focus, setFocus] = useState({ x: 11, y: 9 });
  const [playing, setPlaying] = useState(false);
  const [presetNote, setPresetNote] = useState(PRESETS.sobelX.note);

  const baseImage = useMemo(() => buildImage(GRID), []);

  const sum = useMemo(() => kernelSum(kernel), [kernel]);
  const divisor = normalize && sum !== 0 ? sum : 1;

  const output = useMemo(() => convolve(baseImage, GRID, kernel, divisor), [baseImage, kernel, divisor]);

  const signedOutput = useMemo(() => Math.abs(sum) < 1e-9, [sum]);

  const applyPreset = useCallback((key) => {
    setPresetKey(key);
    setKernel(PRESETS[key].k.map((r) => [...r]));
    setPresetNote(PRESETS[key].note);
    const s = kernelSum(PRESETS[key].k);
    setNormalize(s > 1.5);
  }, []);

  const setCell = useCallback((i, j, raw) => {
    const num = raw === "" || raw === "-" ? 0 : Number(raw);
    if (Number.isNaN(num)) return;
    setKernel((prev) => {
      const next = prev.map((r) => [...r]);
      next[i][j] = num;
      return next;
    });
    setPresetKey("custom");
    setPresetNote("Custom weights. The output recomputes on every edit, exactly as a network would if it nudged these numbers during training.");
  }, []);

  const focusRef = useRef(focus);
  useEffect(() => { focusRef.current = focus; }, [focus]);

  useEffect(() => {
    if (!playing) return;
    const reduce = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    const interval = reduce ? 160 : 90;
    const id = setInterval(() => {
      setFocus((f) => {
        let x = f.x + 1;
        let y = f.y;
        if (x >= GRID) { x = 0; y = y + 1; }
        if (y >= GRID) { y = 0; }
        return { x, y };
      });
    }, interval);
    return () => clearInterval(id);
  }, [playing]);

  const neighborhood = useMemo(() => {
    const cells = [];
    for (let ky = 0; ky < 3; ky++) {
      for (let kx = 0; kx < 3; kx++) {
        const sy = clampIndex(focus.y + ky - 1, GRID);
        const sx = clampIndex(focus.x + kx - 1, GRID);
        const pix = baseImage[sy * GRID + sx];
        const w = kernel[ky][kx];
        const edged = sy !== focus.y + ky - 1 || sx !== focus.x + kx - 1;
        cells.push({ ky, kx, pix, w, product: pix * w, edged });
      }
    }
    return cells;
  }, [focus, baseImage, kernel]);

  const rawSum = neighborhood.reduce((a, c) => a + c.product, 0);
  const result = rawSum / divisor;

  const cellInput = Math.max(7, Math.min(13, Math.floor(300 / GRID)));

  return (
    <div style={{ fontFamily: "'Iowan Old Style', 'Palatino Linotype', Palatino, Georgia, serif", background: C.bg, minHeight: "100vh", padding: "30px 14px 56px", color: C.ink }}>
      <style>{`
        @media (prefers-reduced-motion: reduce) {
          * { transition-duration: 1ms !important; animation-duration: 1ms !important; }
        }
        .ckx-btn:active { transform: scale(0.97); }
        .ckx-btn:focus-visible {
          outline: 2px solid ${C.accent}; outline-offset: 2px;
        }
        input:focus-visible { outline: 2px solid ${C.accent}; outline-offset: 1px; }
        input[type=number]::-webkit-outer-spin-button,
        input[type=number]::-webkit-inner-spin-button { -webkit-appearance: none; margin: 0; }
      `}</style>

      <div style={{ maxWidth: 720, margin: "0 auto" }}>
        <header style={{ marginBottom: 22 }}>
          <Eyebrow>Computer Vision · Convolutional Networks</Eyebrow>
          <h1 style={{ fontSize: 30, fontWeight: 700, margin: "6px 0 0", lineHeight: 1.1, textWrap: "balance" }}>
            Convolution: Kernels on an Image
          </h1>
          <p style={{ color: C.ink, opacity: 0.82, fontSize: 15, margin: "10px 0 0", lineHeight: 1.6, maxWidth: "64ch" }}>
            A convolution slides a tiny grid of weights, the kernel, across an image. At each position it multiplies
            the kernel against the pixels underneath and sums them into one output pixel. That weighted local average
            is the whole operation. The surprise is how much falls out of just nine numbers: change the weights and
            the same machinery detects edges, blurs, or sharpens.
          </p>
        </header>

        <Card style={{ marginBottom: 18 }}>
          <Eyebrow>Kernel</Eyebrow>
          <h2 style={{ fontSize: 19, fontWeight: 700, margin: "5px 0 14px" }}>Pick a kernel, or edit the weights</h2>

          <div style={{ display: "flex", gap: 7, flexWrap: "wrap", marginBottom: 16 }}>
            {PRESET_ORDER.map((key) => (
              <button key={key} className="ckx-btn" onClick={() => applyPreset(key)}
                style={btnStyle(presetKey === key)} aria-pressed={presetKey === key}>
                {PRESETS[key].label}
              </button>
            ))}
            {presetKey === "custom" && (
              <span style={{ ...btnStyle(true), cursor: "default" }}>Custom</span>
            )}
          </div>

          <div style={{ display: "flex", gap: 18, flexWrap: "wrap", alignItems: "flex-start" }}>
            <div style={{ flex: "0 0 168px", maxWidth: 168 }}>
              <KernelEditor kernel={kernel} setCell={setCell}
                focusK={null} />
              <div style={{ marginTop: 10, fontSize: 12.5, color: C.muted, fontFamily: MONO }}>
                sum = <b style={{ color: Math.abs(sum) < 1e-9 ? C.cool : C.ink }}>{sum}</b>
                {normalize && sum !== 0 && <span> &nbsp;&middot;&nbsp; &divide; {sum}</span>}
              </div>
              <label style={{ display: "flex", alignItems: "center", gap: 9, marginTop: 12, fontSize: 13, color: C.ink, cursor: sum === 0 ? "not-allowed" : "pointer", opacity: sum === 0 ? 0.5 : 1 }}>
                <input type="checkbox" checked={normalize && sum !== 0} disabled={sum === 0}
                  onChange={(e) => setNormalize(e.target.checked)}
                  style={{ accentColor: C.accent, width: 16, height: 16 }} />
                Normalize (divide by sum)
              </label>
            </div>

            <div style={{ flex: "1 1 240px", minWidth: 220, fontSize: 13.5, color: C.ink, opacity: 0.88, lineHeight: 1.6 }}>
              {presetNote}
              <div style={{ marginTop: 12, padding: "10px 13px", background: C.bg, borderRadius: 9, border: `1px solid ${C.border}`, fontSize: 12.5, color: C.muted, lineHeight: 1.55 }}>
                Blur kernels sum to 1 (after normalizing) so the picture keeps its overall brightness, they only
                redistribute it. Edge kernels sum to 0 so flat regions cancel to zero and only changes survive. Negative
                weights are drawn in <b style={{ color: C.cool }}>blue</b>.
              </div>
            </div>
          </div>
        </Card>

        <Card style={{ marginBottom: 18 }}>
          <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", gap: 10, flexWrap: "wrap" }}>
            <div>
              <Eyebrow>Sliding window</Eyebrow>
              <h2 style={{ fontSize: 19, fontWeight: 700, margin: "5px 0 0" }}>Input, kernel, output</h2>
            </div>
            <span style={{ fontSize: 12.5, color: C.muted, fontFamily: MONO }}>
              position ({focus.x}, {focus.y})
            </span>
          </div>

          <p style={{ fontSize: 13.5, color: C.ink, opacity: 0.82, margin: "10px 0 16px", lineHeight: 1.6, maxWidth: "64ch" }}>
            Move your pointer over the input to place the window, or press Step to advance it one pixel at a time.
            The highlighted 3x3 patch on the left feeds the one outlined pixel on the right.
          </p>

          <div style={{ display: "flex", gap: 16, flexWrap: "wrap", alignItems: "flex-start", justifyContent: "center" }}>
            <div style={{ textAlign: "center" }}>
              <div style={{ fontSize: 11.5, color: C.muted, marginBottom: 6, fontWeight: 700 }}>Input</div>
              <PixelGrid img={baseImage} n={GRID} cell={cellInput} focus={focus} highlight
                label="Input grayscale image with the active 3 by 3 window highlighted"
                onPick={(x, y) => { setFocus({ x, y }); }} />
              <div style={{ fontSize: 11, color: C.muted, marginTop: 5 }}>{GRID}&times;{GRID} grayscale</div>
            </div>
            <div style={{ textAlign: "center" }}>
              <div style={{ fontSize: 11.5, color: C.muted, marginBottom: 6, fontWeight: 700 }}>Output</div>
              <PixelGrid img={output} n={GRID} cell={cellInput} focus={focus} signed={signedOutput}
                label="Convolved output image with the focused pixel outlined" />
              <div style={{ fontSize: 11, color: C.muted, marginTop: 5 }}>
                {signedOutput ? "signed: blue −, orange +" : "computed live"}
              </div>
            </div>
          </div>

          <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginTop: 16, alignItems: "center", justifyContent: "center" }}>
            <button className="ckx-btn" onClick={() => setPlaying((p) => !p)} style={{ ...btnStyle(true) }}>
              {playing ? "Pause" : "Play sweep"}
            </button>
            <button className="ckx-btn" onClick={() => { setPlaying(false); setFocus((f) => { let x = f.x + 1, y = f.y; if (x >= GRID) { x = 0; y++; } if (y >= GRID) y = 0; return { x, y }; }); }}
              style={btnStyle(false)}>Step</button>
            <button className="ckx-btn" onClick={() => { setPlaying(false); setFocus({ x: 11, y: 9 }); }} style={btnStyle(false)}>Reset</button>
          </div>
        </Card>

        <Card>
          <Eyebrow>The one pixel</Eyebrow>
          <h2 style={{ fontSize: 19, fontWeight: 700, margin: "5px 0 4px" }}>Multiply, then add</h2>
          <p style={{ fontSize: 13.5, color: C.ink, opacity: 0.82, margin: "0 0 16px", lineHeight: 1.6, maxWidth: "64ch" }}>
            Output pixel ({focus.x}, {focus.y}) is the sum of nine products: each kernel weight times the input pixel
            beneath it. Cells marked with a dot read a clamped border pixel.
          </p>

          <div style={{ display: "flex", gap: 14, flexWrap: "wrap", alignItems: "center", justifyContent: "center", marginBottom: 16 }}>
            <BreakdownGrid title="input patch" cells={neighborhood.map((c) => ({ value: c.pix.toFixed(2), edged: c.edged, kind: "pix" }))} />
            <Operator>&times;</Operator>
            <BreakdownGrid title="kernel" cells={neighborhood.map((c) => ({ value: c.w, kind: "w" }))} />
            <Operator>=</Operator>
            <BreakdownGrid title="products" cells={neighborhood.map((c) => ({ value: c.product.toFixed(2), kind: "prod" }))} />
          </div>

          <div style={{ background: C.accentSoft, border: `1px solid ${C.accent}33`, borderRadius: 10, padding: "13px 16px", fontFamily: MONO, fontSize: 13, color: C.ink, lineHeight: 1.7, overflowX: "auto" }}>
            <div style={{ color: C.muted, fontSize: 11, marginBottom: 4 }}>sum of nine products</div>
            {neighborhood.map((c, i) => (
              <span key={i} style={{ color: c.product < 0 ? C.cool : C.ink, whiteSpace: "nowrap" }}>
                {i > 0 && <span style={{ color: C.muted }}> {c.product < 0 ? "−" : "+"} </span>}
                {i === 0 && c.product < 0 && <span style={{ color: C.cool }}>−</span>}
                {Math.abs(c.product).toFixed(2)}
              </span>
            ))}
            <div style={{ marginTop: 8, fontSize: 15, fontWeight: 700 }}>
              = {rawSum.toFixed(3)}
              {divisor !== 1 && <span style={{ color: C.muted }}> &divide; {divisor} = <span style={{ color: C.accent }}>{result.toFixed(3)}</span></span>}
              {divisor === 1 && <span style={{ color: C.accent }}> &rarr; {result.toFixed(3)}</span>}
            </div>
          </div>

          <div style={{ marginTop: 16, padding: "11px 14px", background: C.bg, borderRadius: 10, border: `1px solid ${C.border}`, fontSize: 13, color: C.ink, opacity: 0.85, lineHeight: 1.6 }}>
            <b>Why a CNN cares.</b> A convolutional network does not hand-write these nine numbers; it learns them
            from data by gradient descent. Early layers tend to converge on exactly these kinds of edge and blob
            detectors, then later layers convolve over those responses to find corners, textures, and eventually whole
            objects. The arithmetic on this card is one neuron firing, repeated across every position and every channel.
          </div>

          <div style={{ marginTop: 12, padding: "11px 14px", background: C.bg, borderRadius: 10, border: `1px solid ${C.border}`, fontSize: 12.5, color: C.muted, lineHeight: 1.6 }}>
            <b style={{ color: C.ink }}>Borders.</b> At the edge the 3x3 window hangs off the image. This uses edge
            clamping: out-of-bounds reads repeat the nearest valid pixel, so the output stays the same size as the
            input and the border does not go dark. Zero padding and reflection are the other common choices.
          </div>
        </Card>
      </div>
    </div>
  );
}

function Operator({ children }) {
  return <div style={{ fontSize: 22, color: C.muted, fontWeight: 700 }}>{children}</div>;
}

function BreakdownGrid({ title, cells }) {
  return (
    <div style={{ textAlign: "center" }}>
      <div style={{ fontSize: 10.5, color: C.muted, marginBottom: 6, fontWeight: 700, letterSpacing: "0.04em" }}>{title}</div>
      <div style={{ display: "grid", gridTemplateColumns: "repeat(3, 46px)", gap: 4 }}>
        {cells.map((c, i) => {
          const neg = typeof c.value === "number" ? c.value < 0 : String(c.value).startsWith("-");
          return (
            <div key={i} style={{
              height: 40, display: "flex", alignItems: "center", justifyContent: "center",
              fontFamily: MONO, fontSize: 13, fontWeight: 700,
              borderRadius: 7,
              background: c.kind === "prod" ? C.accentSoft : C.bg,
              border: `1px solid ${c.edged ? C.accent : C.border}`,
              color: neg ? C.cool : C.ink,
              position: "relative",
            }}>
              {c.value}
              {c.edged && <span style={{ position: "absolute", top: 2, right: 4, fontSize: 9, color: C.accent }}>&middot;</span>}
            </div>
          );
        })}
      </div>
    </div>
  );
}
