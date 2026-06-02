import { useState, useEffect, useRef, useMemo, useCallback } from "react";

export const meta = {
  title: "Matrix as a Linear Transformation",
  category: "Linear Algebra",
  description:
    "A 2x2 matrix is a recipe for bending the plane while keeping every line straight and the origin pinned. Drag where the basis vectors land and watch the grid, the determinant as signed area, and the eigenvectors as the directions that refuse to turn.",
  date: "2026-05-24",
  tags: ["linear-algebra", "matrix", "eigenvectors", "determinant"],
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
  grid: "#efebe4",
  gridLine: "#e0d9cd",
  iHat: "#c0561f",
  jHat: "#2f6f9e",
  eigen: "#3f7d52",
  posArea: "rgba(192, 86, 31, 0.16)",
  negArea: "rgba(47, 111, 158, 0.18)",
  shape: "#9a6b1f",
};

const EASE = "cubic-bezier(0.23, 1, 0.32, 1)";
const MONO = "'IBM Plex Mono', ui-monospace, SFMono-Regular, Menlo, monospace";

function applyMatrix(m, x, y) {
  return [m.a * x + m.b * y, m.c * x + m.d * y];
}

function determinant(m) {
  return m.a * m.d - m.b * m.c;
}

function trace(m) {
  return m.a + m.d;
}

// Closed-form eigen-solve for a real 2x2. The characteristic polynomial is
// lambda^2 - tr*lambda + det = 0; the discriminant sign separates the three
// honest cases (real distinct, real repeated, complex conjugate). For each
// real eigenvalue we read an eigenvector from the null space of (M - lambda I)
// without inverting anything, so a singular M is handled too.
function eigen(m) {
  const tr = trace(m);
  const det = determinant(m);
  const disc = tr * tr - 4 * det;
  const eps = 1e-9;

  if (disc < -eps) {
    const re = tr / 2;
    const im = Math.sqrt(-disc) / 2;
    return {
      kind: "complex",
      values: [
        { re, im },
        { re, im: -im },
      ],
      vectors: [],
    };
  }

  const sqrtDisc = Math.sqrt(Math.max(disc, 0));
  const l1 = (tr + sqrtDisc) / 2;
  const l2 = (tr - sqrtDisc) / 2;
  const repeated = Math.abs(disc) <= eps;

  const vectorFor = (lambda) => {
    const r1 = [m.a - lambda, m.b];
    const r2 = [m.c, m.d - lambda];
    let v;
    if (Math.hypot(r1[0], r1[1]) >= Math.hypot(r2[0], r2[1])) {
      v = [r1[1], -r1[0]];
    } else {
      v = [r2[1], -r2[0]];
    }
    const norm = Math.hypot(v[0], v[1]);
    if (norm < eps) return null; // a scalar multiple of I: every direction is an eigenvector
    return [v[0] / norm, v[1] / norm];
  };

  const v1 = vectorFor(l1);
  const v2 = repeated ? null : vectorFor(l2);

  return {
    kind: repeated ? "repeated" : "real",
    values: [
      { re: l1, im: 0 },
      { re: l2, im: 0 },
    ],
    vectors: repeated
      ? [{ lambda: l1, v: v1 }]
      : [
          { lambda: l1, v: v1 },
          { lambda: l2, v: v2 },
        ],
  };
}

function rot(theta) {
  return {
    a: Math.cos(theta),
    b: -Math.sin(theta),
    c: Math.sin(theta),
    d: Math.cos(theta),
  };
}

const IDENTITY = { a: 1, b: 0, c: 0, d: 1 };

const PRESETS = {
  identity: { label: "Identity", m: { a: 1, b: 0, c: 0, d: 1 } },
  rotation: { label: "Rotation 30 degrees", m: rot(Math.PI / 6) },
  scale: { label: "Scale", m: { a: 1.8, b: 0, c: 0, d: 0.6 } },
  shear: { label: "Shear", m: { a: 1, b: 1, c: 0, d: 1 } },
  reflection: { label: "Reflection", m: { a: 1, b: 0, c: 0, d: -1 } },
  singular: { label: "Singular (det 0)", m: { a: 1, b: 0.5, c: 2, d: 1 } },
};

function lerpMatrix(from, to, t) {
  return {
    a: from.a + (to.a - from.a) * t,
    b: from.b + (to.b - from.b) * t,
    c: from.c + (to.c - from.c) * t,
    d: from.d + (to.d - from.d) * t,
  };
}

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

const VIEW = 360;
const UNIT = 52; // pixels per unit length in matrix space
const RANGE = Math.ceil(VIEW / 2 / UNIT) + 1;

function fmt(x, places = 2) {
  if (!Number.isFinite(x)) return "—";
  const v = Math.abs(x) < 1e-9 ? 0 : x;
  return v.toFixed(places);
}

const SHAPE_F = [
  [-0.2, -0.8],
  [-0.2, 0.8],
  [0.6, 0.8],
  [0.6, 0.5],
  [0.1, 0.5],
  [0.1, 0.15],
  [0.5, 0.15],
  [0.5, -0.15],
  [0.1, -0.15],
  [0.1, -0.8],
  [-0.2, -0.8],
];

function Plane({ m, eig, showEigen, shape, dragging, onDrag }) {
  const ref = useRef(null);

  useEffect(() => {
    const cv = ref.current;
    if (!cv) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    cv.width = VIEW * dpr;
    cv.height = VIEW * dpr;
    const ctx = cv.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, VIEW, VIEW);

    const ox = VIEW / 2;
    const oy = VIEW / 2;
    const toPx = (x, y) => [ox + x * UNIT, oy - y * UNIT];

    ctx.strokeStyle = C.grid;
    ctx.lineWidth = 1;
    for (let i = -RANGE; i <= RANGE; i++) {
      const [vx0, vy0] = toPx(i, -RANGE);
      const [vx1, vy1] = toPx(i, RANGE);
      ctx.beginPath();
      ctx.moveTo(vx0, vy0);
      ctx.lineTo(vx1, vy1);
      ctx.stroke();
      const [hx0, hy0] = toPx(-RANGE, i);
      const [hx1, hy1] = toPx(RANGE, i);
      ctx.beginPath();
      ctx.moveTo(hx0, hy0);
      ctx.lineTo(hx1, hy1);
      ctx.stroke();
    }

    // transformed grid: each gridline of the original plane mapped by M, so a
    // straight line stays straight (the defining property of a linear map)
    ctx.strokeStyle = C.gridLine;
    ctx.lineWidth = 1.1;
    const drawMapped = (p0, p1) => {
      const a = applyMatrix(m, p0[0], p0[1]);
      const b = applyMatrix(m, p1[0], p1[1]);
      const [ax, ay] = toPx(a[0], a[1]);
      const [bx, by] = toPx(b[0], b[1]);
      ctx.beginPath();
      ctx.moveTo(ax, ay);
      ctx.lineTo(bx, by);
      ctx.stroke();
    };
    for (let i = -RANGE; i <= RANGE; i++) {
      drawMapped([i, -RANGE], [i, RANGE]);
      drawMapped([-RANGE, i], [RANGE, i]);
    }

    // signed area of the transformed unit square equals the determinant; the
    // fill colour flips when orientation reverses (det < 0)
    const det = determinant(m);
    const corners = [
      applyMatrix(m, 0, 0),
      applyMatrix(m, 1, 0),
      applyMatrix(m, 1, 1),
      applyMatrix(m, 0, 1),
    ];
    ctx.beginPath();
    corners.forEach((c, idx) => {
      const [px, py] = toPx(c[0], c[1]);
      if (idx === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    });
    ctx.closePath();
    ctx.fillStyle = det < 0 ? C.negArea : C.posArea;
    ctx.fill();

    // sample shape (a letter F) carried through the same product
    ctx.strokeStyle = C.shape;
    ctx.lineWidth = 2;
    ctx.lineJoin = "round";
    ctx.lineCap = "round";
    ctx.beginPath();
    shape.forEach((pt, idx) => {
      const [mx, my] = applyMatrix(m, pt[0], pt[1]);
      const [px, py] = toPx(mx, my);
      if (idx === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    });
    ctx.stroke();

    // axes through the origin, which is fixed by every linear map
    ctx.strokeStyle = C.faint;
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    ctx.moveTo(ox, 6);
    ctx.lineTo(ox, VIEW - 6);
    ctx.moveTo(6, oy);
    ctx.lineTo(VIEW - 6, oy);
    ctx.stroke();

    if (showEigen && eig.kind !== "complex") {
      for (const ev of eig.vectors) {
        if (!ev.v) continue;
        const [dx, dy] = ev.v;
        const [sx, sy] = toPx(-dx * RANGE, -dy * RANGE);
        const [ex, ey] = toPx(dx * RANGE, dy * RANGE);
        ctx.strokeStyle = C.eigen;
        ctx.lineWidth = 1.6;
        ctx.setLineDash([6, 5]);
        ctx.beginPath();
        ctx.moveTo(sx, sy);
        ctx.lineTo(ex, ey);
        ctx.stroke();
        ctx.setLineDash([]);

        // the eigenvalue is the stretch along this invariant direction: draw
        // the image of the unit eigenvector as a solid stub whose length is it
        const [tx, ty] = toPx(dx * ev.lambda, dy * ev.lambda);
        ctx.strokeStyle = C.eigen;
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.moveTo(ox, oy);
        ctx.lineTo(tx, ty);
        ctx.stroke();
        ctx.fillStyle = C.eigen;
        ctx.beginPath();
        ctx.arc(tx, ty, 3.4, 0, Math.PI * 2);
        ctx.fill();
      }
    }

    const drawVector = (vx, vy, color, label) => {
      const [tx, ty] = toPx(vx, vy);
      ctx.strokeStyle = color;
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(ox, oy);
      ctx.lineTo(tx, ty);
      ctx.stroke();
      const ang = Math.atan2(ty - oy, tx - ox);
      const head = 9;
      ctx.fillStyle = color;
      ctx.beginPath();
      ctx.moveTo(tx, ty);
      ctx.lineTo(tx - head * Math.cos(ang - 0.4), ty - head * Math.sin(ang - 0.4));
      ctx.lineTo(tx - head * Math.cos(ang + 0.4), ty - head * Math.sin(ang + 0.4));
      ctx.closePath();
      ctx.fill();
      ctx.fillStyle = C.card;
      ctx.strokeStyle = color;
      ctx.lineWidth = 2.4;
      ctx.beginPath();
      ctx.arc(tx, ty, 7, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
      ctx.fillStyle = color;
      ctx.font = `700 13px ${MONO}`;
      ctx.fillText(label, tx + 11, ty - 9);
    };

    // i-hat lands on column 1 of M, j-hat lands on column 2
    drawVector(m.a, m.c, C.iHat, "i");
    drawVector(m.b, m.d, C.jHat, "j");
  }, [m, eig, showEigen, shape]);

  const pointToMatrixSpace = (e) => {
    const rect = ref.current.getBoundingClientRect();
    const scaleX = VIEW / rect.width;
    const scaleY = VIEW / rect.height;
    const px = (e.clientX - rect.left) * scaleX;
    const py = (e.clientY - rect.top) * scaleY;
    return [(px - VIEW / 2) / UNIT, (VIEW / 2 - py) / UNIT];
  };

  const handleDown = (e) => {
    const [mx, my] = pointToMatrixSpace(e);
    const di = Math.hypot(mx - m.a, my - m.c);
    const dj = Math.hypot(mx - m.b, my - m.d);
    const threshold = 20 / UNIT;
    let target = null;
    if (di < threshold && di <= dj) target = "i";
    else if (dj < threshold) target = "j";
    if (target) {
      e.currentTarget.setPointerCapture(e.pointerId);
      onDrag(target);
    }
  };

  const handleMove = (e) => {
    if (!dragging) return;
    onDrag(dragging, pointToMatrixSpace(e));
  };

  const handleUp = (e) => {
    if (dragging) {
      try {
        e.currentTarget.releasePointerCapture(e.pointerId);
      } catch {
        // the pointer can already be released when a gesture ends off-canvas
      }
      onDrag(null);
    }
  };

  return (
    <canvas
      ref={ref}
      onPointerDown={handleDown}
      onPointerMove={handleMove}
      onPointerUp={handleUp}
      onPointerCancel={handleUp}
      style={{
        width: VIEW,
        height: VIEW,
        maxWidth: "100%",
        borderRadius: 12,
        background: C.bg,
        border: `1px solid ${C.border}`,
        display: "block",
        touchAction: "none",
        cursor: dragging ? "grabbing" : "crosshair",
      }}
      role="img"
      aria-label="The plane under a 2 by 2 linear transformation. Drag the labelled i and j vector tips to set the matrix columns; the grid, the shaded unit-square area, and the eigenvectors update live."
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

function Stat({ label, value, accent, swatch }) {
  return (
    <div
      style={{
        background: accent ? C.accentSoft : C.bg,
        borderRadius: 9,
        padding: "9px 12px",
        border: `1px solid ${accent ? C.accent + "33" : C.border}`,
      }}
    >
      <div
        style={{
          fontSize: 10,
          letterSpacing: "0.06em",
          textTransform: "uppercase",
          color: accent ? C.accent : C.muted,
          fontWeight: 700,
          marginBottom: 3,
          display: "flex",
          alignItems: "center",
          gap: 5,
        }}
      >
        {swatch && <span style={{ width: 9, height: 9, borderRadius: 2, background: swatch, display: "inline-block" }} />}
        {label}
      </div>
      <div
        style={{
          fontFamily: MONO,
          fontSize: 16,
          fontWeight: 700,
          color: accent ? C.accent : C.ink,
          lineHeight: 1.15,
          fontVariantNumeric: "tabular-nums",
        }}
      >
        {value}
      </div>
    </div>
  );
}

function btnStyle(active, disabled) {
  return {
    padding: "8px 14px",
    borderRadius: 9,
    border: `1.5px solid ${active ? C.accent : C.border}`,
    background: active ? C.accent : C.card,
    color: active ? "#fff" : C.muted,
    fontSize: 13,
    fontWeight: 600,
    cursor: disabled ? "not-allowed" : "pointer",
    opacity: disabled ? 0.5 : 1,
    fontFamily: "inherit",
    transition: `transform 140ms ${EASE}, background 160ms ${EASE}, border-color 160ms ${EASE}, color 160ms ${EASE}`,
  };
}

function MatrixInput({ value, onChange, color, label }) {
  return (
    <input
      type="number"
      step="0.1"
      value={Number.isFinite(value) ? Math.round(value * 100) / 100 : 0}
      onChange={(e) => onChange(parseFloat(e.target.value))}
      aria-label={label}
      style={{
        width: "100%",
        padding: "9px 4px",
        textAlign: "center",
        fontFamily: MONO,
        fontSize: 16,
        fontWeight: 700,
        color: C.ink,
        background: C.card,
        border: `1.5px solid ${color}55`,
        borderRadius: 8,
        fontVariantNumeric: "tabular-nums",
      }}
    />
  );
}

export default function App() {
  const reduce = usePrefersReducedMotion();
  const [m, setM] = useState({ ...PRESETS.shear.m });
  const [showEigen, setShowEigen] = useState(true);
  const [dragging, setDragging] = useState(null);
  const [animating, setAnimating] = useState(false);

  const animTo = useRef(m);
  const animStart = useRef(0);
  const raf = useRef(0);

  const eig = useMemo(() => eigen(m), [m]);
  const det = determinant(m);
  const tr = trace(m);

  const setEntry = useCallback((key, val) => {
    if (!Number.isFinite(val)) return;
    setM((prev) => ({ ...prev, [key]: val }));
  }, []);

  const handleDrag = useCallback((target, point) => {
    if (target === null) {
      setDragging(null);
      return;
    }
    if (!point) {
      setDragging(target);
      return;
    }
    const [x, y] = point;
    setM((prev) => (target === "i" ? { ...prev, a: x, c: y } : { ...prev, b: x, d: y }));
  }, []);

  const runAnimation = useCallback(() => {
    if (reduce) return;
    cancelAnimationFrame(raf.current);
    animTo.current = { ...m };
    animStart.current = 0;
    setAnimating(true);
    const loop = (ts) => {
      if (!animStart.current) animStart.current = ts;
      const tRaw = Math.min((ts - animStart.current) / 720, 1);
      const eased = 1 - Math.pow(1 - tRaw, 3);
      setM(lerpMatrix(IDENTITY, animTo.current, eased));
      if (tRaw < 1) {
        raf.current = requestAnimationFrame(loop);
      } else {
        setM(animTo.current);
        setAnimating(false);
      }
    };
    raf.current = requestAnimationFrame(loop);
  }, [reduce, m]);

  useEffect(() => () => cancelAnimationFrame(raf.current), []);

  const applyPreset = (key) => {
    cancelAnimationFrame(raf.current);
    setAnimating(false);
    setM({ ...PRESETS[key].m });
  };

  const eigenReadout = () => {
    if (eig.kind === "complex") {
      const { re, im } = eig.values[0];
      return `${fmt(re)} ± ${fmt(Math.abs(im))} i`;
    }
    return eig.values.map((v) => fmt(v.re)).join(",  ");
  };

  return (
    <div
      style={{
        fontFamily: "'Iowan Old Style', 'Palatino Linotype', Palatino, Georgia, serif",
        background: C.bg,
        minHeight: "100vh",
        padding: "30px 14px 56px",
        color: C.ink,
      }}
    >
      <style>{`
        @media (prefers-reduced-motion: reduce) {
          * { transition-duration: 1ms !important; animation-duration: 1ms !important; }
        }
        .mt-btn:active { transform: scale(0.97); }
        .mt-btn:focus-visible, input:focus-visible {
          outline: 2px solid ${C.accent}; outline-offset: 2px;
        }
        input[type=number]::-webkit-inner-spin-button { opacity: 0.4; }
      `}</style>

      <div style={{ maxWidth: 720, margin: "0 auto" }}>
        <header style={{ marginBottom: 22 }}>
          <div style={{ fontSize: 11, letterSpacing: "0.14em", textTransform: "uppercase", color: C.faint, marginBottom: 6 }}>
            Linear algebra / Geometry of matrices
          </div>
          <h1 style={{ fontSize: 30, fontWeight: 700, margin: 0, lineHeight: 1.1, letterSpacing: "-0.02em", textWrap: "balance" }}>
            Matrix as a Linear Transformation
          </h1>
          <p style={{ color: C.muted, fontSize: 15, lineHeight: 1.6, margin: "10px 0 0", maxWidth: "64ch" }}>
            A 2x2 matrix is not a static grid of numbers, it is an instruction for moving every point of the plane
            at once. The rule keeps lines straight, keeps parallel lines parallel, and leaves the origin where it is.
            Everything below is the genuine matrix-vector product applied to the grid, to a sample shape, and to the
            two basis vectors. Drag the labelled tips, or type the entries, and read the geometry off the numbers.
          </p>
        </header>

        <Card style={{ marginBottom: 18 }}>
          <div style={{ display: "flex", gap: 18, flexWrap: "wrap", alignItems: "flex-start", justifyContent: "center" }}>
            <Plane m={m} eig={eig} showEigen={showEigen} shape={SHAPE_F} dragging={dragging} onDrag={handleDrag} />

            <div style={{ flex: "1 1 248px", minWidth: 232 }}>
              <div style={{ fontSize: 13, color: C.muted, marginBottom: 8 }}>
                The columns of M are exactly where the basis vectors land.
              </div>
              <div style={{ display: "flex", alignItems: "stretch", gap: 8, marginBottom: 10 }}>
                <div
                  aria-hidden="true"
                  style={{ width: 8, borderLeft: `2px solid ${C.faint}`, borderTop: `2px solid ${C.faint}`, borderBottom: `2px solid ${C.faint}`, borderRadius: "4px 0 0 4px" }}
                />
                <div style={{ flex: 1, display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
                  <MatrixInput value={m.a} onChange={(v) => setEntry("a", v)} color={C.iHat} label="matrix entry a, top left, x coordinate of i-hat" />
                  <MatrixInput value={m.b} onChange={(v) => setEntry("b", v)} color={C.jHat} label="matrix entry b, top right, x coordinate of j-hat" />
                  <MatrixInput value={m.c} onChange={(v) => setEntry("c", v)} color={C.iHat} label="matrix entry c, bottom left, y coordinate of i-hat" />
                  <MatrixInput value={m.d} onChange={(v) => setEntry("d", v)} color={C.jHat} label="matrix entry d, bottom right, y coordinate of j-hat" />
                </div>
                <div
                  aria-hidden="true"
                  style={{ width: 8, borderRight: `2px solid ${C.faint}`, borderTop: `2px solid ${C.faint}`, borderBottom: `2px solid ${C.faint}`, borderRadius: "0 4px 4px 0" }}
                />
              </div>
              <div style={{ display: "flex", gap: 14, fontSize: 12, color: C.muted, marginBottom: 14, flexWrap: "wrap" }}>
                <span style={{ display: "flex", alignItems: "center", gap: 5 }}>
                  <span style={{ width: 10, height: 10, borderRadius: 2, background: C.iHat }} /> column 1 is i-hat
                </span>
                <span style={{ display: "flex", alignItems: "center", gap: 5 }}>
                  <span style={{ width: 10, height: 10, borderRadius: 2, background: C.jHat }} /> column 2 is j-hat
                </span>
              </div>

              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8 }}>
                <Stat label="determinant" value={fmt(det)} accent />
                <Stat label="trace" value={fmt(tr)} />
                <Stat label="i-hat lands at" value={`(${fmt(m.a)}, ${fmt(m.c)})`} swatch={C.iHat} />
                <Stat label="j-hat lands at" value={`(${fmt(m.b)}, ${fmt(m.d)})`} swatch={C.jHat} />
              </div>

              <div style={{ display: "flex", gap: 8, marginTop: 14, flexWrap: "wrap" }}>
                <button type="button" className="mt-btn" onClick={() => setShowEigen((s) => !s)} style={btnStyle(showEigen)} aria-pressed={showEigen}>
                  {showEigen ? "Eigenvectors on" : "Eigenvectors off"}
                </button>
                <button
                  type="button"
                  className="mt-btn"
                  onClick={runAnimation}
                  disabled={animating || reduce}
                  style={btnStyle(false, animating || reduce)}
                  aria-label="Animate the plane morphing from the identity to the current matrix"
                  title={reduce ? "Reduced motion is on" : undefined}
                >
                  {animating ? "Morphing" : "Play from identity"}
                </button>
                <button type="button" className="mt-btn" onClick={() => applyPreset("identity")} style={btnStyle(false)} aria-label="Reset to the identity matrix">
                  Reset
                </button>
              </div>
            </div>
          </div>
        </Card>

        <Card style={{ marginBottom: 18 }}>
          <div style={{ fontSize: 13, fontWeight: 700, marginBottom: 4 }}>Presets that each teach one thing</div>
          <p style={{ fontSize: 13, color: C.muted, margin: "0 0 14px", lineHeight: 1.55, maxWidth: "62ch" }}>
            Rotation has complex eigenvalues and determinant one. Pure scaling stretches the two axes by different
            factors. Shear slides one axis while the other holds still. Reflection flips orientation, so its
            determinant is negative. The singular preset collapses the whole plane onto a line, so its determinant is
            zero.
          </p>
          <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
            {Object.entries(PRESETS).map(([k, p]) => {
              const active =
                Math.abs(m.a - p.m.a) < 1e-6 &&
                Math.abs(m.b - p.m.b) < 1e-6 &&
                Math.abs(m.c - p.m.c) < 1e-6 &&
                Math.abs(m.d - p.m.d) < 1e-6;
              return (
                <button key={k} type="button" className="mt-btn" onClick={() => applyPreset(k)} style={btnStyle(active)} aria-pressed={active}>
                  {p.label}
                </button>
              );
            })}
          </div>
        </Card>

        <Card style={{ marginBottom: 18 }}>
          <div style={{ fontSize: 13, fontWeight: 700, marginBottom: 12 }}>Eigenvalues and eigenvectors, solved in closed form</div>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))", gap: 8, marginBottom: 14 }}>
            <Stat label="characteristic" value={`λ² − ${fmt(tr)}λ + ${fmt(det)}`} />
            <Stat label="discriminant" value={fmt(tr * tr - 4 * det)} />
            <Stat label="eigenvalues" value={eigenReadout()} accent />
          </div>

          {eig.kind === "complex" ? (
            <p style={{ fontSize: 14, lineHeight: 1.65, color: "#5d4226", margin: 0, maxWidth: "66ch" }}>
              The discriminant is negative, so the eigenvalues are a complex-conjugate pair and there is no real
              direction that stays on its own line. That is the signature of a rotation: every real vector gets turned
              to a new heading, so none of them are merely scaled. The real part of each eigenvalue is how much the
              plane scales per turn and the imaginary part encodes the angle. No real eigenvectors are drawn, because
              honestly none exist.
            </p>
          ) : (
            <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))", gap: 12 }}>
              {eig.vectors.map((ev, idx) => (
                <div key={idx} style={{ background: C.bg, border: `1px solid ${C.border}`, borderRadius: 10, padding: "11px 13px" }}>
                  <div style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 12, fontWeight: 700, color: C.eigen, marginBottom: 6 }}>
                    <span style={{ width: 10, height: 10, borderRadius: 2, background: C.eigen }} />
                    {eig.kind === "repeated" ? "Repeated eigenvalue" : `Eigenvalue ${idx + 1}`}
                  </div>
                  <div style={{ fontFamily: MONO, fontSize: 15, fontWeight: 700, color: C.ink }}>λ = {fmt(ev.lambda)}</div>
                  <div style={{ fontFamily: MONO, fontSize: 13, color: C.muted, marginTop: 4 }}>
                    v = ({ev.v ? `${fmt(ev.v[0])}, ${fmt(ev.v[1])}` : "any direction"})
                  </div>
                </div>
              ))}
            </div>
          )}
          <p style={{ fontSize: 13.5, lineHeight: 1.6, color: C.muted, margin: "14px 0 0", maxWidth: "66ch" }}>
            An eigenvector is a direction the transformation does not turn, it only stretches or squashes along that
            line, and the eigenvalue is the factor. On the plane the dashed green line is the invariant direction and
            the solid green stub is the image of the unit eigenvector, so its length is the eigenvalue. When the
            eigenvalue is negative the stub points the opposite way, because the direction is preserved but flipped
            end to end.
          </p>
        </Card>

        <Card style={{ background: C.accentSoft, borderColor: `${C.accent}33` }}>
          <div style={{ fontSize: 13, fontWeight: 700, marginBottom: 10 }}>Reading the determinant off the picture</div>
          <p style={{ fontSize: 14, lineHeight: 1.65, color: "#5d4226", margin: "0 0 10px", maxWidth: "66ch" }}>
            The shaded parallelogram is the image of the unit square whose sides are i-hat and j-hat. Its area is the
            absolute value of the determinant, {fmt(Math.abs(det))} here, and that is exactly how much the
            transformation multiplies every area in the plane. When the determinant turns negative the fill switches
            colour, which marks the moment the plane gets flipped over and orientation reverses.
          </p>
          <p style={{ fontSize: 14, lineHeight: 1.65, color: "#5d4226", margin: 0, maxWidth: "66ch" }}>
            {Math.abs(det) < 1e-6
              ? "Right now the determinant is zero: the two columns point along the same line, the parallelogram has collapsed to a segment, and the whole plane is squashed onto a one dimensional image. The matrix is singular and cannot be undone, because many input points now share a single output."
              : det < 0
                ? "Right now the determinant is negative, so the transformation reflects the plane: a shape traced clockwise comes out traced counterclockwise. Push the determinant back through zero to restore the original orientation."
                : "Right now the determinant is positive, so orientation is preserved. Drag a tip until the i and j vectors line up and the determinant passes through zero, the instant the plane collapses to a line."}
          </p>
        </Card>

        <p style={{ textAlign: "center", fontSize: 11.5, color: C.faint, marginTop: 22, lineHeight: 1.6 }}>
          Every grid line, the sample shape, and both basis vectors are transformed by the same matrix-vector
          product. Determinant is ad − bc; eigenpairs come from solving λ² − (trace)λ + det = 0 in closed form.
        </p>
      </div>
    </div>
  );
}
