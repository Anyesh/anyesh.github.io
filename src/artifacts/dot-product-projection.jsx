import { useState, useEffect, useRef, useMemo, useCallback } from "react";

export const meta = {
  title: "Dot Product, Cosine, and Projection",
  category: "Linear Algebra",
  description:
    "Two arrows from the origin, and one number that says how much they agree. Drag them apart and watch the dot product fade to zero where they cross at a right angle, then flip negative as they swing opposite.",
  date: "2026-05-09",
  tags: ["linear-algebra", "dot-product", "cosine-similarity", "projection", "embeddings"],
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
  axis: "#cfc7ba",
  vecA: "#c0561f",
  vecB: "#2f6f9e",
  proj: "#3f7d52",
  arc: "#9a6b1f",
  pos: "#3f7d52",
  neg: "#b23b5e",
};

const EASE = "cubic-bezier(0.23, 1, 0.32, 1)";
const MONO = "'IBM Plex Mono', ui-monospace, SFMono-Regular, Menlo, monospace";

const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));

function dot(a, b) {
  return a[0] * b[0] + a[1] * b[1];
}

function mag(v) {
  return Math.hypot(v[0], v[1]);
}

function angleBetween(a, b) {
  const ma = mag(a);
  const mb = mag(b);
  if (ma < 1e-9 || mb < 1e-9) return 0;
  return Math.acos(clamp(dot(a, b) / (ma * mb), -1, 1));
}

function unit(v) {
  const m = mag(v);
  if (m < 1e-9) return [0, 0];
  return [v[0] / m, v[1] / m];
}

// Projection of a onto b is the component of a along b's direction: the signed
// scalar (a.b)/|b| times the unit vector of b. The vector (a minus projection)
// is orthogonal to b, which is what makes the shadow the closest point on b's line.
function projectOnto(a, b) {
  const mb2 = dot(b, b);
  if (mb2 < 1e-12) return { scalar: 0, vec: [0, 0] };
  const k = dot(a, b) / mb2;
  return { scalar: dot(a, b) / Math.sqrt(mb2), vec: [b[0] * k, b[1] * k] };
}

const PRESETS = {
  aligned: { label: "Aligned", a: [2.6, 1.0], b: [1.7, 0.65] },
  perpendicular: { label: "Perpendicular", a: [2.2, 1.3], b: [-1.0, 1.69] },
  opposite: { label: "Opposite", a: [2.4, 0.9], b: [-1.6, -0.6] },
  acute: { label: "Acute", a: [2.5, 0.5], b: [1.4, 1.7] },
};

// Fixed direction vectors with deliberately different lengths, so ranking by raw
// dot product would disagree with ranking by cosine. The longest vector is not
// the most aligned, which is exactly the lesson embedding search teaches.
const EMBEDDINGS = [
  { id: "river", label: "river", v: [2.7, 1.7] },
  { id: "ocean", label: "ocean", v: [1.0, 0.62] },
  { id: "mountain", label: "mountain", v: [-1.4, 2.0] },
  { id: "desert", label: "desert", v: [2.4, -1.3] },
  { id: "rain", label: "rain", v: [0.55, 0.95] },
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

function fmt(x, places = 2) {
  if (!Number.isFinite(x)) return "n/a";
  const v = Math.abs(x) < 1e-9 ? 0 : x;
  return v.toFixed(places);
}

function Card({ children, style }) {
  return (
    <div style={{ background: C.card, border: `1px solid ${C.border}`, borderRadius: 14, padding: 18, ...style }}>
      {children}
    </div>
  );
}

function Stat({ label, value, tone, swatch }) {
  const color = tone === "a" ? C.vecA : tone === "b" ? C.vecB : tone === "proj" ? C.proj : tone === "accent" ? C.accent : C.ink;
  return (
    <div
      style={{
        background: tone ? "transparent" : C.bg,
        borderRadius: 9,
        padding: "8px 11px",
        border: `1px solid ${tone ? color + "33" : C.border}`,
      }}
    >
      <div
        style={{
          fontSize: 10,
          letterSpacing: "0.05em",
          textTransform: "uppercase",
          color: tone ? color : C.muted,
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
          color: tone ? color : C.ink,
          lineHeight: 1.1,
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
    padding: "7px 13px",
    borderRadius: 9,
    border: `1.5px solid ${active ? C.accent : C.border}`,
    background: active ? C.accent : C.card,
    color: active ? "#fff" : C.muted,
    fontSize: 12.5,
    fontWeight: 600,
    cursor: disabled ? "not-allowed" : "pointer",
    opacity: disabled ? 0.5 : 1,
    fontFamily: "inherit",
    transition: `transform 140ms ${EASE}, background 160ms ${EASE}, border-color 160ms ${EASE}, color 160ms ${EASE}`,
  };
}

function toggleStyle(active) {
  return {
    padding: "6px 12px",
    borderRadius: 8,
    border: `1px solid ${active ? C.accent : C.border}`,
    background: active ? C.accentSoft : "transparent",
    color: active ? C.accent : C.muted,
    fontSize: 12,
    fontWeight: active ? 700 : 500,
    cursor: "pointer",
    fontFamily: "inherit",
    transition: `background 150ms ${EASE}, border-color 150ms ${EASE}, color 150ms ${EASE}`,
  };
}

const VIEW = 360;
const UNIT = 56;
const RANGE = Math.ceil(VIEW / 2 / UNIT) + 1;

function Plane({ a, b, opts, dragging, onDrag }) {
  const ref = useRef(null);
  const ox = VIEW / 2;
  const oy = VIEW / 2;

  const toPx = (x, y) => [ox + x * UNIT, oy - y * UNIT];

  const drawA = opts.normalize ? unit(a) : a;
  const drawB = opts.normalize ? unit(b) : b;
  const proj = projectOnto(drawA, drawB);

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

    ctx.strokeStyle = C.axis;
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    ctx.moveTo(ox, 6);
    ctx.lineTo(ox, VIEW - 6);
    ctx.moveTo(6, oy);
    ctx.lineTo(VIEW - 6, oy);
    ctx.stroke();

    const dval = dot(drawA, drawB);

    if (opts.arc) {
      const angA = Math.atan2(drawA[1], drawA[0]);
      const angB = Math.atan2(drawB[1], drawB[0]);
      let diff = angB - angA;
      while (diff > Math.PI) diff -= 2 * Math.PI;
      while (diff < -Math.PI) diff += 2 * Math.PI;
      const r = 38;
      ctx.beginPath();
      ctx.moveTo(ox, oy);
      ctx.arc(ox, oy, r, -angA, -angA - diff, diff > 0);
      ctx.closePath();
      ctx.fillStyle = dval >= 0 ? "rgba(63, 125, 82, 0.16)" : "rgba(178, 59, 94, 0.16)";
      ctx.fill();
      ctx.strokeStyle = C.arc;
      ctx.lineWidth = 1.3;
      ctx.beginPath();
      ctx.arc(ox, oy, r, -angA, -angA - diff, diff > 0);
      ctx.stroke();
    }

    if (opts.proj && mag(drawB) > 1e-6) {
      const [pxv, pyv] = toPx(proj.vec[0], proj.vec[1]);
      const [axv, ayv] = toPx(drawA[0], drawA[1]);
      ctx.strokeStyle = C.faint;
      ctx.lineWidth = 1.3;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(axv, ayv);
      ctx.lineTo(pxv, pyv);
      ctx.stroke();
      ctx.setLineDash([]);

      ctx.strokeStyle = C.proj;
      ctx.lineWidth = 5;
      ctx.lineCap = "round";
      ctx.beginPath();
      ctx.moveTo(ox, oy);
      ctx.lineTo(pxv, pyv);
      ctx.stroke();
      ctx.lineCap = "butt";

      const ux = drawB[0] / mag(drawB);
      const uy = drawB[1] / mag(drawB);
      const tx = ux * 0.12;
      const ty = uy * 0.12;
      const [c1x, c1y] = toPx(proj.vec[0] - tx + uy * 0.12, proj.vec[1] - ty - ux * 0.12);
      const [c2x, c2y] = toPx(proj.vec[0] - tx, proj.vec[1] - ty);
      const [c3x, c3y] = toPx(proj.vec[0] + uy * 0.12, proj.vec[1] - ux * 0.12);
      if (Math.abs(proj.scalar) > 0.18) {
        ctx.strokeStyle = C.proj + "99";
        ctx.lineWidth = 1.2;
        ctx.beginPath();
        ctx.moveTo(c1x, c1y);
        ctx.lineTo(c2x, c2y);
        ctx.lineTo(c3x, c3y);
        ctx.stroke();
      }
    }

    const drawVec = (v, color, label, width) => {
      const [tx, ty] = toPx(v[0], v[1]);
      ctx.strokeStyle = color;
      ctx.lineWidth = width;
      ctx.beginPath();
      ctx.moveTo(ox, oy);
      ctx.lineTo(tx, ty);
      ctx.stroke();
      const ang = Math.atan2(ty - oy, tx - ox);
      const head = 10;
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
      ctx.font = `700 14px ${MONO}`;
      ctx.fillText(label, tx + 11, ty - 9);
    };

    drawVec(drawB, C.vecB, "b", 3);
    drawVec(drawA, C.vecA, "a", 3.4);
  }, [a, b, opts, drawA, drawB, proj]);

  const pointToWorld = (e) => {
    const rect = ref.current.getBoundingClientRect();
    const scaleX = VIEW / rect.width;
    const scaleY = VIEW / rect.height;
    const px = (e.clientX - rect.left) * scaleX;
    const py = (e.clientY - rect.top) * scaleY;
    return [(px - VIEW / 2) / UNIT, (VIEW / 2 - py) / UNIT];
  };

  const handleDown = (e) => {
    const [mx, my] = pointToWorld(e);
    const da = Math.hypot(mx - drawA[0], my - drawA[1]);
    const db = Math.hypot(mx - drawB[0], my - drawB[1]);
    const threshold = 22 / UNIT;
    let target = null;
    if (da < threshold && da <= db) target = "a";
    else if (db < threshold) target = "b";
    if (target) {
      e.currentTarget.setPointerCapture(e.pointerId);
      onDrag(target);
    }
  };

  const handleMove = (e) => {
    if (!dragging) return;
    onDrag(dragging, pointToWorld(e));
  };

  const handleUp = (e) => {
    if (dragging) {
      try {
        e.currentTarget.releasePointerCapture(e.pointerId);
      } catch {
        // pointer may already be released when a gesture ends off-canvas
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
      aria-label={`Two vectors from the origin. Vector a points to (${fmt(drawA[0])}, ${fmt(drawA[1])}), vector b to (${fmt(drawB[0])}, ${fmt(drawB[1])}). Drag either tip. The green shadow is the projection of a onto b.`}
    />
  );
}

const MINI_SIZE = 200;
const MINI_UNIT = 32;

function MiniPlane({ query, ranked, dragging, onQuery }) {
  const ref = useRef(null);
  const size = MINI_SIZE;
  const u = MINI_UNIT;
  const o = size / 2;
  const toPx = (x, y) => [o + x * u, o - y * u];

  useEffect(() => {
    const cv = ref.current;
    if (!cv) return;
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    cv.width = size * dpr;
    cv.height = size * dpr;
    const ctx = cv.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, size, size);

    ctx.strokeStyle = C.axis;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(o, 4);
    ctx.lineTo(o, size - 4);
    ctx.moveTo(4, o);
    ctx.lineTo(size - 4, o);
    ctx.stroke();

    for (const item of ranked) {
      const [tx, ty] = toPx(item.v[0], item.v[1]);
      const top = item.rank === 0;
      ctx.strokeStyle = top ? C.proj : C.faint;
      ctx.lineWidth = top ? 2.6 : 1.6;
      ctx.beginPath();
      ctx.moveTo(o, o);
      ctx.lineTo(tx, ty);
      ctx.stroke();
      ctx.fillStyle = top ? C.proj : C.muted;
      ctx.beginPath();
      ctx.arc(tx, ty, top ? 4 : 3, 0, Math.PI * 2);
      ctx.fill();
      ctx.font = `600 10px ${MONO}`;
      ctx.fillText(item.label, tx + 5, ty - 3);
    }

    const [qx, qy] = toPx(query[0], query[1]);
    ctx.strokeStyle = C.accent;
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(o, o);
    ctx.lineTo(qx, qy);
    ctx.stroke();
    const ang = Math.atan2(qy - o, qx - o);
    ctx.fillStyle = C.accent;
    ctx.beginPath();
    ctx.moveTo(qx, qy);
    ctx.lineTo(qx - 9 * Math.cos(ang - 0.4), qy - 9 * Math.sin(ang - 0.4));
    ctx.lineTo(qx - 9 * Math.cos(ang + 0.4), qy - 9 * Math.sin(ang + 0.4));
    ctx.closePath();
    ctx.fill();
    ctx.fillStyle = C.card;
    ctx.strokeStyle = C.accent;
    ctx.lineWidth = 2.2;
    ctx.beginPath();
    ctx.arc(qx, qy, 6, 0, Math.PI * 2);
    ctx.fill();
    ctx.stroke();
  }, [query, ranked]);

  const pointToWorld = (e) => {
    const rect = ref.current.getBoundingClientRect();
    const sx = size / rect.width;
    const sy = size / rect.height;
    const px = (e.clientX - rect.left) * sx;
    const py = (e.clientY - rect.top) * sy;
    return [(px - size / 2) / u, (size / 2 - py) / u];
  };

  return (
    <canvas
      ref={ref}
      onPointerDown={(e) => {
        e.currentTarget.setPointerCapture(e.pointerId);
        onQuery("start", pointToWorld(e));
      }}
      onPointerMove={(e) => {
        if (dragging) onQuery("move", pointToWorld(e));
      }}
      onPointerUp={(e) => {
        onQuery("end");
        try {
          e.currentTarget.releasePointerCapture(e.pointerId);
        } catch {
          // pointer may already be released off-canvas
        }
      }}
      onPointerCancel={() => onQuery("end")}
      style={{
        width: size,
        height: size,
        maxWidth: "100%",
        borderRadius: 12,
        background: C.bg,
        border: `1px solid ${C.border}`,
        display: "block",
        touchAction: "none",
        cursor: dragging ? "grabbing" : "crosshair",
      }}
      role="img"
      aria-label={`Query vector and five item vectors from the origin. The query points to (${fmt(query[0])}, ${fmt(query[1])}). Drag it to re-rank the items by cosine similarity. The current best match is ${ranked[0].label}.`}
    />
  );
}

export default function App() {
  const reduce = usePrefersReducedMotion();
  const [a, setA] = useState([...PRESETS.acute.a]);
  const [b, setB] = useState([...PRESETS.acute.b]);
  const [dragging, setDragging] = useState(null);
  const [opts, setOpts] = useState({ proj: true, arc: true, formula: true, normalize: false });

  const [query, setQuery] = useState([2.3, 1.5]);
  const [qDragging, setQDragging] = useState(false);

  const drawA = opts.normalize ? unit(a) : a;
  const drawB = opts.normalize ? unit(b) : b;

  const dval = dot(drawA, drawB);
  const ma = mag(drawA);
  const mb = mag(drawB);
  const theta = angleBetween(drawA, drawB);
  const thetaDeg = (theta * 180) / Math.PI;
  const cosSim = ma > 1e-9 && mb > 1e-9 ? clamp(dval / (ma * mb), -1, 1) : 0;
  const proj = projectOnto(drawA, drawB);
  const formProduct = ma * mb * Math.cos(theta);

  const handleDrag = useCallback(
    (target, point) => {
      if (target === null) {
        setDragging(null);
        return;
      }
      if (!point) {
        setDragging(target);
        return;
      }
      const [x, y] = point;
      // When normalize is on the user grabs a unit-length tip; store the dragged
      // direction at its on-screen length so toggling normalize off keeps the angle.
      const set = target === "a" ? setA : setB;
      set([x, y]);
    },
    []
  );

  const applyPreset = (key) => {
    setA([...PRESETS[key].a]);
    setB([...PRESETS[key].b]);
  };

  const toggle = (key) => setOpts((o) => ({ ...o, [key]: !o[key] }));

  const ranked = useMemo(() => {
    const qu = unit(query);
    return EMBEDDINGS.map((e) => {
      const ev = e.v;
      const cs = mag(qu) > 1e-9 && mag(ev) > 1e-9 ? clamp(dot(qu, unit(ev)), -1, 1) : 0;
      return { ...e, cos: cs, raw: dot(query, ev) };
    })
      .sort((x, y) => y.cos - x.cos)
      .map((e, i) => ({ ...e, rank: i }));
  }, [query]);

  const onQuery = useCallback((phase, point) => {
    if (phase === "end") {
      setQDragging(false);
      return;
    }
    if (phase === "start") {
      setQDragging(true);
      if (point) setQuery(point);
      return;
    }
    if (point) setQuery(point);
  }, []);

  const dotSign = dval > 0.02 ? "positive" : dval < -0.02 ? "negative" : "zero";

  const css = `
    .dp-root :focus-visible { outline: 2px solid ${C.accent}; outline-offset: 2px; border-radius: 9px; }
    .dp-btn:active { transform: scale(0.97); }
    @media (prefers-reduced-motion: reduce) {
      .dp-root * { transition-duration: 0.001ms !important; animation-duration: 0.001ms !important; }
    }
  `;

  return (
    <div
      className="dp-root"
      style={{
        fontFamily: "'Iowan Old Style', 'Palatino Linotype', Palatino, Georgia, serif",
        background: C.bg,
        minHeight: "100vh",
        padding: "30px 14px 56px",
        color: C.ink,
      }}
    >
      <style>{css}</style>
      <div style={{ maxWidth: 720, margin: "0 auto" }}>
        <header style={{ marginBottom: 22 }}>
          <div style={{ fontSize: 11, letterSpacing: "0.14em", textTransform: "uppercase", color: C.faint, marginBottom: 6 }}>
            Linear algebra / Two vectors and one number
          </div>
          <h1 style={{ fontSize: 30, fontWeight: 700, margin: 0, lineHeight: 1.1, letterSpacing: "-0.02em", textWrap: "balance" }}>
            Dot Product, Cosine, and Projection
          </h1>
          <p style={{ color: C.ink, fontSize: 15, lineHeight: 1.6, margin: "12px 0 0", maxWidth: "66ch" }}>
            The dot product is usually taught as a formula: multiply matching coordinates and add. That hides what
            it measures. Drag the two arrows below and the same number reads two ways at once: the coordinate sum,
            and the length of a times the length of b times the cosine of the angle between them. It is large when
            the arrows point the same way, zero when they meet at a right angle, and negative when they oppose.
          </p>
        </header>

        <Card style={{ marginBottom: 18 }}>
          <div style={{ display: "flex", gap: 18, flexWrap: "wrap", alignItems: "flex-start", justifyContent: "center" }}>
            <Plane a={a} b={b} opts={opts} dragging={dragging} onDrag={handleDrag} />

            <div style={{ flex: "1 1 250px", minWidth: 234 }}>
              <div style={{ fontSize: 13, color: C.muted, marginBottom: 10, lineHeight: 1.5 }}>
                Drag the tip of <span style={{ color: C.vecA, fontWeight: 700 }}>a</span> or{" "}
                <span style={{ color: C.vecB, fontWeight: 700 }}>b</span>. The shadow of a falling onto b is the
                projection.
              </div>

              <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 8, marginBottom: 12 }}>
                <Stat label="vector a" value={`(${fmt(drawA[0])}, ${fmt(drawA[1])})`} tone="a" />
                <Stat label="vector b" value={`(${fmt(drawB[0])}, ${fmt(drawB[1])})`} tone="b" />
                <Stat label="length |a|" value={fmt(ma)} tone="a" />
                <Stat label="length |b|" value={fmt(mb)} tone="b" />
                <Stat label="angle theta" value={`${fmt(thetaDeg, 1)} deg`} swatch={C.arc} />
                <Stat label="dot a . b" value={fmt(dval)} tone="accent" />
              </div>

              {opts.formula && (
                <div
                  style={{
                    background: C.accentSoft,
                    border: `1px solid ${C.accent}33`,
                    borderRadius: 10,
                    padding: "10px 12px",
                    marginBottom: 12,
                    fontFamily: MONO,
                    fontSize: 12.5,
                    color: "#5d4226",
                    lineHeight: 1.65,
                  }}
                >
                  <div>
                    a . b = ({fmt(drawA[0])})({fmt(drawB[0])}) + ({fmt(drawA[1])})({fmt(drawB[1])}) ={" "}
                    <strong style={{ color: C.accent }}>{fmt(dval)}</strong>
                  </div>
                  <div>
                    |a||b|cos(theta) = ({fmt(ma)})({fmt(mb)})({fmt(Math.cos(theta))}) ={" "}
                    <strong style={{ color: C.accent }}>{fmt(formProduct)}</strong>
                  </div>
                </div>
              )}

              <div style={{ display: "flex", gap: 7, flexWrap: "wrap", marginBottom: 10 }}>
                <button type="button" className="dp-btn" onClick={() => toggle("proj")} style={toggleStyle(opts.proj)} aria-pressed={opts.proj}>
                  Projection
                </button>
                <button type="button" className="dp-btn" onClick={() => toggle("arc")} style={toggleStyle(opts.arc)} aria-pressed={opts.arc}>
                  Angle arc
                </button>
                <button type="button" className="dp-btn" onClick={() => toggle("formula")} style={toggleStyle(opts.formula)} aria-pressed={opts.formula}>
                  |a||b|cos form
                </button>
                <button type="button" className="dp-btn" onClick={() => toggle("normalize")} style={toggleStyle(opts.normalize)} aria-pressed={opts.normalize}>
                  Normalize
                </button>
              </div>

              <div style={{ display: "flex", gap: 7, flexWrap: "wrap" }}>
                {Object.entries(PRESETS).map(([k, p]) => (
                  <button key={k} type="button" className="dp-btn" onClick={() => applyPreset(k)} style={btnStyle(false)}>
                    {p.label}
                  </button>
                ))}
              </div>
            </div>
          </div>

          <div style={{ display: "flex", gap: 16, justifyContent: "center", flexWrap: "wrap", marginTop: 16, fontSize: 12 }}>
            <span style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
              <span style={{ width: 16, height: 3, background: C.vecA, borderRadius: 2 }} />
              <span style={{ color: C.ink, fontWeight: 600 }}>vector a</span>
            </span>
            <span style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
              <span style={{ width: 16, height: 3, background: C.vecB, borderRadius: 2 }} />
              <span style={{ color: C.ink, fontWeight: 600 }}>vector b</span>
            </span>
            <span style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
              <span style={{ width: 16, height: 4, background: C.proj, borderRadius: 2 }} />
              <span style={{ color: C.ink, fontWeight: 600 }}>projection of a onto b</span>
            </span>
          </div>
        </Card>

        <Card style={{ marginBottom: 18 }}>
          <div style={{ fontSize: 13, fontWeight: 700, marginBottom: 4 }}>The projection: a casting its shadow on b</div>
          <p style={{ fontSize: 13.5, color: C.muted, lineHeight: 1.55, margin: "0 0 12px", maxWidth: "64ch" }}>
            Shine a light straight down onto the line through b. The shadow of a is its projection, the closest point
            on that line. Its signed length is the dot product divided by the length of b, so the dot product is just
            this shadow length scaled back up by how long b is. When a and b sit at a right angle the shadow shrinks
            to nothing, which is the exact moment the dot product is zero.
          </p>
          <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))", gap: 8 }}>
            <Stat label="projection length (a . b)/|b|" value={fmt(proj.scalar)} tone="proj" />
            <Stat label="projection vector" value={`(${fmt(proj.vec[0])}, ${fmt(proj.vec[1])})`} tone="proj" />
            <Stat label="dot product" value={`${fmt(proj.scalar)} x ${fmt(mb)} = ${fmt(dval)}`} tone="accent" />
          </div>
          <p style={{ fontSize: 12.5, color: C.muted, margin: "12px 0 0", lineHeight: 1.55 }}>
            Right now the dot product is{" "}
            <strong style={{ color: dotSign === "positive" ? C.pos : dotSign === "negative" ? C.neg : C.muted }}>{dotSign}</strong>
            {dotSign === "positive"
              ? ": the angle is under 90 degrees, so a leans the same way as b and its shadow points along b."
              : dotSign === "negative"
                ? ": the angle is over 90 degrees, so a leans against b and its shadow points backward, opposite to b."
                : ": a and b are perpendicular, the shadow has collapsed onto the origin, and neither vector carries any of the other's direction."}
          </p>
        </Card>

        <Card style={{ marginBottom: 18 }}>
          <div style={{ fontSize: 13, fontWeight: 700, marginBottom: 4 }}>Cosine similarity: the angle alone, length thrown away</div>
          <p style={{ fontSize: 13.5, color: C.muted, lineHeight: 1.55, margin: "0 0 12px", maxWidth: "64ch" }}>
            Divide the dot product by both lengths and the magnitudes cancel, leaving the cosine of the angle on its
            own. That is cosine similarity, a number from minus one to one: one when the arrows point the same way,
            zero at a right angle, minus one when they point opposite. It is exactly the dot product of the two unit
            vectors, so it sees direction and ignores how long the arrows are.
          </p>

          <div style={{ display: "flex", alignItems: "center", gap: 12, flexWrap: "wrap", marginBottom: 10 }}>
            <span style={{ fontSize: 12, color: C.muted, minWidth: 130 }}>cosine similarity</span>
            <div style={{ flex: 1, minWidth: 160, position: "relative", height: 10, borderRadius: 6, background: C.grid, overflow: "hidden" }} aria-hidden="true">
              <div
                style={{
                  position: "absolute",
                  left: "50%",
                  top: 0,
                  bottom: 0,
                  width: `${Math.abs(cosSim) * 50}%`,
                  transform: cosSim >= 0 ? "translateX(0)" : "translateX(-100%)",
                  background: cosSim >= 0 ? C.pos : C.neg,
                  transition: reduce ? "none" : `width 90ms ${EASE}`,
                }}
              />
              <div style={{ position: "absolute", left: "50%", top: -2, bottom: -2, width: 1, background: C.faint }} />
            </div>
            <span style={{ fontFamily: MONO, fontSize: 15, fontWeight: 700, color: cosSim >= 0 ? C.pos : C.neg, minWidth: 56, textAlign: "right" }}>
              {fmt(cosSim)}
            </span>
          </div>

          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: 10,
              padding: "10px 12px",
              borderRadius: 10,
              background: opts.normalize ? C.accentSoft : C.bg,
              border: `1px solid ${opts.normalize ? C.accent + "33" : C.border}`,
            }}
          >
            <button type="button" className="dp-btn" onClick={() => toggle("normalize")} style={toggleStyle(opts.normalize)} aria-pressed={opts.normalize}>
              {opts.normalize ? "Normalized" : "Normalize both to unit length"}
            </button>
            <span style={{ fontSize: 12.5, color: C.muted, lineHeight: 1.5 }}>
              Rescaling both arrows to length one changes |a| and |b| and the raw dot product, yet cosine similarity
              holds at <strong style={{ color: C.accent }}>{fmt(cosSim)}</strong>. Direction is all it ever measured.
            </span>
          </div>
        </Card>

        <Card style={{ marginBottom: 18 }}>
          <div style={{ fontSize: 13, fontWeight: 700, marginBottom: 4 }}>Why embeddings rank by cosine</div>
          <p style={{ fontSize: 13.5, color: C.muted, lineHeight: 1.55, margin: "0 0 14px", maxWidth: "64ch" }}>
            A search model turns each item into a vector. To find the closest match it compares the query's direction
            to every item's direction, not their lengths, because length often tracks something incidental like how
            often a word appears. Drag the query arrow and watch the ranking reshuffle by cosine. The top match is the
            one pointing most nearly the same way, even when a longer vector has a larger raw dot product.
          </p>

          <div style={{ display: "flex", gap: 18, flexWrap: "wrap", alignItems: "flex-start" }}>
            <MiniPlane query={query} ranked={ranked} dragging={qDragging} onQuery={onQuery} />

            <div style={{ flex: "1 1 240px", minWidth: 220 }}>
              <div style={{ fontSize: 11, letterSpacing: "0.05em", textTransform: "uppercase", color: C.muted, fontWeight: 700, marginBottom: 6 }}>
                ranked by cosine to query
              </div>
              <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
                {ranked.map((item) => (
                  <div
                    key={item.id}
                    style={{
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "space-between",
                      gap: 10,
                      padding: "7px 11px",
                      borderRadius: 9,
                      background: item.rank === 0 ? "rgba(63, 125, 82, 0.1)" : C.bg,
                      border: `1px solid ${item.rank === 0 ? C.proj + "55" : C.border}`,
                    }}
                  >
                    <span style={{ display: "flex", alignItems: "center", gap: 8 }}>
                      <span style={{ fontFamily: MONO, fontSize: 11, color: C.faint, width: 14 }}>{item.rank + 1}</span>
                      <span style={{ fontSize: 13.5, fontWeight: item.rank === 0 ? 700 : 500, color: item.rank === 0 ? C.proj : C.ink }}>
                        {item.label}
                      </span>
                    </span>
                    <span style={{ display: "flex", alignItems: "center", gap: 10 }}>
                      <span style={{ width: 54, height: 6, borderRadius: 4, background: C.grid, overflow: "hidden", position: "relative" }} aria-hidden="true">
                        <span
                          style={{
                            position: "absolute",
                            inset: 0,
                            width: `${clamp(item.cos, 0, 1) * 100}%`,
                            background: item.rank === 0 ? C.proj : C.faint,
                          }}
                        />
                      </span>
                      <span style={{ fontFamily: MONO, fontSize: 12, fontWeight: 700, color: item.rank === 0 ? C.proj : C.muted, minWidth: 40, textAlign: "right" }}>
                        {fmt(item.cos)}
                      </span>
                    </span>
                  </div>
                ))}
              </div>
              <p style={{ fontSize: 11.5, color: C.faint, margin: "10px 0 0", lineHeight: 1.5 }}>
                The bar is cosine similarity. Notice the longest item arrow is not always on top: raw dot product
                rewards length, cosine does not.
              </p>
            </div>
          </div>
        </Card>

        <Card style={{ background: C.accentSoft, borderColor: `${C.accent}33` }}>
          <div style={{ fontSize: 13, fontWeight: 700, marginBottom: 10, color: "#5d4226" }}>One number, three readings</div>
          <p style={{ fontSize: 14, lineHeight: 1.65, color: "#5d4226", margin: "0 0 10px", maxWidth: "66ch" }}>
            The dot product is alignment weighted by size: take how much a points along b (its shadow), then scale by
            how long b is. That is why a longer b or a longer a both inflate it, and why two perpendicular vectors
            give zero no matter how long they are.
          </p>
          <p style={{ fontSize: 14, lineHeight: 1.65, color: "#5d4226", margin: 0, maxWidth: "66ch" }}>
            Cosine similarity is the same idea with the lengths divided out, the angle by itself. That makes it the
            natural way to ask whether two embeddings mean similar things rather than which one happens to be bigger,
            which is why search and recommendation systems reach for it first.
          </p>
        </Card>

        <p style={{ textAlign: "center", fontSize: 11.5, color: C.faint, marginTop: 22, lineHeight: 1.6 }}>
          Every quantity here is computed from the live coordinates: the dot product as both the coordinate sum and
          |a||b|cos(theta), the projection as (a . b)/|b| along b, and each ranking as the cosine between unit vectors.
        </p>
      </div>
    </div>
  );
}
