import { useState, useEffect, useMemo, useCallback } from "react";

export const meta = {
  title: "Two's Complement: Signed vs Unsigned",
  category: "Computer Systems",
  description:
    "One row of bits, read two ways. Toggle bits and watch the unsigned and signed values move together, walk the number wheel to see where 255 wraps to 0 and 127 wraps to -128, then negate by flipping bits and adding one and add two signed numbers until they overflow.",
  date: "2026-05-08",
  tags: ["twos-complement", "binary", "signed-integers", "overflow"],
};

const C = {
  bg: "#f7f5f2",
  card: "#ffffff",
  border: "#e4dfd8",
  ink: "#1c1a17",
  muted: "#6f675d",
  faint: "#9b938a",
  accent: "#c0561f",
  accentSoft: "#f6ece5",
  panel: "#faf8f5",
  sign: "#a8453f",
  signSoft: "#fbecea",
  unsigned: "#2e6f8e",
  unsignedSoft: "#eaf1f4",
  good: "#2f7d53",
  goodSoft: "#e9f2ec",
  zero: "#e9e4dc",
};

const EASE = "cubic-bezier(0.23, 1, 0.32, 1)";
const MONO = "'IBM Plex Mono', ui-monospace, SFMono-Regular, Menlo, monospace";
const TAU = Math.PI * 2;

function mask(value, width) {
  if (width >= 32) return value >>> 0;
  return value & ((1 << width) - 1);
}

function toSigned(unsigned, width) {
  const half = 1 << (width - 1);
  return unsigned >= half ? unsigned - (1 << width) : unsigned;
}

function bitsOf(unsigned, width) {
  const out = [];
  for (let i = width - 1; i >= 0; i--) out.push((unsigned >> i) & 1);
  return out;
}

function negate(unsigned, width) {
  return mask(mask(~unsigned, width) + 1, width);
}

function usePrefersReducedMotion() {
  const [reduce, setReduce] = useState(false);
  useEffect(() => {
    if (typeof window === "undefined" || !window.matchMedia) return;
    const mq = window.matchMedia("(prefers-reduced-motion: reduce)");
    const update = () => setReduce(mq.matches);
    update();
    mq.addEventListener("change", update);
    return () => mq.removeEventListener("change", update);
  }, []);
  return reduce;
}

function Card({ children, style }) {
  return (
    <div
      style={{
        background: C.card,
        border: `1px solid ${C.border}`,
        borderRadius: 14,
        padding: 20,
        ...style,
      }}
    >
      {children}
    </div>
  );
}

function Btn({ children, onClick, disabled, primary, tone, ariaLabel, type, title }) {
  const bg = primary ? C.accent : tone === "sign" ? C.signSoft : "transparent";
  const fg = primary ? "#fff" : tone === "sign" ? C.sign : C.ink;
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      type={type || "button"}
      aria-label={ariaLabel}
      title={title}
      style={{
        padding: primary ? "8px 16px" : "7px 13px",
        borderRadius: 8,
        border: primary ? "none" : `1px solid ${tone === "sign" ? C.sign + "44" : C.border}`,
        background: bg,
        color: fg,
        fontSize: 13,
        fontWeight: primary ? 600 : 500,
        fontFamily: "inherit",
        cursor: disabled ? "not-allowed" : "pointer",
        opacity: disabled ? 0.4 : 1,
        transition: `transform 140ms ${EASE}, background 160ms ease`,
        WebkitTapHighlightColor: "transparent",
      }}
      onPointerDown={(e) => {
        if (!disabled) e.currentTarget.style.transform = "scale(0.97)";
      }}
      onPointerUp={(e) => {
        e.currentTarget.style.transform = "scale(1)";
      }}
      onPointerLeave={(e) => {
        e.currentTarget.style.transform = "scale(1)";
      }}
    >
      {children}
    </button>
  );
}

function BitRow({ bits, width, onToggle, reduce }) {
  return (
    <div
      role="group"
      aria-label={`${width}-bit pattern, most significant bit on the left`}
      style={{
        display: "flex",
        gap: width > 4 ? 6 : 9,
        justifyContent: "center",
        flexWrap: "nowrap",
      }}
    >
      {bits.map((b, idx) => {
        const isSign = idx === 0;
        const place = width - 1 - idx;
        const size = width > 4 ? 40 : 54;
        const on = b === 1;
        return (
          <button
            key={idx}
            type="button"
            onClick={() => onToggle(idx)}
            aria-label={`Bit at place 2 to the ${place}${
              isSign ? ", the sign bit" : ""
            }, currently ${b}. Toggle.`}
            aria-pressed={on}
            style={{
              width: size,
              height: size + 6,
              borderRadius: 9,
              border: `1.5px solid ${
                isSign ? C.sign + "88" : on ? C.accent : C.border
              }`,
              background: on ? (isSign ? C.signSoft : C.accentSoft) : C.panel,
              color: on ? (isSign ? C.sign : C.accent) : C.faint,
              fontFamily: MONO,
              fontSize: width > 4 ? 19 : 23,
              fontWeight: 700,
              cursor: "pointer",
              display: "flex",
              flexDirection: "column",
              alignItems: "center",
              justifyContent: "center",
              lineHeight: 1,
              padding: 0,
              transition: reduce
                ? "none"
                : `transform 150ms ${EASE}, background 160ms ${EASE}, border-color 160ms ${EASE}, color 160ms ${EASE}`,
              WebkitTapHighlightColor: "transparent",
            }}
            onPointerDown={(e) => {
              e.currentTarget.style.transform = "scale(0.94)";
            }}
            onPointerUp={(e) => {
              e.currentTarget.style.transform = "scale(1)";
            }}
            onPointerLeave={(e) => {
              e.currentTarget.style.transform = "scale(1)";
            }}
          >
            {b}
            <span style={{ fontSize: 8.5, fontWeight: 600, marginTop: 3, opacity: 0.7 }}>
              {isSign ? "sign" : `2${superscript(place)}`}
            </span>
          </button>
        );
      })}
    </div>
  );
}

function superscript(n) {
  const map = { "0": "⁰", "1": "¹", "2": "²", "3": "³", "4": "⁴", "5": "⁵", "6": "⁶", "7": "⁷" };
  return String(n)
    .split("")
    .map((d) => map[d] || d)
    .join("");
}

function PlaceTable({ bits, width, signed }) {
  const cells = bits.map((b, idx) => {
    const place = width - 1 - idx;
    const isSign = idx === 0;
    const weight = isSign && signed ? -(1 << place) : 1 << place;
    const contribution = b === 1 ? weight : 0;
    return { b, place, isSign, weight, contribution };
  });
  const total = cells.reduce((s, c) => s + c.contribution, 0);
  const accent = signed ? C.sign : C.unsigned;
  const soft = signed ? C.signSoft : C.unsignedSoft;
  return (
    <div
      style={{
        background: soft,
        borderRadius: 11,
        padding: "12px 13px",
        border: `1px solid ${accent}2e`,
      }}
    >
      <div
        style={{
          fontSize: 10.5,
          fontWeight: 700,
          letterSpacing: "0.06em",
          textTransform: "uppercase",
          color: accent,
          marginBottom: 9,
        }}
      >
        {signed ? "Signed (two's complement)" : "Unsigned"}
      </div>
      <div
        style={{
          display: "flex",
          flexWrap: "wrap",
          gap: 5,
          fontFamily: MONO,
          fontSize: 12,
          alignItems: "center",
        }}
      >
        {cells
          .filter((c) => c.b === 1)
          .map((c, i, arr) => (
            <span key={c.place} style={{ color: c.isSign && signed ? C.sign : C.ink }}>
              {c.contribution > 0 ? c.contribution : `(${c.contribution})`}
              {i < arr.length - 1 ? " +" : ""}
            </span>
          ))}
        {cells.every((c) => c.b === 0) && <span style={{ color: C.faint }}>0</span>}
        <span style={{ color: C.faint }}>=</span>
        <span style={{ fontWeight: 700, fontSize: 16, color: accent }}>{total}</span>
      </div>
      <div style={{ fontSize: 11, color: C.muted, marginTop: 8, lineHeight: 1.5 }}>
        {signed
          ? `The top bit carries weight -2${superscript(width - 1)} = ${-(1 << (width - 1))}. Every other place stays positive.`
          : `Every place is a positive power of two, the top one is +2${superscript(width - 1)} = ${1 << (width - 1)}.`}
      </div>
    </div>
  );
}

function NumberWheel({ width, value, reduce }) {
  const count = 1 << width;
  const size = 300;
  const cx = size / 2;
  const cy = size / 2;
  const radius = size / 2 - 34;
  const positions = useMemo(() => {
    const out = [];
    for (let i = 0; i < count; i++) {
      const angle = -Math.PI / 2 + (i / count) * TAU;
      out.push({
        i,
        x: cx + radius * Math.cos(angle),
        y: cy + radius * Math.sin(angle),
        angle,
      });
    }
    return out;
  }, [count, radius, cx, cy]);

  const pointer = positions[value];
  const signed = toSigned(value, width);

  return (
    <svg
      viewBox={`0 0 ${size} ${size}`}
      width="100%"
      style={{ maxWidth: size, display: "block", margin: "0 auto" }}
      role="img"
      aria-label={`Number wheel of all ${count} ${width}-bit patterns. The pointer is at unsigned ${value}, signed ${signed}. Unsigned wraps from ${count - 1} to 0 and signed wraps from ${(1 << (width - 1)) - 1} to ${-(1 << (width - 1))} at the bottom.`}
    >
      <circle cx={cx} cy={cy} r={radius} fill="none" stroke={C.border} strokeWidth={1.5} />
      <path
        d={`M ${cx} ${cy - radius - 9} l -7 -11 l 14 0 z`}
        fill={C.good}
        opacity={0.85}
      />
      <text
        x={cx}
        y={cy - radius - 24}
        textAnchor="middle"
        fontFamily={MONO}
        fontSize={10}
        fill={C.good}
        fontWeight={700}
      >
        0
      </text>
      <path
        d={`M ${cx} ${cy + radius + 9} l -7 11 l 14 0 z`}
        fill={C.sign}
        opacity={0.85}
      />
      <text
        x={cx}
        y={cy + radius + 32}
        textAnchor="middle"
        fontFamily={MONO}
        fontSize={9.5}
        fill={C.sign}
        fontWeight={700}
      >
        {`${count - 1}→ 0  /  ${(1 << (width - 1)) - 1}→${-(1 << (width - 1))}`}
      </text>

      <line
        x1={cx}
        y1={cy}
        x2={pointer.x}
        y2={pointer.y}
        stroke={C.accent}
        strokeWidth={2.4}
        strokeLinecap="round"
        style={{ transition: reduce ? "none" : `all 280ms ${EASE}` }}
      />
      <circle cx={cx} cy={cy} r={4} fill={C.accent} />

      {positions.map((p) => {
        const sv = toSigned(p.i, width);
        const isHere = p.i === value;
        const isWrapUnsigned = p.i === count - 1;
        const isMostNeg = p.i === 1 << (width - 1);
        let dotColor = C.faint;
        if (isWrapUnsigned) dotColor = C.good;
        if (isMostNeg) dotColor = C.sign;
        return (
          <g key={p.i}>
            <circle
              cx={p.x}
              cy={p.y}
              r={isHere ? 7 : 3.4}
              fill={isHere ? C.accent : dotColor}
              opacity={isHere ? 1 : 0.55}
              style={{ transition: reduce ? "none" : `r 180ms ${EASE}, fill 180ms ${EASE}` }}
            />
            {(isHere || count <= 16) && (
              <text
                x={cx + (radius + 17) * Math.cos(p.angle)}
                y={cy + (radius + 17) * Math.sin(p.angle) + 3.5}
                textAnchor="middle"
                fontFamily={MONO}
                fontSize={isHere ? 10.5 : 9}
                fontWeight={isHere ? 700 : 500}
                fill={isHere ? C.accent : C.muted}
              >
                {p.i}
              </text>
            )}
          </g>
        );
      })}

      <text x={cx} y={cy - 16} textAnchor="middle" fontFamily={MONO} fontSize={11} fill={C.unsigned} fontWeight={700}>
        u {value}
      </text>
      <text x={cx} y={cy + 26} textAnchor="middle" fontFamily={MONO} fontSize={11} fill={C.sign} fontWeight={700}>
        s {signed}
      </text>
    </svg>
  );
}

function ValueChip({ label, value, color, soft }) {
  return (
    <div
      style={{
        background: soft,
        borderRadius: 10,
        padding: "11px 14px",
        border: `1px solid ${color}2e`,
        flex: 1,
        minWidth: 120,
      }}
    >
      <div
        style={{
          fontSize: 10.5,
          fontWeight: 700,
          letterSpacing: "0.06em",
          textTransform: "uppercase",
          color,
          marginBottom: 4,
        }}
      >
        {label}
      </div>
      <div
        style={{
          fontFamily: MONO,
          fontSize: 26,
          fontWeight: 700,
          color,
          lineHeight: 1,
          fontVariantNumeric: "tabular-nums",
        }}
      >
        {value}
      </div>
    </div>
  );
}

function binStr(unsigned, width) {
  return bitsOf(unsigned, width).join("");
}

export default function App() {
  const reduce = usePrefersReducedMotion();
  const [width, setWidth] = useState(8);
  const [value, setValue] = useState(0);

  const [opA, setOpA] = useState(40);
  const [opB, setOpB] = useState(70);

  const unsignedMax = (1 << width) - 1;
  const signedMin = -(1 << (width - 1));
  const signedMax = (1 << (width - 1)) - 1;

  const bits = bitsOf(value, width);
  const signed = toSigned(value, width);

  const changeWidth = useCallback(
    (w) => {
      setWidth(w);
      setValue((v) => mask(v, w));
      setOpA((a) => toSigned(mask(a < 0 ? mask(a, w) : a, w), w));
      setOpB((b) => toSigned(mask(b < 0 ? mask(b, w) : b, w), w));
    },
    []
  );

  const toggleBit = useCallback(
    (idx) => {
      const place = width - 1 - idx;
      setValue((v) => mask(v ^ (1 << place), width));
    },
    [width]
  );

  const step = useCallback(
    (delta) => {
      setValue((v) => mask(v + delta, width));
    },
    [width]
  );

  const setSignedEntry = useCallback(
    (raw) => {
      const n = parseInt(raw, 10);
      if (Number.isNaN(n)) return;
      const clamped = Math.max(signedMin, Math.min(signedMax, n));
      setValue(mask(clamped, width));
    },
    [width, signedMin, signedMax]
  );

  const setUnsignedEntry = useCallback(
    (raw) => {
      const n = parseInt(raw, 10);
      if (Number.isNaN(n)) return;
      const clamped = Math.max(0, Math.min(unsignedMax, n));
      setValue(mask(clamped, width));
    },
    [width, unsignedMax]
  );

  const negated = negate(value, width);
  const flipped = mask(~value, width);

  const aMasked = mask(opA, width);
  const bMasked = mask(opB, width);
  const sumMasked = mask(opA + opB, width);
  const sumSignedTrue = toSigned(aMasked, width) + toSigned(bMasked, width);
  const sumSignedWrapped = toSigned(sumMasked, width);
  const signedOverflow = sumSignedTrue < signedMin || sumSignedTrue > signedMax;

  const presets = [
    { label: "0", v: 0 },
    { label: "-1 (all ones)", v: mask(-1, width) },
    { label: `min (${signedMin})`, v: 1 << (width - 1) },
    { label: `max (${signedMax})`, v: signedMax },
  ];

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
        @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600;700&display=swap');
        input[type="number"]:focus-visible,
        input[type="range"]:focus-visible,
        button:focus-visible {
          outline: 2px solid ${C.accent};
          outline-offset: 2px;
        }
        input[type="number"] {
          font-family: ${MONO};
        }
        @media (prefers-reduced-motion: reduce) {
          * { transition-duration: 1ms !important; animation-duration: 1ms !important; }
        }
      `}</style>

      <div style={{ maxWidth: 720, margin: "0 auto" }}>
        <header style={{ marginBottom: 22 }}>
          <div
            style={{
              fontSize: 11,
              letterSpacing: "0.14em",
              textTransform: "uppercase",
              color: C.faint,
              marginBottom: 6,
            }}
          >
            Computer Systems / Number representation
          </div>
          <h1
            style={{
              fontSize: 30,
              fontWeight: 700,
              margin: 0,
              lineHeight: 1.1,
              letterSpacing: "-0.02em",
              textWrap: "balance",
            }}
          >
            Two's Complement: Signed vs Unsigned
          </h1>
          <p
            style={{
              color: C.muted,
              fontSize: 15,
              lineHeight: 1.6,
              margin: "10px 0 0",
              maxWidth: "64ch",
              textWrap: "pretty",
            }}
          >
            A register holds nothing but bits. Whether a pattern means a small positive number or a
            large negative one depends entirely on how you decide to read it. Two's complement is the
            convention that lets the same adder serve both readings, gives zero a single encoding, and
            turns the top bit into a negative place value.
          </p>
        </header>

        <Card style={{ marginBottom: 16 }}>
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              flexWrap: "wrap",
              gap: 12,
              marginBottom: 18,
            }}
          >
            <div style={{ display: "flex", gap: 7 }} role="group" aria-label="Bit width">
              {[4, 8].map((w) => (
                <button
                  key={w}
                  type="button"
                  onClick={() => changeWidth(w)}
                  aria-pressed={width === w}
                  style={{
                    padding: "7px 15px",
                    borderRadius: 8,
                    border: `1px solid ${width === w ? C.accent : C.border}`,
                    background: width === w ? C.accent : "transparent",
                    color: width === w ? "#fff" : C.ink,
                    fontFamily: MONO,
                    fontSize: 13,
                    fontWeight: 600,
                    cursor: "pointer",
                    transition: `background 160ms ease, border-color 160ms ease`,
                  }}
                >
                  {w}-bit
                </button>
              ))}
            </div>
            <div style={{ fontFamily: MONO, fontSize: 13, color: C.muted, letterSpacing: "0.04em" }}>
              0b{binStr(value, width)}
            </div>
          </div>

          <BitRow bits={bits} width={width} onToggle={toggleBit} reduce={reduce} />

          <div style={{ display: "flex", gap: 11, marginTop: 18, flexWrap: "wrap" }}>
            <ValueChip label="Unsigned" value={value} color={C.unsigned} soft={C.unsignedSoft} />
            <ValueChip label="Signed" value={signed} color={C.sign} soft={C.signSoft} />
          </div>

          <div style={{ display: "grid", gridTemplateColumns: "1fr", gap: 11, marginTop: 11 }}>
            <PlaceTable bits={bits} width={width} signed={false} />
            <PlaceTable bits={bits} width={width} signed={true} />
          </div>

          <div
            style={{
              display: "flex",
              gap: 8,
              marginTop: 16,
              flexWrap: "wrap",
              alignItems: "center",
            }}
          >
            <Btn onClick={() => step(-1)} ariaLabel="Decrement the value by one">
              {"− 1"}
            </Btn>
            <Btn onClick={() => step(1)} ariaLabel="Increment the value by one">
              {"+ 1"}
            </Btn>
            <Btn
              onClick={() => setValue(negate(value, width))}
              tone="sign"
              ariaLabel="Negate the value"
              title="Replace the value with its two's complement negation"
            >
              negate (-x)
            </Btn>
            <div style={{ flex: 1 }} />
            <label style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 12.5, color: C.unsigned }}>
              <span style={{ fontWeight: 600 }}>set u</span>
              <input
                type="number"
                value={value}
                min={0}
                max={unsignedMax}
                onChange={(e) => setUnsignedEntry(e.target.value)}
                aria-label="Set the unsigned value"
                style={inputStyle(C.unsigned)}
              />
            </label>
            <label style={{ display: "flex", alignItems: "center", gap: 6, fontSize: 12.5, color: C.sign }}>
              <span style={{ fontWeight: 600 }}>set s</span>
              <input
                type="number"
                value={signed}
                min={signedMin}
                max={signedMax}
                onChange={(e) => setSignedEntry(e.target.value)}
                aria-label="Set the signed value"
                style={inputStyle(C.sign)}
              />
            </label>
          </div>

          <div style={{ display: "flex", gap: 7, marginTop: 13, flexWrap: "wrap" }}>
            {presets.map((p) => (
              <button
                key={p.label}
                type="button"
                onClick={() => setValue(mask(p.v, width))}
                style={{
                  padding: "5px 12px",
                  borderRadius: 999,
                  border: `1px solid ${C.border}`,
                  background: "#fff",
                  color: C.ink,
                  fontSize: 12,
                  fontFamily: "inherit",
                  cursor: "pointer",
                  transition: `transform 140ms ${EASE}`,
                }}
                onPointerDown={(e) => (e.currentTarget.style.transform = "scale(0.96)")}
                onPointerUp={(e) => (e.currentTarget.style.transform = "scale(1)")}
                onPointerLeave={(e) => (e.currentTarget.style.transform = "scale(1)")}
              >
                {p.label}
              </button>
            ))}
          </div>
        </Card>

        <Card style={{ marginBottom: 16 }}>
          <div style={{ fontSize: 14, fontWeight: 700, marginBottom: 4 }}>The number wheel</div>
          <p style={{ fontSize: 12.5, color: C.muted, lineHeight: 1.6, margin: "0 0 14px", maxWidth: "62ch" }}>
            All {1 << width} patterns laid out in a ring, counting up clockwise from 0 at the top. The
            pointer sits at the current pattern. Increment past the bottom and the unsigned reading wraps
            from {unsignedMax} back to 0 while the signed reading wraps from {signedMax} to {signedMin}.
            Both wraps happen at the same physical place: that is arithmetic modulo 2{superscript(width)}.
          </p>
          <NumberWheel width={width} value={value} reduce={reduce} />
          <div style={{ display: "flex", gap: 8, justifyContent: "center", marginTop: 14, flexWrap: "wrap" }}>
            <Btn onClick={() => step(-1)} ariaLabel="Step the wheel back by one">
              {"← step -1"}
            </Btn>
            <Btn onClick={() => step(1)} ariaLabel="Step the wheel forward by one">
              {"step +1 →"}
            </Btn>
          </div>
          <div
            style={{
              display: "flex",
              gap: 16,
              justifyContent: "center",
              marginTop: 14,
              flexWrap: "wrap",
              fontSize: 11.5,
              color: C.muted,
            }}
          >
            <LegendDot color={C.good} label={`unsigned wrap (${unsignedMax}→ 0)`} />
            <LegendDot color={C.sign} label={`signed wrap (most negative ${signedMin})`} />
            <LegendDot color={C.accent} label="current pattern" />
          </div>
        </Card>

        <Card style={{ marginBottom: 16 }}>
          <div style={{ fontSize: 14, fontWeight: 700, marginBottom: 4 }}>Negation: flip the bits, add one</div>
          <p style={{ fontSize: 12.5, color: C.muted, lineHeight: 1.6, margin: "0 0 14px", maxWidth: "62ch" }}>
            To negate a two's complement number you invert every bit, then add one. This works because a
            number plus its bitwise complement is all ones, which is {-1} in signed terms, so adding one
            lands on exactly {-1} times the original. Watch it on the current value.
          </p>
          <div style={{ display: "grid", gap: 9, fontFamily: MONO, fontSize: 13 }}>
            <NegStep label="x" bin={binStr(value, width)} note={`signed ${signed}`} color={C.ink} />
            <NegStep label="~x" bin={binStr(flipped, width)} note="every bit inverted" color={C.unsigned} />
            <NegStep
              label="~x + 1"
              bin={binStr(negated, width)}
              note={`signed ${toSigned(negated, width)}`}
              color={C.sign}
            />
          </div>
          <div
            style={{
              marginTop: 13,
              background: C.goodSoft,
              border: `1px solid ${C.good}33`,
              borderRadius: 10,
              padding: "10px 14px",
              fontSize: 12.5,
              color: C.good,
              lineHeight: 1.55,
            }}
          >
            <b>Check:</b> -({signed}) should be {-signed}. Flip-and-add-one gives{" "}
            {toSigned(negated, width)}.{" "}
            {toSigned(negated, width) === -signed
              ? "They match."
              : signed === signedMin
              ? `The most negative value ${signedMin} is its own negation here: there is no +${-signedMin} to land on, which is the asymmetry of two's complement.`
              : "Mismatch."}
          </div>
        </Card>

        <Card style={{ marginBottom: 16 }}>
          <div style={{ fontSize: 14, fontWeight: 700, marginBottom: 4 }}>Signed addition and overflow</div>
          <p style={{ fontSize: 12.5, color: C.muted, lineHeight: 1.6, margin: "0 0 14px", maxWidth: "62ch" }}>
            The processor adds the bit patterns and keeps the low {width} bits. If the true signed sum
            falls outside [{signedMin}, {signedMax}] the result wraps and the sign flips the wrong way.
            That is signed overflow.
          </p>
          <div style={{ display: "flex", gap: 12, flexWrap: "wrap", alignItems: "flex-end", marginBottom: 14 }}>
            <OperandInput
              label="a (signed)"
              value={toSigned(aMasked, width)}
              min={signedMin}
              max={signedMax}
              onChange={(n) => setOpA(Math.max(signedMin, Math.min(signedMax, n)))}
              bin={binStr(aMasked, width)}
            />
            <div style={{ fontSize: 22, color: C.muted, paddingBottom: 22 }}>+</div>
            <OperandInput
              label="b (signed)"
              value={toSigned(bMasked, width)}
              min={signedMin}
              max={signedMax}
              onChange={(n) => setOpB(Math.max(signedMin, Math.min(signedMax, n)))}
              bin={binStr(bMasked, width)}
            />
          </div>
          <div
            style={{
              background: signedOverflow ? C.signSoft : C.goodSoft,
              border: `1px solid ${signedOverflow ? C.sign : C.good}33`,
              borderRadius: 11,
              padding: "13px 15px",
            }}
          >
            <div style={{ fontFamily: MONO, fontSize: 13, display: "flex", flexWrap: "wrap", gap: 8, alignItems: "baseline" }}>
              <span>{binStr(aMasked, width)}</span>
              <span style={{ color: C.muted }}>+</span>
              <span>{binStr(bMasked, width)}</span>
              <span style={{ color: C.muted }}>=</span>
              <span style={{ fontWeight: 700 }}>{binStr(sumMasked, width)}</span>
            </div>
            <div style={{ marginTop: 9, fontSize: 13, lineHeight: 1.6 }}>
              <span style={{ color: C.muted }}>true signed sum </span>
              <b style={{ color: signedOverflow ? C.sign : C.good }}>{sumSignedTrue}</b>
              <span style={{ color: C.muted }}>, stored as </span>
              <b style={{ color: signedOverflow ? C.sign : C.ink }}>{sumSignedWrapped}</b>
              <span style={{ color: C.muted }}> (unsigned {sumMasked})</span>
            </div>
            <div
              style={{
                marginTop: 9,
                fontSize: 12.5,
                fontWeight: 700,
                color: signedOverflow ? C.sign : C.good,
                lineHeight: 1.5,
              }}
              aria-live="polite"
            >
              {signedOverflow
                ? `Signed overflow. ${sumSignedTrue} does not fit in [${signedMin}, ${signedMax}], so the stored value ${sumSignedWrapped} has the wrong sign.`
                : "No overflow. The true sum fits, so the stored signed value is correct."}
            </div>
          </div>
          <div style={{ display: "flex", gap: 7, marginTop: 12, flexWrap: "wrap" }}>
            <Btn
              onClick={() => {
                setOpA(signedMax);
                setOpB(1);
              }}
              ariaLabel="Load an overflowing example"
            >
              max + 1 (overflows)
            </Btn>
            <Btn
              onClick={() => {
                setOpA(signedMin);
                setOpB(-1);
              }}
              ariaLabel="Load a second overflowing example"
            >
              min + (-1) (overflows)
            </Btn>
            <Btn
              onClick={() => {
                setOpA(width === 4 ? 3 : 40);
                setOpB(width === 4 ? 2 : 70);
              }}
              ariaLabel="Load a non-overflowing example"
            >
              a + b (fits)
            </Btn>
          </div>
        </Card>

        <Card style={{ background: C.accentSoft, border: `1px solid ${C.accent}22` }}>
          <div
            style={{
              fontSize: 11,
              fontWeight: 700,
              letterSpacing: "0.08em",
              textTransform: "uppercase",
              color: C.accent,
              marginBottom: 10,
            }}
          >
            Why two's complement wins
          </div>
          <div style={{ fontSize: 13.5, color: C.ink, lineHeight: 1.7, display: "grid", gap: 10 }}>
            <p style={{ margin: 0 }}>
              <b>One zero, and addition that just works.</b> Sign-magnitude and ones' complement both
              waste a pattern on negative zero. Two's complement has a single zero, and the same binary
              adder gives the right answer for signed and unsigned alike, because it is really addition
              modulo 2{superscript(width)}.
            </p>
            <p style={{ margin: 0 }}>
              <b>The sign bit is a negative place value.</b> Reading signed, the top bit is not a flag
              bolted on the side, it contributes -2{superscript(width - 1)}. Set it alone and you get the
              most negative number; the lower bits then only ever add back toward zero.
            </p>
            <p style={{ margin: 0 }}>
              <b>One more negative than positive.</b> The range is [{signedMin}, {signedMax}], asymmetric
              by one. There is a pattern for {signedMin} but none for {-signedMin}, which is why negating
              the most negative value lands back on itself and is the classic overflow trap.
            </p>
          </div>
        </Card>

        <p style={{ textAlign: "center", fontSize: 11.5, color: C.faint, marginTop: 22, lineHeight: 1.6 }}>
          Every value here comes from real bit math: masking to {width} bits, sign extension from the top
          bit, and addition modulo 2{superscript(width)} = {1 << width}.
        </p>
      </div>
    </div>
  );
}

function inputStyle(color) {
  return {
    width: 64,
    padding: "6px 8px",
    borderRadius: 7,
    border: `1px solid ${color}55`,
    fontSize: 13,
    color,
    background: "#fff",
    fontWeight: 600,
  };
}

function LegendDot({ color, label }) {
  return (
    <span style={{ display: "inline-flex", alignItems: "center", gap: 6 }}>
      <span
        style={{ width: 10, height: 10, borderRadius: 999, background: color, display: "inline-block" }}
      />
      {label}
    </span>
  );
}

function NegStep({ label, bin, note, color }) {
  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        gap: 12,
        background: C.panel,
        border: `1px solid ${C.border}`,
        borderRadius: 9,
        padding: "9px 13px",
      }}
    >
      <span style={{ width: 52, color, fontWeight: 700 }}>{label}</span>
      <span style={{ letterSpacing: "0.12em", color: C.ink }}>{bin}</span>
      <span style={{ flex: 1 }} />
      <span style={{ fontSize: 11.5, color: C.muted, fontFamily: "inherit" }}>{note}</span>
    </div>
  );
}

function OperandInput({ label, value, min, max, onChange, bin }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 6, minWidth: 120 }}>
      <label style={{ fontSize: 11.5, color: C.muted, fontWeight: 600 }}>{label}</label>
      <input
        type="number"
        value={value}
        min={min}
        max={max}
        onChange={(e) => {
          const n = parseInt(e.target.value, 10);
          if (!Number.isNaN(n)) onChange(n);
        }}
        aria-label={label}
        style={{
          width: 96,
          padding: "8px 10px",
          borderRadius: 8,
          border: `1px solid ${C.border}`,
          fontSize: 15,
          fontWeight: 700,
          color: C.ink,
          background: "#fff",
        }}
      />
      <span style={{ fontFamily: MONO, fontSize: 11.5, color: C.faint, letterSpacing: "0.08em" }}>{bin}</span>
    </div>
  );
}
