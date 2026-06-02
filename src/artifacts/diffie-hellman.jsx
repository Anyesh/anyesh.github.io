import { useState, useMemo, useEffect, useRef, useCallback } from "react";

export const meta = {
  title: "Diffie-Hellman Key Exchange",
  category: "Cryptography",
  description:
    "Two strangers end up holding the same secret number, yet that number is never sent across the wire. Watch them build it in the open while an eavesdropper reads every message and still cannot follow.",
  date: "2026-02-24",
  tags: ["diffie-hellman", "cryptography", "key-exchange", "modular-arithmetic", "discrete-log"],
};

const C = {
  bg: "#f7f5f2",
  card: "#ffffff",
  border: "#e4dfd8",
  ink: "#1c1a17",
  muted: "#6f675d",
  accent: "#c0561f",
  accentSoft: "#f6ece5",
  faint: "#efeae3",
  alice: "#2e6f8e",
  aliceSoft: "#eaf2f5",
  bob: "#5a7d3c",
  bobSoft: "#eef3e8",
  eve: "#a8453f",
  eveSoft: "#fbecea",
  good: "#2f7d53",
  goodSoft: "#e9f2ec",
};

const EASE = "cubic-bezier(0.23, 1, 0.32, 1)";

function modpow(base, exp, mod) {
  if (mod === 1n) return 0n;
  let result = 1n;
  let b = base % mod;
  let e = exp;
  while (e > 0n) {
    if (e & 1n) result = (result * b) % mod;
    b = (b * b) % mod;
    e >>= 1n;
  }
  return result;
}

function isPrime(n) {
  if (n < 2n) return false;
  if (n % 2n === 0n) return n === 2n;
  for (let i = 3n; i * i <= n; i += 2n) {
    if (n % i === 0n) return false;
  }
  return true;
}

function primeFactors(n) {
  const factors = [];
  let m = n;
  for (let d = 2n; d * d <= m; d++) {
    if (m % d === 0n) {
      factors.push(d);
      while (m % d === 0n) m /= d;
    }
  }
  if (m > 1n) factors.push(m);
  return factors;
}

// A generator (primitive root) of the field has order p-1, so g^((p-1)/q) != 1
// for every prime q dividing p-1. Checking those exponents is enough.
function isGenerator(g, p) {
  if (g < 2n || g >= p) return false;
  const order = p - 1n;
  for (const q of primeFactors(order)) {
    if (modpow(g, order / q, p) === 1n) return false;
  }
  return true;
}

function smallestGenerator(p) {
  for (let g = 2n; g < p; g++) {
    if (isGenerator(g, p)) return g;
  }
  return 2n;
}

const PRIMES = [23n, 47n, 97n, 227n, 467n, 1019n, 3011n, 7919n, 15559n, 49991n, 104729n, 611953n];

function sanitizedPrimeList() {
  return PRIMES.filter(isPrime);
}

function randomBigBelow(n) {
  const upper = Number(n);
  return BigInt(2 + Math.floor(Math.random() * Math.max(1, upper - 3)));
}

// Eve only knows p, g, A; recovering the exponent is the discrete logarithm.
// For small p we walk g, g^2, g^3, ... until we hit the target, counting steps.
function bruteForceDiscreteLog(g, target, p, cap) {
  let cur = 1n;
  const limit = Number(p - 1n);
  const ceiling = Math.min(limit, cap);
  for (let x = 0; x <= ceiling; x++) {
    if (cur === target) return { x, attempts: x + 1, exhausted: false };
    cur = (cur * g) % p;
  }
  return { x: null, attempts: ceiling + 1, exhausted: ceiling < limit };
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

function Btn({ children, onClick, disabled, primary, ariaLabel, type }) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      type={type || "button"}
      aria-label={ariaLabel}
      style={{
        padding: primary ? "8px 18px" : "7px 14px",
        borderRadius: 8,
        border: primary ? "none" : `1px solid ${C.border}`,
        background: primary ? C.accent : "transparent",
        color: primary ? "#fff" : C.ink,
        fontSize: 13,
        fontWeight: primary ? 600 : 500,
        fontFamily: "inherit",
        cursor: disabled ? "not-allowed" : "pointer",
        opacity: disabled ? 0.4 : 1,
        transition: `transform 140ms ${EASE}, background 160ms ease`,
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

const PAINT = {
  common: "#d6c48a",
  alice: "#3f8fb0",
  bob: "#7faa52",
};

function mixColors(hexes) {
  let r = 0;
  let g = 0;
  let b = 0;
  for (const h of hexes) {
    r += parseInt(h.slice(1, 3), 16);
    g += parseInt(h.slice(3, 5), 16);
    b += parseInt(h.slice(5, 7), 16);
  }
  const n = hexes.length;
  const to = (v) => Math.round(v / n).toString(16).padStart(2, "0");
  return `#${to(r)}${to(g)}${to(b)}`;
}

function CheckMark() {
  return (
    <svg width="9" height="9" viewBox="0 0 12 12" fill="none" aria-hidden="true">
      <path
        d="M2.5 6.4 L5 8.6 L9.5 3.4"
        stroke="#fff"
        strokeWidth="1.9"
        strokeLinecap="round"
        strokeLinejoin="round"
      />
    </svg>
  );
}

function PaintBlob({ color, label, sub, reduce, pulse }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 6, minWidth: 64 }}>
      <span
        style={{
          width: 40,
          height: 40,
          borderRadius: "50%",
          background: color,
          border: `1px solid rgba(0,0,0,0.12)`,
          boxShadow: pulse && !reduce ? `0 0 0 4px ${C.accent}26` : "inset 0 2px 4px rgba(0,0,0,0.12)",
          transition: reduce ? "none" : `background 320ms ${EASE}, box-shadow 320ms ${EASE}`,
        }}
        aria-hidden="true"
      />
      <div style={{ fontSize: 10.5, fontWeight: 700, color: C.ink, textAlign: "center", lineHeight: 1.2 }}>
        {label}
      </div>
      {sub && <div style={{ fontSize: 9.5, color: C.muted, textAlign: "center" }}>{sub}</div>}
    </div>
  );
}

function StepRow({ active, done, label, math, color, soft }) {
  return (
    <div
      style={{
        display: "flex",
        alignItems: "baseline",
        gap: 10,
        padding: "9px 12px",
        borderRadius: 9,
        background: active ? soft : "transparent",
        border: `1px solid ${active ? color + "44" : "transparent"}`,
        opacity: done || active ? 1 : 0.42,
        transition: `opacity 200ms ease, background 200ms ${EASE}`,
      }}
    >
      <span
        style={{
          flex: "0 0 auto",
          width: 16,
          height: 16,
          marginTop: 2,
          borderRadius: "50%",
          border: `2px solid ${color}`,
          background: done ? color : "transparent",
          display: "inline-flex",
          alignItems: "center",
          justifyContent: "center",
        }}
        aria-hidden="true"
      >
        {done && <CheckMark />}
      </span>
      <div style={{ minWidth: 0 }}>
        <div style={{ fontSize: 12.5, color: C.ink, lineHeight: 1.45 }}>{label}</div>
        {math && (
          <div
            style={{
              fontFamily: "'SF Mono', 'Menlo', 'Consolas', monospace",
              fontSize: 12,
              color,
              fontWeight: 600,
              marginTop: 3,
              wordBreak: "break-word",
            }}
          >
            {math}
          </div>
        )}
      </div>
    </div>
  );
}

function PartyCard({ name, color, soft, secretLabel, secretVal, publicLabel, publicVal, sharedVal, step }) {
  return (
    <div
      style={{
        background: soft,
        border: `1px solid ${color}33`,
        borderRadius: 12,
        padding: 14,
        display: "flex",
        flexDirection: "column",
        gap: 8,
      }}
    >
      <div style={{ fontSize: 13, fontWeight: 700, color }}>{name}</div>
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11.5 }}>
        <span style={{ color: C.muted }}>{secretLabel} (private)</span>
        <span style={{ fontFamily: "monospace", fontWeight: 700, color: C.ink }}>
          {step >= 1 ? secretVal : "?"}
        </span>
      </div>
      <div style={{ display: "flex", justifyContent: "space-between", fontSize: 11.5 }}>
        <span style={{ color: C.muted }}>{publicLabel} (public)</span>
        <span style={{ fontFamily: "monospace", fontWeight: 700, color }}>
          {step >= 2 ? publicVal : "—"}
        </span>
      </div>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          fontSize: 11.5,
          paddingTop: 7,
          borderTop: `1px solid ${color}22`,
        }}
      >
        <span style={{ color: C.muted }}>shared secret</span>
        <span style={{ fontFamily: "monospace", fontWeight: 700, color: step >= 4 ? C.good : C.muted }}>
          {step >= 4 ? sharedVal : "?"}
        </span>
      </div>
    </div>
  );
}

const TOTAL_STEPS = 4;
const BRUTE_CAP = 4_000_000;

export default function App() {
  const reduce = usePrefersReducedMotion();
  const primeList = useMemo(sanitizedPrimeList, []);

  const [p, setP] = useState(23n);
  const [g, setG] = useState(5n);
  const [a, setA] = useState(6n);
  const [b, setB] = useState(15n);
  const [step, setStep] = useState(0);
  const [eve, setEve] = useState(null);
  const eveTimer = useRef(null);

  const A = useMemo(() => modpow(g, a, p), [g, a, p]);
  const B = useMemo(() => modpow(g, b, p), [g, b, p]);
  const sharedAlice = useMemo(() => modpow(B, a, p), [B, a, p]);
  const sharedBob = useMemo(() => modpow(A, b, p), [A, b, p]);
  const sharedDirect = useMemo(() => modpow(g, a * b, p), [g, a, b, p]);
  const gIsGen = useMemo(() => isGenerator(g, p), [g, p]);

  const clearEveTimer = useCallback(() => {
    if (eveTimer.current) {
      clearTimeout(eveTimer.current);
      eveTimer.current = null;
    }
  }, []);

  useEffect(() => clearEveTimer, [clearEveTimer]);

  const resetEve = useCallback(() => {
    clearEveTimer();
    setEve(null);
  }, [clearEveTimer]);

  function pickPrime(nextP) {
    resetEve();
    const ng = smallestGenerator(nextP);
    setP(nextP);
    setG(ng);
    setA(randomBigBelow(nextP));
    setB(randomBigBelow(nextP));
    setStep(0);
  }

  function randomizePrime() {
    const next = primeList[Math.floor(Math.random() * primeList.length)];
    pickPrime(next);
  }

  function randomizeSecrets() {
    resetEve();
    setA(randomBigBelow(p));
    setB(randomBigBelow(p));
    setStep(Math.min(step, 1));
  }

  function nextGenerator() {
    resetEve();
    let cand = g + 1n;
    while (cand < p && !isGenerator(cand, p)) cand += 1n;
    if (cand >= p) cand = smallestGenerator(p);
    setG(cand);
    setStep(Math.min(step, 1));
  }

  function advance() {
    resetEve();
    setStep((s) => Math.min(TOTAL_STEPS, s + 1));
  }

  function restart() {
    resetEve();
    setStep(0);
  }

  function runEve(targetName) {
    clearEveTimer();
    const target = targetName === "alice" ? A : B;
    const secret = targetName === "alice" ? a : b;
    const res = bruteForceDiscreteLog(g, target, p, BRUTE_CAP);
    setEve({
      target: targetName,
      running: true,
      attempts: res.attempts,
      found: res.x,
      secret,
      exhausted: res.exhausted,
    });
    eveTimer.current = setTimeout(
      () => {
        setEve((prev) => (prev ? { ...prev, running: false } : prev));
      },
      reduce ? 0 : 650
    );
  }

  const matches = sharedAlice === sharedBob && sharedBob === sharedDirect;
  const searchSpace = p - 2n;

  const commonPaint = PAINT.common;
  const alicePub = mixColors([PAINT.common, PAINT.alice]);
  const bobPub = mixColors([PAINT.common, PAINT.bob]);
  const aliceFinal = mixColors([PAINT.common, PAINT.alice, PAINT.bob]);

  const stepCopy = [
    "Agree on the public setup: a prime p and a generator g. Anyone may know these.",
    "Each side rolls a private secret and tells no one.",
    "Each side sends one public number across the wire. Eve sees both.",
    "Each side raises the number it received to its own private power.",
    "Both arrive at the same shared secret, and it was never transmitted.",
  ];

  return (
    <div
      style={{
        fontFamily: "'Palatino Linotype', 'Book Antiqua', Palatino, serif",
        background: C.bg,
        minHeight: "100vh",
        padding: "24px 14px",
        color: C.ink,
      }}
    >
      <style>{`
        input[type="range"]:focus-visible { outline: 2px solid ${C.accent}; outline-offset: 3px; border-radius: 4px; }
        button:focus-visible { outline: 2px solid ${C.accent}; outline-offset: 2px; }
        select:focus-visible { outline: 2px solid ${C.accent}; outline-offset: 2px; }
        @media (prefers-reduced-motion: reduce) {
          * { transition: none !important; animation: none !important; }
        }
      `}</style>

      <div style={{ maxWidth: 720, margin: "0 auto" }}>
        <div style={{ marginBottom: 22 }}>
          <div
            style={{
              fontSize: 10,
              letterSpacing: "0.15em",
              textTransform: "uppercase",
              color: C.muted,
              marginBottom: 5,
            }}
          >
            Cryptography · Key Exchange
          </div>
          <h1 style={{ fontSize: 25, fontWeight: 700, margin: 0 }}>Diffie-Hellman Key Exchange</h1>
          <p
            style={{
              color: C.ink,
              fontSize: 13.5,
              margin: "8px 0 0",
              lineHeight: 1.65,
              maxWidth: "64ch",
            }}
          >
            Alice and Bob have never met and share no password, yet they need one secret number that
            only the two of them know. Everything they say travels over a wire Eve is listening to.
            The trick is that each keeps a private number, folds it into a public number they are
            happy to shout, and after one swap both can compute the same secret. Eve hears every
            public number and still cannot reconstruct it. Step through it and watch where the secret
            comes from.
          </p>
        </div>

        <Card style={{ marginBottom: 16 }}>
          <div
            style={{
              display: "flex",
              gap: 12,
              flexWrap: "wrap",
              alignItems: "flex-end",
              marginBottom: 16,
            }}
          >
            <div>
              <label
                htmlFor="prime-select"
                style={{ display: "block", fontSize: 11, color: C.muted, marginBottom: 5 }}
              >
                Prime field p
              </label>
              <select
                id="prime-select"
                value={String(p)}
                onChange={(e) => pickPrime(BigInt(e.target.value))}
                style={{
                  padding: "7px 10px",
                  borderRadius: 8,
                  border: `1px solid ${C.border}`,
                  background: "#fff",
                  color: C.ink,
                  fontSize: 13,
                  fontFamily: "inherit",
                }}
              >
                {primeList.map((pr) => (
                  <option key={String(pr)} value={String(pr)}>
                    p = {String(pr)}
                  </option>
                ))}
              </select>
            </div>
            <div>
              <div style={{ fontSize: 11, color: C.muted, marginBottom: 5 }}>Generator g</div>
              <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
                <span
                  style={{
                    fontFamily: "monospace",
                    fontSize: 15,
                    fontWeight: 700,
                    color: C.accent,
                    minWidth: 40,
                  }}
                >
                  g = {String(g)}
                </span>
                <Btn onClick={nextGenerator} ariaLabel="Pick the next valid generator">
                  next g
                </Btn>
              </div>
            </div>
            <div style={{ marginLeft: "auto", display: "flex", gap: 8, flexWrap: "wrap" }}>
              <Btn onClick={randomizePrime} ariaLabel="Randomize the prime and generator">
                random p
              </Btn>
              <Btn onClick={randomizeSecrets} ariaLabel="Randomize both private secrets">
                random secrets
              </Btn>
            </div>
          </div>

          <div
            style={{
              fontSize: 11.5,
              color: gIsGen ? C.good : C.eve,
              background: gIsGen ? C.goodSoft : C.eveSoft,
              border: `1px solid ${gIsGen ? C.good : C.eve}33`,
              borderRadius: 8,
              padding: "7px 12px",
              marginBottom: 4,
              lineHeight: 1.5,
            }}
          >
            {gIsGen ? (
              <>
                g = {String(g)} is a generator of the field: its powers cycle through every nonzero
                value mod {String(p)}, so every shared secret is reachable.
              </>
            ) : (
              <>
                g = {String(g)} is not a full generator mod {String(p)}: its powers only reach part of
                the field, which shrinks the space of possible secrets. Use <b>next g</b> for a proper
                one.
              </>
            )}
          </div>
        </Card>

        <Card style={{ marginBottom: 16 }}>
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              flexWrap: "wrap",
              gap: 10,
              marginBottom: 14,
            }}
          >
            <div style={{ fontSize: 14, fontWeight: 700 }}>The exchange, one step at a time</div>
            <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
              <Btn onClick={restart} disabled={step === 0} ariaLabel="Restart the walkthrough">
                Restart
              </Btn>
              <Btn
                primary
                onClick={advance}
                disabled={step >= TOTAL_STEPS}
                ariaLabel="Advance to the next step"
              >
                {step === 0 ? "Begin" : step >= TOTAL_STEPS ? "Done" : "Next step"}
              </Btn>
            </div>
          </div>

          <div
            aria-live="polite"
            style={{ fontSize: 12.5, color: C.muted, lineHeight: 1.55, marginBottom: 14, minHeight: 34 }}
          >
            <b style={{ color: C.accent }}>
              Step {Math.min(step, TOTAL_STEPS)} of {TOTAL_STEPS}.
            </b>{" "}
            {stepCopy[Math.min(step, TOTAL_STEPS)]}
          </div>

          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(220px, 1fr))",
              gap: 12,
              marginBottom: 16,
            }}
          >
            <PartyCard
              name="Alice"
              color={C.alice}
              soft={C.aliceSoft}
              secretLabel="secret a"
              secretVal={String(a)}
              publicLabel="sends A"
              publicVal={String(A)}
              sharedVal={String(sharedAlice)}
              step={step}
            />
            <PartyCard
              name="Bob"
              color={C.bob}
              soft={C.bobSoft}
              secretLabel="secret b"
              secretVal={String(b)}
              publicLabel="sends B"
              publicVal={String(B)}
              sharedVal={String(sharedBob)}
              step={step}
            />
          </div>

          <div style={{ display: "flex", flexDirection: "column", gap: 4 }}>
            <StepRow
              active={step === 1}
              done={step > 1}
              color={C.accent}
              soft={C.accentSoft}
              label="Private secrets stay home"
              math={step >= 1 ? `a = ${String(a)}   b = ${String(b)}   (never sent)` : null}
            />
            <StepRow
              active={step === 2}
              done={step > 2}
              color={C.alice}
              soft={C.aliceSoft}
              label="Each sends one public number across the wire"
              math={
                step >= 2
                  ? `A = ${String(g)}^${String(a)} mod ${String(p)} = ${String(A)}    B = ${String(g)}^${String(b)} mod ${String(p)} = ${String(B)}`
                  : null
              }
            />
            <StepRow
              active={step === 3}
              done={step > 3}
              color={C.bob}
              soft={C.bobSoft}
              label="Each raises the other's number to its own secret"
              math={
                step >= 3
                  ? `Alice: ${String(B)}^${String(a)} mod ${String(p)}    Bob: ${String(A)}^${String(b)} mod ${String(p)}`
                  : null
              }
            />
            <StepRow
              active={step === 4}
              done={step >= 4}
              color={C.good}
              soft={C.goodSoft}
              label="Both land on the same secret"
              math={
                step >= 4
                  ? `${String(sharedAlice)} = ${String(sharedBob)} = ${String(g)}^(${String(a)}·${String(b)}) mod ${String(p)} = ${String(sharedDirect)}`
                  : null
              }
            />
          </div>

          {step >= 4 && (
            <div
              style={{
                marginTop: 14,
                background: matches ? C.goodSoft : C.eveSoft,
                border: `1px solid ${matches ? C.good : C.eve}44`,
                borderRadius: 10,
                padding: "12px 16px",
                fontSize: 12.5,
                lineHeight: 1.55,
                color: matches ? C.good : C.eve,
                textAlign: "center",
                transition: reduce ? "none" : `opacity 220ms ${EASE}`,
              }}
            >
              {matches ? (
                <>
                  <b>Both hold {String(sharedAlice)}.</b> Alice computed (g<sup>b</sup>)<sup>a</sup>{" "}
                  and Bob computed (g<sup>a</sup>)<sup>b</sup>, which are both g<sup>ab</sup> mod{" "}
                  {String(p)}. The number {String(sharedDirect)} never crossed the wire. Only{" "}
                  {String(A)} and {String(B)} did.
                </>
              ) : (
                <>Mismatch, which should not happen for a valid p and g. Try resetting.</>
              )}
            </div>
          )}
        </Card>

        <Card style={{ marginBottom: 16 }}>
          <div style={{ fontSize: 14, fontWeight: 700, marginBottom: 4 }}>
            The paint analogy, in the same steps
          </div>
          <div style={{ fontSize: 12, color: C.muted, lineHeight: 1.6, marginBottom: 16 }}>
            Mixing paint is easy, separating a blend back into its ingredients is not. That one-way
            gap is exactly what the math leans on. Public mixtures travel openly; the private tints
            never do.
          </div>
          <div
            style={{
              display: "flex",
              justifyContent: "space-around",
              alignItems: "flex-start",
              gap: 8,
              flexWrap: "wrap",
            }}
          >
            <PaintBlob color={commonPaint} label="Common paint" sub="public (= g, p)" reduce={reduce} pulse={step === 0} />
            <PaintBlob color={PAINT.alice} label="Alice's tint" sub="private (= a)" reduce={reduce} pulse={step === 1} />
            <PaintBlob
              color={step >= 2 ? alicePub : commonPaint}
              label="Alice sends"
              sub={step >= 2 ? "public (= A)" : "mixing"}
              reduce={reduce}
              pulse={step === 2}
            />
            <PaintBlob color={PAINT.bob} label="Bob's tint" sub="private (= b)" reduce={reduce} pulse={step === 1} />
            <PaintBlob
              color={step >= 2 ? bobPub : commonPaint}
              label="Bob sends"
              sub={step >= 2 ? "public (= B)" : "mixing"}
              reduce={reduce}
              pulse={step === 2}
            />
            <PaintBlob
              color={step >= 4 ? aliceFinal : "#cfc7bb"}
              label="Shared color"
              sub={step >= 4 ? "secret (= g^ab)" : "after swap"}
              reduce={reduce}
              pulse={step >= 4}
            />
          </div>
          <div style={{ fontSize: 11.5, color: C.muted, lineHeight: 1.55, marginTop: 14, textAlign: "center" }}>
            Alice stirs her tint into Bob's mixture, Bob stirs his into Alice's, and both reach the
            same three-paint blend (
            {step >= 4 ? <b style={{ color: C.ink }}>{aliceFinal.toUpperCase()}</b> : "shown once you finish the steps"}
            ). Eve saw the two public mixtures but cannot pull a private tint back out.
          </div>
        </Card>

        <Card style={{ marginBottom: 16, background: C.eveSoft, border: `1px solid ${C.eve}33` }}>
          <div
            style={{
              fontSize: 11,
              fontWeight: 700,
              letterSpacing: "0.08em",
              textTransform: "uppercase",
              color: C.eve,
              marginBottom: 8,
            }}
          >
            Eve, the eavesdropper
          </div>
          <div style={{ fontSize: 12.5, color: C.ink, lineHeight: 1.6, marginBottom: 14 }}>
            Eve has copied everything public: p = {String(p)}, g = {String(g)}, A = {String(A)}, and B
            = {String(B)}. To get the shared secret she needs one private exponent. Recovering a from{" "}
            A = g<sup>a</sup> mod p is the discrete logarithm, and the only move she has on a field
            this size is to try exponents one by one.
          </div>

          <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: 14 }}>
            <Btn onClick={() => runEve("alice")} ariaLabel="Have Eve brute force Alice's secret">
              Brute-force a from A
            </Btn>
            <Btn onClick={() => runEve("bob")} ariaLabel="Have Eve brute force Bob's secret">
              Brute-force b from B
            </Btn>
            {eve && (
              <Btn onClick={resetEve} ariaLabel="Clear Eve's attempt">
                Clear
              </Btn>
            )}
          </div>

          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))",
              gap: 10,
              marginBottom: 4,
            }}
          >
            <div style={{ background: "#fff", borderRadius: 10, padding: "12px 14px", border: `1px solid ${C.border}` }}>
              <div
                style={{
                  fontSize: 10.5,
                  color: C.muted,
                  fontWeight: 700,
                  letterSpacing: "0.05em",
                  textTransform: "uppercase",
                  marginBottom: 4,
                }}
              >
                Eve's search space
              </div>
              <div style={{ fontSize: 22, fontWeight: 700, color: C.eve, lineHeight: 1.05 }}>
                {searchSpace.toLocaleString()}
              </div>
              <div style={{ fontSize: 11, color: C.muted, marginTop: 4 }}>exponents to try (p − 2)</div>
            </div>
            <div style={{ background: "#fff", borderRadius: 10, padding: "12px 14px", border: `1px solid ${C.border}` }}>
              <div
                style={{
                  fontSize: 10.5,
                  color: C.muted,
                  fontWeight: 700,
                  letterSpacing: "0.05em",
                  textTransform: "uppercase",
                  marginBottom: 4,
                }}
              >
                Alice & Bob's work
              </div>
              <div style={{ fontSize: 22, fontWeight: 700, color: C.good, lineHeight: 1.05 }}>
                ~{String(BigInt(p.toString().length) * 2n)}
              </div>
              <div style={{ fontSize: 11, color: C.muted, marginTop: 4 }}>
                multiplications via square-and-multiply
              </div>
            </div>
          </div>

          <div aria-live="polite" style={{ minHeight: 44, marginTop: 12 }}>
            {eve && (
              <div
                style={{
                  background: "#fff",
                  border: `1px solid ${eve.exhausted ? C.good : C.eve}44`,
                  borderRadius: 10,
                  padding: "11px 15px",
                  fontSize: 12.5,
                  lineHeight: 1.55,
                  color: C.ink,
                }}
              >
                {eve.exhausted ? (
                  <>
                    Eve gave up after <b style={{ color: C.eve }}>{eve.attempts.toLocaleString()}</b>{" "}
                    tries (capped at {BRUTE_CAP.toLocaleString()}). On a prime this large the linear
                    search is hopeless, and this is still tiny next to the primes real systems use.
                  </>
                ) : (
                  <>
                    Eve recovered {eve.target === "alice" ? "Alice's a" : "Bob's b"} ={" "}
                    <b style={{ color: C.eve }}>{eve.found}</b> after{" "}
                    <b style={{ color: C.eve }}>{eve.attempts.toLocaleString()}</b> multiplications,
                    matching the real secret {String(eve.secret)}. With both p and the secret this
                    small the discrete log falls fast. Now grow p and run it again.
                  </>
                )}
              </div>
            )}
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
            Why the secret never travels
          </div>
          <div style={{ fontSize: 13, color: C.ink, lineHeight: 1.7, display: "grid", gap: 10 }}>
            <p style={{ margin: 0 }}>
              <b>Public versus private.</b> The prime p, the generator g, and the two transmitted
              numbers A and B are all public. The two exponents a and b are the only secrets, and
              neither one is ever sent. What crosses the wire is g raised to a private power, which
              tells you nothing about the power itself.
            </p>
            <p style={{ margin: 0 }}>
              <b>Why both sides match.</b> Alice computes B<sup>a</sup> = (g<sup>b</sup>)<sup>a</sup>{" "}
              and Bob computes A<sup>b</sup> = (g<sup>a</sup>)<sup>b</sup>. Exponents multiply the same
              way in either order, so both equal g<sup>ab</sup> mod p. The shared secret is built from
              both private numbers, yet each side supplies only its own.
            </p>
            <p style={{ margin: 0 }}>
              <b>The asymmetry that protects it.</b> Going forward is cheap: g<sup>a</sup> mod p takes
              about as many multiplications as p has digits, using square-and-multiply. Going
              backward, finding a from g<sup>a</sup> mod p, is the discrete logarithm, and the best
              known methods still scale roughly with the size of the field. Add one digit to p and
              Alice's work barely moves while Eve's search grows by a whole factor.
            </p>
            <p style={{ margin: 0 }}>
              <b>What real systems use.</b> The toy primes here top out at a few digits so every
              number fits on screen. Production Diffie-Hellman uses primes of 2048 bits or more, or
              switches to elliptic curves where the same one-way trick holds with much smaller keys.
              The brute force you watched stall on a five-digit prime becomes astronomically out of
              reach.
            </p>
          </div>
        </Card>

        <div style={{ marginTop: 18, textAlign: "center", fontSize: 11, color: C.muted }}>
          Every value is computed with BigInt modular exponentiation; Eve's counter is a step-by-step
          discrete-log search over the field mod {String(p)}.
        </div>
      </div>
    </div>
  );
}
