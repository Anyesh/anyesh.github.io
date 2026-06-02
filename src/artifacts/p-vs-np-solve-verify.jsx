import { useState, useMemo, useEffect, useRef, useCallback } from "react";

export const meta = {
  title: "Solve vs Verify: P vs NP Made Concrete",
  category: "Complexity Theory",
  description:
    "Checking an answer can be instant while finding one takes forever. Watch a DPLL solver grind through an exponential search tree while a checker verifies any solution in a glance: the gap behind P vs NP.",
  date: "2026-04-30",
  tags: ["p-vs-np", "complexity", "sat", "dpll", "np-complete"],
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
  panel: "#efeae3",
  solve: "#a8453f",
  solveSoft: "#fbecea",
  verify: "#2f6d4f",
  verifySoft: "#e2efe7",
  sat: "#2f7d53",
  unsat: "#a8453f",
  decide: "#2e6f8e",
  decideSoft: "#e6eef3",
  unit: "#9a7020",
};

const EASE = "cubic-bezier(0.23, 1, 0.32, 1)";

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

// A 3-SAT clause is three signed literals over variables 1..n.
// A literal is +v (variable v must be true) or -v (variable v must be false).
function generateInstance(n, seed, ratio = 4.2) {
  const rng = mulberry32(seed);
  const m = Math.max(1, Math.round(n * ratio));
  const clauses = [];
  for (let i = 0; i < m; i++) {
    const vars = new Set();
    while (vars.size < 3) vars.add(1 + Math.floor(rng() * n));
    const clause = [...vars].map((v) => (rng() < 0.5 ? v : -v));
    clauses.push(clause);
  }
  return { n, clauses };
}

// A pigeonhole-style unsatisfiable core: every assignment of three variables
// fails at least one clause, because all eight clauses over (a, b, c) are forbidden.
function unsatInstance() {
  const a = 1, b = 2, c = 3;
  const clauses = [];
  for (let i = 0; i < 8; i++) {
    const la = (i & 1 ? a : -a);
    const lb = (i & 2 ? b : -b);
    const lc = (i & 4 ? c : -c);
    clauses.push([la, lb, lc]);
  }
  return { n: 3, clauses };
}

function clauseValue(clause, assign) {
  // Returns "sat" if any literal is true, "unsat" if all literals are false,
  // "open" if undecided literals remain.
  let hasOpen = false;
  for (const lit of clause) {
    const v = Math.abs(lit);
    const val = assign[v];
    if (val === undefined) {
      hasOpen = true;
      continue;
    }
    const litTrue = lit > 0 ? val === true : val === false;
    if (litTrue) return "sat";
  }
  return hasOpen ? "open" : "unsat";
}

// Verifier: given a full assignment, confirm every clause is satisfied.
// Each literal is touched at most once, so the work is linear in the formula size.
function verify(instance, assign) {
  const trace = [];
  let ops = 0;
  let ok = true;
  let firstFail = null;
  for (let ci = 0; ci < instance.clauses.length; ci++) {
    const clause = instance.clauses[ci];
    let clauseSat = false;
    let satLit = null;
    for (const lit of clause) {
      ops++;
      const v = Math.abs(lit);
      const val = assign[v];
      const litTrue = lit > 0 ? val === true : val === false;
      if (litTrue && !clauseSat) {
        clauseSat = true;
        satLit = lit;
      }
    }
    trace.push({ clause: ci, sat: clauseSat, satLit });
    if (!clauseSat) {
      ok = false;
      if (firstFail === null) firstFail = ci;
    }
  }
  return { ops, ok, trace, firstFail, literals: instance.clauses.length * 3 };
}

// Real DPLL: unit propagation, then branch on the first unassigned variable.
// Records every event so the UI can step, watch backtracks, and count work.
function dpll(instance) {
  const { n, clauses } = instance;
  const events = [];
  let decisions = 0;
  let propagations = 0;
  let backtracks = 0;
  const assign = {};

  function snapshot() {
    const a = {};
    for (let v = 1; v <= n; v++) if (assign[v] !== undefined) a[v] = assign[v];
    return a;
  }

  function findUnit() {
    // A unit clause has exactly one open literal and no satisfying literal yet,
    // so that literal is forced.
    for (let ci = 0; ci < clauses.length; ci++) {
      const clause = clauses[ci];
      let open = null;
      let openCount = 0;
      let satisfied = false;
      for (const lit of clause) {
        const v = Math.abs(lit);
        const val = assign[v];
        if (val === undefined) {
          openCount++;
          open = lit;
        } else if (lit > 0 ? val === true : val === false) {
          satisfied = true;
          break;
        }
      }
      if (!satisfied && openCount === 1) return { lit: open, clause: ci };
    }
    return null;
  }

  function hasConflict() {
    for (let ci = 0; ci < clauses.length; ci++) {
      if (clauseValue(clauses[ci], assign) === "unsat") return ci;
    }
    return -1;
  }

  function allSat() {
    for (const clause of clauses) {
      if (clauseValue(clause, assign) !== "sat") return false;
    }
    return true;
  }

  function nextVar() {
    for (let v = 1; v <= n; v++) if (assign[v] === undefined) return v;
    return 0;
  }

  // Iterative depth-first search over the decision tree. Each frame remembers
  // which value it tried so it can flip on backtrack before giving up.
  const stack = [];
  let result = "unsat";

  function propagate() {
    // Force all unit literals before the next decision. Returns the conflicting
    // clause index, or -1 if propagation completed cleanly.
    for (;;) {
      const conflict = hasConflict();
      if (conflict !== -1) return conflict;
      const unit = findUnit();
      if (!unit) return -1;
      const v = Math.abs(unit.lit);
      assign[v] = unit.lit > 0;
      propagations++;
      stack.push({ v, kind: "unit", tried: 2 });
      events.push({
        type: "unit",
        v,
        value: assign[v],
        clause: unit.clause,
        assign: snapshot(),
        decisions,
        propagations,
        backtracks,
        depth: stack.length,
      });
    }
  }

  function backtrack() {
    // Undo frames until we find a decision we can flip to its other value.
    while (stack.length > 0) {
      const top = stack[stack.length - 1];
      delete assign[top.v];
      if (top.kind === "decision" && top.tried === 1) {
        assign[top.v] = false;
        top.tried = 2;
        backtracks++;
        events.push({
          type: "flip",
          v: top.v,
          value: false,
          assign: snapshot(),
          decisions,
          propagations,
          backtracks,
          depth: stack.length,
        });
        return true;
      }
      stack.pop();
    }
    return false;
  }

  outer: for (;;) {
    const conflict = propagate();
    if (conflict !== -1) {
      events.push({
        type: "conflict",
        clause: conflict,
        assign: snapshot(),
        decisions,
        propagations,
        backtracks,
        depth: stack.length,
      });
      if (!backtrack()) {
        result = "unsat";
        break;
      }
      continue;
    }
    if (allSat()) {
      result = "sat";
      events.push({
        type: "sat",
        assign: snapshot(),
        decisions,
        propagations,
        backtracks,
        depth: stack.length,
      });
      break;
    }
    const v = nextVar();
    if (v === 0) {
      // No conflict, no unassigned variable, yet not all satisfied: cannot happen
      // for well-formed clauses, but guard against an infinite loop just in case.
      result = "unsat";
      break outer;
    }
    assign[v] = true;
    decisions++;
    stack.push({ v, kind: "decision", tried: 1 });
    events.push({
      type: "decide",
      v,
      value: true,
      assign: snapshot(),
      decisions,
      propagations,
      backtracks,
      depth: stack.length,
    });
  }

  const certificate = result === "sat" ? snapshotFull(assign, n) : null;
  return {
    result,
    events,
    certificate,
    decisions,
    propagations,
    backtracks,
    steps: events.length,
  };
}

function snapshotFull(assign, n) {
  const a = {};
  for (let v = 1; v <= n; v++) a[v] = assign[v] === undefined ? true : assign[v];
  return a;
}

function fmtBig(x) {
  if (x < 1000) return String(x);
  if (x < 1e6) return `${(x / 1000).toFixed(x < 1e4 ? 1 : 0)}k`;
  if (x < 1e9) return `${(x / 1e6).toFixed(1)}M`;
  if (x < 1e12) return `${(x / 1e9).toFixed(1)}B`;
  return x.toExponential(1);
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

function Btn({ children, onClick, disabled, primary, ariaLabel, type, reduce }) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      type={type || "button"}
      aria-label={ariaLabel}
      style={{
        padding: primary ? "9px 18px" : "8px 14px",
        borderRadius: 9,
        border: primary ? "none" : `1.5px solid ${C.border}`,
        background: primary ? C.accent : C.card,
        color: primary ? "#fff" : C.ink,
        fontSize: 13,
        fontWeight: primary ? 600 : 500,
        fontFamily: "inherit",
        cursor: disabled ? "not-allowed" : "pointer",
        opacity: disabled ? 0.4 : 1,
        transition: reduce ? "none" : `transform 140ms ${EASE}, background 160ms ease`,
        boxShadow: primary ? "0 1px 3px rgba(192,86,31,0.3)" : "none",
      }}
      onPointerDown={(e) => {
        if (!disabled && !reduce) e.currentTarget.style.transform = "scale(0.97)";
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

function Stat({ label, value, sub, color, bg, big }) {
  return (
    <div style={{ background: bg, borderRadius: 10, padding: "12px 14px", flex: 1, minWidth: 120 }}>
      <div
        style={{
          fontSize: 10.5,
          color,
          fontWeight: 700,
          letterSpacing: "0.06em",
          textTransform: "uppercase",
          marginBottom: 4,
        }}
      >
        {label}
      </div>
      <div
        style={{
          fontSize: big ? 28 : 22,
          fontWeight: 700,
          color,
          lineHeight: 1.05,
          fontVariantNumeric: "tabular-nums",
        }}
      >
        {value}
      </div>
      {sub && <div style={{ fontSize: 11, color: C.muted, marginTop: 4 }}>{sub}</div>}
    </div>
  );
}

function Literal({ lit, assign, reduce }) {
  const v = Math.abs(lit);
  const val = assign ? assign[v] : undefined;
  let state = "open";
  if (val !== undefined) {
    const litTrue = lit > 0 ? val === true : val === false;
    state = litTrue ? "true" : "false";
  }
  const palette = {
    open: { bg: C.panel, color: C.muted, border: C.border },
    true: { bg: C.verifySoft, color: C.verify, border: `${C.verify}55` },
    false: { bg: "#f1ece6", color: C.faint, border: C.border },
  }[state];
  return (
    <span
      style={{
        display: "inline-flex",
        alignItems: "center",
        padding: "1px 6px",
        borderRadius: 5,
        background: palette.bg,
        color: palette.color,
        border: `1px solid ${palette.border}`,
        fontFamily: "ui-monospace, 'SF Mono', Menlo, monospace",
        fontSize: 12,
        fontWeight: 600,
        textDecoration: state === "false" ? "line-through" : "none",
        transition: reduce ? "none" : `background 160ms ${EASE}, color 160ms ${EASE}`,
      }}
    >
      {lit < 0 ? <span aria-hidden="true" style={{ marginRight: 1 }}>&not;</span> : null}x
      <sub style={{ fontSize: 9 }}>{v}</sub>
    </span>
  );
}

function ClauseRow({ clause, idx, assign, status, highlight, reduce, mode }) {
  const statusColor = {
    sat: C.sat,
    unsat: C.unsat,
    open: C.muted,
  }[status];
  const statusBg = {
    sat: C.verifySoft,
    unsat: C.solveSoft,
    open: "transparent",
  }[status];
  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        gap: 8,
        padding: "5px 9px",
        borderRadius: 8,
        background: highlight ? (mode === "solve" ? C.solveSoft : C.verifySoft) : statusBg,
        outline: highlight ? `2px solid ${mode === "solve" ? C.solve : C.verify}` : "none",
        transition: reduce ? "none" : `background 180ms ${EASE}, outline-color 180ms ${EASE}`,
      }}
    >
      <span
        style={{
          fontSize: 10,
          color: C.faint,
          fontFamily: "ui-monospace, monospace",
          minWidth: 22,
          fontVariantNumeric: "tabular-nums",
        }}
      >
        c{idx}
      </span>
      <span style={{ display: "flex", alignItems: "center", gap: 5, flex: 1, flexWrap: "wrap" }}>
        <Literal lit={clause[0]} assign={assign} reduce={reduce} />
        <span style={{ color: C.faint, fontSize: 11 }}>&or;</span>
        <Literal lit={clause[1]} assign={assign} reduce={reduce} />
        <span style={{ color: C.faint, fontSize: 11 }}>&or;</span>
        <Literal lit={clause[2]} assign={assign} reduce={reduce} />
      </span>
      <span
        style={{
          fontSize: 10,
          fontWeight: 700,
          color: statusColor,
          minWidth: 38,
          textAlign: "right",
          textTransform: "uppercase",
          letterSpacing: "0.04em",
        }}
      >
        {status === "open" ? "" : status}
      </span>
    </div>
  );
}

const SIZES = [4, 6, 8, 10, 12, 14, 16, 18, 20];

export default function App() {
  const reduce = usePrefersReducedMotion();
  const [n, setN] = useState(8);
  const [seed, setSeed] = useState(20260602);
  const [forceUnsat, setForceUnsat] = useState(false);
  const [stepIdx, setStepIdx] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [showVerify, setShowVerify] = useState(false);
  const [verifyIdx, setVerifyIdx] = useState(0);
  const [verifyPlaying, setVerifyPlaying] = useState(false);
  const [tamper, setTamper] = useState(null);
  const playRef = useRef(null);
  const verifyRef = useRef(null);

  const instance = useMemo(
    () => (forceUnsat ? unsatInstance() : generateInstance(n, seed)),
    [n, seed, forceUnsat]
  );

  const solver = useMemo(() => dpll(instance), [instance]);

  const totalSteps = solver.events.length;
  const clampedIdx = Math.min(stepIdx, totalSteps);
  const current = clampedIdx > 0 ? solver.events[clampedIdx - 1] : null;
  const currentAssign = current ? current.assign : {};
  const searchSpace = useMemo(() => Math.pow(2, instance.n), [instance.n]);

  useEffect(() => {
    setStepIdx(0);
    setPlaying(false);
    setShowVerify(false);
    setVerifyIdx(0);
    setVerifyPlaying(false);
    setTamper(null);
  }, [instance]);

  useEffect(() => {
    if (!playing) {
      clearInterval(playRef.current);
      return;
    }
    playRef.current = setInterval(
      () => {
        setStepIdx((s) => {
          if (s >= totalSteps) {
            setPlaying(false);
            return s;
          }
          return s + 1;
        });
      },
      reduce ? 12 : 55
    );
    return () => clearInterval(playRef.current);
  }, [playing, totalSteps, reduce]);

  // The certificate handed to the verifier: the solver's answer, optionally
  // tampered so the user can watch a wrong certificate get rejected.
  const certificate = useMemo(() => {
    if (!solver.certificate) return null;
    if (tamper === null) return solver.certificate;
    const copy = { ...solver.certificate };
    copy[tamper] = !copy[tamper];
    return copy;
  }, [solver.certificate, tamper]);

  const verifyResult = useMemo(
    () => (certificate ? verify(instance, certificate) : null),
    [instance, certificate]
  );

  useEffect(() => {
    setVerifyIdx(0);
    setVerifyPlaying(false);
  }, [certificate, showVerify]);

  useEffect(() => {
    if (!verifyPlaying || !verifyResult) {
      clearInterval(verifyRef.current);
      return;
    }
    verifyRef.current = setInterval(
      () => {
        setVerifyIdx((s) => {
          if (s >= verifyResult.trace.length) {
            setVerifyPlaying(false);
            return s;
          }
          return s + 1;
        });
      },
      reduce ? 10 : 45
    );
    return () => clearInterval(verifyRef.current);
  }, [verifyPlaying, verifyResult, reduce]);

  const clauseStatus = useMemo(() => {
    return instance.clauses.map((clause) => clauseValue(clause, currentAssign));
  }, [instance, currentAssign]);

  const highlightClause = current && (current.type === "conflict" || current.type === "unit") ? current.clause : null;

  const solverDone = clampedIdx >= totalSteps && totalSteps > 0;
  const solverOps = current ? current.decisions + current.propagations : 0;

  const verifyOpsShown = useMemo(() => {
    if (!verifyResult) return 0;
    let ops = 0;
    for (let i = 0; i < Math.min(verifyIdx, verifyResult.trace.length); i++) {
      ops += instance.clauses[verifyResult.trace[i].clause].length;
    }
    return ops;
  }, [verifyResult, verifyIdx, instance]);

  const regenerate = useCallback(() => {
    setForceUnsat(false);
    setSeed((s) => (s * 1664525 + 1013904223) >>> 0 || 1);
  }, []);

  const stage = useMemo(() => {
    if (totalSteps === 0) return { title: "Ready to search", body: "" };
    if (clampedIdx === 0) {
      return {
        title: "The search has not started",
        body:
          "Every variable is unassigned. The solver will pick a variable, try true, propagate every forced consequence, and only branch again when nothing is forced. A conflict sends it back to flip an earlier choice.",
      };
    }
    if (current.type === "decide") {
      return {
        title: `Decision: try x${current.v} = ${current.value ? "true" : "false"}`,
        body:
          "No clause forced a value, so the solver guesses. This guess opens a new branch of the search tree. Every wrong guess that survives propagation will eventually cost a backtrack.",
      };
    }
    if (current.type === "unit") {
      return {
        title: `Unit propagation: x${current.v} = ${current.value ? "true" : "false"} is forced`,
        body:
          `Clause c${current.clause} had only one literal left undecided, so that literal must be made true or the clause fails. Propagation is free progress: it prunes the tree without a guess.`,
      };
    }
    if (current.type === "conflict") {
      return {
        title: `Conflict in clause c${current.clause}`,
        body:
          "Every literal in this clause is now false, so the current partial assignment can never satisfy the formula. The solver must backtrack to the most recent decision it has not yet flipped.",
      };
    }
    if (current.type === "flip") {
      return {
        title: `Backtrack: flip x${current.v} to false`,
        body:
          "The true branch failed, so the solver undoes everything below the last open decision and tries the other value. This is the expensive part: in the worst case it happens an exponential number of times.",
      };
    }
    if (current.type === "sat") {
      return {
        title: "Satisfied: a full assignment makes every clause true",
        body:
          "The solver found a setting of all variables with no open or failing clause. That assignment is a certificate. Hand it to the verifier and watch how little work it takes to confirm.",
      };
    }
    return { title: "", body: "" };
  }, [current, clampedIdx, totalSteps]);

  const polyBound = instance.clauses.length * 3;

  return (
    <div
      style={{
        fontFamily: "'Palatino Linotype', 'Book Antiqua', Palatino, serif",
        background: C.bg,
        minHeight: "100vh",
        padding: "24px 14px 48px",
        color: C.ink,
      }}
    >
      <style>{`
        input[type="range"]:focus-visible { outline: 2px solid ${C.accent}; outline-offset: 3px; border-radius: 4px; }
        button:focus-visible { outline: 2px solid ${C.accent}; outline-offset: 2px; }
        @media (prefers-reduced-motion: reduce) {
          * { transition: none !important; animation: none !important; }
        }
      `}</style>

      <div style={{ maxWidth: 720, margin: "0 auto" }}>
        <header style={{ marginBottom: 22 }}>
          <div
            style={{
              fontSize: 10.5,
              letterSpacing: "0.15em",
              textTransform: "uppercase",
              color: C.faint,
              marginBottom: 6,
            }}
          >
            Complexity Theory / The open question
          </div>
          <h1 style={{ fontSize: 27, fontWeight: 700, margin: 0, letterSpacing: "-0.01em", textWrap: "balance" }}>
            Solve vs Verify
          </h1>
          <p
            style={{
              color: C.ink,
              fontSize: 14,
              lineHeight: 1.65,
              margin: "10px 0 0",
              maxWidth: "64ch",
            }}
          >
            Here is one Boolean formula in conjunctive normal form: a stack of clauses, each a group
            of three literals joined by <i>or</i>, all clauses joined by <i>and</i>. Two questions
            look almost identical. <b style={{ color: C.solve }}>Solving</b> asks whether any
            assignment of true and false to the variables makes the whole formula true.{" "}
            <b style={{ color: C.verify }}>Verifying</b> asks whether one specific assignment you are
            handed actually works. The first can take an exponential search. The second is a quick
            scan. That gap is what P vs NP is about.
          </p>
        </header>

        <Card style={{ marginBottom: 16 }}>
          <div style={{ display: "flex", flexWrap: "wrap", gap: 16, alignItems: "flex-end" }}>
            <div style={{ flex: "1 1 240px" }}>
              <label
                htmlFor="n-slider"
                style={{
                  display: "flex",
                  justifyContent: "space-between",
                  fontSize: 12,
                  color: C.muted,
                  marginBottom: 6,
                }}
              >
                <span>n — Boolean variables</span>
                <span style={{ fontWeight: 700, color: C.accent }}>
                  {instance.n} vars, {instance.clauses.length} clauses
                </span>
              </label>
              <input
                id="n-slider"
                type="range"
                min={0}
                max={SIZES.length - 1}
                value={Math.max(0, SIZES.indexOf(n))}
                onChange={(e) => {
                  setForceUnsat(false);
                  setN(SIZES[+e.target.value]);
                }}
                disabled={forceUnsat}
                style={{ width: "100%", accentColor: C.accent, opacity: forceUnsat ? 0.4 : 1 }}
                aria-label="Number of Boolean variables"
              />
              <div style={{ fontSize: 11.5, color: C.muted, marginTop: 6 }}>
                Full search space is 2
                <sup>{instance.n}</sup> = <b style={{ color: C.solve }}>{fmtBig(searchSpace)}</b>{" "}
                possible assignments. Adding one variable doubles it.
              </div>
            </div>
            <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
              <Btn onClick={regenerate} reduce={reduce} ariaLabel="Generate a new random formula">
                New formula
              </Btn>
              <Btn
                onClick={() => setForceUnsat(true)}
                reduce={reduce}
                primary={false}
                ariaLabel="Load a formula that has no solution"
              >
                Load unsatisfiable
              </Btn>
            </div>
          </div>
          {forceUnsat && (
            <div
              style={{
                marginTop: 12,
                fontSize: 12,
                color: C.unsat,
                background: C.solveSoft,
                border: `1px solid ${C.unsat}33`,
                borderRadius: 8,
                padding: "8px 12px",
                lineHeight: 1.55,
              }}
            >
              This is a hand-built unsatisfiable instance: all eight clauses over three variables are
              listed, so every one of the eight assignments fails somewhere. The solver must explore
              the whole tree to prove no solution exists.
            </div>
          )}
        </Card>

        <div
          style={{
            display: "grid",
            gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))",
            gap: 12,
            marginBottom: 16,
          }}
        >
          <Stat
            label="Solver steps so far"
            value={fmtBig(solverOps)}
            sub={`of up to 2^${instance.n} = ${fmtBig(searchSpace)} in the worst case`}
            color={C.solve}
            bg={C.solveSoft}
            big
          />
          <Stat
            label="Backtracks"
            value={current ? current.backtracks : 0}
            sub="wrong branches abandoned"
            color={C.unit}
            bg="#f5edd8"
          />
          <Stat
            label="Verifier ops"
            value={showVerify ? fmtBig(verifyOpsShown) : "—"}
            sub={showVerify ? `of ${polyBound} total, linear in clauses` : "run the check below"}
            color={C.verify}
            bg={C.verifySoft}
            big
          />
        </div>

        <div style={{ display: "grid", gridTemplateColumns: "1fr", gap: 16 }}>
          <Card>
            <div
              style={{
                display: "flex",
                justifyContent: "space-between",
                alignItems: "center",
                marginBottom: 4,
                flexWrap: "wrap",
                gap: 8,
              }}
            >
              <div style={{ fontSize: 14, fontWeight: 700, color: C.solve }}>
                The hard job: search for a solution
              </div>
              <div
                style={{
                  fontSize: 11,
                  color: C.muted,
                  fontFamily: "ui-monospace, monospace",
                }}
              >
                {totalSteps > 0 ? `step ${clampedIdx} / ${totalSteps}` : ""}
              </div>
            </div>
            <div style={{ fontSize: 12.5, color: C.muted, lineHeight: 1.6, marginBottom: 14 }}>
              A real DPLL solver: pick a variable, propagate everything forced, branch when stuck,
              backtrack on conflict. Step through it or let it run.
            </div>

            <div
              style={{
                display: "flex",
                flexDirection: "column",
                gap: 3,
                maxHeight: 280,
                overflowY: "auto",
                marginBottom: 14,
                paddingRight: 4,
              }}
            >
              {instance.clauses.map((clause, idx) => (
                <ClauseRow
                  key={idx}
                  clause={clause}
                  idx={idx}
                  assign={currentAssign}
                  status={clauseStatus[idx]}
                  highlight={highlightClause === idx}
                  reduce={reduce}
                  mode="solve"
                />
              ))}
            </div>

            <div style={{ display: "flex", gap: 9, flexWrap: "wrap", marginBottom: 14 }}>
              <Btn
                onClick={() => setStepIdx((s) => Math.min(s + 1, totalSteps))}
                disabled={clampedIdx >= totalSteps}
                reduce={reduce}
                ariaLabel="Advance the solver one step"
              >
                Step
              </Btn>
              <Btn
                primary
                onClick={() => {
                  if (clampedIdx >= totalSteps) setStepIdx(0);
                  setPlaying((p) => !p);
                }}
                disabled={totalSteps === 0}
                reduce={reduce}
                ariaLabel={playing ? "Pause the solver" : "Run the solver"}
              >
                {playing ? "Pause" : clampedIdx >= totalSteps && totalSteps > 0 ? "Replay" : "Run solver"}
              </Btn>
              <Btn
                onClick={() => {
                  setStepIdx(totalSteps);
                  setPlaying(false);
                }}
                disabled={clampedIdx >= totalSteps}
                reduce={reduce}
                ariaLabel="Jump to the end of the search"
              >
                Skip to end
              </Btn>
              <Btn
                onClick={() => {
                  setStepIdx(0);
                  setPlaying(false);
                }}
                reduce={reduce}
                ariaLabel="Reset the solver to the start"
              >
                Reset
              </Btn>
            </div>

            <div
              aria-live="polite"
              style={{
                background: C.panel,
                borderRadius: 10,
                padding: "12px 15px",
                minHeight: 72,
              }}
            >
              <div style={{ fontSize: 13, fontWeight: 700, marginBottom: 4, color: C.ink }}>
                {stage.title}
              </div>
              {stage.body && (
                <div style={{ fontSize: 12.5, color: C.muted, lineHeight: 1.6 }}>{stage.body}</div>
              )}
            </div>

            {solverDone && (
              <div
                style={{
                  marginTop: 12,
                  display: "flex",
                  alignItems: "center",
                  gap: 10,
                  flexWrap: "wrap",
                  background: solver.result === "sat" ? C.verifySoft : C.solveSoft,
                  border: `1px solid ${(solver.result === "sat" ? C.sat : C.unsat)}33`,
                  borderRadius: 10,
                  padding: "10px 14px",
                }}
              >
                <span
                  style={{
                    fontSize: 11,
                    fontWeight: 700,
                    letterSpacing: "0.06em",
                    textTransform: "uppercase",
                    color: solver.result === "sat" ? C.sat : C.unsat,
                  }}
                >
                  {solver.result === "sat" ? "Satisfiable" : "Unsatisfiable"}
                </span>
                <span style={{ fontSize: 12.5, color: C.muted, flex: 1, minWidth: 180 }}>
                  {solver.result === "sat"
                    ? `Found after ${solver.decisions} decisions, ${solver.propagations} propagations, ${solver.backtracks} backtracks.`
                    : `Proved after exploring the tree: ${solver.decisions} decisions, ${solver.backtracks} backtracks, no assignment works.`}
                </span>
                {solver.result === "sat" && (
                  <Btn
                    primary
                    onClick={() => setShowVerify(true)}
                    reduce={reduce}
                    ariaLabel="Send the found assignment to the verifier"
                  >
                    Verify this certificate
                  </Btn>
                )}
              </div>
            )}
          </Card>

          {showVerify && verifyResult && certificate && (
            <Card style={{ border: `1px solid ${C.verify}44` }}>
              <div style={{ fontSize: 14, fontWeight: 700, color: C.verify, marginBottom: 4 }}>
                The easy job: verify a certificate
              </div>
              <div style={{ fontSize: 12.5, color: C.muted, lineHeight: 1.6, marginBottom: 14 }}>
                The certificate is a full assignment of every variable. Verifying reads each clause
                once and checks that at least one of its literals is true. The work is linear in the
                size of the formula, no search, no backtracking.
              </div>

              <div
                style={{
                  display: "flex",
                  flexWrap: "wrap",
                  gap: 6,
                  marginBottom: 14,
                  padding: "10px 12px",
                  background: C.panel,
                  borderRadius: 10,
                }}
              >
                <span style={{ fontSize: 11.5, color: C.muted, marginRight: 4, alignSelf: "center" }}>
                  certificate:
                </span>
                {Object.keys(certificate)
                  .map(Number)
                  .sort((a, b) => a - b)
                  .map((v) => {
                    const tampered = tamper === v;
                    return (
                      <button
                        key={v}
                        type="button"
                        onClick={() => setTamper((t) => (t === v ? null : v))}
                        aria-label={`Toggle variable x${v} in the certificate, currently ${certificate[v] ? "true" : "false"}`}
                        style={{
                          fontFamily: "ui-monospace, monospace",
                          fontSize: 12,
                          fontWeight: 600,
                          padding: "3px 8px",
                          borderRadius: 6,
                          cursor: "pointer",
                          border: `1px solid ${tampered ? C.unsat : certificate[v] ? `${C.verify}55` : C.border}`,
                          background: tampered ? C.solveSoft : certificate[v] ? C.verifySoft : "#fff",
                          color: tampered ? C.unsat : certificate[v] ? C.verify : C.faint,
                          transition: reduce ? "none" : `transform 120ms ${EASE}`,
                        }}
                        onPointerDown={(e) => {
                          if (!reduce) e.currentTarget.style.transform = "scale(0.94)";
                        }}
                        onPointerUp={(e) => (e.currentTarget.style.transform = "scale(1)")}
                        onPointerLeave={(e) => (e.currentTarget.style.transform = "scale(1)")}
                      >
                        x{v}={certificate[v] ? "T" : "F"}
                      </button>
                    );
                  })}
              </div>
              <div style={{ fontSize: 11.5, color: C.muted, marginBottom: 14, lineHeight: 1.55 }}>
                Click any variable to flip it and corrupt the certificate. A single wrong bit can
                break a clause, and the verifier catches it just as fast as it confirms a correct one.
                Verification never has to search; it only reads.
              </div>

              <div
                style={{
                  display: "flex",
                  flexDirection: "column",
                  gap: 3,
                  maxHeight: 280,
                  overflowY: "auto",
                  marginBottom: 14,
                  paddingRight: 4,
                }}
              >
                {instance.clauses.map((clause, idx) => {
                  const checked = idx < verifyIdx;
                  const traceItem = verifyResult.trace[idx];
                  const status = checked ? (traceItem.sat ? "sat" : "unsat") : "open";
                  return (
                    <ClauseRow
                      key={idx}
                      clause={clause}
                      idx={idx}
                      assign={certificate}
                      status={status}
                      highlight={idx === verifyIdx - 1}
                      reduce={reduce}
                      mode="verify"
                    />
                  );
                })}
              </div>

              <div style={{ display: "flex", gap: 9, flexWrap: "wrap", marginBottom: 14 }}>
                <Btn
                  onClick={() => setVerifyIdx((s) => Math.min(s + 1, verifyResult.trace.length))}
                  disabled={verifyIdx >= verifyResult.trace.length}
                  reduce={reduce}
                  ariaLabel="Check one more clause"
                >
                  Check next clause
                </Btn>
                <Btn
                  primary
                  onClick={() => {
                    if (verifyIdx >= verifyResult.trace.length) setVerifyIdx(0);
                    setVerifyPlaying((p) => !p);
                  }}
                  reduce={reduce}
                  ariaLabel={verifyPlaying ? "Pause verification" : "Run verification"}
                >
                  {verifyPlaying
                    ? "Pause"
                    : verifyIdx >= verifyResult.trace.length
                    ? "Replay"
                    : "Run verify"}
                </Btn>
                <Btn
                  onClick={() => {
                    setVerifyIdx(verifyResult.trace.length);
                    setVerifyPlaying(false);
                  }}
                  disabled={verifyIdx >= verifyResult.trace.length}
                  reduce={reduce}
                  ariaLabel="Check all clauses at once"
                >
                  Check all
                </Btn>
              </div>

              {verifyIdx >= verifyResult.trace.length && (
                <div
                  aria-live="polite"
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: 10,
                    flexWrap: "wrap",
                    background: verifyResult.ok ? C.verifySoft : C.solveSoft,
                    border: `1px solid ${(verifyResult.ok ? C.sat : C.unsat)}33`,
                    borderRadius: 10,
                    padding: "10px 14px",
                  }}
                >
                  <span
                    style={{
                      fontSize: 11,
                      fontWeight: 700,
                      letterSpacing: "0.06em",
                      textTransform: "uppercase",
                      color: verifyResult.ok ? C.sat : C.unsat,
                    }}
                  >
                    {verifyResult.ok ? "Certificate accepted" : "Certificate rejected"}
                  </span>
                  <span style={{ fontSize: 12.5, color: C.muted }}>
                    {verifyResult.ok
                      ? `All ${instance.clauses.length} clauses satisfied in ${verifyResult.ops} literal reads. No search.`
                      : `Clause c${verifyResult.firstFail} has no true literal, so this assignment fails. Caught in ${verifyResult.ops} reads.`}
                  </span>
                </div>
              )}
            </Card>
          )}
        </div>

        <Card style={{ background: C.accentSoft, border: `1px solid ${C.accent}22`, marginTop: 16 }}>
          <div
            style={{
              fontSize: 11,
              fontWeight: 700,
              letterSpacing: "0.08em",
              textTransform: "uppercase",
              color: C.accent,
              marginBottom: 12,
            }}
          >
            The gap, and the open question
          </div>
          <div style={{ fontSize: 13.5, color: C.ink, lineHeight: 1.7, display: "grid", gap: 11 }}>
            <p style={{ margin: 0 }}>
              <b>NP is the class of verify-fast problems.</b> A decision problem is in NP when every
              yes-instance has a certificate that a polynomial-time checker can confirm. SAT fits: the
              certificate is a satisfying assignment, and the verifier above reads each clause once.
              P is the class of problems you can <i>solve</i> in polynomial time. Every problem in P
              is also in NP, because solving fast is at least as good as checking fast.
            </p>
            <p style={{ margin: 0 }}>
              <b>SAT is NP-complete.</b> It is in NP, and every other NP problem can be rewritten as a
              SAT instance in polynomial time. That reduction is why these problems travel together:
              a fast algorithm for SAT would give a fast algorithm for every problem in NP, from graph
              colouring to the travelling salesman decision problem. They are all the same hardness in
              disguise.
            </p>
            <p style={{ margin: 0 }}>
              <b>P vs NP asks whether solving is really harder than verifying.</b> For SAT and the
              thousands of other NP-complete problems, no polynomial-time solver is known: the best
              general methods still search a tree that can blow up exponentially, exactly as you saw
              above. But nobody has proved that a fast solver cannot exist. P = NP would mean every
              quickly-checkable problem is also quickly solvable; P &ne; NP would mean some are
              genuinely harder to solve than to check. Which one is true is still open. The
              exponential search you watched is the best we currently know how to do, not a proof that
              nothing better exists.
            </p>
          </div>
        </Card>

        <p
          style={{
            textAlign: "center",
            fontSize: 11.5,
            color: C.faint,
            marginTop: 22,
            lineHeight: 1.6,
          }}
        >
          Real 3-SAT at clause-to-variable ratio 4.2, instances seeded by mulberry32 so they are
          reproducible. DPLL with unit propagation drives the search; the verifier is a single linear
          pass over the clauses.
        </p>
      </div>
    </div>
  );
}
