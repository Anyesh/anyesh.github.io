import { useState, useEffect, useRef, useMemo, useCallback } from "react";

export const meta = {
  title: "Q-learning on a Gridworld",
  category: "Reinforcement Learning",
  description:
    "An agent starts knowing nothing and learns to reach a goal from reward alone. Watch the value of each cell fill backward from the reward, the greedy arrows snap into a path, and see why cutting off exploration too early traps the agent on a worse route.",
  date: "2026-05-06",
  tags: ["reinforcement-learning", "q-learning", "temporal-difference", "exploration"],
};

const C = {
  bg: "#f7f5f2",
  card: "#ffffff",
  border: "#e4dfd8",
  ink: "#1c1a17",
  muted: "#6f675e",
  faint: "#9b938a",
  accent: "#c0561f",
  accentSoft: "#f5e7df",
  start: "#2f6d4f",
  startSoft: "#dcebe2",
  goal: "#2e7d51",
  trap: "#b3341b",
  trapSoft: "#f3ddd8",
  wall: "#3a352f",
  agent: "#1f4f8f",
  cold: "#eef2f4",
};

const EASE = "cubic-bezier(0.23, 1, 0.32, 1)";

const ROWS = 6;
const COLS = 6;
const N = ROWS * COLS;

const STEP_REWARD = -0.04;
const GOAL_REWARD = 1.0;
const TRAP_REWARD = -1.0;
const MAX_STEPS = 120;

const ACTIONS = [
  { dr: -1, dc: 0, name: "up", glyph: "↑" },
  { dr: 1, dc: 0, name: "down", glyph: "↓" },
  { dr: 0, dc: -1, name: "left", glyph: "←" },
  { dr: 0, dc: 1, name: "right", glyph: "→" },
];

const idx = (r, c) => r * COLS + c;
const rowOf = (s) => Math.floor(s / COLS);
const colOf = (s) => s % COLS;
const inBounds = (r, c) => r >= 0 && r < ROWS && c >= 0 && c < COLS;

function makeRng(seed) {
  let s = seed >>> 0;
  return () => {
    s = (s * 1103515245 + 12345) & 0x7fffffff;
    return s / 0x7fffffff;
  };
}

function zeroQ() {
  return Array.from({ length: N }, () => [0, 0, 0, 0]);
}

const PRESETS = {
  detour: {
    label: "Detour",
    start: idx(5, 0),
    goal: idx(0, 5),
    walls: new Set([idx(1, 2), idx(2, 2), idx(3, 2), idx(4, 2)]),
    traps: new Set([idx(2, 4)]),
  },
  twoPaths: {
    label: "Two paths",
    start: idx(5, 0),
    goal: idx(0, 5),
    walls: new Set([idx(2, 1), idx(2, 2), idx(2, 3), idx(3, 3), idx(3, 4)]),
    traps: new Set([idx(4, 3), idx(1, 2)]),
  },
  open: {
    label: "Open field",
    start: idx(5, 0),
    goal: idx(0, 5),
    walls: new Set(),
    traps: new Set(),
  },
};

function isTerminalState(s, goal, traps) {
  return s === goal || traps.has(s);
}

function stepEnv(s, a, env, slip, rng) {
  if (isTerminalState(s, env.goal, env.traps)) return s;
  let act = a;
  // a slip rotates the intended move to a random one, modelling stochastic dynamics
  if (slip > 0 && rng() < slip) act = Math.floor(rng() * 4);
  const r = rowOf(s);
  const c = colOf(s);
  const nr = r + ACTIONS[act].dr;
  const nc = c + ACTIONS[act].dc;
  if (!inBounds(nr, nc)) return s;
  const ns = idx(nr, nc);
  if (env.walls.has(ns)) return s;
  return ns;
}

function rewardOf(ns, env) {
  if (ns === env.goal) return GOAL_REWARD;
  if (env.traps.has(ns)) return TRAP_REWARD;
  return STEP_REWARD;
}

function argmaxAction(q) {
  let best = 0;
  for (let i = 1; i < 4; i++) if (q[i] > q[best]) best = i;
  return best;
}

function maxQ(q) {
  let m = q[0];
  for (let i = 1; i < 4; i++) if (q[i] > m) m = q[i];
  return m;
}

function greedyRollout(Q, env) {
  let s = env.start;
  const visited = new Set([s]);
  const path = [s];
  let steps = 0;
  while (!isTerminalState(s, env.goal, env.traps) && steps < N * 2) {
    const a = argmaxAction(Q[s]);
    const r = rowOf(s);
    const c = colOf(s);
    const nr = r + ACTIONS[a].dr;
    const nc = c + ACTIONS[a].dc;
    let ns = s;
    if (inBounds(nr, nc) && !env.walls.has(idx(nr, nc))) ns = idx(nr, nc);
    if (ns === s || visited.has(ns)) {
      return { path, solves: false, steps, looped: true };
    }
    visited.add(ns);
    path.push(ns);
    s = ns;
    steps++;
  }
  return { path, solves: s === env.goal, steps, looped: false };
}

function bfsOptimalLength(env) {
  const dist = new Array(N).fill(-1);
  const queue = [env.start];
  dist[env.start] = 0;
  while (queue.length) {
    const s = queue.shift();
    if (s === env.goal) return dist[s];
    const r = rowOf(s);
    const c = colOf(s);
    for (const a of ACTIONS) {
      const nr = r + a.dr;
      const nc = c + a.dc;
      if (!inBounds(nr, nc)) continue;
      const ns = idx(nr, nc);
      if (env.walls.has(ns) || env.traps.has(ns)) continue;
      if (dist[ns] === -1) {
        dist[ns] = dist[s] + 1;
        queue.push(ns);
      }
    }
  }
  return -1;
}

function Slider({ label, value, min, max, step, onChange, fmt, color }) {
  return (
    <label style={{ display: "block" }}>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "baseline",
          marginBottom: 5,
        }}
      >
        <span style={{ fontSize: 12.5, color: C.muted, fontWeight: 500 }}>{label}</span>
        <span
          style={{
            fontSize: 13,
            fontWeight: 700,
            color: color || C.ink,
            fontVariantNumeric: "tabular-nums",
            fontFamily: "ui-monospace, 'SF Mono', Menlo, monospace",
          }}
        >
          {fmt(value)}
        </span>
      </div>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(+e.target.value)}
        style={{ width: "100%", accentColor: color || C.accent }}
      />
    </label>
  );
}

function ControlButton({ onClick, children, primary, disabled, ariaLabel }) {
  return (
    <button
      type="button"
      onClick={onClick}
      disabled={disabled}
      aria-label={ariaLabel}
      style={{
        padding: primary ? "9px 18px" : "9px 13px",
        borderRadius: 9,
        border: primary ? "none" : `1.5px solid ${C.border}`,
        background: primary ? C.accent : C.card,
        color: primary ? "#fff" : C.ink,
        fontSize: 13,
        fontWeight: primary ? 600 : 500,
        fontFamily: "inherit",
        cursor: disabled ? "not-allowed" : "pointer",
        opacity: disabled ? 0.4 : 1,
        transition: `transform 140ms ${EASE}, background 160ms ease`,
        boxShadow: primary ? "0 1px 3px rgba(192,86,31,0.3)" : "none",
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

function StatBox({ label, value, sub, color, bg }) {
  return (
    <div style={{ background: bg, borderRadius: 10, padding: "11px 13px", flex: 1, minWidth: 96 }}>
      <div
        style={{
          fontSize: 10,
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
          fontSize: 22,
          fontWeight: 700,
          color,
          lineHeight: 1,
          fontVariantNumeric: "tabular-nums",
        }}
      >
        {value}
      </div>
      {sub && <div style={{ fontSize: 10.5, color, opacity: 0.75, marginTop: 4 }}>{sub}</div>}
    </div>
  );
}

function valueColor(v, vmax) {
  if (vmax <= 1e-6) return C.cold;
  const t = Math.max(0, Math.min(1, v / vmax));
  // interpolate cold paper to terracotta so high value reads warm, matching "reward flows here"
  const cold = [238, 242, 244];
  const warm = [192, 86, 31];
  const ch = cold.map((x, i) => Math.round(x + (warm[i] - x) * t));
  return `rgb(${ch[0]}, ${ch[1]}, ${ch[2]})`;
}

export default function App() {
  const [presetKey, setPresetKey] = useState("detour");
  const [env, setEnv] = useState(() => {
    const p = PRESETS.detour;
    return {
      start: p.start,
      goal: p.goal,
      walls: new Set(p.walls),
      traps: new Set(p.traps),
    };
  });

  const [Q, setQ] = useState(() => zeroQ());
  const [agent, setAgent] = useState(() => PRESETS.detour.start);
  const [episode, setEpisode] = useState(0);
  const [stepInEp, setStepInEp] = useState(0);
  const [epReturn, setEpReturn] = useState(0);
  const [lastEpReturn, setLastEpReturn] = useState(null);
  const [totalSteps, setTotalSteps] = useState(0);

  const [alpha, setAlpha] = useState(0.5);
  const [gamma, setGamma] = useState(0.95);
  const [epsilon, setEpsilon] = useState(1.0);
  const [decay, setDecay] = useState(true);
  const [slip, setSlip] = useState(0);

  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(2);
  const [showArrows, setShowArrows] = useState(true);
  const [selected, setSelected] = useState(null);
  const [editMode, setEditMode] = useState("off");
  const [reduceMotion, setReduceMotion] = useState(false);

  const rngRef = useRef(makeRng(7));
  const effEpsRef = useRef(1.0);
  const envRef = useRef(env);
  const stateRef = useRef({ agent, episode, stepInEp, epReturn, totalSteps });

  useEffect(() => {
    stateRef.current = { agent, episode, stepInEp, epReturn, totalSteps };
  });
  useEffect(() => {
    envRef.current = env;
  }, [env]);

  useEffect(() => {
    if (typeof window === "undefined" || !window.matchMedia) return;
    const mq = window.matchMedia("(prefers-reduced-motion: reduce)");
    const apply = () => setReduceMotion(mq.matches);
    apply();
    mq.addEventListener("change", apply);
    return () => mq.removeEventListener("change", apply);
  }, []);

  const effectiveEpsilon = useMemo(() => {
    if (!decay) return epsilon;
    // exponential schedule anchored at the slider value so the slider stays the ceiling
    return Math.max(0.02, epsilon * Math.pow(0.997, episode));
  }, [epsilon, decay, episode]);

  useEffect(() => {
    effEpsRef.current = effectiveEpsilon;
  }, [effectiveEpsilon]);

  const resetLearning = useCallback((nextEnv) => {
    const e = nextEnv || envRef.current;
    setQ(zeroQ());
    setAgent(e.start);
    setEpisode(0);
    setStepInEp(0);
    setEpReturn(0);
    setLastEpReturn(null);
    setTotalSteps(0);
    setSelected(null);
    setPlaying(false);
    rngRef.current = makeRng(7);
  }, []);

  const applyPreset = useCallback(
    (key) => {
      const p = PRESETS[key];
      const nextEnv = {
        start: p.start,
        goal: p.goal,
        walls: new Set(p.walls),
        traps: new Set(p.traps),
      };
      setPresetKey(key);
      setEnv(nextEnv);
      envRef.current = nextEnv;
      setEditMode("off");
      resetLearning(nextEnv);
    },
    [resetLearning]
  );

  const doAction = useCallback(() => {
    const rng = rngRef.current;
    const e = envRef.current;
    setQ((prevQ) => {
      const st = stateRef.current;
      let s = st.agent;
      if (isTerminalState(s, e.goal, e.traps)) {
        setAgent(e.start);
        setStepInEp(0);
        setEpReturn(0);
        return prevQ;
      }
      const eps = effEpsRef.current;
      let a;
      if (rng() < eps) a = Math.floor(rng() * 4);
      else a = argmaxAction(prevQ[s]);

      const ns = stepEnv(s, a, e, slip, rng);
      const r = rewardOf(ns, e);
      const terminal = isTerminalState(ns, e.goal, e.traps);
      const target = terminal ? r : r + gamma * maxQ(prevQ[ns]);

      const nextQ = prevQ.slice();
      const row = prevQ[s].slice();
      row[a] = prevQ[s][a] + alpha * (target - prevQ[s][a]);
      nextQ[s] = row;

      const newReturn = st.epReturn + r;
      setAgent(ns);
      setTotalSteps((x) => x + 1);

      if (terminal || st.stepInEp + 1 >= MAX_STEPS) {
        setLastEpReturn(newReturn);
        setEpisode((ep) => ep + 1);
        setAgent(e.start);
        setStepInEp(0);
        setEpReturn(0);
      } else {
        setStepInEp((x) => x + 1);
        setEpReturn(newReturn);
      }
      return nextQ;
    });
  }, [alpha, gamma, slip]);

  // run whole episodes synchronously against a working copy, then commit once
  const runEpisodes = useCallback(
    (count) => {
      const rng = rngRef.current;
      const e = envRef.current;
      setPlaying(false);
      setQ((prevQ) => {
        const work = prevQ.map((row) => row.slice());
        let ep = stateRef.current.episode;
        let steps = stateRef.current.totalSteps;
        let lastReturn = null;
        for (let n = 0; n < count; n++) {
          let s = e.start;
          let inEp = 0;
          let ret = 0;
          const eps = decay ? Math.max(0.02, epsilon * Math.pow(0.997, ep)) : epsilon;
          while (!isTerminalState(s, e.goal, e.traps) && inEp < MAX_STEPS) {
            let a;
            if (rng() < eps) a = Math.floor(rng() * 4);
            else a = argmaxAction(work[s]);
            const ns = stepEnv(s, a, e, slip, rng);
            const r = rewardOf(ns, e);
            const terminal = isTerminalState(ns, e.goal, e.traps);
            const target = terminal ? r : r + gamma * maxQ(work[ns]);
            work[s][a] = work[s][a] + alpha * (target - work[s][a]);
            ret += r;
            s = ns;
            inEp++;
            steps++;
          }
          ep++;
          lastReturn = ret;
        }
        setEpisode(ep);
        setTotalSteps(steps);
        setLastEpReturn(lastReturn);
        setAgent(e.start);
        setStepInEp(0);
        setEpReturn(0);
        return work;
      });
    },
    [alpha, gamma, epsilon, decay, slip]
  );

  const stepOneEpisode = useCallback(() => runEpisodes(1), [runEpisodes]);

  useEffect(() => {
    if (!playing) return;
    const interval = reduceMotion ? 16 : Math.max(12, 110 / speed);
    const id = setInterval(() => {
      doAction();
    }, interval);
    return () => clearInterval(id);
  }, [playing, speed, doAction, reduceMotion]);

  const V = useMemo(() => Q.map((q) => maxQ(q)), [Q]);
  const vmax = useMemo(() => {
    let m = 0;
    for (let s = 0; s < N; s++) {
      if (s === env.goal || env.walls.has(s) || env.traps.has(s)) continue;
      if (V[s] > m) m = V[s];
    }
    return m > 0 ? m : 1;
  }, [V, env]);

  const rollout = useMemo(() => greedyRollout(Q, env), [Q, env]);
  const optimalLen = useMemo(() => bfsOptimalLength(env), [env]);
  const greedyPathSet = useMemo(() => new Set(rollout.path), [rollout]);

  const editWorld = useCallback((s, mode) => {
    setEnv((prev) => {
      const walls = new Set(prev.walls);
      const traps = new Set(prev.traps);
      let { start, goal } = prev;
      if (mode === "wall") {
        if (s === start || s === goal) return prev;
        if (walls.has(s)) walls.delete(s);
        else {
          walls.add(s);
          traps.delete(s);
        }
      } else if (mode === "trap") {
        if (s === start || s === goal) return prev;
        if (traps.has(s)) traps.delete(s);
        else {
          traps.add(s);
          walls.delete(s);
        }
      } else if (mode === "goal") {
        if (walls.has(s) || traps.has(s) || s === start) return prev;
        goal = s;
      } else if (mode === "start") {
        if (walls.has(s) || traps.has(s) || s === goal) return prev;
        start = s;
      }
      const next = { start, goal, walls, traps };
      envRef.current = next;
      return next;
    });
    setPresetKey("custom");
    resetLearning();
  }, [resetLearning]);

  const onCellClick = (s) => {
    if (editMode === "off") {
      setSelected((cur) => (cur === s ? null : s));
      return;
    }
    editWorld(s, editMode);
  };

  const cellSize = `min(${Math.floor(420 / COLS)}px, calc((100vw - 56px) / ${COLS}))`;

  const selQ = selected !== null ? Q[selected] : null;
  const selIsSpecial =
    selected !== null &&
    (selected === env.goal || env.walls.has(selected) || env.traps.has(selected));

  const editModes = [
    { id: "off", label: "Inspect" },
    { id: "wall", label: "Wall" },
    { id: "trap", label: "Trap" },
    { id: "goal", label: "Goal" },
    { id: "start", label: "Start" },
  ];

  const presetButtons = [
    { id: "detour", label: "Detour" },
    { id: "twoPaths", label: "Two paths" },
    { id: "open", label: "Open field" },
  ];

  return (
    <div
      style={{
        fontFamily:
          "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif",
        background: C.bg,
        minHeight: "100vh",
        padding: "26px 16px 48px",
        color: C.ink,
      }}
    >
      <style>{`
        .ql-cell:focus-visible {
          outline: 3px solid ${C.accent};
          outline-offset: -3px;
          z-index: 4;
        }
        .ql-grid button { -webkit-tap-highlight-color: transparent; }
        @media (prefers-reduced-motion: reduce) {
          .ql-anim { transition: none !important; }
        }
      `}</style>

      <div style={{ maxWidth: 720, margin: "0 auto" }}>
        <header style={{ marginBottom: 20 }}>
          <div
            style={{
              fontSize: 11,
              letterSpacing: "0.14em",
              textTransform: "uppercase",
              color: C.faint,
              marginBottom: 6,
            }}
          >
            Reinforcement Learning / Tabular control
          </div>
          <h1 style={{ fontSize: 28, fontWeight: 700, margin: 0, letterSpacing: "-0.02em" }}>
            Q-learning on a Gridworld
          </h1>
          <p
            style={{
              color: C.muted,
              fontSize: 14.5,
              lineHeight: 1.6,
              margin: "10px 0 0",
              maxWidth: "64ch",
            }}
          >
            The agent begins with every value at zero and no map of the world. It only feels reward:
            a small cost each step, a big payoff at the goal, a penalty in a trap. Run episodes and
            watch value seep backward from the goal until the greedy arrows form a path.
          </p>
        </header>

        <div
          style={{
            display: "inline-flex",
            background: C.card,
            border: `1px solid ${C.border}`,
            borderRadius: 11,
            padding: 4,
            marginBottom: 16,
            gap: 4,
            flexWrap: "wrap",
          }}
          role="radiogroup"
          aria-label="Choose a gridworld layout"
        >
          {presetButtons.map((p) => {
            const active = presetKey === p.id;
            return (
              <button
                key={p.id}
                type="button"
                role="radio"
                aria-checked={active}
                onClick={() => applyPreset(p.id)}
                style={{
                  padding: "7px 15px",
                  borderRadius: 8,
                  border: "none",
                  background: active ? C.accent : "transparent",
                  color: active ? "#fff" : C.muted,
                  fontSize: 13,
                  fontWeight: active ? 600 : 500,
                  fontFamily: "inherit",
                  cursor: "pointer",
                  transition: "background 160ms ease, color 160ms ease",
                }}
              >
                {p.label}
              </button>
            );
          })}
        </div>

        <div style={{ display: "flex", gap: 9, marginBottom: 16, flexWrap: "wrap" }}>
          <StatBox label="Episodes" value={episode} sub="goal or trap reached" color={C.accent} bg={C.accentSoft} />
          <StatBox
            label="Greedy policy"
            value={rollout.solves ? `${rollout.steps}` : "—"}
            sub={
              rollout.solves
                ? optimalLen > 0 && rollout.steps === optimalLen
                  ? "steps (optimal)"
                  : `steps (best ${optimalLen})`
                : "does not reach goal"
            }
            color={rollout.solves ? C.goal : C.faint}
            bg={rollout.solves ? C.startSoft : C.cold}
          />
          <StatBox
            label="Last return"
            value={lastEpReturn === null ? "—" : lastEpReturn.toFixed(2)}
            sub="reward summed over episode"
            color={C.muted}
            bg={C.bg}
          />
          <StatBox
            label="Epsilon"
            value={effectiveEpsilon.toFixed(2)}
            sub={decay ? "decaying with episodes" : "fixed exploration"}
            color={C.agent}
            bg="#e6edf6"
          />
        </div>

        <div
          className="ql-grid"
          role="grid"
          aria-label={`Gridworld, ${ROWS} rows by ${COLS} columns. Cell color shows learned value; arrows show the greedy action.`}
          style={{
            display: "grid",
            gridTemplateColumns: `repeat(${COLS}, ${cellSize})`,
            gap: 3,
            background: C.border,
            padding: 3,
            borderRadius: 12,
            width: "fit-content",
            margin: "0 auto",
            userSelect: "none",
          }}
        >
          {Array.from({ length: N }, (_, s) => {
            const isStart = s === env.start;
            const isGoal = s === env.goal;
            const isWall = env.walls.has(s);
            const isTrap = env.traps.has(s);
            const isAgent = s === agent && !isGoal;
            const isSelected = s === selected;
            const onGreedyPath = greedyPathSet.has(s) && rollout.solves && !isGoal && !isStart;

            let bg = valueColor(V[s], vmax);
            let fg = V[s] / vmax > 0.55 ? "#fff" : C.muted;
            if (isWall) {
              bg = C.wall;
              fg = "#fff";
            } else if (isGoal) {
              bg = C.goal;
              fg = "#fff";
            } else if (isTrap) {
              bg = C.trap;
              fg = "#fff";
            }

            const greedyA = argmaxAction(Q[s]);
            const hasValue = maxQ(Q[s]) > 1e-6 || Math.min(...Q[s]) < -1e-6;
            const showArrow = showArrows && !isWall && !isGoal && !isTrap && hasValue;

            const label = isStart
              ? "Start cell"
              : isGoal
              ? "Goal cell, reward plus one"
              : isWall
              ? "Wall"
              : isTrap
              ? "Trap cell, reward minus one"
              : `Cell row ${rowOf(s)} column ${colOf(s)}, value ${V[s].toFixed(2)}`;

            return (
              <button
                key={s}
                type="button"
                role="gridcell"
                aria-label={label}
                className="ql-cell ql-anim"
                onClick={() => onCellClick(s)}
                style={{
                  width: cellSize,
                  height: cellSize,
                  border: isSelected ? `2.5px solid ${C.ink}` : "none",
                  padding: 0,
                  borderRadius: 4,
                  background: bg,
                  color: fg,
                  cursor: "pointer",
                  position: "relative",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  fontFamily: "ui-monospace, 'SF Mono', Menlo, monospace",
                  transition: reduceMotion ? "none" : `background 260ms ${EASE}`,
                  overflow: "hidden",
                  zIndex: isSelected ? 3 : 1,
                }}
              >
                {onGreedyPath && (
                  <span
                    aria-hidden="true"
                    style={{
                      position: "absolute",
                      inset: 0,
                      boxShadow: `inset 0 0 0 2px ${C.accent}`,
                      borderRadius: 4,
                      pointerEvents: "none",
                    }}
                  />
                )}
                {showArrow && (
                  <span
                    aria-hidden="true"
                    style={{
                      fontSize: `clamp(14px, ${Math.floor(380 / COLS)}px, 22px)`,
                      lineHeight: 1,
                      fontWeight: 700,
                      opacity: 0.92,
                    }}
                  >
                    {ACTIONS[greedyA].glyph}
                  </span>
                )}
                {isStart && (
                  <span
                    style={{
                      position: "absolute",
                      top: 3,
                      left: 4,
                      fontSize: 9,
                      fontWeight: 700,
                      color: C.start,
                    }}
                  >
                    S
                  </span>
                )}
                {isGoal && <span style={{ fontSize: 13, fontWeight: 700 }}>+1</span>}
                {isTrap && <span style={{ fontSize: 13, fontWeight: 700 }}>−1</span>}
                {!isWall && !isGoal && !isTrap && V[s] !== 0 && (
                  <span
                    style={{
                      position: "absolute",
                      bottom: 2,
                      right: 4,
                      fontSize: 8.5,
                      opacity: 0.75,
                      fontWeight: 600,
                    }}
                  >
                    {V[s].toFixed(2)}
                  </span>
                )}
                {isAgent && (
                  <span
                    aria-hidden="true"
                    className="ql-anim"
                    style={{
                      position: "absolute",
                      width: "46%",
                      height: "46%",
                      borderRadius: "50%",
                      background: C.agent,
                      boxShadow: "0 0 0 3px rgba(31,79,143,0.25)",
                      transition: reduceMotion ? "none" : `transform 120ms ${EASE}`,
                    }}
                  />
                )}
              </button>
            );
          })}
        </div>

        <div
          aria-live="polite"
          style={{
            background: C.card,
            border: `1px solid ${C.border}`,
            borderRadius: 12,
            padding: "13px 15px",
            margin: "14px auto 0",
            maxWidth: 460,
            minHeight: 64,
          }}
        >
          {selected === null ? (
            <div style={{ fontSize: 12.5, color: C.muted, lineHeight: 1.55 }}>
              Click any cell to read its four Q-values, one per action. Q(s, a) is the expected
              discounted future reward of taking action a in state s, then acting greedily ever
              after. The greedy arrow points at the action with the largest Q.
            </div>
          ) : selIsSpecial ? (
            <div style={{ fontSize: 12.5, color: C.muted, lineHeight: 1.55 }}>
              {selected === env.goal
                ? "The goal is terminal. The agent collects +1 and the episode ends, so no actions are taken from here."
                : env.walls.has(selected)
                ? "A wall. The agent cannot enter it; bumping into it wastes a step and leaves the agent in place."
                : "A trap is terminal. The agent collects −1 and the episode ends, which is why nearby values stay low."}
            </div>
          ) : (
            <div>
              <div
                style={{
                  fontSize: 11,
                  color: C.faint,
                  marginBottom: 8,
                  fontFamily: "ui-monospace, 'SF Mono', Menlo, monospace",
                }}
              >
                State ({rowOf(selected)}, {colOf(selected)}) &middot; V = {V[selected].toFixed(3)}
              </div>
              <div style={{ display: "grid", gridTemplateColumns: "repeat(4, 1fr)", gap: 6 }}>
                {ACTIONS.map((act, i) => {
                  const best = argmaxAction(selQ) === i;
                  return (
                    <div
                      key={act.name}
                      style={{
                        textAlign: "center",
                        padding: "7px 4px",
                        borderRadius: 8,
                        background: best ? C.accentSoft : C.bg,
                        border: best ? `1.5px solid ${C.accent}66` : `1px solid ${C.border}`,
                      }}
                    >
                      <div style={{ fontSize: 16, color: best ? C.accent : C.muted, lineHeight: 1 }}>
                        {act.glyph}
                      </div>
                      <div
                        style={{
                          fontSize: 12,
                          fontWeight: 700,
                          color: best ? C.accent : C.ink,
                          marginTop: 4,
                          fontVariantNumeric: "tabular-nums",
                          fontFamily: "ui-monospace, 'SF Mono', Menlo, monospace",
                        }}
                      >
                        {selQ[i].toFixed(2)}
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </div>

        <div
          style={{
            display: "flex",
            gap: 9,
            flexWrap: "wrap",
            margin: "18px 0 14px",
            justifyContent: "center",
          }}
        >
          <ControlButton
            primary
            onClick={() => setPlaying((p) => !p)}
            ariaLabel={playing ? "Pause learning" : "Play learning"}
          >
            {playing ? "Pause" : "Play"}
          </ControlButton>
          <ControlButton onClick={doAction} disabled={playing} ariaLabel="Take one action">
            Step action
          </ControlButton>
          <ControlButton onClick={stepOneEpisode} ariaLabel="Run one full episode">
            Step episode
          </ControlButton>
          <ControlButton onClick={() => runEpisodes(50)} ariaLabel="Run fifty episodes">
            Run 50
          </ControlButton>
          <ControlButton onClick={() => runEpisodes(500)} ariaLabel="Run five hundred episodes">
            Run 500
          </ControlButton>
          <ControlButton onClick={() => resetLearning()} ariaLabel="Reset and zero the Q-table">
            Reset Q
          </ControlButton>
        </div>

        <section
          style={{
            background: C.card,
            border: `1px solid ${C.border}`,
            borderRadius: 14,
            padding: "16px 18px",
            marginBottom: 14,
          }}
        >
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(150px, 1fr))",
              gap: "14px 20px",
            }}
          >
            <Slider
              label="Learning rate α"
              value={alpha}
              min={0.05}
              max={1}
              step={0.05}
              onChange={setAlpha}
              fmt={(v) => v.toFixed(2)}
              color={C.accent}
            />
            <Slider
              label="Discount γ"
              value={gamma}
              min={0.5}
              max={0.99}
              step={0.01}
              onChange={setGamma}
              fmt={(v) => v.toFixed(2)}
              color={C.agent}
            />
            <Slider
              label="Exploration ε"
              value={epsilon}
              min={0}
              max={1}
              step={0.05}
              onChange={setEpsilon}
              fmt={(v) => v.toFixed(2)}
              color={C.goal}
            />
            <Slider
              label="Slip probability"
              value={slip}
              min={0}
              max={0.5}
              step={0.05}
              onChange={setSlip}
              fmt={(v) => `${Math.round(v * 100)}%`}
              color={C.trap}
            />
            <Slider
              label="Speed"
              value={speed}
              min={1}
              max={8}
              step={1}
              onChange={setSpeed}
              fmt={(v) => `${v}x`}
              color={C.muted}
            />
          </div>

          <div
            style={{
              display: "flex",
              gap: 16,
              flexWrap: "wrap",
              marginTop: 16,
              paddingTop: 14,
              borderTop: `1px solid ${C.border}`,
              alignItems: "center",
            }}
          >
            <label
              style={{
                display: "flex",
                alignItems: "center",
                gap: 7,
                cursor: "pointer",
                fontSize: 13,
                color: C.muted,
              }}
            >
              <input
                type="checkbox"
                checked={decay}
                onChange={(e) => setDecay(e.target.checked)}
                style={{ accentColor: C.accent, width: 15, height: 15 }}
              />
              Decay ε over episodes
            </label>
            <label
              style={{
                display: "flex",
                alignItems: "center",
                gap: 7,
                cursor: "pointer",
                fontSize: 13,
                color: C.muted,
              }}
            >
              <input
                type="checkbox"
                checked={showArrows}
                onChange={(e) => setShowArrows(e.target.checked)}
                style={{ accentColor: C.accent, width: 15, height: 15 }}
              />
              Show policy arrows
            </label>
          </div>
        </section>

        <section
          style={{
            background: C.card,
            border: `1px solid ${C.border}`,
            borderRadius: 14,
            padding: "16px 18px",
            marginBottom: 14,
          }}
        >
          <div
            style={{
              fontSize: 11,
              letterSpacing: "0.08em",
              textTransform: "uppercase",
              color: C.muted,
              fontWeight: 700,
              marginBottom: 10,
            }}
          >
            Edit the world
          </div>
          <div style={{ display: "flex", gap: 6, flexWrap: "wrap", marginBottom: 10 }}>
            {editModes.map((m) => {
              const active = editMode === m.id;
              return (
                <button
                  key={m.id}
                  type="button"
                  aria-pressed={active}
                  onClick={() => {
                    setEditMode(m.id);
                    if (m.id !== "off") setSelected(null);
                  }}
                  style={{
                    padding: "6px 13px",
                    borderRadius: 8,
                    border: `1.5px solid ${active ? C.accent : C.border}`,
                    background: active ? C.accentSoft : "transparent",
                    color: active ? C.accent : C.muted,
                    fontSize: 12.5,
                    fontWeight: active ? 700 : 500,
                    fontFamily: "inherit",
                    cursor: "pointer",
                  }}
                >
                  {m.label}
                </button>
              );
            })}
          </div>
          <div style={{ fontSize: 12, color: C.faint, lineHeight: 1.5 }}>
            {editMode === "off"
              ? "Inspect mode: clicking a cell reads its Q-values. Switch to a tool to reshape the world; any edit zeroes the Q-table so learning restarts."
              : editMode === "wall"
              ? "Click cells to add or remove walls."
              : editMode === "trap"
              ? "Click cells to place or clear traps (terminal, reward −1)."
              : editMode === "goal"
              ? "Click a cell to move the goal (terminal, reward +1)."
              : "Click a cell to move the start."}
          </div>
        </section>

        <section
          style={{
            background: C.accentSoft,
            border: `1px solid ${C.accent}33`,
            borderRadius: 14,
            padding: "18px 20px",
          }}
        >
          <h2 style={{ fontSize: 16, fontWeight: 700, margin: "0 0 10px" }}>What to watch for</h2>
          <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
            <div>
              <div style={{ fontSize: 13.5, fontWeight: 700, color: "#7a3a17", marginBottom: 3 }}>
                Value flows backward from reward
              </div>
              <p
                style={{
                  fontSize: 13.5,
                  lineHeight: 1.6,
                  color: "#6b4b32",
                  margin: 0,
                  maxWidth: "66ch",
                }}
              >
                Reset and run episodes. Only cells next to the goal warm up first, because only they
                ever see the +1. Each update carries a slice of that value one cell further back:
                Q(s, a) moves toward r + &gamma; max Q(s', a'), the reward you just got plus the best
                value of where you landed. After enough episodes the warm front reaches the start.
              </p>
            </div>
            <div>
              <div style={{ fontSize: 13.5, fontWeight: 700, color: "#7a3a17", marginBottom: 3 }}>
                Explore too little and you get stuck
              </div>
              <p
                style={{
                  fontSize: 13.5,
                  lineHeight: 1.6,
                  color: "#6b4b32",
                  margin: 0,
                  maxWidth: "66ch",
                }}
              >
                Drop &epsilon; to 0 and reset: the agent always takes its current best guess, which
                early on is an arbitrary tie, so it can lock onto a dead end and never discover the
                goal. Keep &epsilon; high, or leave decay on so it explores first and exploits later,
                and the greedy path turns optimal.
              </p>
            </div>
            <div>
              <div style={{ fontSize: 13.5, fontWeight: 700, color: "#7a3a17", marginBottom: 3 }}>
                Gamma sets how far the agent looks
              </div>
              <p
                style={{
                  fontSize: 13.5,
                  lineHeight: 1.6,
                  color: "#6b4b32",
                  margin: 0,
                  maxWidth: "66ch",
                }}
              >
                With &gamma; near its floor the agent leans on immediate reward, so distant cells stay
                faint and a long route is hard to value. Push &gamma; toward 1 and value carries across
                the whole grid, letting the agent trade many small step costs for the payoff far away.
              </p>
            </div>
          </div>
        </section>

        <p
          style={{
            textAlign: "center",
            fontSize: 11.5,
            color: C.faint,
            marginTop: 20,
            lineHeight: 1.6,
            maxWidth: "70ch",
            marginLeft: "auto",
            marginRight: "auto",
          }}
        >
          Tabular Q-learning, four actions, step reward {STEP_REWARD}, goal +{GOAL_REWARD}, trap{" "}
          {TRAP_REWARD}. Cell shade is V(s) = max over actions of Q(s, a); the terracotta outline
          traces the current greedy path. Update rule: Q(s, a) &larr; Q(s, a) + &alpha;[r + &gamma;
          max Q(s', a') &minus; Q(s, a)].
        </p>
      </div>
    </div>
  );
}
