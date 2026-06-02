import { useState, useEffect, useMemo, useRef, useCallback } from "react";

export const meta = {
  title: "Monty Hall",
  category: "Probability",
  description:
    "After the host opens a goat door it feels like a coin flip, so switching cannot matter. Play the real game with a host who knows where the prize is, then run thousands of games and watch staying settle at 1/3 while switching settles at 2/3.",
  date: "2026-03-02",
  tags: ["monty-hall", "probability", "conditional-probability", "simulation"],
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
  stay: "#9a7020",
  staySoft: "#f5edd8",
  switch: "#2e6f8e",
  switchSoft: "#e6eff3",
  good: "#2f7d53",
  goodSoft: "#e6f1ea",
  bad: "#a8453f",
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

function pickInt(rng, n) {
  return Math.floor(rng() * n);
}

// Plays one full game under the real host rule and returns whether the strategy won.
// The host never opens the player's door and never opens the prize, so it opens
// every other door, leaving the player's pick plus exactly one other door closed.
function playGame(rng, n, doSwitch) {
  const prize = pickInt(rng, n);
  const pick = pickInt(rng, n);
  let remaining;
  if (pick === prize) {
    const goats = [];
    for (let d = 0; d < n; d++) {
      if (d !== pick) goats.push(d);
    }
    remaining = goats[pickInt(rng, goats.length)];
  } else {
    // when the first pick is wrong the only non-pick non-prize doors get opened,
    // so the prize is forced to be the door left closed
    remaining = prize;
  }
  const finalDoor = doSwitch ? remaining : pick;
  return finalDoor === prize;
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

function Btn({ children, onClick, disabled, primary, tone, ariaLabel, ariaPressed }) {
  const bg = primary ? (tone || C.accent) : "transparent";
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      type="button"
      aria-label={ariaLabel}
      aria-pressed={ariaPressed}
      style={{
        padding: primary ? "8px 16px" : "7px 13px",
        borderRadius: 8,
        border: primary ? "none" : `1px solid ${C.border}`,
        background: bg,
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

function Stat({ label, value, sub, color, bg }) {
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
      <div style={{ fontSize: 22, fontWeight: 700, color, lineHeight: 1.05 }}>{value}</div>
      {sub && <div style={{ fontSize: 11, color: C.muted, marginTop: 4 }}>{sub}</div>}
    </div>
  );
}

function GoatGlyph({ size = 30, color }) {
  return (
    <svg width={size} height={size} viewBox="0 0 32 32" fill="none" aria-hidden="true">
      <path
        d="M7 19c0-4 2-7 5-7 1.6 0 2.4 1 4 1s2.4-1 4-1c3 0 5 3 5 7 0 3-2 6-9 6s-9-3-9-6z"
        fill={color}
        opacity="0.16"
      />
      <path
        d="M8 12c-1.5-1-2.5-3-2-5 1.5.5 3 1.5 3.5 3M24 12c1.5-1 2.5-3 2-5-1.5.5-3 1.5-3.5 3"
        stroke={color}
        strokeWidth="1.6"
        strokeLinecap="round"
      />
      <path
        d="M7 18c0-4 2.2-6.5 5-6.5 1.6 0 2.4.8 4 .8s2.4-.8 4-.8c2.8 0 5 2.5 5 6.5 0 3.2-2.4 5.8-9 5.8S7 21.2 7 18z"
        stroke={color}
        strokeWidth="1.6"
        strokeLinejoin="round"
      />
      <circle cx="13" cy="17" r="1.1" fill={color} />
      <circle cx="19" cy="17" r="1.1" fill={color} />
      <path d="M14.5 21h3" stroke={color} strokeWidth="1.4" strokeLinecap="round" />
    </svg>
  );
}

function CarGlyph({ size = 30, color }) {
  return (
    <svg width={size} height={size} viewBox="0 0 32 32" fill="none" aria-hidden="true">
      <path
        d="M5 21l1.5-5c.4-1.4 1.6-2.4 3-2.4h13c1.4 0 2.6 1 3 2.4L30 21v3a1 1 0 0 1-1 1h-2a1 1 0 0 1-1-1v-1H9v1a1 1 0 0 1-1 1H6a1 1 0 0 1-1-1z"
        fill={color}
        opacity="0.16"
      />
      <path
        d="M5 21l1.5-5c.4-1.4 1.6-2.4 3-2.4h13c1.4 0 2.6 1 3 2.4L29 21M5 21h24M5 21v3a1 1 0 0 0 1 1h2a1 1 0 0 0 1-1v-1h16v1a1 1 0 0 0 1 1h2a1 1 0 0 0 1-1v-3"
        stroke={color}
        strokeWidth="1.6"
        strokeLinejoin="round"
        strokeLinecap="round"
      />
      <circle cx="10" cy="21" r="1.4" fill={color} />
      <circle cx="22" cy="21" r="1.4" fill={color} />
    </svg>
  );
}

const DOOR_STATE = { closed: "closed", open: "open", chosen: "chosen", winner: "winner" };

function Door({ index, content, state, picked, onPick, disabled, reduce, label }) {
  const isOpen = state === DOOR_STATE.open || state === DOOR_STATE.winner;
  const isWinner = state === DOOR_STATE.winner;
  const glyphColor = content === "car" ? C.accent : C.muted;
  let border = `2px solid ${C.border}`;
  let bg = "#fbf8f4";
  if (picked) border = `2px solid ${C.accent}`;
  if (isWinner) {
    border = `2px solid ${C.good}`;
    bg = C.goodSoft;
  } else if (isOpen) {
    bg = C.faint;
  }
  return (
    <button
      type="button"
      onClick={() => !disabled && onPick(index)}
      disabled={disabled}
      aria-label={label}
      style={{
        position: "relative",
        flex: "1 1 0",
        minWidth: 0,
        aspectRatio: "3 / 4",
        borderRadius: 12,
        border,
        background: bg,
        cursor: disabled ? "default" : "pointer",
        padding: 0,
        overflow: "hidden",
        fontFamily: "inherit",
        transition: reduce
          ? "none"
          : `transform 220ms ${EASE}, border-color 200ms ${EASE}, background 200ms ${EASE}`,
        transform: picked && !isOpen && !reduce ? "translateY(-4px)" : "translateY(0)",
        boxShadow: picked && !isOpen ? `0 6px 16px ${C.accent}22` : "none",
      }}
      onPointerDown={(e) => {
        if (!disabled && !reduce) e.currentTarget.style.transform = "scale(0.98)";
      }}
      onPointerUp={(e) => {
        if (!disabled && !reduce)
          e.currentTarget.style.transform = picked && !isOpen ? "translateY(-4px)" : "translateY(0)";
      }}
      onPointerLeave={(e) => {
        if (!disabled && !reduce)
          e.currentTarget.style.transform = picked && !isOpen ? "translateY(-4px)" : "translateY(0)";
      }}
    >
      <div
        style={{
          position: "absolute",
          inset: 0,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          opacity: isOpen ? 1 : 0,
          transition: reduce ? "none" : `opacity 260ms ${EASE} 60ms`,
        }}
      >
        {content === "car" ? <CarGlyph color={glyphColor} /> : <GoatGlyph color={glyphColor} />}
      </div>
      <div
        style={{
          position: "absolute",
          inset: 0,
          background: isWinner ? C.goodSoft : "#f3ede4",
          borderRadius: 10,
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
          gap: 6,
          transformOrigin: "left center",
          transform: isOpen ? "perspective(520px) rotateY(-105deg)" : "rotateY(0deg)",
          opacity: isOpen ? 0 : 1,
          transition: reduce
            ? "opacity 160ms ease"
            : `transform 440ms ${EASE}, opacity 200ms ease 240ms`,
          backfaceVisibility: "hidden",
        }}
      >
        <div
          style={{
            width: 7,
            height: 7,
            borderRadius: "50%",
            background: picked ? C.accent : C.muted,
            alignSelf: "flex-end",
            marginRight: "20%",
          }}
        />
        <div
          style={{
            fontSize: 30,
            fontWeight: 700,
            color: picked ? C.accent : C.muted,
            opacity: 0.5,
          }}
        >
          {index + 1}
        </div>
      </div>
    </button>
  );
}

const PHASE = { pick: "pick", revealed: "revealed", done: "done" };

function buildRound(rng, n) {
  const prize = pickInt(rng, n);
  return { prize, n };
}

function RunningBar({ label, rate, theory, color, bg, wins, games }) {
  const pct = Math.max(0, Math.min(1, rate));
  const theoryPct = Math.max(0, Math.min(1, theory));
  return (
    <div style={{ flex: 1, minWidth: 150 }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "baseline", marginBottom: 6 }}>
        <span style={{ fontSize: 12, fontWeight: 700, color }}>{label}</span>
        <span style={{ fontSize: 20, fontWeight: 700, color, fontVariantNumeric: "tabular-nums" }}>
          {games > 0 ? `${(rate * 100).toFixed(1)}%` : "—"}
        </span>
      </div>
      <div
        role="img"
        aria-label={`${label} win rate ${(rate * 100).toFixed(1)} percent, theoretical ${(theory * 100).toFixed(1)} percent`}
        style={{
          position: "relative",
          height: 30,
          background: bg,
          borderRadius: 8,
          overflow: "hidden",
        }}
      >
        <div
          style={{
            position: "absolute",
            inset: 0,
            width: `${pct * 100}%`,
            background: color,
            opacity: 0.85,
            transition: `width 320ms ${EASE}`,
          }}
        />
        <div
          style={{
            position: "absolute",
            top: -2,
            bottom: -2,
            left: `${theoryPct * 100}%`,
            width: 2,
            background: C.ink,
          }}
          title={`theory ${(theory * 100).toFixed(1)}%`}
        />
      </div>
      <div style={{ fontSize: 11, color: C.muted, marginTop: 5, fontVariantNumeric: "tabular-nums" }}>
        {games > 0 ? `${wins.toLocaleString()} wins of ${games.toLocaleString()}` : "no games yet"}
        {" · theory "}
        <b style={{ color }}>{(theory * 100).toFixed(1)}%</b>
      </div>
    </div>
  );
}

// Lays out the three equally likely worlds for a 3-door game given the player picked door 1,
// tracing what the host must open and what stay vs switch yields in each world.
function ThreeCases({ reduce }) {
  const cases = [
    { prize: 0, opens: 1, leaves: 2 },
    { prize: 1, opens: 2, leaves: 1 },
    { prize: 2, opens: 1, leaves: 2 },
  ];
  return (
    <div style={{ display: "grid", gap: 10 }}>
      {cases.map((cs, i) => {
        const stayWins = cs.prize === 0;
        const cell = (idx) => {
          const isPrize = idx === cs.prize;
          const isOpened = idx === cs.opens;
          const isPick = idx === 0;
          let bg = "#fbf8f4";
          let bd = C.border;
          if (isOpened) {
            bg = C.faint;
            bd = C.border;
          } else if (isPrize) {
            bg = C.accentSoft;
            bd = `${C.accent}55`;
          }
          return (
            <div
              key={idx}
              style={{
                flex: 1,
                borderRadius: 8,
                border: `1.5px solid ${bd}`,
                background: bg,
                padding: "8px 4px",
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                gap: 3,
                opacity: isOpened ? 0.7 : 1,
              }}
            >
              <div style={{ fontSize: 10, color: C.muted, fontWeight: 700 }}>{idx + 1}</div>
              {isOpened ? (
                <GoatGlyph size={22} color={C.muted} />
              ) : isPrize ? (
                <CarGlyph size={22} color={C.accent} />
              ) : (
                <GoatGlyph size={22} color={C.muted} />
              )}
              <div style={{ fontSize: 9, color: isPick ? C.accent : C.muted, fontWeight: isPick ? 700 : 500 }}>
                {isPick ? "your pick" : isOpened ? "host opens" : "stays shut"}
              </div>
            </div>
          );
        };
        return (
          <div
            key={i}
            style={{
              display: "grid",
              gridTemplateColumns: "1fr auto",
              gap: 10,
              alignItems: "center",
              padding: 10,
              borderRadius: 10,
              background: C.card,
              border: `1px solid ${C.border}`,
            }}
          >
            <div>
              <div style={{ fontSize: 11, color: C.muted, marginBottom: 6 }}>
                Prize behind door {cs.prize + 1}{" "}
                <span style={{ color: C.muted }}>(probability 1/3)</span>
              </div>
              <div style={{ display: "flex", gap: 6 }}>{[0, 1, 2].map((idx) => cell(idx))}</div>
            </div>
            <div style={{ display: "flex", flexDirection: "column", gap: 6, minWidth: 92 }}>
              <Outcome label="Stay" win={stayWins} reduce={reduce} />
              <Outcome label="Switch" win={!stayWins} reduce={reduce} />
            </div>
          </div>
        );
      })}
    </div>
  );
}

function Outcome({ label, win }) {
  return (
    <div
      style={{
        display: "flex",
        alignItems: "center",
        justifyContent: "space-between",
        gap: 6,
        padding: "4px 9px",
        borderRadius: 6,
        background: win ? C.goodSoft : C.faint,
        border: `1px solid ${win ? `${C.good}44` : C.border}`,
      }}
    >
      <span style={{ fontSize: 11, fontWeight: 600, color: win ? C.good : C.muted }}>{label}</span>
      <span style={{ fontSize: 11, fontWeight: 700, color: win ? C.good : C.muted }}>
        {win ? "wins" : "loses"}
      </span>
    </div>
  );
}

export default function App() {
  const reduce = usePrefersReducedMotion();

  const [seed, setSeed] = useState(20260518);
  const playRng = useRef(mulberry32(20260518));
  const [round, setRound] = useState(() => buildRound(mulberry32(20260518), 3));
  const [phase, setPhase] = useState(PHASE.pick);
  const [pick, setPick] = useState(null);
  const [opened, setOpened] = useState(null);
  const [finalDoor, setFinalDoor] = useState(null);
  const [won, setWon] = useState(null);
  const [tally, setTally] = useState({ stayWins: 0, stayGames: 0, switchWins: 0, switchGames: 0 });

  const [simN, setSimN] = useState(3);
  const [sim, setSim] = useState({ stayWins: 0, stayGames: 0, switchWins: 0, switchGames: 0 });
  const simRng = useRef(mulberry32(20260518 ^ 0x5bd1e995));
  const [running, setRunning] = useState(false);

  function freshPlay(useSeed) {
    playRng.current = mulberry32(useSeed >>> 0);
    setRound(buildRound(playRng.current, 3));
    setPhase(PHASE.pick);
    setPick(null);
    setOpened(null);
    setFinalDoor(null);
    setWon(null);
  }

  function handlePick(doorIdx) {
    if (phase !== PHASE.pick) return;
    const { prize } = round;
    const goats = [];
    for (let d = 0; d < 3; d++) {
      if (d !== doorIdx && d !== prize) goats.push(d);
    }
    const hostOpens = goats[pickInt(playRng.current, goats.length)];
    setPick(doorIdx);
    setOpened(hostOpens);
    setPhase(PHASE.revealed);
  }

  function decide(doSwitch) {
    if (phase !== PHASE.revealed) return;
    const other = [0, 1, 2].find((d) => d !== pick && d !== opened);
    const chosen = doSwitch ? other : pick;
    const didWin = chosen === round.prize;
    setFinalDoor(chosen);
    setWon(didWin);
    setPhase(PHASE.done);
    setTally((t) => ({
      stayWins: t.stayWins + (!doSwitch && didWin ? 1 : 0),
      stayGames: t.stayGames + (!doSwitch ? 1 : 0),
      switchWins: t.switchWins + (doSwitch && didWin ? 1 : 0),
      switchGames: t.switchGames + (doSwitch ? 1 : 0),
    }));
  }

  function nextRound() {
    const nextSeed = (playRng.current() * 0xffffffff) >>> 0;
    freshPlay(nextSeed);
  }

  function runGames(count) {
    setRunning(true);
    const rng = simRng.current;
    let { stayWins, stayGames, switchWins, switchGames } = sim;
    for (let i = 0; i < count; i++) {
      if (playGame(rng, simN, false)) stayWins++;
      stayGames++;
      if (playGame(rng, simN, true)) switchWins++;
      switchGames++;
    }
    setSim({ stayWins, stayGames, switchWins, switchGames });
    setRunning(false);
  }

  function resetSim() {
    simRng.current = mulberry32((seed ^ 0x5bd1e995) >>> 0);
    setSim({ stayWins: 0, stayGames: 0, switchWins: 0, switchGames: 0 });
  }

  function changeN(nextN) {
    setSimN(nextN);
    simRng.current = mulberry32((seed ^ 0x5bd1e995) >>> 0);
    setSim({ stayWins: 0, stayGames: 0, switchWins: 0, switchGames: 0 });
  }

  function reseed() {
    const next = (Math.floor(Math.random() * 0xffffffff)) >>> 0;
    setSeed(next);
    freshPlay(next);
    simRng.current = mulberry32((next ^ 0x5bd1e995) >>> 0);
    setSim({ stayWins: 0, stayGames: 0, switchWins: 0, switchGames: 0 });
    setTally({ stayWins: 0, stayGames: 0, switchWins: 0, switchGames: 0 });
  }

  const stayRate = sim.stayGames > 0 ? sim.stayWins / sim.stayGames : 0;
  const switchRate = sim.switchGames > 0 ? sim.switchWins / sim.switchGames : 0;
  const staTheory = 1 / simN;
  const swTheory = (simN - 1) / simN;

  const manualStayRate = tally.stayGames > 0 ? tally.stayWins / tally.stayGames : null;
  const manualSwitchRate = tally.switchGames > 0 ? tally.switchWins / tally.switchGames : null;

  const otherDoor = useMemo(
    () => (pick != null && opened != null ? [0, 1, 2].find((d) => d !== pick && d !== opened) : null),
    [pick, opened]
  );

  function doorState(idx) {
    if (phase === PHASE.pick) return idx === pick ? DOOR_STATE.chosen : DOOR_STATE.closed;
    if (idx === opened) return DOOR_STATE.open;
    if (phase === PHASE.done) {
      if (idx === round.prize) return DOOR_STATE.winner;
      return DOOR_STATE.open;
    }
    return DOOR_STATE.closed;
  }

  function doorContent(idx) {
    return idx === round.prize ? "car" : "goat";
  }

  let statusText;
  if (phase === PHASE.pick) {
    statusText = "Pick a door. One hides a car, two hide goats.";
  } else if (phase === PHASE.revealed) {
    statusText = `You picked door ${pick + 1}. The host, who knows where the car is, opened door ${
      opened + 1
    } to show a goat. Door ${otherDoor + 1} is still shut. Stay or switch?`;
  } else {
    statusText = won
      ? `The car was behind door ${round.prize + 1}. You ${
          finalDoor === round.prize ? "landed on it" : "missed"
        }. You won.`
      : `The car was behind door ${round.prize + 1}. You picked door ${finalDoor + 1}. No car this time.`;
  }

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
            Probability · Conditional reasoning
          </div>
          <h1 style={{ fontSize: 25, fontWeight: 700, margin: 0, textWrap: "balance" }}>
            The Monty Hall Problem
          </h1>
          <p
            style={{
              color: C.ink,
              fontSize: 13.5,
              margin: "8px 0 0",
              lineHeight: 1.65,
              maxWidth: "66ch",
              textWrap: "pretty",
            }}
          >
            Three doors, one car, two goats. You pick a door. The host, who knows what is behind
            each one, opens a different door to reveal a goat, then offers you the choice to stay or
            switch. Two doors are left, so it feels like a coin flip and switching should not matter.
            That intuition is wrong, and the rest of this page is built to show you exactly why.
          </p>
        </div>

        <Card style={{ marginBottom: 16 }}>
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "baseline",
              marginBottom: 14,
              flexWrap: "wrap",
              gap: 8,
            }}
          >
            <div style={{ fontSize: 14, fontWeight: 700 }}>Play a round</div>
            <div style={{ fontSize: 11, color: C.muted }}>
              Seeded placement · seed {seed}
            </div>
          </div>

          <div style={{ display: "flex", gap: 10, marginBottom: 16 }}>
            {[0, 1, 2].map((idx) => (
              <Door
                key={idx}
                index={idx}
                content={doorContent(idx)}
                state={doorState(idx)}
                picked={idx === pick}
                onPick={handlePick}
                disabled={phase !== PHASE.pick}
                reduce={reduce}
                label={
                  phase === PHASE.pick
                    ? `Pick door ${idx + 1}`
                    : `Door ${idx + 1}, ${doorState(idx) === DOOR_STATE.closed ? "closed" : doorContent(idx) === "car" ? "car" : "goat"}`
                }
              />
            ))}
          </div>

          <div
            aria-live="polite"
            style={{
              minHeight: 44,
              fontSize: 12.5,
              lineHeight: 1.55,
              color: C.ink,
              background: phase === PHASE.done ? (won ? C.goodSoft : C.faint) : C.faint,
              border: `1px solid ${phase === PHASE.done && won ? `${C.good}44` : C.border}`,
              borderRadius: 10,
              padding: "10px 14px",
              marginBottom: 14,
              transition: reduce ? "none" : `background 220ms ${EASE}`,
            }}
          >
            {phase === PHASE.done && (
              <b style={{ color: won ? C.good : C.bad, marginRight: 5 }}>{won ? "Won." : "Lost."}</b>
            )}
            {statusText}
          </div>

          <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
            {phase === PHASE.revealed && (
              <>
                <Btn primary tone={C.stay} onClick={() => decide(false)} ariaLabel={`Stay with door ${pick + 1}`}>
                  Stay with door {pick + 1}
                </Btn>
                <Btn primary tone={C.switch} onClick={() => decide(true)} ariaLabel={`Switch to door ${otherDoor + 1}`}>
                  Switch to door {otherDoor + 1}
                </Btn>
              </>
            )}
            {phase === PHASE.done && (
              <Btn primary onClick={nextRound} ariaLabel="Play another round">
                Play again
              </Btn>
            )}
            {phase === PHASE.pick && pick == null && (
              <div style={{ fontSize: 12, color: C.muted, alignSelf: "center" }}>
                Click a door above to begin.
              </div>
            )}
          </div>

          {(tally.stayGames > 0 || tally.switchGames > 0) && (
            <div
              style={{
                marginTop: 16,
                paddingTop: 14,
                borderTop: `1px solid ${C.border}`,
                display: "flex",
                gap: 10,
                flexWrap: "wrap",
              }}
            >
              <Stat
                label="Your stays"
                value={manualStayRate == null ? "—" : `${Math.round(manualStayRate * 100)}%`}
                sub={`${tally.stayWins} won of ${tally.stayGames}`}
                color={C.stay}
                bg={C.staySoft}
              />
              <Stat
                label="Your switches"
                value={manualSwitchRate == null ? "—" : `${Math.round(manualSwitchRate * 100)}%`}
                sub={`${tally.switchWins} won of ${tally.switchGames}`}
                color={C.switch}
                bg={C.switchSoft}
              />
            </div>
          )}
        </Card>

        <Card style={{ marginBottom: 16 }}>
          <div style={{ fontSize: 14, fontWeight: 700, marginBottom: 4 }}>Why it is not a coin flip</div>
          <p
            style={{
              fontSize: 12.5,
              color: C.muted,
              lineHeight: 1.6,
              margin: "0 0 14px",
              maxWidth: "66ch",
            }}
          >
            Say you always pick door 1. The car is equally likely to sit behind any door, so there
            are three equally likely worlds. The host is not free: it must open a door that is not
            yours and not the car. Trace each world and read off what stay and switch give you.
          </p>
          <ThreeCases reduce={reduce} />
          <p
            style={{
              fontSize: 12.5,
              color: C.ink,
              lineHeight: 1.65,
              margin: "14px 0 0",
              maxWidth: "66ch",
              textWrap: "pretty",
            }}
          >
            Staying wins only in the one world where your first guess was already right, which is 1
            in 3. Switching wins in the other two worlds, where your first guess was wrong and the
            host was forced to clear away the only other goat, leaving the car behind the door you
            switch to. That is 2 in 3. The opened door is not new randomness: the host knew where
            the car was, so its choice carries information about the doors you did not pick.
          </p>
        </Card>

        <Card style={{ marginBottom: 16 }}>
          <div
            style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "baseline",
              flexWrap: "wrap",
              gap: 8,
              marginBottom: 4,
            }}
          >
            <div style={{ fontSize: 14, fontWeight: 700 }}>Let the games settle it</div>
            <div style={{ fontSize: 11, color: C.muted }}>thousands of simulated rounds</div>
          </div>
          <p
            style={{
              fontSize: 12.5,
              color: C.muted,
              lineHeight: 1.6,
              margin: "0 0 16px",
              maxWidth: "66ch",
            }}
          >
            Each game runs the full rule: random car, random first pick, host opens goats it is
            allowed to open, then a fixed strategy decides. Run enough of them and the two rates pull
            apart toward 1/{simN} and {simN - 1}/{simN}. The black tick marks the theoretical value.
          </p>

          <div style={{ display: "flex", gap: 16, flexWrap: "wrap", marginBottom: 16 }}>
            <RunningBar
              label="Always stay"
              rate={stayRate}
              theory={staTheory}
              color={C.stay}
              bg={C.staySoft}
              wins={sim.stayWins}
              games={sim.stayGames}
            />
            <RunningBar
              label="Always switch"
              rate={switchRate}
              theory={swTheory}
              color={C.switch}
              bg={C.switchSoft}
              wins={sim.switchWins}
              games={sim.switchGames}
            />
          </div>

          <div style={{ display: "flex", gap: 8, flexWrap: "wrap", alignItems: "center" }}>
            <span style={{ fontSize: 12, color: C.muted, marginRight: 2 }}>Run</span>
            {[1, 100, 1000, 10000].map((count) => (
              <Btn
                key={count}
                onClick={() => runGames(count)}
                disabled={running}
                ariaLabel={`Run ${count} games per strategy`}
              >
                +{count.toLocaleString()}
              </Btn>
            ))}
            <Btn onClick={() => resetSim()} ariaLabel="Reset the simulation counts">
              Reset counts
            </Btn>
          </div>

          <div
            style={{
              marginTop: 16,
              paddingTop: 16,
              borderTop: `1px solid ${C.border}`,
            }}
          >
            <label
              htmlFor="ndoors"
              style={{
                display: "flex",
                justifyContent: "space-between",
                fontSize: 12,
                color: C.muted,
                marginBottom: 6,
              }}
            >
              <span>Number of doors (host opens all but one of the rest)</span>
              <span style={{ fontWeight: 700, color: C.accent }}>{simN}</span>
            </label>
            <input
              id="ndoors"
              type="range"
              min={3}
              max={100}
              value={simN}
              onChange={(e) => changeN(+e.target.value)}
              style={{ width: "100%", accentColor: C.accent }}
            />
            <p style={{ fontSize: 12, color: C.ink, lineHeight: 1.6, margin: "10px 0 0", maxWidth: "66ch" }}>
              With <b>{simN}</b> doors your first pick is right only{" "}
              <b style={{ color: C.stay }}>{(staTheory * 100).toFixed(simN > 20 ? 1 : 0)}%</b> of
              the time. The host then opens every goat door except one, so switching collects the
              whole rest of the probability:{" "}
              <b style={{ color: C.switch }}>{(swTheory * 100).toFixed(simN > 20 ? 1 : 0)}%</b>. Push
              the slider to 100 and the gap is impossible to miss, which is the cleanest way to feel
              why switching wins.
            </p>
          </div>
        </Card>

        <Card style={{ background: C.accentSoft, border: `1px solid ${C.accent}22`, marginBottom: 16 }}>
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
            The mistake, named
          </div>
          <div style={{ fontSize: 13, color: C.ink, lineHeight: 1.7, display: "grid", gap: 10 }}>
            <p style={{ margin: 0 }}>
              <b>The 50/50 trap.</b> Two doors remain, so it looks like even odds. That would be
              true if a door had been opened at random and happened to show a goat. It was not. The
              host always reveals a goat and never touches your door, so the reveal tells you nothing
              new about your own door: it was a 1 in 3 guess when you made it and it stays 1 in 3.
            </p>
            <p style={{ margin: 0 }}>
              <b>Where the missing 1/3 goes.</b> The other two doors started with 2/3 of the
              probability between them. The host opens one of those two and shows it is empty, so its
              whole share collapses onto the single door left standing. Switch and you claim that
              2/3.
            </p>
            <p style={{ margin: 0 }}>
              <b>The rule is everything.</b> If the host did not know where the car was and opened a
              random door, the puzzle really would be 50/50 on the games where a goat happened to
              appear. The asymmetry comes entirely from a host who is constrained to avoid the car
              and avoid your door.
            </p>
          </div>
        </Card>

        <div style={{ display: "flex", justifyContent: "center", marginBottom: 14 }}>
          <Btn onClick={reseed} ariaLabel="Reseed the prize placement and clear all counts">
            Reseed and clear all
          </Btn>
        </div>

        <div style={{ textAlign: "center", fontSize: 11, color: C.muted, lineHeight: 1.6 }}>
          Prize placement comes from a seeded mulberry32 generator, not Math.random, so a given seed
          replays the same games. Win rates are counted from the simulated rounds.
        </div>
      </div>
    </div>
  );
}
