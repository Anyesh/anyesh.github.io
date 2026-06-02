import { useState, useEffect, useRef, useCallback } from "react";

export const meta = {
  title: "Raft: Leader Election & Replication",
  category: "Distributed Systems",
  description:
    "Five nodes have to agree on one log, even when servers crash. Kill the leader and watch randomized timeouts elect a new one, replicate to a majority, and recover without losing a committed entry.",
  date: "2026-03-11",
  tags: ["raft", "consensus", "distributed-systems", "replication"],
};

const C = {
  bg: "#f7f5f2",
  card: "#ffffff",
  border: "#e4dfd8",
  ink: "#1c1a17",
  muted: "#857c72",
  accent: "#c0561f",
  accentSoft: "#f6ece5",
  faint: "#efeae3",
};

const ROLE = {
  follower: { label: "Follower", color: "#3f5f9e", soft: "#e7ecf6" },
  candidate: { label: "Candidate", color: "#b8902a", soft: "#f6efd9" },
  leader: { label: "Leader", color: "#c0561f", soft: "#f6ece5" },
  dead: { label: "Down", color: "#9a938a", soft: "#efeae3" },
};

const EASE = "cubic-bezier(0.23, 1, 0.32, 1)";
const N = 5;
const MAJORITY = Math.floor(N / 2) + 1;
const TICK_MS = 100;
const ELECTION_MIN = 1500;
const ELECTION_SPAN = 1500;
const HEARTBEAT_INTERVAL = 600;
const RPC_FLIGHT = 350;

function makeRng(seed) {
  let s = seed >>> 0;
  return function next() {
    s ^= s << 13;
    s ^= s >>> 17;
    s ^= s << 5;
    s >>>= 0;
    return s / 0x100000000;
  };
}

function electionTimeout(rng, nodeId) {
  const jitter = (nodeId * 211 + 97) % 400;
  return Math.round(ELECTION_MIN + jitter + rng() * (ELECTION_SPAN - 400));
}

function initialState(seed) {
  const rng = makeRng(seed);
  const nodes = [];
  for (let i = 0; i < N; i++) {
    nodes.push({
      id: i,
      role: "follower",
      currentTerm: 0,
      votedFor: null,
      log: [],
      commitIndex: 0,
      timeoutAt: electionTimeout(rng, i),
      timer: 0,
      votesGranted: new Set(),
      nextHeartbeatIn: 0,
    });
  }
  return {
    nodes,
    clock: 0,
    rngState: rng.__peek ? rng.__peek() : seed,
    messages: [],
    nextCommand: 1,
    narrative: {
      phase: "boot",
      text: "Five nodes start as followers in term 0. Each waits a randomized election timeout before assuming the leader is gone. The shortest timer fires first.",
    },
    seed,
  };
}

function cloneNode(n) {
  return {
    ...n,
    log: n.log.map((e) => ({ ...e })),
    votesGranted: new Set(n.votesGranted),
  };
}

function lastLogTerm(node) {
  return node.log.length ? node.log[node.log.length - 1].term : 0;
}

function lastLogIndex(node) {
  return node.log.length;
}

function logIsUpToDate(candidate, voter) {
  const cTerm = lastLogTerm(candidate);
  const vTerm = lastLogTerm(voter);
  if (cTerm !== vTerm) return cTerm > vTerm;
  return lastLogIndex(candidate) >= lastLogIndex(voter);
}

function step(prev) {
  const rng = makeRng((prev.seedCursor ?? prev.seed) + prev.clock * 2654435761);
  const nodes = prev.nodes.map(cloneNode);
  let messages = prev.messages
    .map((m) => ({ ...m }))
    .filter((m) => m.arriveAt > prev.clock - RPC_FLIGHT * 2);
  const clock = prev.clock + TICK_MS;
  let narrative = prev.narrative;

  const deliver = messages.filter((m) => !m.delivered && m.arriveAt <= clock);
  for (const msg of deliver) {
    msg.delivered = true;
    const from = nodes[msg.from];
    const to = nodes[msg.to];
    if (!to || to.role === "dead") continue;

    if (msg.type === "RequestVote") {
      if (msg.term > to.currentTerm) {
        to.currentTerm = msg.term;
        to.votedFor = null;
        to.role = "follower";
        to.votesGranted = new Set();
      }
      let granted = false;
      const candidateSnapshot = { log: msg.candLog };
      if (
        msg.term === to.currentTerm &&
        (to.votedFor === null || to.votedFor === msg.from) &&
        logIsUpToDate(candidateSnapshot, to)
      ) {
        granted = true;
        to.votedFor = msg.from;
        to.timer = 0;
        to.timeoutAt = electionTimeout(rng, to.id);
      }
      messages.push({
        id: `${clock}-vr-${msg.to}-${msg.from}`,
        type: "VoteReply",
        from: msg.to,
        to: msg.from,
        term: to.currentTerm,
        granted,
        sentAt: clock,
        arriveAt: clock + RPC_FLIGHT,
        delivered: false,
      });
    } else if (msg.type === "VoteReply") {
      if (to.role !== "candidate" || msg.term < to.currentTerm) continue;
      if (msg.term > to.currentTerm) {
        to.currentTerm = msg.term;
        to.role = "follower";
        to.votedFor = null;
        to.votesGranted = new Set();
        continue;
      }
      if (msg.granted) {
        to.votesGranted.add(msg.from);
        if (to.votesGranted.size >= MAJORITY && to.role === "candidate") {
          to.role = "leader";
          to.nextHeartbeatIn = 0;
          narrative = {
            phase: "leader",
            text: `Node ${to.id} collected ${to.votesGranted.size} of ${N} votes, a majority, so it becomes leader for term ${to.currentTerm}. Because two leaders would each need a majority and any two majorities of ${N} must overlap in at least one node, no other node can win this term. The leader now sends heartbeats to suppress new elections.`,
          };
        }
      }
    } else if (msg.type === "AppendEntries") {
      if (msg.term < to.currentTerm) {
        messages.push({
          id: `${clock}-ar-${msg.to}-${msg.from}`,
          type: "AppendReply",
          from: msg.to,
          to: msg.from,
          term: to.currentTerm,
          success: false,
          matchIndex: 0,
          sentAt: clock,
          arriveAt: clock + RPC_FLIGHT,
          delivered: false,
        });
        continue;
      }
      to.currentTerm = msg.term;
      to.role = "follower";
      to.votedFor = msg.leaderId;
      to.timer = 0;
      to.timeoutAt = electionTimeout(rng, to.id);
      to.log = msg.entries.map((e) => ({ ...e }));
      to.commitIndex = Math.min(msg.leaderCommit, to.log.length);
      messages.push({
        id: `${clock}-ar-${msg.to}-${msg.from}`,
        type: "AppendReply",
        from: msg.to,
        to: msg.from,
        term: to.currentTerm,
        success: true,
        matchIndex: to.log.length,
        sentAt: clock,
        arriveAt: clock + RPC_FLIGHT,
        delivered: false,
      });
    } else if (msg.type === "AppendReply") {
      if (to.role !== "leader") continue;
      if (msg.term > to.currentTerm) {
        to.currentTerm = msg.term;
        to.role = "follower";
        to.votedFor = null;
        to.votesGranted = new Set();
        continue;
      }
      if (msg.success) {
        to.matchIndex = to.matchIndex || {};
        to.matchIndex[msg.from] = msg.matchIndex;
      }
    }
  }

  for (const node of nodes) {
    if (node.role === "dead") continue;

    if (node.role === "leader") {
      node.timer = 0;
      node.nextHeartbeatIn -= TICK_MS;
      if (node.nextHeartbeatIn <= 0) {
        node.nextHeartbeatIn = HEARTBEAT_INTERVAL;
        for (const peer of nodes) {
          if (peer.id === node.id) continue;
          messages.push({
            id: `${clock}-ae-${node.id}-${peer.id}`,
            type: "AppendEntries",
            from: node.id,
            to: peer.id,
            term: node.currentTerm,
            leaderId: node.id,
            entries: node.log.map((e) => ({ ...e })),
            leaderCommit: node.commitIndex,
            sentAt: clock,
            arriveAt: clock + RPC_FLIGHT,
            delivered: false,
          });
        }
      }
      const match = node.matchIndex || {};
      for (let idx = node.log.length; idx > node.commitIndex; idx--) {
        let count = 1;
        for (const peer of nodes) {
          if (peer.id !== node.id && (match[peer.id] || 0) >= idx) count++;
        }
        if (
          count >= MAJORITY &&
          node.log[idx - 1] &&
          node.log[idx - 1].term === node.currentTerm
        ) {
          if (idx > node.commitIndex) {
            node.commitIndex = idx;
            narrative = {
              phase: "commit",
              text: `Entry ${idx} is now stored on ${count} of ${N} nodes, a majority, so the leader marks it committed. An entry is never counted as committed until it reaches a majority, which guarantees any future leader (itself elected by a majority) already holds it. That overlap is why committed entries survive a leader change.`,
            };
          }
          break;
        }
      }
    } else {
      node.timer += TICK_MS;
      if (node.timer >= node.timeoutAt) {
        node.role = "candidate";
        node.currentTerm += 1;
        node.votedFor = node.id;
        node.votesGranted = new Set([node.id]);
        node.timer = 0;
        node.timeoutAt = electionTimeout(rng, node.id);
        narrative = {
          phase: "election",
          text: `Node ${node.id} timed out without hearing a heartbeat, so it starts an election: it increments its term to ${node.currentTerm}, votes for itself, and sends RequestVote to everyone. Randomized timeouts mean nodes rarely time out at the same instant, which keeps split votes (where no one reaches a majority) uncommon.`,
        };
        for (const peer of nodes) {
          if (peer.id === node.id) continue;
          messages.push({
            id: `${clock}-rv-${node.id}-${peer.id}`,
            type: "RequestVote",
            from: node.id,
            to: peer.id,
            term: node.currentTerm,
            candLog: node.log.map((e) => ({ ...e })),
            sentAt: clock,
            arriveAt: clock + RPC_FLIGHT,
            delivered: false,
          });
        }
      }
    }
  }

  const leaders = nodes.filter((n) => n.role === "leader");
  if (leaders.length === 0 && narrative.phase !== "election" && narrative.phase !== "boot") {
    const anyCandidate = nodes.some((n) => n.role === "candidate");
    if (!anyCandidate) {
      narrative = {
        phase: "waiting",
        text: "No leader right now. Followers are counting down their election timers; whichever fires first will stand for election in the next term.",
      };
    }
  }

  return {
    ...prev,
    nodes,
    messages,
    clock,
    narrative,
    seedCursor: prev.seedCursor ?? prev.seed,
  };
}

function submitCommand(state) {
  const nodes = state.nodes.map(cloneNode);
  const leader = nodes.find((n) => n.role === "leader");
  if (!leader) return state;
  leader.log.push({ term: leader.currentTerm, cmd: `x${state.nextCommand}` });
  leader.matchIndex = leader.matchIndex || {};
  leader.nextHeartbeatIn = 0;
  return {
    ...state,
    nodes,
    nextCommand: state.nextCommand + 1,
    narrative: {
      phase: "replicate",
      text: `A client command (x${state.nextCommand}) lands on leader ${leader.id}. It appends to its own log first, then ships the whole log to followers via AppendEntries. The entry stays uncommitted (hollow) until a majority acknowledge it.`,
    },
  };
}

function killLeader(state) {
  const nodes = state.nodes.map(cloneNode);
  const leader = nodes.find((n) => n.role === "leader");
  if (!leader) return state;
  leader.role = "dead";
  return {
    ...state,
    nodes,
    messages: state.messages.filter((m) => m.from !== leader.id && m.to !== leader.id),
    narrative: {
      phase: "failover",
      text: `Leader ${leader.id} is down. Its heartbeats stop, so the remaining followers will time out and one of them starts a new election in a higher term. Any uncommitted entries that only lived on the dead leader can be safely overwritten; committed ones already sit on a majority, so they survive.`,
    },
  };
}

function healNode(state) {
  const nodes = state.nodes.map(cloneNode);
  const dead = nodes.find((n) => n.role === "dead");
  if (!dead) return state;
  const rng = makeRng(state.seed + state.clock);
  dead.role = "follower";
  dead.votedFor = null;
  dead.votesGranted = new Set();
  dead.timer = 0;
  dead.timeoutAt = electionTimeout(rng, dead.id);
  dead.nextHeartbeatIn = 0;
  return {
    ...state,
    nodes,
    narrative: {
      phase: "rejoin",
      text: `Node ${dead.id} rejoins as a follower. When the current leader's next AppendEntries arrives, it overwrites this node's log with the leader's, discarding any stale uncommitted entries it held from a term it lost. The cluster converges back to one consistent log.`,
    },
  };
}

function nodePoint(i, cx, cy, r) {
  const angle = (i / N) * 2 * Math.PI - Math.PI / 2;
  return { x: cx + r * Math.cos(angle), y: cy + r * Math.sin(angle) };
}

function Cluster({ state, reduceMotion }) {
  const size = 320;
  const cx = size / 2;
  const cy = size / 2;
  const ringR = 110;
  const nodeR = 30;

  const pts = state.nodes.map((_, i) => nodePoint(i, cx, cy, ringR));
  const live = state.clock;
  const active = state.messages.filter(
    (m) => !m.delivered && m.arriveAt > live && m.sentAt <= live
  );

  return (
    <svg
      viewBox={`0 0 ${size} ${size}`}
      width="100%"
      style={{ maxWidth: 340, display: "block", margin: "0 auto" }}
      role="img"
      aria-label="Five Raft nodes arranged in a circle, colored by role, with vote and heartbeat messages animating along the edges between them"
    >
      {state.nodes.map((_, i) =>
        state.nodes.slice(i + 1).map((__, k) => {
          const j = i + 1 + k;
          return (
            <line
              key={`e-${i}-${j}`}
              x1={pts[i].x}
              y1={pts[i].y}
              x2={pts[j].x}
              y2={pts[j].y}
              stroke={C.faint}
              strokeWidth={1.5}
            />
          );
        })
      )}

      {active.map((m) => {
        const a = pts[m.from];
        const b = pts[m.to];
        const frac = Math.max(
          0,
          Math.min(1, (live - m.sentAt) / (m.arriveAt - m.sentAt))
        );
        const x = a.x + (b.x - a.x) * frac;
        const y = a.y + (b.y - a.y) * frac;
        const isVote = m.type === "RequestVote" || m.type === "VoteReply";
        const isAppend = m.type === "AppendEntries";
        const col = isVote ? ROLE.candidate.color : isAppend ? ROLE.leader.color : C.muted;
        const pulse = m.type === "AppendEntries" || m.type === "RequestVote";
        return (
          <g key={m.id}>
            <circle
              cx={x}
              cy={y}
              r={pulse ? 6 : 4}
              fill={col}
              opacity={0.9}
              style={
                reduceMotion
                  ? undefined
                  : { transition: `cx ${TICK_MS}ms linear, cy ${TICK_MS}ms linear` }
              }
            />
            {!reduceMotion && pulse && (
              <circle cx={x} cy={y} r={6} fill="none" stroke={col} strokeWidth={1.5} opacity={0.4}>
                <animate attributeName="r" from="6" to="13" dur="0.7s" repeatCount="indefinite" />
                <animate attributeName="opacity" from="0.4" to="0" dur="0.7s" repeatCount="indefinite" />
              </circle>
            )}
          </g>
        );
      })}

      {state.nodes.map((node, i) => {
        const p = pts[i];
        const role = ROLE[node.role];
        const isCand = node.role === "candidate";
        const timeFrac =
          node.role === "follower" || node.role === "candidate"
            ? Math.max(0, Math.min(1, 1 - node.timer / node.timeoutAt))
            : 1;
        const circ = 2 * Math.PI * (nodeR + 6);
        return (
          <g key={node.id}>
            {(node.role === "follower" || node.role === "candidate") && (
              <circle
                cx={p.x}
                cy={p.y}
                r={nodeR + 6}
                fill="none"
                stroke={role.color}
                strokeWidth={3}
                strokeDasharray={circ}
                strokeDashoffset={circ * (1 - timeFrac)}
                strokeLinecap="round"
                transform={`rotate(-90 ${p.x} ${p.y})`}
                opacity={0.55}
                style={reduceMotion ? undefined : { transition: `stroke-dashoffset ${TICK_MS}ms linear` }}
              />
            )}
            <circle
              cx={p.x}
              cy={p.y}
              r={nodeR}
              fill={role.soft}
              stroke={role.color}
              strokeWidth={node.role === "leader" ? 3 : 2}
              style={reduceMotion ? undefined : { transition: `fill 250ms ${EASE}, stroke 250ms ${EASE}` }}
            />
            <text
              x={p.x}
              y={p.y - 6}
              textAnchor="middle"
              fontSize={14}
              fontWeight={700}
              fill={role.color}
            >
              N{node.id}
            </text>
            <text x={p.x} y={p.y + 9} textAnchor="middle" fontSize={9} fill={C.ink} opacity={0.75}>
              term {node.currentTerm}
            </text>
            {node.role === "leader" && (
              <text x={p.x} y={p.y + 21} textAnchor="middle" fontSize={8} fontWeight={700} fill={role.color}>
                LEADER
              </text>
            )}
            {isCand && (
              <text x={p.x} y={p.y + 21} textAnchor="middle" fontSize={8} fontWeight={700} fill={role.color}>
                {node.votesGranted.size}/{N} votes
              </text>
            )}
          </g>
        );
      })}
    </svg>
  );
}

function LogRow({ node }) {
  const role = ROLE[node.role];
  return (
    <div style={{ display: "flex", alignItems: "center", gap: 10, padding: "6px 0" }}>
      <div style={{ width: 70, flexShrink: 0, display: "flex", alignItems: "center", gap: 6 }}>
        <span
          aria-hidden
          style={{
            width: 9,
            height: 9,
            borderRadius: "50%",
            background: role.color,
            display: "inline-block",
          }}
        />
        <span style={{ fontSize: 12, fontWeight: 700, color: C.ink }}>N{node.id}</span>
      </div>
      <div style={{ display: "flex", gap: 4, flexWrap: "wrap", flex: 1 }}>
        {node.log.length === 0 && (
          <span style={{ fontSize: 11, color: C.muted, fontStyle: "italic" }}>empty</span>
        )}
        {node.log.map((entry, idx) => {
          const committed = idx + 1 <= node.commitIndex;
          return (
            <div
              key={idx}
              title={`${entry.cmd} · term ${entry.term} · ${committed ? "committed" : "uncommitted"}`}
              style={{
                minWidth: 30,
                textAlign: "center",
                fontSize: 11,
                fontWeight: 700,
                padding: "3px 5px",
                borderRadius: 5,
                background: committed ? ROLE.leader.soft : C.card,
                color: committed ? ROLE.leader.color : C.muted,
                border: `1.5px solid ${committed ? ROLE.leader.color : C.border}`,
                borderStyle: committed ? "solid" : "dashed",
                transition: `all 250ms ${EASE}`,
              }}
            >
              {entry.term}
            </div>
          );
        })}
      </div>
    </div>
  );
}

function Btn({ children, onClick, disabled, primary, ariaLabel }) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      aria-label={ariaLabel}
      style={{
        padding: primary ? "8px 16px" : "7px 13px",
        borderRadius: 8,
        border: primary ? "none" : `1px solid ${C.border}`,
        background: primary ? C.accent : "transparent",
        color: primary ? "#fff" : C.ink,
        fontSize: 12.5,
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

function Card({ children, style }) {
  return (
    <div
      style={{
        background: C.card,
        border: `1px solid ${C.border}`,
        borderRadius: 14,
        padding: 18,
        ...style,
      }}
    >
      {children}
    </div>
  );
}

const SPEEDS = [
  { label: "0.5x", mult: 2 },
  { label: "1x", mult: 1 },
  { label: "2x", mult: 0.5 },
  { label: "4x", mult: 0.25 },
];

const SEED = 0x5eed1234;

export default function App() {
  const [state, setState] = useState(() => initialState(SEED));
  const [playing, setPlaying] = useState(true);
  const [speedIdx, setSpeedIdx] = useState(1);
  const stateRef = useRef(state);
  const intervalRef = useRef(null);
  stateRef.current = state;

  const reduceMotion =
    typeof window !== "undefined" &&
    window.matchMedia &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  useEffect(() => {
    if (!playing) {
      clearInterval(intervalRef.current);
      return;
    }
    intervalRef.current = setInterval(() => {
      setState((s) => step(s));
    }, TICK_MS * SPEEDS[speedIdx].mult);
    return () => clearInterval(intervalRef.current);
  }, [playing, speedIdx]);

  const leader = state.nodes.find((n) => n.role === "leader");
  const hasDead = state.nodes.some((n) => n.role === "dead");
  const committedCount = leader ? leader.commitIndex : 0;
  const term = Math.max(...state.nodes.map((n) => n.currentTerm));

  const doStep = useCallback(() => setState((s) => step(s)), []);
  const doReset = useCallback(() => {
    setState(initialState(SEED));
    setPlaying(true);
  }, []);

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
        button:focus-visible { outline: 2px solid ${C.accent}; outline-offset: 2px; }
        .raft-seg:focus-visible { outline: 2px solid ${C.accent}; outline-offset: 2px; }
        @media (prefers-reduced-motion: reduce) {
          * { transition: none !important; }
        }
      `}</style>

      <div style={{ maxWidth: 720, margin: "0 auto" }}>
        <div style={{ marginBottom: 20 }}>
          <div
            style={{
              fontSize: 10,
              letterSpacing: "0.15em",
              textTransform: "uppercase",
              color: C.muted,
              marginBottom: 5,
            }}
          >
            Distributed Systems · Consensus
          </div>
          <h1 style={{ fontSize: 25, fontWeight: 700, margin: 0 }}>
            Raft: Leader Election &amp; Replication
          </h1>
          <p
            style={{
              color: C.muted,
              fontSize: 13.5,
              margin: "7px 0 0",
              lineHeight: 1.6,
              maxWidth: "64ch",
            }}
          >
            Five nodes have to agree on a single ordered log while any of them can crash at any moment. Raft
            pulls this off by electing one leader on randomized timeouts, replicating its log to a majority,
            and re-electing the moment that leader goes dark. Kill the leader below and watch the cluster heal.
          </p>
        </div>

        <Card style={{ marginBottom: 16 }}>
          <div style={{ display: "flex", gap: 7, flexWrap: "wrap", marginBottom: 16 }}>
            <Btn primary onClick={() => setPlaying((p) => !p)} ariaLabel={playing ? "Pause the simulation" : "Play the simulation"}>
              {playing ? "Pause" : "Play"}
            </Btn>
            <Btn onClick={doStep} disabled={playing} ariaLabel="Advance one tick">
              Step
            </Btn>
            <Btn
              onClick={() => setState((s) => submitCommand(s))}
              disabled={!leader}
              ariaLabel="Submit a client command to the leader"
            >
              Submit command
            </Btn>
            <Btn
              onClick={() => setState((s) => killLeader(s))}
              disabled={!leader}
              ariaLabel="Kill the current leader to trigger a re-election"
            >
              Kill leader
            </Btn>
            <Btn
              onClick={() => setState((s) => healNode(s))}
              disabled={!hasDead}
              ariaLabel="Heal and restart a downed node"
            >
              Heal node
            </Btn>
            <Btn onClick={doReset} ariaLabel="Reset the cluster to its initial state">
              Reset
            </Btn>
          </div>

          <div
            style={{
              display: "flex",
              alignItems: "center",
              gap: 10,
              marginBottom: 14,
              flexWrap: "wrap",
            }}
          >
            <span style={{ fontSize: 12, color: C.muted }}>Speed</span>
            <div
              role="group"
              aria-label="Simulation speed"
              style={{ display: "inline-flex", border: `1px solid ${C.border}`, borderRadius: 8, overflow: "hidden" }}
            >
              {SPEEDS.map((s, i) => (
                <button
                  key={s.label}
                  className="raft-seg"
                  onClick={() => setSpeedIdx(i)}
                  aria-pressed={speedIdx === i}
                  style={{
                    padding: "5px 11px",
                    border: "none",
                    borderRight: i < SPEEDS.length - 1 ? `1px solid ${C.border}` : "none",
                    background: speedIdx === i ? C.accent : "transparent",
                    color: speedIdx === i ? "#fff" : C.muted,
                    fontSize: 12,
                    fontWeight: speedIdx === i ? 700 : 500,
                    fontFamily: "inherit",
                    cursor: "pointer",
                  }}
                >
                  {s.label}
                </button>
              ))}
            </div>
            <div style={{ marginLeft: "auto", fontSize: 12, color: C.muted }}>
              Highest term <b style={{ color: C.ink }}>{term}</b> · committed{" "}
              <b style={{ color: C.ink }}>{committedCount}</b>
            </div>
          </div>

          <Cluster state={state} reduceMotion={reduceMotion} />

          <div style={{ display: "flex", gap: 14, justifyContent: "center", flexWrap: "wrap", marginTop: 10 }}>
            {Object.entries(ROLE).map(([k, v]) => (
              <div key={k} style={{ display: "flex", alignItems: "center", gap: 5, fontSize: 11, color: C.muted }}>
                <span
                  aria-hidden
                  style={{ width: 10, height: 10, borderRadius: "50%", background: v.color, display: "inline-block" }}
                />
                {v.label}
              </div>
            ))}
          </div>
        </Card>

        <Card
          aria-live="polite"
          style={{
            marginBottom: 16,
            background: C.accentSoft,
            border: `1px solid ${C.accent}22`,
          }}
        >
          <div
            style={{
              fontSize: 11,
              fontWeight: 700,
              letterSpacing: "0.08em",
              textTransform: "uppercase",
              color: C.accent,
              marginBottom: 8,
            }}
          >
            What is happening
          </div>
          <p style={{ margin: 0, fontSize: 13.5, lineHeight: 1.65, color: C.ink, minHeight: 60 }}>
            {state.narrative.text}
          </p>
        </Card>

        <Card style={{ marginBottom: 16 }}>
          <div style={{ fontSize: 14, fontWeight: 700, marginBottom: 4 }}>Replicated logs</div>
          <div style={{ fontSize: 12.5, color: C.muted, marginBottom: 12, lineHeight: 1.6 }}>
            Each box is a log entry labelled with the term it was created in. Solid terracotta means
            committed (safely stored on a majority); dashed gray means replicated but not yet
            committed. Press <b>Submit command</b> while a leader exists and watch an entry spread
            outward, then commit once a majority hold it.
          </div>
          <div>
            {state.nodes.map((n) => (
              <LogRow key={n.id} node={n} />
            ))}
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
            The two safety rules
          </div>
          <div style={{ fontSize: 13, color: C.ink, lineHeight: 1.7, display: "grid", gap: 10 }}>
            <p style={{ margin: 0 }}>
              <b>At most one leader per term.</b> To win, a candidate needs votes from a strict
              majority ({MAJORITY} of {N}), and each node casts at most one vote per term. Two
              different candidates would each need their own majority, but any two majorities of {N}
              nodes must share at least one node, and that node only voted once. So a term can never
              produce two leaders. Randomized timeouts make ties rare; when a term does split with no
              winner, everyone just times out again and tries a higher term.
            </p>
            <p style={{ margin: 0 }}>
              <b>Committed entries are never lost.</b> An entry commits only after a majority store
              it. A new leader is also elected by a majority, and the election rule only grants a
              vote to a candidate whose log is at least as up to date as the voter's. Those two
              majorities overlap, so any committed entry is already present in the new leader's log.
              Kill the leader after an entry commits and you will see the next leader keep it; kill it
              before commit and the half-replicated entry can vanish, which is exactly why commit
              waits for a majority.
            </p>
          </div>
        </Card>

        <div style={{ marginTop: 18, textAlign: "center", fontSize: 11, color: C.muted }}>
          Election timeouts are drawn from a seeded PRNG (per-node jitter), so the run is reproducible
          across reloads
        </div>
      </div>
    </div>
  );
}
