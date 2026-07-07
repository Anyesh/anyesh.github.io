import { useState } from "react";
import LensReadout, {
  LC,
  LMONO,
  LSERIF,
} from "../components/lens/LensReadout.jsx";
import data from "./data/lens/planning.json";

export const meta = {
  title: "The Word It Has Already Chosen",
  category: "Interpretability",
  description:
    "Misspell langauge and a model reads it as language anyway. Set up a rhyme and it locks the last word in long before writing it. A lens that decodes the model's internal guess at each layer shows the correction and the rhyme are settled well before the output. Real readouts from Qwen3.5-4B.",
  date: "2026-07-07",
  tags: ["interpretability", "transformers", "jacobian-lens", "planning", "typos"],
};

const byName = Object.fromEntries(data.items.map((it) => [it.name, it]));
const C = LC;

const CASES = [
  {
    name: "typo-language",
    label: "reading a typo",
    concepts: [
      { word: "language", label: "language  (what it reads)", color: LC.answer },
      { word: "langauge", label: "langauge  (what is on the page)", color: LC.op },
    ],
    milestones: [
      { layer: 8, kind: "answer", label: "the model has quietly corrected the typo: language is its top read of the word, not the misspelling actually printed in front of it." },
      { layer: 16, kind: "note", label: "having understood the word, later layers move on to guessing what comes next, so a period and continuation tokens take the lead." },
    ],
  },
  {
    name: "couplet-breath-death",
    label: "settling a rhyme",
    concepts: [
      { word: "death", label: "death  (the rhyme for breath)", color: LC.answer },
      { word: "God", label: "God  (a plausible alternative)", color: LC.op },
      { word: "peace", label: "peace  (another candidate)", color: LC.note },
    ],
    milestones: [
      { layer: 4, kind: "note", label: "early layers hold only punctuation and ellipses. No word is committed." },
      { layer: 18, kind: "answer", label: "death locks in as the rhyme for breath, and it stays on top through every remaining layer, decided long before it is written." },
    ],
  },
];

function Card({ children, style }) {
  return (
    <div style={{ border: `1px solid ${C.border}`, background: C.card, borderRadius: 14, padding: "18px 20px", ...style }}>
      {children}
    </div>
  );
}

export default function PlanningAhead() {
  const [active, setActive] = useState("typo-language");
  const cfg = CASES.find((c) => c.name === active);
  const item = byName[active];

  return (
    <div style={{ background: C.bg, minHeight: "100vh", color: C.ink, fontFamily: LSERIF, padding: "0 0 80px" }}>
      <style>{`@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600;700&display=swap');`}</style>
      <div style={{ maxWidth: 720, margin: "0 auto", padding: "48px 22px 0" }}>
        <div style={{ fontFamily: LMONO, fontSize: 12, letterSpacing: 1, textTransform: "uppercase", color: C.faint, marginBottom: 14 }}>
          Interpretability
        </div>
        <h1 style={{ fontSize: 34, lineHeight: 1.15, margin: "0 0 20px", fontWeight: 700 }}>
          The word it has already chosen
        </h1>

        <div style={{ fontSize: 17, lineHeight: 1.7, color: "#2c2823" }}>
          <p style={{ margin: "0 0 16px" }}>
            Write <code style={codeS}>langauge</code>, with the letters flipped, and a
            model still understands <strong>language</strong>. It does not stumble over
            the typo the way a strict string match would. Somewhere inside, the messy
            characters on the page get read as the word you meant. Where does that happen,
            and how early?
          </p>
          <p style={{ margin: "0 0 16px" }}>
            The lens from{" "}
            <a href="/a/watch-it-calculate" style={linkS}>the arithmetic walkthrough</a>{" "}
            answers it directly: decode the model's internal guess at each layer and the
            corrected word <strong>language</strong> is already its top pick by the lower
            middle of the network, while the literal misspelling never leads. Switch to
            the rhyming couplet and you see the same early commitment: the model fixes on{" "}
            <strong>death</strong> to rhyme with <em>breath</em> and then holds it, steady,
            for the rest of the stack.
          </p>
        </div>

        <div style={{ display: "flex", gap: 8, margin: "22px 0 14px", flexWrap: "wrap" }}>
          {CASES.map((c) => (
            <button
              key={c.name}
              onClick={() => setActive(c.name)}
              style={{ fontFamily: LMONO, fontSize: 13.5, padding: "8px 14px", borderRadius: 9, cursor: "pointer", border: `1px solid ${active === c.name ? C.ink : C.border}`, background: active === c.name ? C.ink : C.card, color: active === c.name ? "#fff" : C.muted }}
            >
              {c.label}
            </button>
          ))}
        </div>

        <LensReadout item={item} concepts={cfg.concepts} milestones={cfg.milestones} />

        <Card style={{ marginTop: 28 }}>
          <h3 style={h3S}>Read at the right spot</h3>
          <p style={pS}>
            Both readouts are taken at the last token of the prompt, the point where the
            model is deciding what comes next. For the typo that means it has folded the
            garbled spelling into a clean sense of the word before it does anything else
            with it. For the rhyme it means the payoff word is chosen up front, not
            improvised at the last instant. The paper shows a sharper version of this on
            larger models, where the rhyme for the <em>next</em> line is already planned
            several words in advance; here the effect is smaller but the shape is the same.
          </p>
        </Card>

        <div style={{ marginTop: 30, paddingTop: 18, borderTop: `1px solid ${C.border}`, fontFamily: LSERIF, fontSize: 14, color: C.muted, lineHeight: 1.6 }}>
          <p style={{ margin: "0 0 10px" }}>More of the same lens at work:</p>
          <ul style={{ margin: 0, paddingLeft: 18 }}>
            <li><a href="/a/watch-it-calculate" style={linkS}>Doing arithmetic</a> : the sub-result, then the answer.</li>
            <li><a href="/a/multi-hop-readout" style={linkS}>A silent reasoning step</a> : reaching "Italy" without saying it.</li>
            <li><a href="/a/global-workspace" style={linkS}>The global workspace</a> : the bigger claim behind all of these.</li>
          </ul>
          <p style={{ marginTop: 16, fontSize: 13, color: C.faint }}>
            Readouts are real, computed on {data.model} with the lens {data.lens}. From{" "}
            <a href="https://transformer-circuits.pub/2026/workspace/index.html" target="_blank" rel="noreferrer" style={linkS}>
              Verbalizable Representations Form a Global Workspace in Language Models
            </a>{" "}
            (Anthropic, 2026). Lens code:{" "}
            <a href="https://github.com/anthropics/jacobian-lens" target="_blank" rel="noreferrer" style={linkS}>anthropics/jacobian-lens</a>.
          </p>
        </div>
      </div>
    </div>
  );
}

const codeS = { fontFamily: LMONO, fontSize: 15, background: "#f2efe9", padding: "1px 6px", borderRadius: 4 };
const h3S = { fontSize: 18, margin: "0 0 8px", fontWeight: 700 };
const pS = { fontSize: 15, lineHeight: 1.7, color: "#2c2823", margin: 0 };
const linkS = { color: "#1d4ed8", textDecoration: "none", borderBottom: "1px solid #bcd" };
