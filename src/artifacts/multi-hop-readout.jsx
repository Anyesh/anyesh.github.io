import { useState } from "react";
import LensReadout, {
  LC,
  LMONO,
  LSERIF,
} from "../components/lens/LensReadout.jsx";
import data from "./data/lens/multihop.json";

export const meta = {
  title: "The Reasoning Step It Never Says Out Loud",
  category: "Interpretability",
  description:
    "I asked a model for the currency of the country shaped like a boot and it answered euro. To get there it first has to decide the country is Italy, but it never writes Italy. That step happens silently, inside. Using the paper's lens I caught Italy surfacing mid-network before it hands off to euro. The readouts are from a 4B model I ran myself.",
  date: "2026-07-07",
  tags: ["interpretability", "transformers", "reasoning", "jacobian-lens", "multi-hop"],
};

const byName = Object.fromEntries(data.items.map((it) => [it.name, it]));
const C = LC;

const CASES = [
  {
    name: "boot-currency",
    label: "boot-shaped country → currency",
    concepts: [
      { word: "Italy", label: "Italy  (the silent middle step)", color: LC.inter },
      { word: "euro", label: "euro  (the answer)", color: LC.answer },
      { word: "lira", label: "lira  (the old currency, a near miss)", color: LC.op },
    ],
    milestones: [
      { layer: 10, kind: "note", label: "the list fills with countries (France, Germany, America). The model knows a country is coming but has not singled one out yet; Italy is still well down." },
      { layer: 20, kind: "inter", label: "Italy is now among the top words, sitting right there in the list, though the model will never write it. That is the silent middle step." },
      { layer: 26, kind: "answer", label: "euro takes over the list. With the country pinned, the model retrieves its currency and commits." },
    ],
  },
  {
    name: "carnival-ocean",
    label: "Carnival country → ocean",
    concepts: [
      { word: "Brazil", label: "Brazil  (the silent middle step)", color: LC.inter },
      { word: "Atlantic", label: "Atlantic  (the answer)", color: LC.answer },
      { word: "Pacific", label: "Pacific  (the wrong coast)", color: LC.op },
    ],
    milestones: [
      { layer: 8, kind: "note", label: "the model knows it is an ocean (ocean, sea top the list) but not which one. Brazil, the country that decides the coast, is still buried." },
      { layer: 20, kind: "inter", label: "Atlantic and Pacific are both near the top now. The model has the right region but has not chosen the coast." },
      { layer: 26, kind: "answer", label: "Atlantic settles just ahead of Pacific. Brazil never surfaced strongly here, so the tie stays close." },
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

export default function MultiHopReadout() {
  const [active, setActive] = useState("boot-currency");
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
          The reasoning step it never says out loud
        </h1>

        <div style={{ fontSize: 17, lineHeight: 1.7, color: "#2c2823" }}>
          <p style={{ margin: "0 0 16px" }}>
            I gave a model "The currency used in the country shaped like a boot is..." and
            it finished with <strong>euro</strong>. That takes two hops: first work out
            that the boot-shaped country is <strong>Italy</strong>, then recall that Italy
            uses the euro. But the model never writes "Italy". It jumps straight from the
            riddle to the currency, so from the outside the middle step is invisible.
          </p>
          <p style={{ margin: "0 0 16px" }}>
            The paper below is where this hidden step was found, and the lens that reveals
            it is theirs, the same one from{" "}
            <a href="/a/watch-it-calculate" style={linkS}>the arithmetic walkthrough</a>. I
            ran it on the boot-currency example on a small model of my own, and could watch{" "}
            <strong>Italy</strong> rise out of the noise in the middle of the network, hold
            near the top for a while, then hand off to <strong>euro</strong> near the end.
            The model needed that thought and never said it.
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
          <h3 style={h3S}>Why the buried step matters</h3>
          <p style={pS}>
            There are two ways a model could answer this. It could pattern-match "boot
            currency" straight to "euro" from training text, or it could actually compose
            two separate facts. The lens shows Italy present as its own distinct thing
            before the currency forms, which points to the second: the country gets
            computed, held, and then used. In the Carnival case the buried country
            (Brazil) is fainter, which you can see in the softer orange line. Not every
            hop reads out this cleanly at 4B.
          </p>
        </Card>

        <div style={{ marginTop: 30, paddingTop: 18, borderTop: `1px solid ${C.border}`, fontFamily: LSERIF, fontSize: 14, color: C.muted, lineHeight: 1.6 }}>
          <p style={{ margin: "0 0 10px" }}>Related readouts from the same lens:</p>
          <ul style={{ margin: 0, paddingLeft: 18 }}>
            <li><a href="/a/watch-it-calculate" style={linkS}>Doing arithmetic</a> : the sub-result then the answer, layer by layer.</li>
            <li><a href="/a/planning-ahead" style={linkS}>Planning the next word</a> : rhymes and typo fixes settled before they are written.</li>
            <li><a href="/a/global-workspace" style={linkS}>The global workspace</a> : what the paper argues these readable thoughts really are.</li>
          </ul>
          <p style={{ marginTop: 16, fontSize: 13, color: C.faint }}>
            I computed these readouts on {data.model} with the lens {data.lens}. From{" "}
            <a href="https://transformer-circuits.pub/2026/workspace/index.html" target="_blank" rel="noreferrer" style={linkS}>
              Verbalizable Representations Form a Global Workspace in Language Models
            </a>{" "}
            (Anthropic, 2026). Lens code:{" "}
            <a href="https://github.com/anthropics/jacobian-lens" target="_blank" rel="noreferrer" style={linkS}>anthropics/jacobian-lens</a>. I generated every number here with scripts/gen-lens-data.py in this repo; rerun it to reproduce.
          </p>
        </div>
      </div>
    </div>
  );
}

const h3S = { fontSize: 18, margin: "0 0 8px", fontWeight: 700 };
const pS = { fontSize: 15, lineHeight: 1.7, color: "#2c2823", margin: 0 };
const linkS = { color: "#1d4ed8", textDecoration: "none", borderBottom: "1px solid #bcd" };
