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
    "Ask for the currency of the country shaped like a boot and a model answers euro. To get there it must first decide the country is Italy, yet it never writes Italy. That middle step happens silently, inside. A lens catches Italy surfacing mid-network, then handing off to euro. Real readouts from Qwen3.5-4B.",
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
      { layer: 11, kind: "inter", label: "Italy has quietly climbed into view. The model worked out which country is shaped like a boot, and it will never write the word." },
      { layer: 20, kind: "inter", label: "Italy is now near the very top of its own guesses. The hidden middle step is fully formed." },
      { layer: 25, kind: "answer", label: "euro overtakes everything. With the country fixed, the model looks up its currency and commits." },
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
      { layer: 8, kind: "note", label: "the model knows an ocean is coming (ocean tops the list) but not which one." },
      { layer: 21, kind: "inter", label: "Atlantic and Pacific both surge. It has the right kind of answer, and Brazil, never spoken, is what decides between the two coasts." },
      { layer: 25, kind: "answer", label: "Atlantic settles in front of Pacific, because Brazil's coastline faces east." },
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
            "The currency used in the country shaped like a boot is..." and the model
            finishes with <strong>euro</strong>. Getting there takes two hops: first work
            out that the boot-shaped country is <strong>Italy</strong>, then recall that
            Italy uses the euro. But the model never writes "Italy". It goes straight from
            the riddle to the currency, so the middle step is invisible from the outside.
          </p>
          <p style={{ margin: "0 0 16px" }}>
            The same lens from{" "}
            <a href="/a/watch-it-calculate" style={linkS}>the arithmetic walkthrough</a>{" "}
            makes it visible. It decodes what the model is leaning toward saying at each
            layer, and here you can watch <strong>Italy</strong> rise out of the noise in
            the middle of the network, hold at the top for a while, then hand off to{" "}
            <strong>euro</strong> near the end. A thought the model needed but never
            spoke.
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
            This is the difference between a model that pattern-matches "boot currency"
            straight to "euro" from training text, and one that genuinely composes two
            separate facts. The lens shows Italy present as its own distinct thing before
            the currency ever forms, which is evidence for the second story: the country
            is computed, held, and then used. In the Carnival case the buried country
            (Brazil) is fainter, and you can see it in the softer orange line, an honest
            reminder that not every hop reads out as cleanly at this model size.
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

const h3S = { fontSize: 18, margin: "0 0 8px", fontWeight: 700 };
const pS = { fontSize: 15, lineHeight: 1.7, color: "#2c2823", margin: 0 };
const linkS = { color: "#1d4ed8", textDecoration: "none", borderBottom: "1px solid #bcd" };
