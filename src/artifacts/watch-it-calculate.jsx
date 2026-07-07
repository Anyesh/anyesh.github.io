import { useState } from "react";
import LensReadout, {
  LC,
  LMONO,
  LSERIF,
} from "../components/lens/LensReadout.jsx";
import data from "./data/lens/calculate.json";

export const meta = {
  title: "Watch a Language Model Do Arithmetic",
  category: "Interpretability",
  description:
    "I gave a language model (2 + 3) * 4 and it wrote 20 without running a calculator. Using the paper's lens I read the model's half-formed answer out of each layer and watched it reach the sub-sum 5 partway up, then twenty near the top. The readouts are from a 4B model I ran myself.",
  date: "2026-07-07",
  tags: ["interpretability", "transformers", "reasoning", "jacobian-lens", "arithmetic"],
};

const byName = Object.fromEntries(data.items.map((it) => [it.name, it]));

const CASES = [
  {
    name: "sum-times",
    label: "(2 + 3) × 4",
    concepts: [
      { word: "5", label: "5  (the sub-sum 2+3)", color: LC.inter },
      { word: "twenty", label: "twenty  (the answer)", color: LC.answer },
      { word: "multiplication", label: "the idea of multiplying", color: LC.op },
    ],
    milestones: [
      { layer: 15, kind: "note", label: "still nothing but punctuation and question marks. It has not started computing." },
      { layer: 21, kind: "inter", label: "the sub-sum 5 (that is 2 + 3) surfaces as the top guess. The inner bracket has been worked out." },
      { layer: 27, kind: "answer", label: "twenty appears. The model has multiplied 5 by 4, and it is holding the answer as a number word before it writes the digit." },
    ],
  },
  {
    name: "add-then-mult",
    label: "2 + 3 × 4",
    concepts: [
      { word: "twelve", label: "twelve  (3 × 4, done first)", color: LC.inter },
      { word: "fourteen", label: "fourteen  (the answer)", color: LC.answer },
      { word: "addition", label: "the idea of adding", color: LC.op },
    ],
    milestones: [
      { layer: 24, kind: "inter", label: "twelve surfaces first. There is no 12 written anywhere in the prompt: the model multiplied 3 by 4 before touching the 2, exactly the order of operations." },
      { layer: 27, kind: "answer", label: "fourteen appears. Only now does it add the 2, landing on the correct 14." },
    ],
  },
  {
    name: "paper-4-17",
    label: "(4 + 17) × 2",
    concepts: [
      { word: "forty", label: "forty  (the answer, 42)", color: LC.answer },
      { word: "twenty", label: "twenty  (a near miss)", color: LC.op },
    ],
    milestones: [
      { layer: 21, kind: "note", label: "the top guesses are still stray digits and question marks. The 21 from the bracket is not cleanly readable here." },
      { layer: 27, kind: "answer", label: "forty appears: the model is closing in on 42. The paper's own example runs the same shape, (4 + 17) * 2 + 7, on a far larger model where every step reads out clean." },
    ],
  },
];

const C = LC;

function Card({ children, style }) {
  return (
    <div
      style={{
        border: `1px solid ${C.border}`,
        background: C.card,
        borderRadius: 14,
        padding: "18px 20px",
        ...style,
      }}
    >
      {children}
    </div>
  );
}

export default function WatchItCalculate() {
  const [active, setActive] = useState("sum-times");
  const cfg = CASES.find((c) => c.name === active);
  const item = byName[active];

  return (
    <div
      style={{
        background: C.bg,
        minHeight: "100vh",
        color: C.ink,
        fontFamily: LSERIF,
        padding: "0 0 80px",
      }}
    >
      <style>{`@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600;700&display=swap');`}</style>

      <div style={{ maxWidth: 720, margin: "0 auto", padding: "48px 22px 0" }}>
        <div
          style={{
            fontFamily: LMONO,
            fontSize: 12,
            letterSpacing: 1,
            textTransform: "uppercase",
            color: C.faint,
            marginBottom: 14,
          }}
        >
          Interpretability
        </div>
        <h1 style={{ fontSize: 34, lineHeight: 1.15, margin: "0 0 20px", fontWeight: 700 }}>
          Watch a language model do arithmetic
        </h1>

        <div style={{ fontSize: 17, lineHeight: 1.7, color: "#2c2823" }}>
          <p style={{ margin: "0 0 16px" }}>
            I typed <code style={codeS}>(2 + 3) * 4</code> into a language model and it
            answered <strong>20</strong>. It didn't run a calculator or any code, it just
            did one forward pass, the same single sweep through the network it uses to
            guess the next word in a sentence. So where does the 20 come from?
          </p>
          <p style={{ margin: "0 0 16px" }}>
            So I looked. A transformer builds its answer in stages, one per layer, and the
            paper below comes with a <em>lens</em> that decodes the half-formed guess
            sitting in the model at any layer into plain words, a bit like clipping a probe
            onto a chip and reading the bus while it computes. I ran that lens on every
            layer. Drag the slider and you see what I saw: the guess reaches the inner sum{" "}
            <strong>5</strong> partway up, then <strong>twenty</strong> near the top,
            before it writes a single digit.
          </p>
        </div>

        <div style={{ display: "flex", gap: 8, margin: "22px 0 14px", flexWrap: "wrap" }}>
          {CASES.map((c) => (
            <button
              key={c.name}
              onClick={() => setActive(c.name)}
              style={{
                fontFamily: LMONO,
                fontSize: 14,
                padding: "8px 14px",
                borderRadius: 9,
                cursor: "pointer",
                border: `1px solid ${active === c.name ? C.ink : C.border}`,
                background: active === c.name ? C.ink : C.card,
                color: active === c.name ? "#fff" : C.muted,
              }}
            >
              {c.label}
            </button>
          ))}
        </div>

        <LensReadout item={item} concepts={cfg.concepts} milestones={cfg.milestones} />

        <div
          style={{
            fontFamily: LSERIF,
            fontSize: 13,
            color: C.faint,
            margin: "10px 2px 0",
            lineHeight: 1.55,
          }}
        >
          I give the model two worked examples first, then the expression. Without that
          nudge the 4B model reads a bare <code style={codeSmall}>expr =</code> as a quiz
          to echo back rather than solve, and computes nothing. The middle layers are
          genuinely noisy: stray tokens and question marks turn up between the clean
          milestones. I left that noise in rather than smooth it out.
        </div>

        <Card style={{ marginTop: 28 }}>
          <h3 style={h3S}>The answer shows up as a word, not a digit</h3>
          <p style={pS}>
            The answer shows up as <code style={codeSmall}>twenty</code> or{" "}
            <code style={codeSmall}>forty</code>, not <code style={codeSmall}>2</code> or{" "}
            <code style={codeSmall}>4</code>. Inside the network the result is a number as
            a <em>concept</em>, and only in the last few layers does it commit to the
            actual digit it will write. That is what makes the lens worth using to me: I
            can read the meaning the model is holding before it turns into output.
          </p>
        </Card>

        <Card style={{ marginTop: 16 }}>
          <h3 style={h3S}>What the lens actually is</h3>
          <p style={pS}>
            A transformer keeps a running vector for each token, the{" "}
            <em>residual stream</em>, and refines it layer by layer until the top layer
            turns it into a next-word guess. The natural way to peek at an early layer is
            to shove its vector straight into that final guess step (the{" "}
            <em>logit lens</em>), but early vectors live in a different internal
            coordinate system, so the readout is mush. The Jacobian lens first applies the
            average linear map from that layer to the top before decoding, which is why
            <code style={codeSmall}>5</code> and <code style={codeSmall}>twenty</code> come
            out as real words instead of noise. It is a read-only probe: it changes
            nothing about the model, it just reports what a layer is leaning toward saying.
          </p>
        </Card>

        <div
          style={{
            marginTop: 30,
            paddingTop: 18,
            borderTop: `1px solid ${C.border}`,
            fontFamily: LSERIF,
            fontSize: 14,
            color: C.muted,
            lineHeight: 1.6,
          }}
        >
          <p style={{ margin: "0 0 10px" }}>
            Next, the same lens on a different kind of thought:
          </p>
          <ul style={{ margin: 0, paddingLeft: 18 }}>
            <li>
              <a href="/a/multi-hop-readout" style={linkS}>
                A silent chain of reasoning
              </a>{" "}
              : watch the model reach "Italy" without ever saying it.
            </li>
            <li>
              <a href="/a/planning-ahead" style={linkS}>
                Planning the next word
              </a>{" "}
              : rhymes and typo fixes it settles on before writing them.
            </li>
            <li>
              <a href="/a/global-workspace" style={linkS}>
                The global workspace
              </a>{" "}
              : the paper's larger claim about what these readable thoughts are.
            </li>
          </ul>
          <p style={{ marginTop: 16, fontSize: 13, color: C.faint }}>
            I computed these readouts on {data.model} with the lens{" "}
            {data.lens}. The method and the original arithmetic example come from{" "}
            <a
              href="https://transformer-circuits.pub/2026/workspace/index.html"
              target="_blank"
              rel="noreferrer"
              style={linkS}
            >
              Verbalizable Representations Form a Global Workspace in Language Models
            </a>{" "}
            (Anthropic, 2026). Lens code:{" "}
            <a
              href="https://github.com/anthropics/jacobian-lens"
              target="_blank"
              rel="noreferrer"
              style={linkS}
            >
              anthropics/jacobian-lens
            </a>
            .
          </p>
        </div>
      </div>
    </div>
  );
}

const codeS = {
  fontFamily: LMONO,
  fontSize: 15,
  background: "#f2efe9",
  padding: "1px 6px",
  borderRadius: 4,
};
const codeSmall = {
  fontFamily: LMONO,
  fontSize: 12.5,
  background: "#f2efe9",
  padding: "1px 5px",
  borderRadius: 4,
};
const h3S = { fontSize: 18, margin: "0 0 8px", fontWeight: 700 };
const pS = { fontSize: 15, lineHeight: 1.7, color: "#2c2823", margin: 0 };
const linkS = { color: "#1d4ed8", textDecoration: "none", borderBottom: "1px solid #bcd" };
