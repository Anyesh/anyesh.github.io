import LensReadout, {
  LC,
  LMONO,
  LSERIF,
} from "../components/lens/LensReadout.jsx";
import data from "./data/lens/workspace.json";

export const meta = {
  title: "A Global Workspace Inside the Model",
  category: "Interpretability",
  description:
    "Nobody told the model the author was British. It worked that out from a few spellings and then started spelling colour, not color, on its own. With the paper's lens I can read that unstated decision forming and steering the output. It is a small window onto the paper's larger claim, that language models hold a shared, readable set of active ideas. The readouts are from a 4B model I ran myself.",
  date: "2026-07-07",
  tags: ["interpretability", "transformers", "jacobian-lens", "global-workspace", "cognition"],
};

const item = data.items.find((it) => it.name === "summoning-spelling");
const C = LC;

const concepts = [
  { word: "colour", label: "colour  (British spelling)", color: LC.answer },
  { word: "color", label: "color  (American, same meaning)", color: LC.op },
  { word: "shade", label: "shade  (a plain synonym)", color: LC.note },
];
const milestones = [
  { layer: 8, kind: "note", label: "colour is already the model's preferred continuation. The British spellings earlier in the passage (organised, labelled, favourite) have quietly set its sense of the author." },
  { layer: 19, kind: "answer", label: "colour jumps to the top of the entire vocabulary while American color, which means exactly the same thing, stays far behind. The model settled on a British writer and now spells like one, though nobody said so." },
];

function Card({ children, style }) {
  return (
    <div style={{ border: `1px solid ${C.border}`, background: C.card, borderRadius: 14, padding: "18px 20px", ...style }}>
      {children}
    </div>
  );
}

function Idea({ term, children }) {
  return (
    <div style={{ marginBottom: 14 }}>
      <div style={{ fontFamily: LMONO, fontSize: 13.5, fontWeight: 700, color: C.ink, marginBottom: 2 }}>
        {term}
      </div>
      <div style={{ fontSize: 14.5, lineHeight: 1.65, color: "#2c2823" }}>{children}</div>
    </div>
  );
}

export default function GlobalWorkspace() {
  return (
    <div style={{ background: C.bg, minHeight: "100vh", color: C.ink, fontFamily: LSERIF, padding: "0 0 80px" }}>
      <style>{`@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600;700&display=swap');`}</style>
      <div style={{ maxWidth: 720, margin: "0 auto", padding: "48px 22px 0" }}>
        <div style={{ fontFamily: LMONO, fontSize: 12, letterSpacing: 1, textTransform: "uppercase", color: C.faint, marginBottom: 14 }}>
          Interpretability
        </div>
        <h1 style={{ fontSize: 34, lineHeight: 1.15, margin: "0 0 20px", fontWeight: 700 }}>
          A global workspace inside the model
        </h1>

        <div style={{ fontSize: 17, lineHeight: 1.7, color: "#2c2823" }}>
          <p style={{ margin: "0 0 16px" }}>
            Here is a short passage: a person sorting samples by shade, who has{" "}
            <em>organised</em> and <em>labelled</em> them, for whom red was never a{" "}
            <em>favourite</em>. Nobody states that the writer is British. Yet when the
            model continues the sentence, it reaches for <strong>colour</strong>, the
            British spelling, over the identical-meaning American <strong>color</strong>.
            It picked up a fact nobody supplied, the author's dialect, and let it steer a
            choice further down the line.
          </p>
          <p style={{ margin: "0 0 16px" }}>
            The same lens catches that inference forming. When I run it, the British
            spelling pulls clear of both the American one and plain synonyms and climbs to
            the top of the vocabulary. This is what the paper is about, and not just in
            this one example. The model works something out, keeps it active while it keeps
            writing, and holds it in a form I can read back out.
          </p>
        </div>

        <LensReadout item={item} concepts={concepts} milestones={milestones} />

        <Card style={{ marginTop: 28 }}>
          <h3 style={h3S}>What "global workspace" means</h3>
          <p style={{ ...pS, marginBottom: 16 }}>
            The term is borrowed from cognitive science, where it names the small set of
            things a mind is currently aware of and can act on. The paper argues language
            models have a working analogue, with four properties that the earlier readouts
            already hint at:
          </p>
          <Idea term="verbalizable">
            The active idea can be decoded into words. That is the whole game the lens
            plays: <code style={codeSmall}>Italy</code>, <code style={codeSmall}>colour</code>,{" "}
            <code style={codeSmall}>twenty</code> come out as real tokens, so a hidden state
            is legible rather than opaque.
          </Idea>
          <Idea term="ignition">
            An idea can go from absent to fully present fairly sharply, rather than
            drifting up gradually. The colour and Italy curves climb steeply once they
            catch: the rank trajectory in the chart above is a small version of that
            snap-into-focus.
          </Idea>
          <Idea term="top-down control">
            The workspace can be steered by the task, not just by the raw input. Ask the
            model to attend to a topic and the corresponding idea brightens; the British
            author here is summoned by evidence rather than a direct statement.
          </Idea>
          <Idea term="limited capacity">
            Only so much can be held active at once, which is exactly what makes it a
            workspace and not just "all of memory". The paper probes this directly with
            dual-task and capacity experiments.
          </Idea>
        </Card>

        <Card style={{ marginTop: 16, background: "#fbfaf7" }}>
          <h3 style={h3S}>Where it does not reproduce</h3>
          <p style={pS}>
            I also tried the classic probe of asking the model to think about its greatest
            fear without saying it. At this model size the fear concept didn't surface
            cleanly, so I left it out rather than dress up noise as a result. The effects on
            this page are real but modest. A 4B model with about thirty layers is a small
            stand-in for the frontier systems the paper studies, where the workspace is
            sharper and the experiments go much further.
          </p>
        </Card>

        <div style={{ marginTop: 30, paddingTop: 18, borderTop: `1px solid ${C.border}`, fontFamily: LSERIF, fontSize: 14, color: C.muted, lineHeight: 1.6 }}>
          <p style={{ margin: "0 0 10px" }}>Start from the concrete readouts if you have not:</p>
          <ul style={{ margin: 0, paddingLeft: 18 }}>
            <li><a href="/a/watch-it-calculate" style={linkS}>Doing arithmetic</a> : a thought you can check against a right answer.</li>
            <li><a href="/a/multi-hop-readout" style={linkS}>A silent reasoning step</a> : reaching "Italy" without saying it.</li>
            <li><a href="/a/planning-ahead" style={linkS}>Planning the next word</a> : rhymes and typo fixes settled early.</li>
          </ul>
          <p style={{ marginTop: 16, fontSize: 13, color: C.faint }}>
            I computed these readouts on {data.model} with the lens {data.lens}. The
            framework, and the experiments this page only gestures at, come from{" "}
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

const codeSmall = { fontFamily: LMONO, fontSize: 12.5, background: "#f2efe9", padding: "1px 5px", borderRadius: 4 };
const h3S = { fontSize: 18, margin: "0 0 8px", fontWeight: 700 };
const pS = { fontSize: 15, lineHeight: 1.7, color: "#2c2823", margin: 0 };
const linkS = { color: "#1d4ed8", textDecoration: "none", borderBottom: "1px solid #bcd" };
