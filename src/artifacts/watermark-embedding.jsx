import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import watermarkLM from "./data/watermark-lm.json";
import { mulberry32, hashSeed, sampleNextWord, detectWatermark } from "./data/watermark-lm-core.js";

export const meta = {
  title: "Claude's Watermark: Hidden in Word Choice",
  category: "LLM Systems",
  description:
    "Anthropic announced that Claude now watermarks generated text, and the natural guess is a hidden character or a metadata tag. Neither is true: the signal is which ordinary words get picked. Step through real word-by-word generation on a bigram model (a lookup table of word to next-word odds) trained on real text, watch the model hash the previous word into a seed, split its whole vocabulary into a green and red list from that seed, nudge the odds toward green, and see the statistical fingerprint that leaves behind.",
  date: "2026-08-14",
  tags: ["watermarking", "llm", "text-generation", "provenance", "hashing"],
};

const C = {
  bg: "#f7f5f2",
  card: "#ffffff",
  border: "#e4dfd8",
  ink: "#1c1a17",
  muted: "#857c72",
  faint: "#efeae3",
  accent: "#c0561f",
  accentSoft: "#f6ece5",
  blue: "#1e6fa0",
  blueSoft: "#e8f0f5",
  green: "#5a7d3c",
  greenSoft: "#eaf1e2",
  red: "#a8453f",
  redSoft: "#f7e9e7",
};

const MONO = "'IBM Plex Mono', ui-monospace, SFMono-Regular, Menlo, monospace";
const EASE = "cubic-bezier(0.23, 1, 0.32, 1)";

const KEY = 20260814;
const GAMMA = 0.25;
const DELTA = 4;
const TEMPERATURE = 1;

function usePrefersReducedMotion() {
  const [reduce, setReduce] = useState(false);
  useEffect(() => {
    if (typeof window === "undefined" || !window.matchMedia) return;
    const mq = window.matchMedia("(prefers-reduced-motion: reduce)");
    const apply = () => setReduce(mq.matches);
    apply();
    mq.addEventListener("change", apply);
    return () => mq.removeEventListener("change", apply);
  }, []);
  return reduce;
}

function Card({ children, style }) {
  return (
    <div style={{ background: C.card, border: `1px solid ${C.border}`, borderRadius: 14, padding: 20, ...style }}>
      {children}
    </div>
  );
}

function SectionTitle({ children }) {
  return (
    <h2 style={{ fontSize: 19, fontWeight: 700, margin: "34px 0 8px", lineHeight: 1.25, textWrap: "balance" }}>
      {children}
    </h2>
  );
}

function Prose({ children, style }) {
  return (
    <p style={{ fontSize: 14, lineHeight: 1.65, color: C.ink, margin: "0 0 14px", maxWidth: "68ch", ...style }}>
      {children}
    </p>
  );
}

function Caption({ children, style }) {
  return <p style={{ fontSize: 12, lineHeight: 1.6, color: C.muted, margin: "10px 0 0", ...style }}>{children}</p>;
}

function Btn({ children, onClick, disabled, primary, ariaLabel, ariaPressed }) {
  return (
    <button
      type="button"
      className="wme-press"
      onClick={onClick}
      disabled={disabled}
      aria-label={ariaLabel}
      aria-pressed={ariaPressed}
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
    >
      {children}
    </button>
  );
}

function Pill({ children, active, onClick, ariaLabel }) {
  return (
    <button
      type="button"
      className="wme-press"
      onClick={onClick}
      aria-pressed={active}
      aria-label={ariaLabel}
      style={{
        padding: "6px 12px",
        borderRadius: 999,
        border: `1.5px solid ${active ? C.accent : C.border}`,
        background: active ? C.accentSoft : "transparent",
        color: active ? C.accent : C.muted,
        fontWeight: active ? 700 : 500,
        fontSize: 12.5,
        cursor: "pointer",
        fontFamily: "inherit",
        transition: "background 160ms ease, border-color 160ms ease, color 160ms ease",
      }}
    >
      {children}
    </button>
  );
}

function Stat({ label, value, unit, tone }) {
  return (
    <div style={{ flex: "1 1 140px", background: C.faint, borderRadius: 10, padding: "12px 14px" }}>
      <div style={{ fontSize: 20, fontWeight: 700, color: tone || C.ink, lineHeight: 1.1 }}>
        {value}
        {unit && <span style={{ fontSize: 11, fontWeight: 500, color: C.muted, marginLeft: 4 }}>{unit}</span>}
      </div>
      <div style={{ fontSize: 11, color: C.muted, marginTop: 4, lineHeight: 1.4 }}>{label}</div>
    </div>
  );
}

function WordChip({ word, isGreen, sampled, reduce }) {
  return (
    <span
      style={{
        fontFamily: MONO,
        fontSize: 12.5,
        lineHeight: "20px",
        borderRadius: 6,
        padding: "1px 6px",
        display: "inline-block",
        background: isGreen ? C.greenSoft : C.redSoft,
        color: isGreen ? C.green : C.red,
        border: sampled ? `1.5px solid ${isGreen ? C.green : C.red}` : "1px solid transparent",
        fontWeight: sampled ? 700 : 500,
        transition: reduce ? "none" : `transform 200ms ${EASE}`,
        animation: sampled && !reduce ? "wme-settle 260ms " + EASE : "none",
      }}
    >
      {word}
    </span>
  );
}

const VOCAB = watermarkLM.vocab;
const VOCAB_SIZE = VOCAB.length;
const STOI = new Map(VOCAB.map((w, i) => [w, i]));

const PRESETS = [
  { label: "once upon a time", words: ["once", "upon", "a", "time", ","] },
  { label: "the little girl", words: ["the", "little", "girl"] },
  { label: "one day", words: ["one", "day", ","] },
  { label: "tom and his mom", words: ["tom", "and", "his", "mom"] },
];
const FALLBACK_WORD = "once";

function resolvePrompt(words) {
  const ids = words.map((w) => STOI.get(w)).filter((v) => v !== undefined);
  return ids.length ? ids : [STOI.get(FALLBACK_WORD)];
}

function BarPair({ pre, post, isGreen }) {
  const tone = isGreen ? C.green : C.red;
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 2, width: 34 }}>
      <div
        title={`before bias: ${(pre * 100).toFixed(1)}%`}
        style={{ height: 5, borderRadius: 3, background: C.faint, overflow: "hidden" }}
      >
        <div style={{ width: `${Math.min(100, pre * 100)}%`, height: "100%", background: `${tone}88` }} />
      </div>
      <div
        title={`after bias: ${(post * 100).toFixed(1)}%`}
        style={{ height: 5, borderRadius: 3, background: C.faint, overflow: "hidden" }}
      >
        <div style={{ width: `${Math.min(100, post * 100)}%`, height: "100%", background: tone }} />
      </div>
    </div>
  );
}

function DistributionPanel({ prevWord, hash, distribution, chosenWordIdx, watermark, reduce }) {
  const sortedByPost = [...distribution].sort((a, b) => b.postProb - a.postProb);
  return (
    <div>
      <div style={{ display: "flex", flexWrap: "wrap", gap: 10, alignItems: "baseline", marginBottom: 10 }}>
        <span style={{ fontSize: 12, color: C.muted }}>
          previous word <b style={{ color: C.ink, fontFamily: MONO }}>{prevWord}</b>
        </span>
        <span style={{ fontSize: 12, color: C.muted }}>
          hash(prev, key) &rarr; seed{" "}
          <b style={{ color: C.accent, fontFamily: MONO }}>{hash.toString(16).padStart(8, "0")}</b>
        </span>
        <span style={{ fontSize: 12, color: C.muted }}>
          green list &asymp; {Math.round(GAMMA * 100)}% of {VOCAB_SIZE} words
        </span>
      </div>
      <div
        style={{
          display: "flex",
          flexWrap: "wrap",
          gap: 10,
          padding: 12,
          borderRadius: 10,
          background: C.faint,
        }}
        role="img"
        aria-label={`Top candidate next words, ${watermark ? "with" : "without"} the green-list nudge applied. ${sortedByPost
          .map((c) => `${c.word}, ${c.isGreen ? "green" : "red"}, ${(c.postProb * 100).toFixed(1)} percent`)
          .join(". ")}`}
      >
        {sortedByPost.map((c) => (
          <div key={c.wordIdx} style={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 5 }}>
            <BarPair pre={c.preProb} post={c.postProb} isGreen={c.isGreen} />
            <WordChip word={c.word} isGreen={c.isGreen} sampled={c.wordIdx === chosenWordIdx} reduce={reduce} />
          </div>
        ))}
      </div>
      <Caption>
        Top row of each bar is the odds straight from the bigram table (a lookup of previous word &rarr; next-word
        counts, no different from a hash map you'd build for autocomplete); bottom row is the odds after{" "}
        {watermark ? (
          <>green-listed candidates get {`+${DELTA}`} added to their log-odds</>
        ) : (
          <>no nudge, since this is the Plain card</>
        )}
        . The outlined chip is the word this step actually sampled.
      </Caption>
    </div>
  );
}

// Sequences and step records live in refs, not state, so `step` reads and
// writes its own source of truth synchronously; setState only ever receives
// a plain snapshot. That keeps this safe under StrictMode's double-invoked
// updaters, which would otherwise double-consume the seeded rng.
function useGenerator(promptIds) {
  const rngRef = useRef(null);
  const seedRef = useRef(0);
  const wmSeqRef = useRef([]);
  const plainSeqRef = useRef([]);
  const wmStepsRef = useRef([]);
  const plainStepsRef = useRef([]);
  const [wmSeq, setWmSeq] = useState([]);
  const [plainSeq, setPlainSeq] = useState([]);
  const [wmSteps, setWmSteps] = useState([]);
  const [plainSteps, setPlainSteps] = useState([]);

  const reset = useCallback(() => {
    seedRef.current += 1;
    rngRef.current = { wm: mulberry32(1000 + seedRef.current), plain: mulberry32(1000 + seedRef.current) };
    wmSeqRef.current = [...promptIds];
    plainSeqRef.current = [...promptIds];
    wmStepsRef.current = [];
    plainStepsRef.current = [];
    setWmSeq(wmSeqRef.current);
    setPlainSeq(plainSeqRef.current);
    setWmSteps(wmStepsRef.current);
    setPlainSteps(plainStepsRef.current);
  }, [promptIds]);

  useEffect(() => {
    reset();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [promptIds.join(",")]);

  const step = useCallback(() => {
    if (!rngRef.current) return;
    const wmPrev = wmSeqRef.current[wmSeqRef.current.length - 1];
    const wmResult = sampleNextWord(wmPrev, watermarkLM, {
      watermark: true,
      key: KEY,
      gamma: GAMMA,
      delta: DELTA,
      temperature: TEMPERATURE,
      rng: rngRef.current.wm,
    });
    wmSeqRef.current = [...wmSeqRef.current, wmResult.wordIdx];
    wmStepsRef.current = [...wmStepsRef.current, { prevIdx: wmPrev, hash: hashSeed(wmPrev, KEY), ...wmResult }];

    const plainPrev = plainSeqRef.current[plainSeqRef.current.length - 1];
    const plainResult = sampleNextWord(plainPrev, watermarkLM, {
      watermark: false,
      key: KEY,
      gamma: GAMMA,
      delta: DELTA,
      temperature: TEMPERATURE,
      rng: rngRef.current.plain,
    });
    plainSeqRef.current = [...plainSeqRef.current, plainResult.wordIdx];
    plainStepsRef.current = [
      ...plainStepsRef.current,
      { prevIdx: plainPrev, hash: hashSeed(plainPrev, KEY), ...plainResult },
    ];

    setWmSeq(wmSeqRef.current);
    setPlainSeq(plainSeqRef.current);
    setWmSteps(wmStepsRef.current);
    setPlainSteps(plainStepsRef.current);
  }, []);

  return { wmSeq, plainSeq, wmSteps, plainSteps, step, reset };
}

function Generator({ reduce }) {
  const [presetIdx, setPresetIdx] = useState(0);
  const [customWord, setCustomWord] = useState("");
  const [useCustom, setUseCustom] = useState(false);
  const [focus, setFocus] = useState("watermarked");
  const [playing, setPlaying] = useState(false);

  const promptIds = useMemo(() => {
    if (useCustom) {
      const w = customWord.trim().toLowerCase();
      const id = STOI.get(w);
      return [id !== undefined ? id : STOI.get(FALLBACK_WORD)];
    }
    return resolvePrompt(PRESETS[presetIdx].words);
  }, [useCustom, customWord, presetIdx]);

  const { wmSeq, plainSeq, wmSteps, plainSteps, step, reset } = useGenerator(promptIds);

  useEffect(() => {
    if (!playing) return;
    if (wmSteps.length >= 24) {
      setPlaying(false);
      return;
    }
    const id = setTimeout(step, reduce ? 620 : 460);
    return () => clearTimeout(id);
  }, [playing, wmSteps.length, step, reduce]);

  const customInvalid = useCustom && customWord.trim() !== "" && !STOI.has(customWord.trim().toLowerCase());
  const focusSteps = focus === "watermarked" ? wmSteps : plainSteps;
  const lastStep = focusSteps[focusSteps.length - 1];

  const detection = useMemo(() => {
    if (wmSteps.length < 4) return null;
    return {
      wm: detectWatermark(wmSeq, KEY, VOCAB_SIZE, GAMMA),
      plain: detectWatermark(plainSeq, KEY, VOCAB_SIZE, GAMMA),
    };
  }, [wmSeq, plainSeq, wmSteps.length]);

  return (
    <Card>
      <div style={{ display: "flex", gap: 14, flexWrap: "wrap", marginBottom: 14 }}>
        <div>
          <div style={{ fontSize: 11, fontWeight: 700, color: C.muted, textTransform: "uppercase", letterSpacing: 0.8, marginBottom: 6 }}>
            Seed word
          </div>
          <div style={{ display: "flex", gap: 6, flexWrap: "wrap", alignItems: "center" }}>
            {PRESETS.map((p, i) => (
              <Pill key={p.label} active={!useCustom && presetIdx === i} onClick={() => { setPresetIdx(i); setUseCustom(false); }}>
                {p.label}
              </Pill>
            ))}
            <input
              type="text"
              value={customWord}
              placeholder="type a word..."
              aria-label="Type a custom seed word from the model's vocabulary"
              onFocus={() => setUseCustom(true)}
              onChange={(e) => {
                setUseCustom(true);
                setCustomWord(e.target.value);
              }}
              style={{
                fontFamily: "inherit",
                fontSize: 12.5,
                padding: "5px 10px",
                borderRadius: 999,
                border: `1.5px solid ${useCustom ? (customInvalid ? C.red : C.accent) : C.border}`,
                background: useCustom ? C.accentSoft : "transparent",
                color: C.ink,
                width: 120,
              }}
            />
          </div>
          {customInvalid && (
            <Caption style={{ color: C.red, margin: "6px 0 0" }}>
              &ldquo;{customWord.trim()}&rdquo; isn&apos;t in this toy model&apos;s 500-word vocabulary, so it&apos;ll fall back to
              &ldquo;{FALLBACK_WORD}&rdquo; until you type a word it knows.
            </Caption>
          )}
        </div>
      </div>

      <div style={{ display: "flex", gap: 18, flexWrap: "wrap", marginBottom: 14 }}>
        {[
          { id: "watermarked", label: "Watermarked", seq: wmSeq, tone: C.accent },
          { id: "plain", label: "Plain (no bias)", seq: plainSeq, tone: C.blue },
        ].map((track) => (
          <button
            key={track.id}
            type="button"
            className="wme-press"
            onClick={() => setFocus(track.id)}
            aria-pressed={focus === track.id}
            aria-label={`Show the mechanism detail for the ${track.label} continuation`}
            style={{
              flex: "1 1 260px",
              minWidth: 0,
              textAlign: "left",
              fontFamily: "inherit",
              cursor: "pointer",
              padding: "10px 12px",
              borderRadius: 10,
              border: `1.5px solid ${focus === track.id ? track.tone : C.border}`,
              background: focus === track.id ? `${track.tone}11` : C.faint,
            }}
          >
            <div style={{ fontSize: 11.5, fontWeight: 700, color: track.tone, marginBottom: 6 }}>{track.label}</div>
            <div aria-live="polite" style={{ fontFamily: MONO, fontSize: 12.5, lineHeight: 1.7, color: C.ink }}>
              {track.seq.map((id, i) => {
                const isPrompt = i < promptIds.length;
                const stepRec = !isPrompt && (track.id === "watermarked" ? wmSteps : plainSteps)[i - promptIds.length];
                return (
                  <span key={i} style={{ marginRight: 5, display: "inline-block" }}>
                    {isPrompt ? (
                      VOCAB[id]
                    ) : (
                      <span style={{ color: stepRec?.wasGreen ? C.green : C.red, fontWeight: 600 }}>{VOCAB[id]}</span>
                    )}
                  </span>
                );
              })}
            </div>
          </button>
        ))}
      </div>

      {lastStep ? (
        <DistributionPanel
          prevWord={VOCAB[lastStep.prevIdx]}
          hash={lastStep.hash}
          distribution={lastStep.distributionForDisplay}
          chosenWordIdx={lastStep.wordIdx}
          watermark={focus === "watermarked"}
          reduce={reduce}
        />
      ) : (
        <Caption>Press Step to sample the first word after the seed and see the mechanism fire.</Caption>
      )}

      <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginTop: 14, alignItems: "center" }}>
        <Btn primary onClick={step} disabled={wmSteps.length >= 24} ariaLabel="Sample one more word on both tracks">
          Step
        </Btn>
        <Btn onClick={() => setPlaying((p) => !p)} disabled={wmSteps.length >= 24} ariaLabel={playing ? "Pause autoplay" : "Autoplay steps"}>
          {playing ? "Pause" : "Play"}
        </Btn>
        <Btn onClick={reset} ariaLabel="Restart both tracks from the seed word with a fresh run">
          New run
        </Btn>
        <span style={{ marginLeft: "auto", fontSize: 12, color: C.muted }}>{wmSteps.length} / 24 words</span>
      </div>

      {detection && (
        <div style={{ marginTop: 16, paddingTop: 14, borderTop: `1px solid ${C.faint}` }}>
          <div style={{ fontSize: 11, fontWeight: 700, color: C.muted, textTransform: "uppercase", letterSpacing: 0.8, marginBottom: 8 }}>
            Detector readout, recomputed from the actual words above
          </div>
          <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
            <Stat label="watermarked run: green rate" value={`${detection.wm.greenCount}/${detection.wm.T}`} tone={C.accent} />
            <Stat label="watermarked z-score" value={detection.wm.z.toFixed(1)} tone={detection.wm.z > 4 ? C.green : C.ink} />
            <Stat label="plain run: green rate" value={`${detection.plain.greenCount}/${detection.plain.T}`} tone={C.blue} />
            <Stat label="plain z-score" value={detection.plain.z.toFixed(1)} tone={C.ink} />
          </div>
          <Caption>
            Anthropic doesn&apos;t need the model weights to check this, only the key. Recompute the same green list at
            every position and count how often the real text landed there; a z-score past about 4 happens by chance
            roughly once in 30,000 runs, while ordinary writing sits within a couple points of zero, exactly the split
            you can watch appear above as you add words.
          </Caption>
        </div>
      )}

      <Caption style={{ marginTop: detection ? 0 : 10 }}>
        Both tracks consume the same run of random draws, so any word where they differ is the bias, not luck.
        Toggle which one is inspected above with the two cards; keep pressing Step to watch the green share of the
        watermarked run climb while the plain run hovers near {Math.round(GAMMA * 100)}%.
      </Caption>
    </Card>
  );
}

export default function App() {
  const reduce = usePrefersReducedMotion();

  return (
    <div style={{ background: C.bg, minHeight: "100vh", color: C.ink, fontFamily: "system-ui, -apple-system, sans-serif" }}>
      <style>{`
        @keyframes wme-settle {
          from { opacity: 0; transform: translateY(3px) scale(0.94); }
          to { opacity: 1; transform: none; }
        }
        .wme-press:active { transform: scale(0.97); }
        .wme-press:focus-visible, button:focus-visible, input:focus-visible, a:focus-visible {
          outline: 2px solid ${C.accent};
          outline-offset: 2px;
        }
        @media (prefers-reduced-motion: reduce) {
          * { transition-duration: 1ms !important; animation-duration: 1ms !important; }
        }
      `}</style>
      <div style={{ maxWidth: 720, margin: "0 auto", padding: "28px 16px 60px" }}>
        <header style={{ marginBottom: 18 }}>
          <div style={{ fontSize: 11, fontWeight: 700, letterSpacing: 1.2, textTransform: "uppercase", color: C.accent, marginBottom: 5 }}>
            LLM Systems &middot; Provenance
          </div>
          <h1 style={{ fontSize: 26, fontWeight: 700, margin: 0, lineHeight: 1.2, textWrap: "balance" }}>
            Claude&apos;s Watermark: Hidden in Word Choice
          </h1>
          <Prose style={{ color: C.muted, margin: "10px 0 0", fontSize: 13.5 }}>
            Ask Claude to write the same paragraph twice and, character for character, both can look completely
            ordinary: no stray unicode, no invisible tag, nothing a diff would catch. Anthropic can still tell you
            afterward which one it wrote. The signal was never attached to the text; it is the text, specifically
            which of several equally reasonable words got picked at every position. Type a seed word below, step
            through real generation one word at a time, and watch that bias get computed live in your browser.
          </Prose>
        </header>

        <SectionTitle>Three steps, every single word</SectionTitle>
        <Prose>
          The model behind this page is a bigram (a lookup table built from real children&apos;s stories: for every
          word, which word tends to follow, and how often). At each position it does three things before it writes
          anything down. First, hash the previous word together with a secret key into one number, the same way a
          hash map turns a string key into a bucket index. Second, use that number to seed a pseudorandom shuffle of
          the model&apos;s entire vocabulary and take the first slice, about a quarter of it, as this position&apos;s
          &ldquo;green list&rdquo;; a different quarter at every position, but the same quarter every time that exact
          context and key recur. Third, before picking the next word, add a fixed bonus to every green candidate&apos;s
          odds. Nothing about which word looks natural changes: green and red candidates are both plausible, the die
          is just loaded toward green.
        </Prose>

        <SectionTitle>Watch a word get chosen</SectionTitle>
        <Prose>
          Pick where the sentence starts, then press Step. Both a watermarked and a plain continuation grow at once
          from the same random draws, so the only thing that can make them diverge is the bias itself. The panel
          below the two tracks shows the mechanism for whichever one you&apos;re inspecting: the hash, the top
          candidates split into green and red, their odds before and after the nudge, and which word the model
          actually wrote down.
        </Prose>
        <Generator reduce={reduce} />

        <SectionTitle>What this means for you</SectionTitle>
        <Prose>
          This toy runs a 500-word bigram table where the real system runs a full language model, but the check
          Anthropic performs afterward is the same shape: recompute the green list at every position from the key
          and count how often the text landed there. You can&apos;t detect it by reading; the words all read fine.
          You also can&apos;t strip it by asking a different model to lightly rephrase a sentence or two, since the
          statistic only firms up over many words and survives most single-word edits. Paraphrasing the whole thing
          in your own words, translating it, or generating it unwatermarked in the first place all remove it, because
          each replaces the biased word choices with an independent set.
        </Prose>

        <footer style={{ marginTop: 30, paddingTop: 16, borderTop: `1px solid ${C.border}` }}>
          <p style={{ fontSize: 12.5, color: C.muted, lineHeight: 1.7, margin: 0, maxWidth: "68ch" }}>
            Sources:{" "}
            <a
              href="https://support.claude.com/en/articles/16266773-how-claude-marks-ai-generated-content"
              target="_blank"
              rel="noreferrer"
              style={{ color: C.accent, textDecorationColor: `${C.accent}66`, fontWeight: 600 }}
            >
              Anthropic, &ldquo;How Claude marks AI-generated content&rdquo;
            </a>
            ; Kirchenbauer et al.,{" "}
            <a
              href="https://arxiv.org/abs/2301.10226"
              target="_blank"
              rel="noreferrer"
              style={{ color: C.accent, textDecorationColor: `${C.accent}66`, fontWeight: 600 }}
            >
              &ldquo;A Watermark for Large Language Models&rdquo;
            </a>{" "}
            (2023), the green-list-and-bias algorithm this page implements; Dathathri et al.,{" "}
            <a
              href="https://doi.org/10.1038/s41586-024-08025-4"
              target="_blank"
              rel="noreferrer"
              style={{ color: C.accent, textDecorationColor: `${C.accent}66`, fontWeight: 600 }}
            >
              &ldquo;Scalable watermarking for identifying large language model outputs&rdquo;
            </a>{" "}
            (SynthID-Text, Nature 2024), the closest deployed relative outside Anthropic. The bigram table above is
            real corpus statistics from TinyStories, computed by the training script committed alongside this site;
            every hash, shuffle, and sample above runs live in your browser.
          </p>
        </footer>
      </div>
    </div>
  );
}
