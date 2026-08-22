import { useState, useEffect, useRef, useCallback } from "react";

export const meta = {
  title: "What 'Agentic' Actually Means",
  category: "LLM Systems",
  description:
    "When an LLM 'calls a tool,' it never touches a filesystem, a network socket, or an API. It writes one blob of structured JSON and stops. Everything you'd call 'agentic' happens in the code around it. Walk the real request/response cycle, one message at a time.",
  date: "2026-08-22",
  tags: ["agents", "tool-use", "function-calling", "llm-systems"],
};

const C = {
  bg: "#f7f5f2",
  card: "#ffffff",
  border: "#e4dfd8",
  ink: "#1c1a17",
  muted: "#8c8278",
  accent: "#c0561f",
  accentL: "#f6ece5",
  green: "#2e7d51",
  greenL: "#e4f2eb",
  blue: "#2a5298",
  blueL: "#e5ecf8",
  gold: "#9a7020",
  goldL: "#f5edd8",
  faint: "#efeae3",
};

const MONO = "'IBM Plex Mono', ui-monospace, SFMono-Regular, Menlo, monospace";
const SERIF = "'Palatino Linotype', 'Book Antiqua', Palatino, Georgia, serif";

const ACTORS = {
  user: { label: "You", color: C.ink, bg: C.faint },
  model: { label: "The Model", color: C.accent, bg: C.accentL },
  harness: { label: "The Harness (your code)", color: C.blue, bg: C.blueL },
};

const STEPS = [
  {
    actor: "user",
    title: "You send a question, plus a tool the model is allowed to use",
    body:
      "The request has two parts: the conversation, and a `tools` array describing what get_weather takes as input. Nothing has been called yet — this is just a menu the model can read.",
    code: `POST /v1/messages
{
  "model": "claude-opus-5",
  "max_tokens": 1024,
  "tools": [
    {
      "name": "get_weather",
      "description": "Get the current weather for a location",
      "input_schema": {
        "type": "object",
        "properties": {
          "location": { "type": "string" },
          "unit": { "type": "string", "enum": ["celsius", "fahrenheit"] }
        },
        "required": ["location"]
      }
    }
  ],
  "messages": [
    { "role": "user", "content": "What's the weather in Lisbon right now?" }
  ]
}`,
  },
  {
    actor: "model",
    title: "The model replies with a tool_use block — not an answer",
    body:
      "It can't answer yet; it doesn't know the weather. So instead of prose, it emits structured JSON: a tool name, an input object matching the schema, and an id. stop_reason is \"tool_use\", a signal that means \"I'm not done, run this and come back.\" This block is the model's entire contribution to the loop. It has not queried anything.",
    code: `{
  "id": "msg_01A...",
  "role": "assistant",
  "stop_reason": "tool_use",
  "content": [
    {
      "type": "tool_use",
      "id": "toolu_01X8...",
      "name": "get_weather",
      "input": { "location": "Lisbon, Portugal" }
    }
  ]
}`,
  },
  {
    actor: "harness",
    title: "Your code reads the JSON and runs real code",
    body:
      "This step never touches an LLM. Plain application code: parse the tool_use block, look up \"get_weather\" in a dictionary of functions you registered, call it with the model's input, and get back a real value from a real weather API. The model has no idea this step is happening — it's just waiting.",
    code: `// inside your application, not inside the model
const block = response.content.find(b => b.type === "tool_use");
// block.name === "get_weather"
// block.input === { location: "Lisbon, Portugal" }

const registeredTools = { get_weather: fetchWeatherFromRealAPI };
const result = await registeredTools[block.name](block.input);
// result = { tempC: 19, condition: "partly cloudy" }`,
  },
  {
    actor: "harness",
    title: "Your code sends the result back, tagged with the same id",
    body:
      "The harness appends the model's own tool_use turn to the conversation, then adds a tool_result carrying the exact tool_use_id from step 2. That id is the only thing linking \"here is what get_weather returned\" back to \"the call the model asked for.\" Nothing here required a model call — it's message bookkeeping.",
    code: `POST /v1/messages
{
  "model": "claude-opus-5",
  "max_tokens": 1024,
  "tools": [ /* same tools array as step 1 */ ],
  "messages": [
    { "role": "user", "content": "What's the weather in Lisbon right now?" },
    { "role": "assistant", "content": [ /* the tool_use block from step 2 */ ] },
    {
      "role": "user",
      "content": [
        {
          "type": "tool_result",
          "tool_use_id": "toolu_01X8...",
          "content": "{\\"tempC\\":19,\\"condition\\":\\"partly cloudy\\"}"
        }
      ]
    }
  ]
}`,
  },
  {
    actor: "model",
    title: "The model reads the result and finally answers",
    body:
      "This is a fresh prediction pass over the whole conversation, now including the tool_result. The model never executed anything; it just read text you handed it, the same way it reads anything else in context. stop_reason is back to \"end_turn\" because there's nothing left to wait on.",
    code: `{
  "id": "msg_01B...",
  "role": "assistant",
  "stop_reason": "end_turn",
  "content": [
    {
      "type": "text",
      "text": "It's 19°C and partly cloudy in Lisbon right now."
    }
  ]
}`,
  },
];

function Label({ children, color = C.muted }) {
  return (
    <div
      style={{
        fontSize: 10,
        letterSpacing: "0.14em",
        textTransform: "uppercase",
        color,
        fontFamily: MONO,
        marginBottom: 8,
      }}
    >
      {children}
    </div>
  );
}

function Button({ children, onClick, variant = "ghost", disabled, title }) {
  const base = {
    fontFamily: MONO,
    fontSize: 12,
    borderRadius: 9,
    padding: "9px 16px",
    cursor: disabled ? "not-allowed" : "pointer",
    transition:
      "transform 140ms cubic-bezier(0.23,1,0.32,1), background 140ms ease, border-color 140ms ease",
    opacity: disabled ? 0.45 : 1,
    border: "1.5px solid",
    userSelect: "none",
  };
  const styles =
    variant === "solid"
      ? { ...base, background: C.accent, color: "#fff", borderColor: C.accent }
      : variant === "dark"
      ? { ...base, background: C.ink, color: "#fff", borderColor: C.ink }
      : { ...base, background: C.card, color: C.ink, borderColor: C.border };
  const ref = useRef(null);
  return (
    <button
      ref={ref}
      className="atc-focus"
      title={title}
      disabled={disabled}
      onClick={onClick}
      onPointerDown={() => {
        if (!disabled && ref.current) ref.current.style.transform = "scale(0.97)";
      }}
      onPointerUp={() => {
        if (ref.current) ref.current.style.transform = "scale(1)";
      }}
      onPointerLeave={() => {
        if (ref.current) ref.current.style.transform = "scale(1)";
      }}
      style={styles}
    >
      {children}
    </button>
  );
}

function ActorPill({ actorKey }) {
  const a = ACTORS[actorKey];
  return (
    <span
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: 6,
        fontFamily: MONO,
        fontSize: 11,
        fontWeight: 700,
        color: a.color,
        background: a.bg,
        borderRadius: 20,
        padding: "4px 12px",
      }}
    >
      <span
        style={{
          width: 6,
          height: 6,
          borderRadius: "50%",
          background: a.color,
          display: "inline-block",
        }}
      />
      {a.label}
    </span>
  );
}

function Rail({ step, onJump }) {
  return (
    <div
      role="tablist"
      aria-label="tool-calling cycle steps"
      style={{ display: "flex", gap: 6, flexWrap: "wrap", marginBottom: 18 }}
    >
      {STEPS.map((s, i) => {
        const active = i === step;
        const a = ACTORS[s.actor];
        return (
          <button
            key={i}
            role="tab"
            aria-selected={active}
            className="atc-focus"
            onClick={() => onJump(i)}
            title={s.title}
            style={{
              flex: "1 1 60px",
              minWidth: 44,
              height: 8,
              borderRadius: 5,
              border: "none",
              cursor: "pointer",
              background: active ? a.color : C.faint,
              opacity: i < step ? 0.55 : 1,
              transition: "background 200ms ease, opacity 200ms ease",
            }}
          />
        );
      })}
    </div>
  );
}

export default function App() {
  const [step, setStep] = useState(0);
  const [auto, setAuto] = useState(false);

  const reducedMotion =
    typeof window !== "undefined" &&
    window.matchMedia &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  const next = useCallback(() => {
    setStep((s) => {
      if (s >= STEPS.length - 1) {
        setAuto(false);
        return s;
      }
      return s + 1;
    });
  }, []);

  const prev = useCallback(() => {
    setStep((s) => Math.max(0, s - 1));
  }, []);

  useEffect(() => {
    if (!auto) return;
    if (step >= STEPS.length - 1) {
      setAuto(false);
      return;
    }
    const id = setTimeout(next, reducedMotion ? 900 : 2200);
    return () => clearTimeout(id);
  }, [auto, step, next, reducedMotion]);

  const current = STEPS[step];
  const actor = ACTORS[current.actor];

  return (
    <div
      style={{
        fontFamily: SERIF,
        background: C.bg,
        minHeight: "100vh",
        color: C.ink,
        padding: "26px 14px 56px",
      }}
    >
      <style>{`@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600;700&display=swap');
        .atc-focus:focus-visible { outline: 2px solid ${C.accent}; outline-offset: 2px; border-radius: 8px; }
      `}</style>

      <div style={{ maxWidth: 720, margin: "0 auto" }}>
        <header style={{ marginBottom: 22 }}>
          <div
            style={{
              fontSize: 10,
              letterSpacing: "0.16em",
              textTransform: "uppercase",
              color: C.muted,
              fontFamily: MONO,
              marginBottom: 7,
            }}
          >
            LLM Systems
          </div>
          <h1
            style={{
              fontSize: 27,
              fontWeight: 700,
              margin: 0,
              letterSpacing: "-0.01em",
              textWrap: "balance",
            }}
          >
            What &ldquo;Agentic&rdquo; Actually Means
          </h1>
          <p
            style={{
              color: C.ink,
              fontSize: 14.5,
              lineHeight: 1.62,
              margin: "12px 0 0",
              maxWidth: "64ch",
              textWrap: "pretty",
            }}
          >
            Ask an LLM to check the weather and something in your head pictures it reaching out
            and querying an API. It never does. Its entire contribution is one blob of JSON
            describing which function to call and with what arguments, and then it stops and
            waits. The rest, running the function, sending the answer back, deciding to try
            again, is five plain messages bouncing between the model and the code that hosts
            it. Step through the exact cycle below, one real Messages API payload at a time.
          </p>
        </header>

        <section
          style={{
            background: C.accentL,
            border: `1px solid ${C.accent}33`,
            borderRadius: 12,
            padding: "14px 16px",
            marginBottom: 18,
            fontSize: 12.5,
            lineHeight: 1.6,
            color: C.ink,
          }}
        >
          <strong style={{ color: C.accent }}>&ldquo;Agentic&rdquo; is not new model tech.</strong>{" "}
          It's a label for a loop wrapped around an ordinary model call: look at the
          conversation so far, decide on one action, get a result, decide the next action,
          repeat until there's nothing left to do. The model was already capable of the
          &ldquo;decide&rdquo; part. What changed is that people started writing the loop.
        </section>

        <section
          style={{
            background: C.card,
            border: `1px solid ${C.border}`,
            borderRadius: 14,
            padding: "18px 16px",
            marginBottom: 16,
          }}
        >
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
            <Label color={C.ink}>
              step {step + 1} of {STEPS.length}
            </Label>
            <ActorPill actorKey={current.actor} />
          </div>

          <Rail step={step} onJump={(i) => { setStep(i); setAuto(false); }} />

          <h2
            style={{
              fontSize: 16.5,
              fontWeight: 700,
              margin: "0 0 8px",
              lineHeight: 1.4,
              color: actor.color,
            }}
          >
            {current.title}
          </h2>
          <p
            style={{
              fontSize: 13.5,
              lineHeight: 1.65,
              color: C.ink,
              margin: "0 0 14px",
              maxWidth: "62ch",
            }}
          >
            {current.body}
          </p>

          <div
            style={{
              background: C.ink,
              borderRadius: 10,
              padding: "14px 16px",
              overflowX: "auto",
            }}
          >
            <pre
              style={{
                margin: 0,
                fontFamily: MONO,
                fontSize: 12,
                lineHeight: 1.6,
                color: "#f3ede4",
                whiteSpace: "pre",
              }}
            >
              {current.code}
            </pre>
          </div>

          <div
            style={{
              display: "flex",
              gap: 8,
              flexWrap: "wrap",
              marginTop: 16,
              alignItems: "center",
            }}
          >
            <Button onClick={prev} disabled={step === 0}>
              &larr; back
            </Button>
            <Button
              variant="solid"
              onClick={next}
              disabled={step === STEPS.length - 1 && !auto}
              title={step === STEPS.length - 1 ? "cycle complete" : "advance one step"}
            >
              {step === STEPS.length - 1 ? "cycle complete" : "next →"}
            </Button>
            <Button
              variant="dark"
              onClick={() => {
                if (step === STEPS.length - 1) setStep(0);
                setAuto((a) => !a);
              }}
            >
              {auto ? "pause" : "play through"}
            </Button>
          </div>
        </section>

        <section
          style={{
            background: C.blueL,
            border: `1px solid ${C.blue}33`,
            borderRadius: 12,
            padding: "15px 17px",
            marginBottom: 16,
            fontSize: 12.5,
            lineHeight: 1.65,
            color: C.ink,
          }}
        >
          <Label color={C.blue}>what the harness owns</Label>
          <p style={{ margin: 0 }}>
            Everything with side effects: parsing the JSON, matching{" "}
            <code style={{ fontFamily: MONO }}>"get_weather"</code> against a dictionary of
            functions you wrote and registered, actually calling that function, catching its
            errors, and re-serializing the return value into the message the model reads next.
            The model's tool definition is a description it was shown, not a live connection to
            anything. If the harness never runs step 3, the &ldquo;call&rdquo; stays a string
            sitting in a JSON blob forever.
          </p>
        </section>

        <section
          style={{
            background: C.goldL,
            border: `1px solid ${C.gold}33`,
            borderRadius: 12,
            padding: "15px 17px",
            marginBottom: 16,
            fontSize: 12.5,
            lineHeight: 1.65,
            color: C.ink,
          }}
        >
          <Label color={C.gold}>what the model still owns</Label>
          <p style={{ margin: "0 0 9px" }}>
            Owning zero execution doesn't mean owning zero judgment. Three things sit entirely
            on the model's side of the fence, and a harness can't paper over a bad call in any
            of them:
          </p>
          <p style={{ margin: "0 0 9px" }}>
            <strong>Picking the right tool.</strong> With a dozen registered functions instead
            of one, choosing correctly, and not inventing a thirteenth that was never
            registered, is a judgment call made from the tool descriptions alone.
          </p>
          <p style={{ margin: "0 0 9px" }}>
            <strong>Knowing when to stop.</strong> Call again for more information, or answer
            with what's already in hand? Loop forever and the harness eventually has to cut it
            off; stop too early and the answer is wrong.
          </p>
          <p style={{ margin: 0 }}>
            <strong>Staying inside the schema.</strong> Emit malformed or slightly-off-schema
            JSON at scale and the harness's parser rejects the call before it ever reaches step
            3. Reliability here is a property of the model, not the plumbing.
          </p>
        </section>

        <section
          style={{
            background: C.card,
            border: `1px solid ${C.border}`,
            borderRadius: 14,
            padding: "18px 16px",
            marginBottom: 16,
          }}
        >
          <Label>brain and body</Label>
          <p style={{ fontSize: 13, lineHeight: 1.65, margin: 0, color: C.ink }}>
            The model is a brain with no hands: it can want the weather checked, but it cannot
            reach out and check it. The harness is the body, arms that fetch, eyes that read
            the result back. A brain with better judgment picks the right hand for the job and
            knows when to stop reaching; it still never touches the world directly. That
            separation is also why a harness can hand a model a brand-new tool it has never
            seen before, described only by a schema, and get a reasonable first attempt back:
            the judgment generalizes even when the specific hand is new.
          </p>
        </section>

        <footer
          style={{
            marginTop: 22,
            textAlign: "center",
            fontSize: 11,
            color: C.muted,
            fontFamily: MONO,
            lineHeight: 1.6,
          }}
        >
          real Anthropic Messages API tool_use / tool_result shapes &middot; the loop, not the
          model, is what &ldquo;agentic&rdquo; names
        </footer>
      </div>
    </div>
  );
}
