"""Generate the Jacobian-lens readouts baked into the interpretability artifacts.

This is the exact script behind src/artifacts/data/lens/*.json. It applies
Anthropic's released Jacobian lens to an open model and dumps per-layer readouts
for a curated prompt set drawn from the paper's own evaluation data.

Reproduce:

    uv venv && . .venv/bin/activate
    uv pip install torch --index-url https://download.pytorch.org/whl/cu124
    uv pip install git+https://github.com/anthropics/jacobian-lens.git huggingface_hub
    OUT=./out python scripts/gen-lens-data.py

Needs a CUDA GPU with about 10 GB free for Qwen3.5-4B. The lens is
neuronpedia/jacobian-lens@qwen-n1000; lens.apply is one forward pass plus a
linear transport, no fitting. Paper:
https://transformer-circuits.pub/2026/workspace/index.html
"""

from __future__ import annotations

import json
import os

import torch
import transformers

import jlens

MODEL_NAME = "Qwen/Qwen3.5-4B"
LENS_REPO = "neuronpedia/jacobian-lens"
LENS_REVISION = "qwen-n1000"
LENS_FILE = "qwen3.5-4b/jlens/Salesforce-wikitext/Qwen3.5-4B_jacobian_lens_n1000.pt"
OUT = os.environ.get("OUT", "./out")
TOP_K = 10

CALCULATE = [
    {
        "name": "sum-times",
        "expr": "(2 + 3) * 4",
        "primer": "(1 + 4) * 2 = 10\n(2 + 2) * 3 = 12\n",
        "answer": "20",
        "inter": "5",
        "track": ["5", "five", "20", "twenty", "multiplication"],
    },
    {
        "name": "paper-4-17",
        "expr": "(4 + 17) * 2",
        "primer": "(1 + 1) * 5 = 10\n(2 + 2) * 3 = 12\n",
        "answer": "42",
        "inter": "21",
        "track": ["21", "42", "forty", "forty-two", "twenty"],
    },
    {
        "name": "add-then-mult",
        "expr": "2 + 3 * 4",
        "primer": "3 + 2 * 4 = 11\n1 + 6 * 2 = 13\n",
        "answer": "14",
        "inter": "12",
        "track": ["12", "twelve", "14", "fourteen", "addition"],
    },
]
MULTIHOP = [
    {
        "name": "boot-currency",
        "prompt": "Fact: The currency used in the country shaped like a boot is",
        "answer": "euro",
        "track": ["Italy", "euro", "lira", "Rome"],
    },
    {
        "name": "carnival-ocean",
        "prompt": "Fact: The ocean on the coast of the country where Carnival is most famously celebrated is the",
        "answer": "Atlantic",
        "track": ["Brazil", "Atlantic", "Pacific"],
    },
]
PLANNING = [
    {
        "name": "couplet-breath-death",
        "prompt": "A rhyming couplet:\nThe soldier held his final, rattling breath,\nAnd closed his eyes to greet the face of",
        "answer": "death",
        "track": ["death", "God", "peace"],
    },
    {
        "name": "typo-language",
        "prompt": "The course required that every student learn a second langauge",
        "answer": "language",
        "track": ["language", "langauge"],
    },
]
WORKSPACE = [
    {
        "name": "summoning-spelling",
        "prompt": "She had organised the samples by shade and labelled each one before showing him the options. He recognised the effort, but red had never been his favourite",
        "answer": "colour",
        "track": ["colour", "color", "shade", "hue"],
    },
    {
        "name": "dont-say-fear",
        "chat": "Think about your greatest fear, but don't say it.",
        "prefill": "I",
        "answer": None,
        "track": ["death", "fear", "failure", "loss", "dark"],
    },
]
SECTIONS = {
    "calculate": CALCULATE,
    "multihop": MULTIHOP,
    "planning": PLANNING,
    "workspace": WORKSPACE,
}


def wordlike_mask(tok, vocab_size):
    # The paper restricts the displayed top-k to word-like tokens, because the
    # raw top of the distribution is dominated by punctuation and word fragments.
    mask = torch.zeros(vocab_size, dtype=torch.bool)
    for tid in range(vocab_size):
        try:
            dec = tok.decode([tid], clean_up_tokenization_spaces=False)
        except Exception:
            continue
        s = dec.strip()
        if not s:
            continue
        if "<|" in s or (s.startswith("<") and s.endswith(">")):
            continue
        ok = all(
            ch.isalnum() or (0 < pos < len(s) - 1 and ch in "'-’")
            for pos, ch in enumerate(s)
        )
        mask[tid] = ok
    return mask


def resolve_id(tok, word):
    lead = "" if word[0].isdigit() else " "
    return tok.encode(lead + word, add_special_tokens=False)[0]


def build_prompt(tok, item):
    if "chat" in item:
        msgs = [{"role": "user", "content": item["chat"]}]
        if item.get("prefill"):
            msgs.append({"role": "assistant", "content": item["prefill"]})
            return tok.apply_chat_template(
                msgs, tokenize=False, continue_final_message=True
            )
        return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    if "expr" in item:
        return item["primer"] + item["expr"] + " ="
    return item["prompt"]


def main():
    os.makedirs(OUT, exist_ok=True)
    hf = transformers.AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16
    ).cuda()
    tok = transformers.AutoTokenizer.from_pretrained(MODEL_NAME)
    model = jlens.from_hf(hf, tok)
    lens = jlens.JacobianLens.from_pretrained(
        LENS_REPO, filename=LENS_FILE, revision=LENS_REVISION
    )
    layers = list(lens.source_layers)

    vocab = model.unembed(torch.zeros(1, model.d_model)).shape[-1]
    wmask = wordlike_mask(tok, vocab)
    n_wordlike = int(wmask.sum())
    neg = torch.full((vocab,), float("-inf"))

    def masked_topk(logits):
        ml = torch.where(wmask, logits, neg)
        probs = torch.softmax(logits, dim=-1)
        top = ml.topk(TOP_K).indices.tolist()
        return [
            {
                "tok": tok.decode([t]).strip() or tok.decode([t]),
                "prob": round(probs[t].item(), 5),
            }
            for t in top
        ]

    for section, items in SECTIONS.items():
        results = []
        for item in items:
            prompt = build_prompt(tok, item)
            ids_t = model.encode(prompt)
            n_in = ids_t.shape[1]
            gen = hf.generate(
                ids_t.cuda(),
                max_new_tokens=6,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
            )[0].tolist()
            generated = tok.decode(gen[n_in:]).strip().split("\n")[0].strip()

            ids = ids_t[0].tolist()
            tokens = [tok.decode([i]) for i in ids]
            track_ids = {w: resolve_id(tok, w) for w in item["track"]}

            lens_logits, model_logits, _ = lens.apply(
                model, prompt, layers=layers, positions=[-1]
            )

            per_layer = []
            for layer in layers:
                logits = lens_logits[layer][0]
                probs = torch.softmax(logits, dim=-1)
                ranked = logits.argsort(descending=True)
                mranked = torch.where(wmask, logits, neg).argsort(descending=True)
                rank_full = {
                    tid: (ranked == tid).nonzero()[0, 0].item()
                    for tid in set(track_ids.values())
                }
                rank_word = {
                    tid: (mranked == tid).nonzero()[0, 0].item()
                    for tid in set(track_ids.values())
                }
                per_layer.append(
                    {
                        "layer": layer,
                        "top": masked_topk(logits),
                        "track": {
                            w: {
                                "rank": rank_full[track_ids[w]],
                                "rank_word": rank_word[track_ids[w]],
                                "prob": round(probs[track_ids[w]].item(), 6),
                            }
                            for w in item["track"]
                        },
                    }
                )

            entry = {
                "name": item["name"],
                "prompt": prompt,
                "tokens": tokens,
                "read_pos": -1,
                "read_token": tokens[-1],
                "answer": item.get("answer"),
                "generated": generated,
                "track_words": item["track"],
                "layers": layers,
                "n_layers": model.n_layers,
                "wordlike_vocab": n_wordlike,
                "per_layer": per_layer,
                "model_out": masked_topk(model_logits[0]),
            }
            if "expr" in item:
                entry["expr"] = item["expr"]
                entry["inter"] = item["inter"]
            results.append(entry)
            print(f"{section}/{item['name']}: model says {generated!r}", flush=True)

        with open(os.path.join(OUT, f"{section}.json"), "w") as f:
            json.dump(
                {
                    "model": MODEL_NAME,
                    "lens": f"{LENS_REPO}@{LENS_REVISION}",
                    "display": "top-k restricted to word-like tokens (paper mask); ranks full-vocab and word-only",
                    "items": results,
                },
                f,
                indent=1,
            )


if __name__ == "__main__":
    main()
