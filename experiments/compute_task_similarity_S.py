"""Compute the task lexical-similarity matrix S for dil (FUNSD/SROIE/CORD).

Memory-light: reads ONLY the raw OCR `tokens` (text) from each HF dataset, capped to a
small sample, accumulates a bag-of-words per task, discards as it goes (no image decode,
no LayoutLMv3 tokenizer, no full KIE build). This is the go/no-go falsification: does S
have CONTRAST (receipts cluster, form is an outlier) or is it FLAT (idea dead)?

S[i,j] = cosine(idf-weighted bag-of-words_i, ..._j) — the same signal the method's
sparse_doc_vectors uses, but at the raw-word level (tokenizer-independent, so it reflects
true vocabulary overlap, not subword-id artifacts).
"""
import json
import math
from collections import Counter

from datasets import load_dataset

CAP = 200  # docs per task — enough for a stable vocab profile, tiny RAM


def _cord_tokens(ex):
    """cord-v2 nests OCR in ground_truth JSON: valid_line[*].words[*].text."""
    try:
        gt = json.loads(ex["ground_truth"])
    except (KeyError, TypeError, json.JSONDecodeError):
        return []
    out = []
    for line in gt.get("valid_line", []):
        for w in line.get("words", []):
            t = w.get("text", "").strip()
            if t:
                out.append(t)
    return out


def task_bow(name, split, token_col):
    """Streaming bag-of-words over up to CAP docs' raw tokens."""
    bow = Counter()
    ndocs = 0
    ds = load_dataset(name, split=split, streaming=True)
    for ex in ds:
        if token_col == "cord":
            toks = _cord_tokens(ex)
        else:
            toks = ex.get(token_col) or ex.get("words") or ex.get("tokens")
        if not toks:
            continue
        for t in toks:
            w = str(t).lower().strip()
            if w:
                bow[w] += 1
        ndocs += 1
        if ndocs >= CAP:
            break
    return bow, ndocs


def main():
    specs = [
        ("FUNSD", "nielsr/funsd-layoutlmv3", "train", "tokens"),
        ("SROIE", "mp-02/sroie", "train", "words"),
        ("CORD", "naver-clova-ix/cord-v2", "train", "cord"),  # nested ground_truth JSON
    ]
    bows, vocabs = {}, {}
    for label, name, split, col in specs:
        try:
            bow, n = task_bow(name, split, col)
            bows[label] = bow
            vocabs[label] = set(bow)
            print(f"{label}: {n} docs, vocab={len(bow)} unique words, top={bow.most_common(8)}")
        except Exception as e:
            print(f"{label}: FAILED to load ({e}); skipping")

    labels = list(bows)
    if len(labels) < 2:
        print("\nNot enough tasks loaded to compute S.")
        return

    # IDF over the union (a word in fewer tasks is more discriminative).
    df = Counter()
    for v in vocabs.values():
        for w in v:
            df[w] += 1
    nT = len(labels)
    idf = {w: math.log((1 + nT) / (1 + df[w])) + 1 for w in df}

    def vec(bow):
        # L2-normalized idf-weighted tf.
        v = {w: bow[w] * idf[w] for w in bow}
        norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
        return {w: x / norm for w, x in v.items()}

    vecs = {label: vec(bows[label]) for label in labels}

    def cos(a, b):
        keys = set(a) & set(b)
        return sum(a[w] * b[w] for w in keys)

    print("\n=== Task lexical-similarity matrix S (idf-weighted BoW cosine) ===")
    print("        " + "".join(f"{label:>8}" for label in labels))
    for li in labels:
        row = "".join(f"{cos(vecs[li], vecs[lj]):>8.3f}" for lj in labels)
        print(f"{li:>8}{row}")

    # Raw Jaccard too (vocabulary overlap, idf-free) for a sanity cross-check.
    print("\n=== Raw vocab Jaccard (|A∩B|/|A∪B|) ===")
    print("        " + "".join(f"{label:>8}" for label in labels))
    for li in labels:
        row = ""
        for lj in labels:
            inter = len(vocabs[li] & vocabs[lj])
            union = len(vocabs[li] | vocabs[lj]) or 1
            row += f"{inter/union:>8.3f}"
        print(f"{li:>8}{row}")

    print("\nGO/NO-GO: if SROIE-CORD >> FUNSD-{SROIE,CORD}, S has the receipts-cluster")
    print("contrast the method needs. If all off-diagonals are similar, S is FLAT -> idea dead.")


if __name__ == "__main__":
    main()
