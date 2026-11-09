import glob
import os

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from mock_llm import chat

CHUNK_WORDS = 40

SYSTEM = (
    "You are a POLICY ASSISTANT. Answer only from the supplied context. "
    "If the answer is not in the context, say that you could not find the information."
)


def load_chunks(overlap=10):
    step = CHUNK_WORDS - overlap
    chunks = []
    for path in sorted(glob.glob("data/policies/*.txt")):
        words = open(path, encoding="utf-8").read().split()
        start = 0
        while True:
            chunks.append({"source": os.path.basename(path),
                           "text": " ".join(words[start:start + CHUNK_WORDS])})
            if start + CHUNK_WORDS >= len(words):
                break
            start += step
    return chunks


class Retriever:
    def __init__(self, overlap=10):
        self.chunks = load_chunks(overlap)
        self.vec = TfidfVectorizer()
        self.matrix = self.vec.fit_transform([c["text"] for c in self.chunks])

    def retrieve(self, query, k=3):
        sims = cosine_similarity(self.vec.transform([query]), self.matrix)[0]
        order = sims.argsort()[::-1][:k]
        return [{"source": self.chunks[i]["source"],
                 "score": float(sims[i]),
                 "text": self.chunks[i]["text"]} for i in order]

    def answer(self, question):
        hits = self.retrieve(question, 3)
        context = "\n".join(f"[{h['source']}]\n{h['text']}" for h in hits)
        user = f"CONTEXT:\n{context}\nQUESTION:\n{question}"
        resp = chat([{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": user}])
        return resp.choices[0].message.content, [h["source"] for h in hits]


# default retriever (overlap 10) reused by the refund agent
default = Retriever(10)
retrieve = default.retrieve
answer = default.answer


def evaluate(overlap):
    r = Retriever(overlap)
    ev = pd.read_csv("data/raw/policy_eval.csv", keep_default_na=False)
    doc_ok = fact_ok = fallback_ok = n_ans = n_unans = 0
    failures = []
    for _, row in ev.iterrows():
        text, sources = r.answer(row["question"])
        if row["expected_doc"] == "NONE":
            n_unans += 1
            if text.startswith("I could not find"):
                fallback_ok += 1
            else:
                failures.append((row["question"], "no fallback"))
        else:
            n_ans += 1
            d = row["expected_doc"] in sources
            f = row["expected_fact"] in text
            doc_ok += d
            fact_ok += f
            if not (d and f):
                failures.append((row["question"], f"doc_ok={d} fact_ok={f}"))
    print(f"overlap {overlap}: chunks {len(r.chunks)}")
    print(f"  expected_doc in top 3: {doc_ok}/{n_ans}")
    print(f"  answer contains expected_fact: {fact_ok}/{n_ans}")
    print(f"  fallback on unanswerable: {fallback_ok}/{n_unans}")
    for q, why in failures:
        print("  FAIL:", q, "->", why)
    return r


if __name__ == "__main__":
    r = Retriever(10)
    counts = pd.Series([c["source"] for c in r.chunks]).value_counts()
    print("chunks total:", len(r.chunks))
    print(counts.to_string())
    print("top 3 sources for 'refund policy late delivery':",
          [h["source"] for h in retrieve("refund policy late delivery")])
    print("=" * 40)
    evaluate(10)
    print("=" * 40)
    evaluate(0)