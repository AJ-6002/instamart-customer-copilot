import re
import pandas as pd
from mock_llm import est_tokens

MAX_INPUT_TOKENS = 400
PATTERNS = [
    ("EMAIL", re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")),
    ("PHONE", re.compile(r"\b[6-9]\d{9}\b")),
    ("ORDER_ID", re.compile(r"ORD-\d{6}")),
]


def redact(text):
    counts = {}
    for name, pat in PATTERNS:
        text, n = pat.subn(f"[{name}]", text)
        counts[name] = n
    return text, counts


def guard(text):
    if text.strip() == "":
        return "skipped_empty"
    if est_tokens(text) > MAX_INPUT_TOKENS:
        return "flagged_long"
    return "ok"


if __name__ == "__main__":
    t = pd.read_csv("data/raw/tickets.csv", keep_default_na=False)
    totals = {"EMAIL": 0, "PHONE": 0, "ORDER_ID": 0}
    redacted = []
    for text in t["text"]:
        r, c = redact(text)
        redacted.append(r)
        for k in totals:
            totals[k] += c[k]

    status = t["text"].map(guard)
    print("status counts:")
    print(status.value_counts().to_string())
    print("substitutions:", totals, "total:", sum(totals.values()))

    left = sum(len(p.findall(r)) for r in redacted for _, p in PATTERNS)
    print("left after redaction:", left)
    long_text = t.loc[status == "flagged_long", "text"].iloc[0]
    print("long ticket tokens:", est_tokens(long_text))