import json
import random
import re
import time

import pandas as pd

import mock_llm
from mock_llm import chat, TransientLLMError
from part_c_prep import redact, guard, PATTERNS

ALLOWED = ["billing", "delivery", "technical", "fraud"]
BASE_DELAY = 0.05
MAX_RETRIES = 3

ZERO_SHOT = (
    "You CLASSIFY customer support tickets. "
    "Allowed categories: billing, delivery, technical, fraud. "
    'Reply with JSON only, in the form {"category": "...", "reason": "..."}.'
)

FEW_SHOT = (
    "You CLASSIFY customer support tickets. "
    "Allowed categories: billing, delivery, technical, fraud. "
    'Reply with JSON only, in the form {"category": "...", "reason": "..."}.\n'
    "EXAMPLES:\n"
    "Ticket: I was charged twice for [ORDER_ID]. Please refund the extra payment. -> "
    '{"category": "billing", "reason": "double charge"}\n'
    "Ticket: Someone hacked my account and used my wallet balance. -> "
    '{"category": "fraud", "reason": "unauthorized access"}\n'
    "Ticket: The app crashes every time I open the cart. -> "
    '{"category": "technical", "reason": "app error"}\n'
    "Ticket: My order has not arrived and the rider is not picking up calls. -> "
    '{"category": "delivery", "reason": "late or missing order"}'
)

tickets = pd.read_csv("data/raw/tickets.csv", keep_default_na=False)
segs = pd.read_csv("outputs/customer_segments.csv")
segs["is_premium"] = segs["is_premium"].astype(str).str.lower() == "true"
seg_info = segs.set_index("customer_id")[["segment_name", "is_premium"]].to_dict("index")

retry_log = []


def call_with_retry(messages, ticket_id, stats):
    retry = 0
    while True:
        try:
            return chat(messages)
        except TransientLLMError:
            stats["errors"] += 1
            if retry >= MAX_RETRIES:
                raise
            wait = BASE_DELAY * 2 ** retry * random.uniform(0.9, 1.15)
            retry_log.append(f"ticket={ticket_id} retry={retry} wait={wait:.4f}s")
            stats["waits"].append(wait)
            time.sleep(wait)
            retry += 1


def validate(text):
    try:
        data = json.loads(text)
        if isinstance(data, dict) and data.get("category") in ALLOWED:
            return "classified_json", data["category"]
        return "flagged_invalid_output", ""
    except (json.JSONDecodeError, TypeError):
        words = re.findall(r"[a-z]+", (text or "").lower())
        found = {w for w in words if w in ALLOWED}
        if len(found) == 1:
            return "classified_keyword", found.pop()
        return "flagged_invalid_output", ""


def route(status, category, customer_id):
    info = seg_info.get(customer_id)
    premium = bool(info["is_premium"]) if info else False
    if status not in ("classified_json", "classified_keyword"):
        return "human_review", "NORMAL"
    if category == "fraud":
        return "security_team", "HIGH"
    if premium:
        return "premium_desk", "HIGH"
    return "support_queue", "NORMAL"


def run_pipeline(system_prompt, budget):
    mock_llm.reset()
    stats = {"errors": 0, "waits": []}
    total = 0
    rows = []
    for _, r in tickets.iterrows():
        status, category, used = guard(r["text"]), "", 0
        if status == "ok":
            if total >= budget:
                status = "budget_exhausted"
            else:
                red, _ = redact(r["text"])
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": red},
                ]
                for m in messages:
                    assert not any(p.search(m["content"]) for _, p in PATTERNS), "PII in prompt"
                resp = call_with_retry(messages, r["ticket_id"], stats)
                used = resp.usage.total_tokens
                total += used
                status, category = validate(resp.choices[0].message.content)
        route_name, priority = route(status, category, r["customer_id"])
        info = seg_info.get(r["customer_id"])
        rows.append({
            "ticket_id": r["ticket_id"],
            "customer_id": r["customer_id"],
            "status": status,
            "category": category,
            "segment_name": info["segment_name"] if info else "",
            "priority": priority,
            "route": route_name,
            "tokens_used": used,
            "true_category": r["true_category"],
        })
    return pd.DataFrame(rows), stats, total


def report(name, df, stats, total):
    ok = df[df["status"].isin(["classified_json", "classified_keyword"])]
    acc = (ok["category"] == ok["true_category"]).mean()
    print("=" * 50)
    print(name)
    print(df["status"].value_counts().to_string())
    print("transient errors:", stats["errors"])
    print("classified:", len(ok), "correct:", int((ok["category"] == ok["true_category"]).sum()))
    print("accuracy:", round(acc * 100, 1), "%")
    print("total tokens:", total, "tokens per classified ticket:", round(total / len(ok), 1))
    return ok


zero, zs, zt = run_pipeline(ZERO_SHOT, 10**9)
ok_zero = report("ZERO-SHOT (no budget)", zero, zs, zt)
print("confusion table (rows = true, columns = predicted):")
print(pd.crosstab(ok_zero["true_category"], ok_zero["category"]))

few, fs, ft = run_pipeline(FEW_SHOT, 10**9)
report("FEW-SHOT (no budget)", few, fs, ft)

demo, ds, dt = run_pipeline(ZERO_SHOT, 6000)
cols = ["ticket_id", "customer_id", "status", "category", "segment_name", "priority", "route", "tokens_used"]
demo[cols].to_csv("outputs/triage_budget_demo.csv", index=False)
print("=" * 50)
print("BUDGET DEMO (zero-shot, budget 6000)")
print("budget_exhausted tickets:", int((demo["status"] == "budget_exhausted").sum()))

final, fis, fit = run_pipeline(FEW_SHOT, 100000)
final[cols].to_csv("outputs/triage_results.csv", index=False)
print("=" * 50)
print("FINAL (few-shot, budget 100000): rows", len(final))
print(final["route"].value_counts().to_string())

with open("outputs/retry_log.txt", "w") as f:
    f.write("\n".join(retry_log))
waits = [float(x.split("wait=")[1].rstrip("s")) for x in retry_log]
print("=" * 50)
print("retry log lines:", len(retry_log), "min wait:", min(waits), "max wait:", max(waits))
print("first 3 log lines:")
print("\n".join(retry_log[:3]))