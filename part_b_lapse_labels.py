import pandas as pd
from sklearn.model_selection import train_test_split

orders = pd.read_csv("data/clean/orders.csv", parse_dates=["order_ts"])
done = orders[orders["status"] == "completed"]

cutoff = pd.Timestamp("2025-09-01")
end = pd.Timestamp("2025-10-01")

before = done[done["order_ts"] < cutoff]
cohort_ids = sorted(before["customer_id"].unique())

september = done[(done["order_ts"] >= cutoff) & (done["order_ts"] < end)]
active_in_sept = set(september["customer_id"])

labels = pd.DataFrame({"customer_id": cohort_ids})
labels["lapsed"] = (~labels["customer_id"].isin(active_in_sept)).astype(int)

print("cohort size:", len(labels))
print("lapsed:", int(labels["lapsed"].sum()))
print("lapsed percent:", round(labels["lapsed"].mean() * 100, 2))

train, test = train_test_split(
    labels, test_size=0.2, stratify=labels["lapsed"], random_state=42
)
print("train rows:", len(train), "test rows:", len(test))

baseline_accuracy = (test["lapsed"] == 0).mean()
print("always-not-lapsed accuracy:", round(baseline_accuracy * 100, 1))