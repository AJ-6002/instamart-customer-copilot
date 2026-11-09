import os
import pandas as pd

orders = pd.read_csv("data/clean/orders.csv", parse_dates=["order_ts"])
done = orders[orders["status"] == "completed"]

snapshot = pd.Timestamp("2025-10-01 00:00:00")

rfm = done.groupby("customer_id").agg(
    last_order=("order_ts", "max"),
    frequency=("order_id", "nunique"),
    monetary=("order_amount", "sum"),
).reset_index()

rfm["recency"] = (snapshot - rfm["last_order"]).dt.days
rfm = rfm[["customer_id", "recency", "frequency", "monetary"]]

os.makedirs("outputs", exist_ok=True)
rfm.to_csv("outputs/rfm.csv", index=False)

print("customers:", len(rfm))
print("recency median:", rfm["recency"].median(), "min:", rfm["recency"].min(), "max:", rfm["recency"].max())
print("frequency mean:", round(rfm["frequency"].mean(), 2), "max:", rfm["frequency"].max())
print("monetary mean:", round(rfm["monetary"].mean(), 2), "max:", round(rfm["monetary"].max(), 2))