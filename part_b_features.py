import os
import pandas as pd

orders = pd.read_csv("data/clean/orders.csv", parse_dates=["order_ts"])
customers = pd.read_csv("data/clean/customers.csv", parse_dates=["signup_date"])
products = pd.read_csv("data/clean/products.csv")

cutoff = pd.Timestamp("2025-09-01")
end = pd.Timestamp("2025-10-01")

# only orders BEFORE the cutoff are used for features
before_all = orders[orders["order_ts"] < cutoff]
before_done = before_all[before_all["status"] == "completed"]

# numbers from completed orders
feat = before_done.groupby("customer_id").agg(
    last_order=("order_ts", "max"),
    frequency=("order_id", "nunique"),
    monetary=("order_amount", "sum"),
    avg_discount=("discount_pct", "mean"),
    avg_delivery=("delivery_minutes", "mean"),
).reset_index()
feat["recency"] = (cutoff - feat["last_order"]).dt.days

# cancellation rate: cancelled orders divided by all orders
cancel = before_all.groupby("customer_id").agg(
    total_orders=("order_id", "nunique"),
    cancelled=("status", lambda s: (s == "cancelled").sum()),
).reset_index()
cancel["cancel_rate"] = cancel["cancelled"] / cancel["total_orders"]
feat = feat.merge(cancel[["customer_id", "cancel_rate"]], on="customer_id", how="left")

# favourite category (most orders; ties go to the first name in A-Z order)
cat = before_done.merge(products[["product_id", "category"]], on="product_id")
counts = cat.groupby(["customer_id", "category"]).size().reset_index(name="n")
counts = counts.sort_values(["customer_id", "n", "category"], ascending=[True, False, True])
fav = counts.drop_duplicates("customer_id")[["customer_id", "category"]]
fav = fav.rename(columns={"category": "fav_category"})
feat = feat.merge(fav, on="customer_id", how="left")

# customer facts: city, age, gender, tenure
feat = feat.merge(customers[["customer_id", "city", "age", "gender", "signup_date"]],
                  on="customer_id", how="left")
feat["tenure_days"] = (cutoff - feat["signup_date"]).dt.days

# label: 1 = no completed order in September 2025
sept = orders[(orders["status"] == "completed")
              & (orders["order_ts"] >= cutoff) & (orders["order_ts"] < end)]
active = set(sept["customer_id"])
feat["lapsed"] = (~feat["customer_id"].isin(active)).astype(int)

cols = ["customer_id", "recency", "frequency", "monetary", "avg_discount",
        "cancel_rate", "avg_delivery", "fav_category", "city", "age",
        "gender", "tenure_days", "lapsed"]
data = feat[cols]

os.makedirs("outputs", exist_ok=True)
data.to_csv("outputs/lapse_dataset.csv", index=False)

print("rows:", len(data))
print("lapsed:", int(data["lapsed"].sum()))
print("missing values in features:", int(data.drop(columns="lapsed").isna().sum().sum()))
print(data.head(3).to_string(index=False))