import pandas as pd

customers = pd.read_csv("data/raw/customers.csv")

# (a) City: remove spaces at the ends, then Title Case
customers["city"] = customers["city"].str.strip().str.title()

# (b) Gender: map every spelling to male / female / unknown
g = customers["gender"].str.strip().str.lower()
customers["gender"] = g.map(
    {"male": "male", "m": "male", "female": "female", "f": "female"}
).fillna("unknown")

# (c) Age: outside 10-100 becomes missing, then fill with the city's median age
customers.loc[(customers["age"] < 10) | (customers["age"] > 100), "age"] = None
customers["age"] = customers["age"].fillna(
    customers.groupby("city")["age"].transform("median")
)

# (d) Signup date: text to datetime
customers["signup_date"] = pd.to_datetime(customers["signup_date"])

# Checks
print(customers["city"].value_counts())
print(customers["gender"].value_counts())
print("missing ages:", customers["age"].isna().sum())
print("age min:", customers["age"].min(), "age max:", customers["age"].max())
# ================= ORDERS =================
orders = pd.read_csv("data/raw/orders.csv")
products = pd.read_csv("data/raw/products.csv")

# Rule 1: order_ts to datetime
orders["order_ts"] = pd.to_datetime(orders["order_ts"])

# Rule 2: remove exact duplicate rows
before = len(orders)
orders = orders.drop_duplicates()
print("Rule 2 removed rows:", before - len(orders))

# Rule 3: for each order_id keep only the latest order_ts
before = len(orders)
orders = orders.sort_values("order_ts").drop_duplicates(subset="order_id", keep="last")
print("Rule 3 removed rows:", before - len(orders))

# Rule 4: amount to number, fill missing from the product price
orders = orders.merge(products[["product_id", "category", "price_inr"]], on="product_id", how="left")
orders["order_amount"] = pd.to_numeric(orders["order_amount"], errors="coerce")
print("Rule 4 missing amounts to fill:", orders["order_amount"].isna().sum())
calc = (orders["price_inr"] * orders["quantity"] * (1 - orders["discount_pct"] / 100)).round(2)
orders["order_amount"] = orders["order_amount"].fillna(calc)

# Rule 5: negative delivery times become missing
print("Rule 5 negative delivery times:", (orders["delivery_minutes"] < 0).sum())
orders.loc[orders["delivery_minutes"] < 0, "delivery_minutes"] = None
# completed orders: fill with the median of the product category (completed orders only)
done = orders["status"] == "completed"
med = orders[done].groupby("category")["delivery_minutes"].transform("median")
orders.loc[done, "delivery_minutes"] = orders.loc[done, "delivery_minutes"].fillna(med)
# cancelled orders: left missing on purpose

# Checks
print("rows:", len(orders))
print(orders["status"].value_counts())
print("missing order_amount:", orders["order_amount"].isna().sum())
print("missing delivery, completed:", orders.loc[done, "delivery_minutes"].isna().sum())
print("missing delivery, cancelled:", orders.loc[orders["status"] == "cancelled", "delivery_minutes"].isna().sum())
print("sum of completed order_amount:", round(orders.loc[done, "order_amount"].sum(), 2))