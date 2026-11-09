import sqlite3
import pandas as pd

raw_orders = pd.read_csv("data/raw/orders.csv")
raw_cust = pd.read_csv("data/raw/customers.csv")

con = sqlite3.connect("data/instamart.db")
orders = pd.read_sql_query("SELECT * FROM orders", con)
customers = pd.read_sql_query("SELECT * FROM customers", con)
con.close()

done = orders["status"] == "completed"
cancelled = orders["status"] == "cancelled"

checks = [
    # name, your value, expected value
    ("raw order rows", len(raw_orders), 13084),
    ("exact duplicates", int(raw_orders.duplicated().sum()), 119),
    ("pending rows", int((raw_orders["status"] == "pending").sum()), 656),
    ("distinct order_id", raw_orders["order_id"].nunique(), 12309),
    ("non-numeric amounts",
     int(pd.to_numeric(raw_orders["order_amount"], errors="coerce").isna().sum()), 120),
    ("negative delivery_minutes", int((raw_orders["delivery_minutes"] < 0).sum()), 106),
    ("blank delivery_minutes", int(raw_orders["delivery_minutes"].isna().sum()), 2541),
    ("blank ages", int(raw_cust["age"].isna().sum()), 61),
    ("ages outside 10-100",
     int(((raw_cust["age"] < 10) | (raw_cust["age"] > 100)).sum()), 6),
    ("raw city spellings", raw_cust["city"].nunique(), 24),
    ("raw gender spellings", raw_cust["gender"].nunique(), 10),
    ("blank genders", int(raw_cust["gender"].isna().sum()), 30),
    ("clean order rows", len(orders), 12309),
    ("completed orders", int(done.sum()), 11110),
    ("cancelled orders", int(cancelled.sum()), 1199),
    ("pending after cleaning", int((orders["status"] == "pending").sum()), 0),
    ("missing order_amount", int(orders["order_amount"].isna().sum()), 0),
    ("missing delivery, completed", int(orders.loc[done, "delivery_minutes"].isna().sum()), 0),
    ("missing delivery, cancelled", int(orders.loc[cancelled, "delivery_minutes"].isna().sum()), 1199),
    ("sum completed amount", round(float(orders.loc[done, "order_amount"].sum()), 2), 2998668.8),
    ("clean cities", customers["city"].nunique(), 6),
    ("missing ages after cleaning", int(customers["age"].isna().sum()), 0),
    ("min age", int(customers["age"].min()), 18),
    ("max age", int(customers["age"].max()), 60),
]

for name, actual, expected in checks:
    assert actual == expected, f"FAILED: {name}: got {actual}, expected {expected}"
    print("ok:", name, actual)

print("ALL CHECKPOINTS PASSED")