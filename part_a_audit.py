import pandas as pd

orders = pd.read_csv("data/raw/orders.csv")
customers = pd.read_csv("data/raw/customers.csv")
products = pd.read_csv("data/raw/products.csv")

tables = {"orders": orders, "customers": customers, "products": products}

# Shape, data types, missing values for every table
for name, df in tables.items():
    print("=" * 50)
    print(name, "shape:", df.shape)
    print(df.dtypes)
    missing = df.isna().sum()
    percent = (missing / len(df) * 100).round(2)
    print(pd.DataFrame({"missing": missing, "percent": percent}))

# Audit figures for orders
print("=" * 50)
print("ORDERS AUDIT")
print("raw rows:", len(orders))
print("exact duplicate rows:", orders.duplicated().sum())
print("pending rows:", (orders["status"] == "pending").sum())
print("distinct order_id:", orders["order_id"].nunique())
amount_numeric = pd.to_numeric(orders["order_amount"], errors="coerce")
print("non-numeric amounts:", amount_numeric.isna().sum())
print("negative delivery_minutes:", (orders["delivery_minutes"] < 0).sum())
print("blank delivery_minutes:", orders["delivery_minutes"].isna().sum())

# Audit figures for customers
print("=" * 50)
print("CUSTOMERS AUDIT")
print("blank age:", customers["age"].isna().sum())
bad_age = (customers["age"] < 10) | (customers["age"] > 100)
print("age outside 10-100:", bad_age.sum())
print("distinct raw city spellings:", customers["city"].nunique())
print("distinct non-blank gender spellings:", customers["gender"].nunique())
print("blank gender:", customers["gender"].isna().sum())