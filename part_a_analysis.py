import sqlite3
import pandas as pd

con = sqlite3.connect("data/instamart.db")
df = pd.read_sql_query("""
    SELECT o.*, c.city, p.category
    FROM orders o
    JOIN customers c ON o.customer_id = c.customer_id
    JOIN products p ON o.product_id = p.product_id
    WHERE o.status = 'completed'
""", con)
con.close()

# (a) pivot table with totals
pivot = pd.pivot_table(df, values="order_amount", index="city",
                       columns="category", aggfunc="sum", margins=True)
print("pivot grand total:", round(pivot.loc["All", "All"], 2))

city = df.groupby("city").agg(
    total=("order_amount", "sum"),
    average=("order_amount", "mean"),
    count=("order_amount", "count"),
).round(2)
print(city)

# (b) statistics
s = df["order_amount"]
q1, q3 = s.quantile(0.25), s.quantile(0.75)
iqr = q3 - q1
lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
print("mean:", round(s.mean(), 2), "median:", s.median(), "skew:", round(s.skew(), 2))
print("Q1:", q1, "Q3:", q3, "lower fence:", lower, "upper fence:", upper)
print("outliers below:", int((s < lower).sum()), "above:", int((s > upper).sum()))

corr = df[["order_amount", "discount_pct", "quantity", "delivery_minutes"]].corr().round(3)
print(corr)