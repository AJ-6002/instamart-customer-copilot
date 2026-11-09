import sqlite3
import pandas as pd

con = sqlite3.connect("data/instamart.db")
text = open("sql/queries.sql", encoding="utf-8").read()
queries = [q.strip() for q in text.split(";") if q.strip()]

for i, q in enumerate(queries, 1):
    df = pd.read_sql_query(q, con)
    print("=" * 40)
    print(f"Q{i}: {len(df)} rows")
    if i == 2:
        print("total revenue:", round(df["revenue"].sum(), 2))
        print(df.sort_values("revenue", ascending=False).head(1))
    if i == 5:
        print("average:", df["avg_total"].iloc[0])
    if i == 6:
        print(df[df["row_num"] == 1][["customer_id", "city", "total_spend"]])
        print("rows where RANK != ROW_NUMBER:", int((df["rnk"] != df["row_num"]).sum()))
    if i == 7:
        print("last cumulative:", df["cumulative_revenue"].iloc[-1])

con.close()