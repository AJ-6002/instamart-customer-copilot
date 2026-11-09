import sqlite3
import pandas as pd

con = sqlite3.connect("data/instamart.db")
text = open("sql/queries.sql", encoding="utf-8").read()
queries = [q.strip() for q in text.split(";") if q.strip()]

for i, q in enumerate(queries, 1):
    df = pd.read_sql_query(q, con)
    print("=" * 40)
    print(f"Q{i}: {len(df)} rows")
    print(df.head(3))
    if i == 2:
        print("total revenue:", round(df["revenue"].sum(), 2))
        print("largest cell:")
        print(df.sort_values("revenue", ascending=False).head(1))

con.close()