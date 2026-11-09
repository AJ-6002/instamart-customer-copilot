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