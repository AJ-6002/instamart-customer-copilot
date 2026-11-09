import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

data = pd.read_csv("outputs/lapse_dataset.csv")

y = data["lapsed"]
X = data.drop(columns=["customer_id", "lapsed"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

nominal = ["city", "gender", "fav_category"]
numeric = ["recency", "frequency", "monetary", "avg_discount",
           "cancel_rate", "avg_delivery", "age", "tenure_days"]

pre = ColumnTransformer([
    ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), nominal),
    ("num", StandardScaler(), numeric),
])

# learn from TRAIN only, then apply to both
Xtr = pre.fit_transform(X_train)
Xte = pre.transform(X_test)

print("train shape:", Xtr.shape, "test shape:", Xte.shape)

scaler = pre.named_transformers_["num"]
freq_pos = numeric.index("frequency")
print("scaler learned mean of frequency (train only):", round(scaler.mean_[freq_pos], 3))
print("full-cohort mean of frequency:", round(data["frequency"].mean(), 3))

print("missing in X_train:", int(X_train.isna().sum().sum()))
print("missing in X_test:", int(X_test.isna().sum().sum()))