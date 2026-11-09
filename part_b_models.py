import pandas as pd, joblib
from sklearn.base import clone
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, accuracy_score

data = pd.read_csv("outputs/lapse_dataset.csv")
y = data["lapsed"]
X = data.drop(columns=["lapsed"])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
ids_test = X_test["customer_id"].values
X_train = X_train.drop(columns="customer_id")
X_test = X_test.drop(columns="customer_id")

nominal = ["city", "gender", "fav_category"]
numeric = ["recency", "frequency", "monetary", "avg_discount", "cancel_rate", "avg_delivery", "age", "tenure_days"]

def make_pipe(model):
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), nominal),
        ("num", StandardScaler(), numeric)])
    return Pipeline([("pre", pre), ("clf", model)])

def report(name, model):
    p = model.predict(X_test)
    print(name, "confusion matrix:", confusion_matrix(y_test, p).tolist())
    print(name, "precision %.3f recall %.3f F1 %.3f" % (
        precision_score(y_test, p, zero_division=0),
        recall_score(y_test, p, zero_division=0),
        f1_score(y_test, p, zero_division=0)))
    return f1_score(y_test, p, zero_division=0)

base = [0] * len(y_test)
print("BASELINE accuracy %.3f F1 %.3f" % (accuracy_score(y_test, base), f1_score(y_test, base, zero_division=0)))

lr = GridSearchCV(make_pipe(LogisticRegression(max_iter=1000)),
                  {"clf__C": [0.01, 0.1, 1, 10, 100]}, cv=5, scoring="f1")
lr.fit(X_train, y_train)
print("LR best:", lr.best_params_, "cv F1 %.3f" % lr.best_score_)

rf = GridSearchCV(make_pipe(RandomForestClassifier(n_estimators=100, random_state=42)),
                  {"clf__max_depth": [3, 5, 8], "clf__min_samples_leaf": [5, 10, 20]},
                  cv=5, scoring="f1")
rf.fit(X_train, y_train)
print("RF best:", rf.best_params_, "cv F1 %.3f" % rf.best_score_)

f_lr = report("LR", lr.best_estimator_)
f_rf = report("RF", rf.best_estimator_)
if f_rf >= f_lr:
    best, best_name = rf.best_estimator_, "RandomForest"
else:
    best, best_name = lr.best_estimator_, "LogisticRegression"
print("SELECTED:", best_name)

minority = X_train[y_train == 1]
n_add = int((y_train == 0).sum() - (y_train == 1).sum())
extra = minority.sample(n=n_add, replace=True, random_state=42)
Xb = pd.concat([X_train, extra])
yb = pd.concat([y_train, pd.Series([1] * n_add, index=extra.index)])
print("balanced train counts:", yb.value_counts().to_dict())
lr_bal = clone(lr.best_estimator_).fit(Xb, yb)
print("--- LR without balancing")
report("LR", lr.best_estimator_)
print("--- LR with balancing")
report("LR balanced", lr_bal)

rf_best = rf.best_estimator_
names = rf_best.named_steps["pre"].get_feature_names_out()
imp = pd.Series(rf_best.named_steps["clf"].feature_importances_, index=names).sort_values(ascending=False)
print("TOP 5 RF FEATURES:")
print(imp.head(5).round(4))
lr_best = lr.best_estimator_
coef = pd.Series(lr_best.named_steps["clf"].coef_[0],
                 index=lr_best.named_steps["pre"].get_feature_names_out())
print("LR coefficients:")
print(coef[["num__recency", "num__frequency", "num__tenure_days"]].round(3))

joblib.dump(best, "outputs/lapse_model.joblib")
loaded = joblib.load("outputs/lapse_model.joblib")
print("reloaded predictions identical:", bool((loaded.predict(X_test) == best.predict(X_test)).all()))

seg = pd.read_csv("outputs/customer_segments.csv")
t = pd.DataFrame({"customer_id": ids_test,
                  "lapse_probability": best.predict_proba(X_test)[:, 1],
                  "actually_lapsed": y_test.values})
t = t.merge(seg[["customer_id", "is_premium"]], on="customer_id", how="left")
t = t[t["is_premium"] == True].sort_values("lapse_probability", ascending=False).head(10)
t[["customer_id", "lapse_probability", "actually_lapsed"]].to_csv("outputs/retention_list.csv", index=False)
print("retention list rows:", len(t), "truly lapsed:", int(t["actually_lapsed"].sum()))