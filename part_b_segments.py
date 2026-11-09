import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

rfm = pd.read_csv("outputs/rfm.csv")
cols = ["recency", "frequency", "monetary"]
X = StandardScaler().fit_transform(rfm[cols])

km = KMeans(n_clusters=4, n_init=10, random_state=42).fit(X)
rfm["segment_id"] = km.labels_

profile = rfm.groupby("segment_id").agg(
    size=("customer_id", "count"),
    mean_recency=("recency", "mean"),
    mean_frequency=("frequency", "mean"),
    mean_monetary=("monetary", "mean"),
).round(2)
print(profile)

pca = PCA(n_components=2).fit(X)
ratio = pca.explained_variance_ratio_
print("PCA ratios:", ratio.round(3), "total:", round(ratio.sum(), 3))

# name the segments by their profile, not by their number
premium_id = profile["mean_monetary"].idxmax()
lapsing_id = profile["mean_recency"].idxmax()
loyal_id = profile["mean_frequency"].idxmax()

names = {}
for sid in profile.index:
    if sid == premium_id:
        names[sid] = "Premium Big Spenders"
    elif sid == lapsing_id:
        names[sid] = "Lapsing Customers"
    elif sid == loyal_id:
        names[sid] = "Loyal Regulars"
    else:
        names[sid] = "Casual Buyers"

rfm["segment_name"] = rfm["segment_id"].map(names)
rfm["is_premium"] = rfm["segment_id"] == premium_id

out = rfm[["customer_id", "recency", "frequency", "monetary",
           "segment_id", "segment_name", "is_premium"]]
out.to_csv("outputs/customer_segments.csv", index=False)

print(out.groupby(["segment_id", "segment_name", "is_premium"]).size())