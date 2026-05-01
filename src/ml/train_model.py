# backend/src/ml/train_model.py
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import joblib

# 1. LOAD DATASET
df = pd.read_csv("../../data/sri_lanka_heritage_dataset.csv")

# Important Columns
coords = df[['lat', 'lon']].values

# 2. SCALE COORDINATES (BEST PRACTICE)
scaler = StandardScaler()
coords_scaled = scaler.fit_transform(coords)

# 3. TRAIN K-MEANS MODEL

# Choose number of clusters
# Sri Lanka has ~200 heritage sites → 8–15 clusters ideal
k = 12  # ideal for 200–1000 Sri Lankan points

kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(coords_scaled)

df['cluster'] = kmeans.labels_

# 3.1 SILHOUETTE SCORE EVALUATION
sil_score = silhouette_score(coords_scaled, kmeans.labels_)
print(f"Silhouette Score for k={k}: {sil_score:.4f}")

# 4. SAVE MODEL + SCALER
joblib.dump(kmeans, "../../models/model.pkl")
joblib.dump(scaler, "../../models/scaler.pkl")

print("Model training complete! Saved as model.pkl and scaler.pkl")

# 5. CLUSTER VISUALIZATION
plt.figure(figsize=(10, 8))
plt.scatter(df['lon'], df['lat'], c=df['cluster'], cmap='tab20', s=20)
plt.title("Sri Lanka Heritage Sites - KMeans Clustering")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.grid(True)
plt.show()

# 6. EXPORT CLUSTERED DATA
df.to_csv("../../data/heritage_clustered.csv", index=False)
print("Clustered CSV saved as heritage_clustered.csv")
