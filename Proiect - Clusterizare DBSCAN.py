from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score


# Load dataset
data = pd.read_csv("C:/Users/elena/OneDrive/Desktop/Cursuri master/ML/mental_health/Mental_health_Preprocessed_MICE.csv", low_memory=False)

# Verifică structura datelor
print(data.head())  # Primele 5 rânduri
print(data.info())  # Informații generale despre dataset

# Selectăm caracteristicile pentru clustering
features_for_clustering = data[[ 'Depression (%)', 'Drug use disorders (%)']].values

# Standardizăm datele
scaler = StandardScaler()
features_scaled = scaler.fit_transform(features_for_clustering)


# Calculăm distanțele minime pentru a selecta parametrul eps
nearest_neighbors = NearestNeighbors(n_neighbors=2)  # Al doilea vecin cel mai apropiat
nearest_neighbors.fit(features_scaled)
distances, indices = nearest_neighbors.kneighbors(features_scaled)

# Sortăm distanțele și le plotează pentru a alege eps
distances = np.sort(distances[:, 1])  # Distanțele până la al doilea vecin
plt.figure(figsize=(10, 6))
plt.plot(distances)
plt.title("Grafic pentru determinarea parametrului eps (DBSCAN)")
plt.xlabel("Index puncte")
plt.ylabel("Distanța până la al 2-lea cel mai apropiat vecin")
plt.grid(True)
plt.show()


# Setăm eps și min_samples 
eps = 0.5  
min_samples = 5

# Aplicăm DBSCAN
dbscan = DBSCAN(eps=eps, min_samples=min_samples)
dbscan_labels = dbscan.fit_predict(features_scaled)

# Plotează rezultatele clusterizării DBSCAN
plt.figure(figsize=(8, 6))
plt.scatter(features_scaled[:, 0], features_scaled[:, 1], c=dbscan_labels, cmap='viridis', s=50, alpha=0.6)
plt.title(f"Clusterizare DBSCAN (eps={eps}, min_samples={min_samples})")
plt.xlabel("Depression")
plt.ylabel("Drug use disorders")
plt.grid(True)
plt.show()

# Afișăm statistici despre clustere
n_clusters = len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0)
n_noise = list(dbscan_labels).count(-1)

print(f"Numărul de clustere identificate: {n_clusters}")
print(f"Numărul de puncte zgomot (noise): {n_noise}")

# Evaluăm performanța clusterizării DBSCAN (dacă există cel puțin 2 clustere)
if n_clusters > 1:
    silhouette = silhouette_score(features_scaled, dbscan_labels)
    db_score = davies_bouldin_score(features_scaled, dbscan_labels)
    print(f"Silhouette Score = {silhouette:.4f}")
    print(f"Davies-Bouldin Score = {db_score:.4f}")
