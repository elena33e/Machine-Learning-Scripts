# -*- coding: utf-8 -*-
"""
Created on Sat Dec 28 13:38:15 2024

@author: elena
"""

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


# Iterăm prin diferite valori pentru n_clusters
silhouette_scores = []
db_scores = []
range_n_clusters = range(2, 7)

for n_clusters in range_n_clusters:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(features_scaled)
    
    # Silhouette Score
    silhouette = silhouette_score(features_scaled, kmeans.labels_)
    silhouette_scores.append(silhouette)
    
    # DB Score
    db_score = davies_bouldin_score(features_scaled, kmeans.labels_)
    db_scores.append(db_score)
    
    
    # Plotăm clusterele 
    plt.figure(figsize=(8, 6))
    plt.scatter(features_scaled[:, 0], features_scaled[:, 1], c=kmeans.labels_, cmap='viridis', s=50, alpha=0.6)
    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], 
                s=200, c='red', marker='X', label='Centroizi')
    plt.title(f"Clusterele K-Means (n_clusters = {n_clusters})")
    plt.xlabel("Depresssion")
    plt.ylabel("Drug use disorders")
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # Indicii pentru fiecare număr de clustere
    print(f"n_clusters = {n_clusters}")
    print(f"Silhouette Score = {silhouette:.4f}")
    print(f"Davies-Bouldin Score = {db_score:.4f}")
    print("-" * 30)

# Graficul pentru Elbow
inertias = []

for n_clusters in range_n_clusters:
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(features_scaled)
    inertias.append(kmeans.inertia_)  

# Plotează Elbow
plt.figure(figsize=(10, 6))
plt.plot(range_n_clusters, inertias, marker='o', linestyle='-', color='b', label='Inertia')
plt.title("Metoda Elbow")
plt.xlabel("Număr de clustere")
plt.ylabel("Inertia")
plt.xticks(range_n_clusters)
plt.legend()
plt.grid(True)
plt.show()

# Plotăm rezultatele pentru toate metricile
plt.figure(figsize=(10, 6))
plt.plot(range_n_clusters, silhouette_scores, marker='o', label='Silhouette Score')
plt.plot(range_n_clusters, db_scores, marker='s', label='DB Score')
plt.title("Metrici de evaluare pentru diferite valori ale n_clusters")
plt.xlabel("Număr de clustere")
plt.ylabel("Valoare metrică")
plt.legend()
plt.grid(True)
plt.show()
