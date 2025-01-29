# -*- coding: utf-8 -*-
"""
Created on Wed Dec  4 10:56:39 2024

@author: elena
"""

# Autor: Elena
import pandas as pd
from sklearn.impute import SimpleImputer
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
import time
from sklearn.model_selection import train_test_split

# Încarcă dataset-ul cu low_memory=False
data = pd.read_csv("https://github.com/elena33e/Machine-Learning-Scripts/blob/main/Mental_health_Preprocessed_MICE.csv", low_memory=False)

# Verifică structura datelor
print(data.head())
print(data.info())  


numeric_data = data.select_dtypes(include=['number'])

# Calculează matricea de corelație
correlation_matrix = numeric_data.corr()

# Afișează matricea de corelație
print("Matricea de corelație:\n", correlation_matrix)

# Vizualizare cu seaborn heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Matricea de corelație')
plt.show()

# Corelația pentru coloana "Anxiety disorders (%)"
anxiety_correlation = correlation_matrix['Anxiety disorders (%)']

# Sortează coeficienții de corelație în ordine descrescătoare
sorted_anxiety_correlation = anxiety_correlation.abs().sort_values(ascending=False)

# Afișează coeficienții de corelație pentru "Anxiety disorders (%)"
print("Coeficienții de corelație pentru Anxiety disorders (%):\n", sorted_anxiety_correlation)
                         
# Definim y (targetul)
y = data['Anxiety disorders (%)'].values 

# Aplicam regresia liniara pentru 1, 2 si 3 atribute
regressor = LinearRegression()

for features in [['Depression (%)'], 
                 ['Drug use disorders (%)', 'Depression (%)'],
                 ['Drug use disorders (%)', 'Depression (%)', 'Bipolar disorder (%)']]:
    X = data[features].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
    start_time = time.time()
    # Antrenam modelul
    regressor.fit(X_train, y_train)
    
    # Facem predicții pe setul de test
    y_pred = regressor.predict(X_test)
    
    # Calculam coeficientul R^2 pe setul de test
    r_sq = regressor.score(X_test, y_test)
    
    end_time = time.time()  # Timpul de sfârșit
    computation_time = end_time - start_time
    print(f'R^2 pentru predictorii {features}: {r_sq}')
    print(f"Timp de calcul: {computation_time:.4f} secunde\n")
    
    # Verificăm dacă numărul de variabile este 1 sau 2 pentru a genera graficul
    if len(features) == 1:
        # Regresie univariată
        plt.figure(figsize=(10, 6))
        X_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
        y_pred_line = regressor.predict(X_range)
        
        plt.scatter(X_test, y_test, color='blue', label='Valori reale')
        plt.plot(X_range, y_pred_line, color='red', label='Linia de regresie')
        plt.scatter(X_test, y_pred, color='green', alpha=0.5, label='Valori prezise')
        plt.xlabel(features[0])
        plt.ylabel('Anxiety disorders')
        plt.title(f'Regresie univariată: {features[0]} vs Anxiety disorders\nR^2: {r_sq:.4f}')
        plt.legend()
        plt.show()
        
    elif len(features) == 2:
        # Regresie bivariată
        x1_range = np.linspace(X[:, 0].min(), X[:, 0].max(), 50)
        x2_range = np.linspace(X[:, 1].min(), X[:, 1].max(), 50)
        x1_grid, x2_grid = np.meshgrid(x1_range, x2_range)
        
        # Creăm gridul de intrare pentru planul de regresie
        X_grid = np.c_[x1_grid.ravel(), x2_grid.ravel()]
        y_pred_grid = regressor.predict(X_grid).reshape(x1_grid.shape)

        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Scatter plot pentru valorile reale
        ax.scatter(X_test[:, 0], X_test[:, 1], y_test, color='blue', label='Valori reale', alpha=0.5)
        
        # Scatter plot pentru valorile prezise
        ax.scatter(X_test[:, 0], X_test[:, 1], y_pred, color='green', label='Valori prezise', alpha=0.5)
        
        # Planul de regresie
        ax.plot_surface(x1_grid, x2_grid, y_pred_grid, color='red', alpha=0.5, rstride=100, cstride=100)
        
        # Setăm etichetele axelor
        ax.set_xlabel(features[0])
        ax.set_ylabel(features[1])
        ax.set_zlabel('Anxiety disorders')
        plt.title(f'Plan de regresie bivariată: {features[0]}, {features[1]} vs Anxiety disorders\nR^2: {r_sq:.4f}')
        plt.legend()
        plt.show()
