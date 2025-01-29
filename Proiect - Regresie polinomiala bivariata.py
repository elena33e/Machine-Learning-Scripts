# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 15:11:22 2024

@author: elena
"""

import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

# Încarcă dataset-ul cu low_memory=False
data = pd.read_csv("C:/Users/elena/OneDrive/Desktop/Cursuri master/ML/mental_health/Mental_health_Preprocessed_MICE.csv", low_memory=False)

# Verifică structura datelor
print(data.head())  # Primele 5 rânduri
print(data.info())  # Informații generale despre dataset


numeric_data = data.select_dtypes(include=['number'])

#Definim X si y
X = data[['Drug use disorders (%)', 'Depression (%)']].values 
y = data['Anxiety disorders (%)'].values  

print('X: \n', X)
print('y: ', y)  
X, y = np.array(X), np.array(y)

# Transforming input data
# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Test different polynomial degrees and report R^2 

for degree in [2, 3, 4, 5, 6, 7, 8]:
    start_time = time.time()
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_poly_train = poly.fit_transform(X_train)
    X_poly_test = poly.transform(X_test)
    
    model = LinearRegression().fit(X_poly_train, y_train)
    r_sq = model.score(X_poly_train, y_train)
    
    y_pred = model.predict(X_poly_test)
    test_r_sq = model.score(X_poly_test, y_test)
    
    end_time = time.time() 
    computation_time = end_time - start_time
    print("\nDegree:", degree)
    print("Training R^2:", r_sq)
    print("Test R^2:", test_r_sq)
    print(f"Timp de calcul: {computation_time:.4f} secunde\n")


# Plotarea rezultatelor pentru gradul 6
degree = 8
poly = PolynomialFeatures(degree=degree, include_bias=False)
X_poly = poly.fit_transform(X)
model = LinearRegression().fit(X_poly, y)
y_pred = model.predict(X_poly)
# Crearea unui meshgrid pentru Schooling și Adult Mortality
x_min, x_max = X[:, 0].min(), X[:, 0].max()
y_min, y_max = X[:, 1].min(), X[:, 1].max()
x_grid, y_grid = np.meshgrid(np.linspace(x_min, x_max, 50), np.linspace(y_min, y_max, 50))

# Împachetăm valorile pentru grid
X_grid = np.c_[x_grid.ravel(), y_grid.ravel()]
X_grid_poly = poly.transform(X_grid)

# Prezicem pe grid
z_grid = model.predict(X_grid_poly).reshape(x_grid.shape)

# Plotarea graficului 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Scatter plot pentru datele reale
ax.scatter(X[:, 0], X[:, 1], y, color='blue', label='Actual')

# Planul de regresie prezis
ax.plot_surface(x_grid, y_grid, z_grid, color='red', alpha=0.5, rstride=100, cstride=100)

# Etichetele axelor
ax.set_xlabel('Drug use disorders (%)')
ax.set_ylabel('Depression (%)')
ax.set_zlabel('Anxiety disorders (%)')

plt.legend()
plt.show()


plt.show()