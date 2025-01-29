# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 14:56:58 2024

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
                         
# Selectăm predictorul 'Drug use disorders (%)' și variabila țintă 'Anxiety disorders (%)'
y = np.array(data[['Anxiety disorders (%)']].values)
X = np.array(data['Depression (%)'].values).reshape(-1, 1)

# Împărțim datele în seturi de antrenament și testare
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Listă pentru a salva valorile R^2 și timpul de execuție pentru fiecare grad de polinom pe setul de testare
results = []

# Iterăm prin gradele de polinom de la 2 la 8
for degree in range(2, 9):
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_train_poly = poly.fit_transform(X_train)
    X_test_poly = poly.transform(X_test)
    
    # Inițiem și antrenăm modelul de regresie liniară pe variabilele polinomiale
    model = LinearRegression()
    
    start_time = time.time()
    model.fit(X_train_poly, y_train)
    end_time = time.time()
    
    # Calculăm scorul R^2 pentru setul de testare
    r_sq_test = model.score(X_test_poly, y_test)
    exec_time = end_time - start_time
    
    # Adăugăm rezultatele în listă
    results.append((degree, r_sq_test, exec_time))
    
    # Generăm valori de predicție pentru plotare
    X_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    X_range_poly = poly.transform(X_range)
    y_range_pred = model.predict(X_range_poly)

    # Plotare pentru gradul curent
    plt.figure(figsize=(10, 6))
    plt.scatter(X_test, y_test, color='blue', label='Actual test data', alpha=0.5)
    plt.plot(X_range, y_range_pred, color='red', label=f'Polynomial Degree {degree} Fit', linewidth=2)
    plt.xlabel('Depression (%)')
    plt.ylabel('Anxiety disorders (%)')
    plt.title(f'Polynomial Regression (Degree {degree}) Fit for Set\nR^2 Test: {r_sq_test:.4f}')
    plt.legend()
    plt.show()

# Afișăm rezultatele pentru toate gradele
print("Degree | R^2 | Execution Time (s)")
for degree, r_sq_test, exec_time in results:
    print(f"{degree:<6} | {r_sq_test:.4f} | {exec_time:.4f}")
