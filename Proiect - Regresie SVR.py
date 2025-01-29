# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 13:46:33 2024

@author: elena
"""

from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import pandas as pd
import time
from sklearn.impute import SimpleImputer
import numpy as np

# Încarcă dataset-ul cu low_memory=False
data = pd.read_csv("https://github.com/elena33e/Machine-Learning-Scripts/blob/main/Mental_health_Preprocessed_MICE.csv", low_memory=False)

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

# Lista cu kernel-urile de testat
kernels = ['linear', 'poly']

kernel_results = {}

# Testam fiecare kernel
for kernel in kernels:
    print(f"\nEvaluating SVR model with kernel: {kernel}")
    
    start_time = time.time()
    
    # Definim un ipeline pentru SVR
    svm_regressor = Pipeline([
        ("scaler", StandardScaler()), 
        ("svm_regressor", SVR(kernel=kernel, C=100, epsilon=0.001))
    ])
    
    # Antreneam modelul
    svm_regressor.fit(X_train, y_train)
    
    # Facem prezicerile
    y_pred = svm_regressor.predict(X_test)
    
    end_time = time.time()
    
    # Calculam metricile de evaluare
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    # Timpul de calcul
    elapsed_time = end_time - start_time
    
    kernel_results[kernel] = {'MAE': mae, 'MSE': mse, 'R2': r2, 'Time (s)': elapsed_time}
    
    print(f"Kernel: {kernel}")
    print(f"Mean Absolute Error (MAE): {mae:.4f}")
    print(f"Mean Squared Error (MSE): {mse:.4f}")
    print(f"R^2 Score: {r2:.4f}")
    print(f"Computation Time: {elapsed_time:.4f} seconds")

# Cel mai bun kernel, în funcție de scorul R^2
best_kernel = max(kernel_results, key=lambda k: kernel_results[k]['R2'])
best_metrics = kernel_results[best_kernel]

print(f"\nBest Kernel: {best_kernel}")
print(f"Best Kernel Metrics - MAE: {best_metrics['MAE']:.4f}, MSE: {best_metrics['MSE']:.4f}, R^2: {best_metrics['R2']:.4f}")
