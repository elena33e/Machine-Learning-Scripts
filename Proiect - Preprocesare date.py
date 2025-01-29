# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 16:54:02 2025

@author: elena
"""

import pandas as pd
import numpy as np
import miceforest as mf
import seaborn as sns
import matplotlib.pyplot as plt

# Încarcăm dataset-ul
original_data = pd.read_csv("C:/Users/elena/OneDrive/Desktop/Cursuri master/ML/mental_health/Mental health Data.csv", low_memory=False)
data = original_data.copy()

# Verificăm structura datelor
print(data.head())
print(data.info())

# Identificăm datele lipsă
print('Date lipsă inițiale:\n', data.isnull().sum())

# Convertim manual coloanele de procente la tipul numeric (dacă e cazul)
columns_to_convert = ['Schizophrenia (%)', 'Bipolar disorder (%)', 'Eating disorders (%)']
for col in columns_to_convert:
    data[col] = pd.to_numeric(data[col], errors='coerce')

# Imputarea valorilor lipsă din coloana 'Code' cu cea mai frecventă valoare (modul)
if data['Code'].isnull().any():  
    most_frequent_code = data['Code'].mode()[0] 
    data['Code'] = data['Code'].fillna(most_frequent_code)

data = data.drop(columns=['index'])

# Selectăm doar coloanele numerice pentru imputare
numeric_data = data.select_dtypes(include=[np.number])


# Creeăm un kernel miceforest pentru imputare pe coloanele numerice
imputation_kernel = mf.ImputationKernel(
    
    numeric_data, 
    save_all_iterations_data=True,
    random_state=42  
)

# Rulăm imputarea iterativă pe coloanele numerice
imputation_kernel.mice(
    iterations=10,  
    verbose=True  
)

# Extragem dataset-ul completat după imputare
imputed_data = imputation_kernel.complete_data(0)  

# Înlocuim coloanele imputate în dataset-ul original
data[numeric_data.columns] = imputed_data

# Verificăm valorile lipsă după imputare
print("Valori lipsă după imputare:\n", data.isnull().sum())

# Salvăm dataset-ul completat într-un fișier CSV
data.to_csv("C:/Users/elena/OneDrive/Desktop/Cursuri master/ML/mental_health/Mental_health_Preprocessed_MICE.csv", index=False)

# Primele rânduri ale dataset-ului completat
print(data.head())

# Vizualizare pentru comparația valorilor lipsă înainte și după imputare
plt.figure(figsize=(12, 6))
sns.heatmap(original_data.isnull(), cbar=False, cmap='viridis')
plt.title("Valori lipsă înainte de imputare")
plt.show()

plt.figure(figsize=(12, 6))
sns.heatmap(data.isnull(), cbar=False, cmap='viridis')
plt.title("Valori lipsă după imputare")
plt.show()
