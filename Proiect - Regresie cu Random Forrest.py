# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 12:13:32 2024

@author: elena
"""

from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import time
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

# Încarcă dataset-ul cu low_memory=False
data = pd.read_csv("https://github.com/elena33e/Machine-Learning-Scripts/blob/main/Mental_health_Preprocessed_MICE.csv", low_memory=False)

# Verifică structura datelor
print(data.head())  # Primele 5 rânduri
print(data.info())  # Informații generale despre dataset

# Separate predictors and target
predictors = data.select_dtypes(include=['number']).drop(columns=['Anxiety disorders (%)'])
X = predictors.values 
y = data['Anxiety disorders (%)'].values  

print('X: \n', X)
print('y: ', y)  
X, y = np.array(X), np.array(y)

"""
# Impartim dataset-ul in seturi de training si test
#X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#Standardizare
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
"""
# Standardizare
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Testăm Random Forest pentru diverse numere de estimatori 
n_estimators = [10, 50, 100, 200]  
results_cv = []

print("Scorurile R² și timpul de execuție:")
print("------------------------------------------------------------")

kf = KFold(n_splits=5, shuffle=True, random_state=0)  # KFold cu 5 fold-uri

for n in n_estimators:
    rf_regressor = RandomForestRegressor(random_state=0, n_estimators=n)
    
    # Calculăm R² folosind cross_val_score
    start_time = time.time()
    cv_scores = cross_val_score(rf_regressor, X, y, cv=kf, scoring='r2')  # Scorurile R² pentru fiecare fold
    end_time = time.time()
    
    # Salvăm și afișăm rezultatele
    mean_r2 = np.mean(cv_scores)
    elapsed_time = end_time - start_time
    
    results_cv.append({'Estimators': n, 'R²': mean_r2, 'Time (s)': elapsed_time})
    print(f"Estimatori: {n}, R² mediu: {mean_r2:.4f}, Timp: {elapsed_time:.4f} secunde")

# Creăm un DataFrame cu rezultatele
results_cv_df = pd.DataFrame(results_cv)

# Variația lui R^2 în funcție de n_estimators
plt.figure(figsize=(12, 6))
plt.plot(results_cv_df['Estimators'], results_cv_df['R²'], label='R²', marker='o')
plt.xlabel('Număr de estimatori (arbori)')
plt.ylabel('Valoare R²')
plt.title('Evoluția performanței în funcție de numărul de estimatori')
plt.legend()
plt.grid()
plt.show()

# Cel mai bun n_estimators pe baza R²
best_cv_result = results_cv_df.loc[results_cv_df['R²'].idxmax()]
best_estimators = int(best_cv_result['Estimators'])
best_cv_r2 = best_cv_result['R²']
best_cv_time = best_cv_result['Time (s)']

print("\nCel mai bun model Random Forest:")
print(f"Estimatori: {best_estimators}, Mean R²: {best_cv_r2:.4f}, Timp: {best_cv_time:.4f} secunde")

