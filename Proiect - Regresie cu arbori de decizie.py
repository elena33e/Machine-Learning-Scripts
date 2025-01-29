# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 12:00:48 2024

@author: elena
"""

from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree
import pandas as pd
import time
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

data = pd.read_csv("https://github.com/elena33e/Machine-Learning-Scripts/blob/main/Mental_health_Preprocessed_MICE.csv", low_memory=False)

# Verifică structura datelor
print(data.head())  # Primele 5 rânduri
print(data.info())  # Informații generale despre dataset

numeric_data = data.select_dtypes(include=['number'])

# Separate predictors and target
predictors = data.select_dtypes(include=['number']).drop(columns=['Anxiety disorders (%)'])
X = predictors.values 
y = data['Anxiety disorders (%)'].values  

print('X: \n', X)
print('y: ', y)  
X, y = np.array(X), np.array(y)

# Impartim dataset-ul in seturi de training si test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizare
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# Testăm modelul pentru diverse adâncimi ale arborelui
max_depths = range(1, 11) 
results = []

print("Scorurile R² și timpul de execuție pentru fiecare adâncime:")
print("------------------------------------------------------------")

for depth in max_depths:
    start_time = time.time() 
    
    # Inițializăm modelul cu adâncimea curentă
    regressor = DecisionTreeRegressor(random_state=42, max_depth=depth, )
    regressor.fit(X_train, y_train)
    
    # Prezicem valorile
    y_pred = regressor.predict(X_test)
    
    end_time = time.time()  
    
    # Calculăm R^2
    r2 = r2_score(y_test, y_pred)
    elapsed_time = end_time - start_time 
    
    # Salvăm și afișăm rezultatele
    results.append({'Max Depth': depth, 'R²': r2, 'Time (s)': elapsed_time})
    print(f"Adâncime: {depth}, R²: {r2:.4f}, Timp: {elapsed_time:.4f} secunde")

# Creăm un DataFrame cu rezultatele
results_df = pd.DataFrame(results)

# Evoluția metricilor în funcție de adâncimea arborelui
plt.figure(figsize=(12, 6))
plt.plot(results_df['Max Depth'], results_df['R²'], label='R²', marker='o')
plt.xlabel('Adâncimea maximă a arborelui')
plt.ylabel('Valoare R^2')
plt.title('Evoluția performanței în funcție de adâncime')
plt.legend()
plt.grid()
plt.show()

# Determinăm cea mai bună adâncime pe baza R²
best_result = results_df.loc[results_df['R²'].idxmax()]
best_depth = int(best_result['Max Depth'])
best_r2 = best_result['R²']
best_time = best_result['Time (s)']

print("\nCea mai bună adâncime a arborelui:")
print(f"Adâncime: {best_depth}, R²: {best_r2:.4f}, Timp: {best_time:.4f} secunde")

# Inițializăm modelul cu cea mai bună adâncime
best_regressor = DecisionTreeRegressor(random_state=0, max_depth=best_depth)
best_regressor.fit(X_train, y_train)

# Vizualizăm arborele de decizie aferent celei mai bune adâncimi
plt.figure(figsize=(14, 10))
plot_tree(best_regressor, 
          filled=True, 
          feature_names=predictors.columns, 
          rounded=True, 
          fontsize=10,
          max_depth=5)
plt.title(f"Decision Tree Regression (Adâncime optimă: {best_depth})")
plt.show()



