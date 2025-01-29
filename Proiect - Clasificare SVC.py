# -*- coding: utf-8 -*-
"""
Created on Thu Dec 12 13:36:49 2024

@author: elena
"""

import pandas as pd
import numpy as np 
import seaborn as sns
import time
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from mlxtend.plotting import plot_decision_regions
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE

# Load dataset
data = pd.read_csv("C:/Users/elena/OneDrive/Desktop/Cursuri master/ML/mental_health/Mental_health_Preprocessed_MICE.csv", low_memory=False)


target_column = 'Depression (%)'

# Calculează limitele pentru împărțirea în trei intervale
min_value = data[target_column].min()
max_value = data[target_column].max()
bin_edges = [min_value, 
             min_value + (max_value - min_value) / 3, 
             min_value + 2 * (max_value - min_value) / 3, 
             max_value]

# Aplicăm pd.cut() pentru a crea clasele
data['target_class'] = pd.cut(data[target_column], bins=bin_edges, labels=['Low', 'Medium', 'High'], include_lowest=True)

# Verificăm distribuția claselor
print("Distribuția claselor:\n", data['target_class'].value_counts())

# Transformăm etichetele în valori numerice
le = LabelEncoder()
data['target_class_encoded'] = le.fit_transform(data['target_class']) 

# Verificăm primele câteva rânduri din dataset
print(data.head())

# Separate predictors and target
predictors = data.select_dtypes(include=['number']).drop(columns=[target_column, 'target_class_encoded'])
X = predictors.values
y = data['target_class_encoded'].values

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Aplicăm SMOTE pentru echilibrare
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Verificăm distribuția claselor după SMOTE
print("Distribuția claselor după SMOTE:")
print(pd.Series(y_train_resampled).value_counts())


# Iitializam lista de kernels pe care e vom testa
kernels = ['rbf', 'sigmoid']

# Initializam un dictionar pentru a pastra scorurile de acuratete
kernel_scores = {}

# Testam modelul cu fiecare kernel
for kernel in kernels:
    print(f"\nEvaluare pentru kernel: {kernel}")
    
    # Cream un pipeline pentru fiecare kernel
    svm_clf = Pipeline([("scaler", StandardScaler()), ("svm_clf", SVC(kernel=kernel, C=100))])
    
    start_time = time.time() 
    # Antrenăm modelul
    svm_clf.fit(X_train_resampled, y_train_resampled)
    
    # Facem predicții
    y_pred = svm_clf.predict(X_test)
    
    # Evaluăm performanța modelului
    accuracy = accuracy_score(y_test, y_pred)
    kernel_scores[kernel] = accuracy
    computation_time = time.time() - start_time 
    
    # Afișăm rezultatele pentru fiecare kernel
    print(f"Accuracy pentru {kernel}: {accuracy:.4f}")
    print(f"Computation Time = {computation_time:.4f} seconds")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    

# Identificam cel mai bun kernel
best_kernel = max(kernel_scores, key=kernel_scores.get)
best_accuracy = kernel_scores[best_kernel]

print(f"\nBest kernel: {best_kernel} with accuracy: {best_accuracy:.4f}")

# Reantrenăm modelul cu cel mai bun kernel
best_svm_clf = Pipeline([("scaler", StandardScaler()), ("svm_clf", SVC(kernel=best_kernel, C=5))])
best_svm_clf.fit(X_train_resampled, y_train_resampled)

# Facem predicții
y_best_pred = best_svm_clf.predict(X_test)

# Generăm matricea de confuzie și raportul de clasificare
best_cm = confusion_matrix(y_test, y_best_pred)
report = classification_report(y_test, y_best_pred, target_names=['Low', 'Medium', 'High'], output_dict=True)
report_df = pd.DataFrame(report).transpose()

# Afișăm matricea de confuzie
plt.figure(figsize=(8, 6))
sns.heatmap(best_cm, annot=True, fmt='d', cmap='Purples', xticklabels=['Low', 'Medium', 'High'], yticklabels=['Low', 'Medium', 'High'])
plt.xlabel('Etichete prezise')
plt.ylabel('Etichete reale')
plt.title(f'Matricea de confuzie pentru cel mai bun kernel: {best_kernel}')
plt.show()

# Afișăm raportul de clasificare
plt.figure(figsize=(10, 4))
sns.heatmap(report_df.iloc[:-1, :].T, annot=True, cmap="PRGn", cbar=False, fmt=".2f")
plt.title(f"Raport de clasificare pentru cel mai bun kernel: {best_kernel}")
plt.show()

# Reducem dimensiunea la 2 componente cu PCA
pca = PCA(n_components=2)
X_test_pca = pca.fit_transform(X_test)

# Plotăm regiunile de decizie
plt.figure(dpi=100)
plot_decision_regions(X=X_test_pca, y=y_test, clf=best_svm_clf, legend=2)

plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title(f"Regiuni de decizie SVC cu Kernel: {best_kernel}")
plt.show()