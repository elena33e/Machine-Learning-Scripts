# -*- coding: utf-8 -*-
"""
Revised KNN Classification with Proper Cross-Validation Adjustments
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.neighbors import KNeighborsClassifier
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.decomposition import PCA
from mlxtend.plotting import plot_decision_regions

# Încarcă dataset-ul cu low_memory=False
data = pd.read_csv("C:/Users/elena/OneDrive/Desktop/Cursuri master/ML/mental_health/Mental_health_Preprocessed_MICE.csv", low_memory=False)

# Verifică structura datelor
print(data.head())  # Primele 5 rânduri
print(data.info())  # Informații generale despre dataset


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

# Cross-validation setup
kf = KFold(n_splits=5, shuffle=True, random_state=42)
accuracies = []
best_k = 1
best_accuracy = 0

for k in range(5, 15):  # Test k from 1 to 9
    KNN = KNeighborsClassifier(n_neighbors=k)
    cv_scores = []
    
    start_time = time.time()

    for train_index, test_index in kf.split(X):
        # Split data into train and test for this fold
        X_train_cv, X_test_cv = X[train_index], X[test_index]
        y_train_cv, y_test_cv = y[train_index], y[test_index]

     
        # Normalize the data
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_cv)
        X_test_scaled = scaler.transform(X_test_cv)

        # Train KNN and evaluate
        KNN.fit(X_train_scaled, y_train_cv)
        y_pred_cv = KNN.predict(X_test_scaled)
        accuracy_cv = accuracy_score(y_test_cv, y_pred_cv)
        cv_scores.append(accuracy_cv)

    # Average accuracy across folds
    mean_cv_score = np.mean(cv_scores)
    accuracies.append((k, mean_cv_score))

    if mean_cv_score > best_accuracy:
        best_accuracy = mean_cv_score
        best_k = k
        
    end_time = time.time()  # Încheie măsurarea timpului
    elapsed_time = end_time - start_time

    print(f"k = {k}, Cross-validated accuracy = {mean_cv_score:.4f} \u00b1 {np.std(cv_scores):.4f}, Time taken: {elapsed_time:.4f} seconds")

print(f"Best k: {best_k} with accuracy: {best_accuracy:.4f}")

# Final model training and evaluation
best_knn = KNeighborsClassifier(n_neighbors=best_k)

# Normalizare a datelor (fără SMOTE)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)  # Aplicăm scalarea direct pe datele originale

# Împărțim datele în seturi de antrenament și testare
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Antrenăm modelul KNN cu cel mai bun k
best_knn.fit(X_train, y_train)

# Predicții pe setul de testare
y_pred = best_knn.predict(X_test)

# Afișează rezultatele
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))

best_cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=['Low', 'Medium', 'High'], output_dict=True)

report_df = pd.DataFrame(report).transpose()


plt.figure(figsize=(8, 6))
sns.heatmap(best_cm, annot=True, fmt='d', cmap='Purples', xticklabels=['Low', 'Medium', 'High'], yticklabels=['Low', 'Medium', 'High'])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title(f'Confusion Matrix for k = {best_k}')
plt.show()


plt.figure(figsize=(10, 4))
sns.heatmap(report_df.iloc[:-1, :].T, annot=True, cmap="PRGn", cbar=False, fmt=".2f")
plt.title(f"Classification Report for k = {best_k}")
plt.show()


# Plot accuracy scores for different k values
k_values = [x[0] for x in accuracies]
accuracy_scores = [x[1] for x in accuracies]
plt.figure(figsize=(10, 5))
plt.plot(k_values, accuracy_scores, marker='o', color='b')
plt.title('Accuracy Scores for Different k Values')
plt.xlabel('k (Number of Neighbors)')
plt.ylabel('Accuracy')
plt.grid()
plt.show()


# Aplicăm PCA pentru a reduce la 2 dimensiuni
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Împărțim datele în set de antrenament și test
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# Antrenăm modelul KNN cu valoarea optimă pentru k
best_k = 5  # Poți ajusta această valoare pe baza celor mai bune rezultate
best_knn = KNeighborsClassifier(n_neighbors=best_k)
best_knn.fit(X_train, y_train)

# Vizualizează regiunile de clasificare
plt.figure(figsize=(10, 8))
plot_decision_regions(X=X_test, y=y_test, clf=best_knn, legend=2)
plt.title(f'Regions of Classification for k = {best_k} with PCA')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()
