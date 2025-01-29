# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 11:33:56 2024

@author: elena
"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
import time
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import RFE

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

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Apply SMOTE to balance classes
smote = SMOTE(random_state=0)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Definește modelul de bază
base_model = RandomForestClassifier(n_estimators=200, random_state=42, criterion='gini')

# Initializam RFE 
rfe = RFE(estimator=base_model, n_features_to_select=3, step=1)

# Aplicăm RFE pe setul de antrenament
rfe.fit(X_train_balanced, y_train_balanced)

# Caracteristicile selectate
selected_features = predictors.columns[rfe.support_]
print(f"Caracteristicile selectate sunt: {list(selected_features)}")

# Cream un nou set de date cu caracteristicile selectate
X_train_rfe = X_train_balanced[:, rfe.support_]
X_test_rfe = X_test[:, rfe.support_]

# Măsurarea timpului pentru antrenarea modelului final
start_time_train = time.time()
final_model = RandomForestClassifier(n_estimators=10, random_state=42, criterion='gini')
final_model.fit(X_train_rfe, y_train_balanced)
end_time_train = time.time()
print(f"Timpul de antrenare al modelului final: {end_time_train - start_time_train:.4f} secunde")

# Preziceri și evaluare
y_pred_rfe = final_model.predict(X_test_rfe)

# Evaluarea performanței
accuracy_rfe = accuracy_score(y_test, y_pred_rfe)
print(f"Acuratețea modelului cu caracteristicile selectate: {accuracy_rfe:.4f}")


# Confusion matrix for selected features
cm_rfe = confusion_matrix(y_test, y_pred_rfe)
print("Matricea de confuzie pentru caracteristici selectate:")
print(cm_rfe)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm_rfe, annot=True, fmt='d', cmap='Purples', 
            xticklabels=['Low', 'Medium', 'High'], yticklabels=['Low', 'Medium', 'High'])
plt.xlabel('Etichete prezise')
plt.ylabel('Etichete reale')
plt.title('Matricea de confuzie pentru Random Forest cu cele mai importante caracteristici')
plt.show()

# Classification report for selected features
final_report = classification_report(y_test, y_pred_rfe, target_names=['Low', 'Medium', 'High'], output_dict=True)
final_report_df = pd.DataFrame(final_report).transpose()
print(final_report_df)

# Plot classification report heatmap
plt.figure(figsize=(10, 4))
sns.heatmap(final_report_df.iloc[:-1, :].T, annot=True, cmap="PRGn", cbar=False, fmt=".2f")
plt.title('Raportul de clasificare pentru Random Forest cu cele mai importante caracteristici')
plt.show()


