# -*- coding: utf-8 -*-
"""
Created on Sun Dec  8 15:21:42 2024

@author: elena
"""

import pandas as pd
import time
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# Încarcă dataset-ul cu low_memory=False
data = pd.read_csv("https://github.com/elena33e/Machine-Learning-Scripts/blob/main/Mental_health_Preprocessed_MICE.csv", low_memory=False)


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

# Selectăm predictorii și ținta
predictors = data.select_dtypes(include=['number']).drop(columns=[target_column, 'target_class_encoded'])
X = predictors
y = data['target_class_encoded']

# Normalizăm predictorii
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Împărțim datele în seturi de antrenament și testare
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Aplica SMOTE pentru echilibrarea setului de antrenament
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Verificăm distribuția claselor după aplicarea SMOTE
print("Distribuția claselor după SMOTE:\n", pd.Series(y_train_smote).value_counts())

# Măsurăm timpul de antrenare 
start_time = time.time()  

#Antrenarea modelului
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train_smote, y_train_smote)

end_time = time.time()  

training_time = end_time - start_time
print(f"Timpul de antrenare al modelului: {training_time:.4f} secunde")


# Prezicem și evaluăm performanța
y_pred = log_reg.predict(X_test)
print("Matricea de confuzie:")
print(confusion_matrix(y_test, y_pred))
print("\nRaport de clasificare:")
print(classification_report(y_test, y_pred))

# Matricea de confuzie
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', 
            xticklabels=['Low', 'Medium', 'High'], yticklabels=['Low', 'Medium', 'High'])
plt.xlabel('Etichete prezise')
plt.ylabel('Etichete reale')
plt.title('Matricea de confuzie pentru regresia logistică')
plt.show()

# Raportul de clasificare
report = classification_report(y_test, y_pred, target_names=['Low', 'Medium', 'High'], output_dict=True)
report_df = pd.DataFrame(report).transpose()
print(report_df)

plt.figure(figsize=(10, 4))
sns.heatmap(report_df.iloc[:-1, :].T, annot=True, cmap="PRGn", cbar=False, fmt=".2f")
plt.title('Raportul de clasificare pentru Regresia logistica')
plt.show()

# Vizualizarea distribuției claselor pe ținta 'target_class'
plt.figure(figsize=(8, 6))
sns.countplot(x='target_class', data=data, hue='target_class', palette="Set2", legend=False)
plt.title('Distribuția datelor pe clase (Low, Medium, High)')
plt.xlabel('Clasă')
plt.ylabel('Număr de înregistrări')
plt.show()

# Vizualizarea distribuției valorilor din 'Anxiety disorders (%)' pentru fiecare clasă
plt.figure(figsize=(8, 6))
sns.boxplot(x='target_class', y='Drug use disorders (%)', data=data, hue='target_class', palette="Set2", legend=False)
plt.title('Distribuția valorilor pentru Drug use disorders (%) pe clase')
plt.xlabel('Clasă')
plt.ylabel('Drug use disorders (%)')
plt.show()

