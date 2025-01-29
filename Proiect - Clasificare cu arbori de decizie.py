# -*- coding: utf-8 -*-
"""
Created on Fri Dec 13 11:09:28 2024

@author: elena
"""

import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
import time
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from mlxtend.plotting import plot_decision_regions
from sklearn.decomposition import PCA



# Încarcă dataset-ul
data = pd.read_csv("C:/Users/elena/OneDrive/Desktop/Cursuri master/ML/mental_health/Mental_health_Preprocessed_MICE.csv", low_memory=False)

# Verifică structura datelor
print(data.head())  
print(data.info())  


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

# Impartim dataset-ul in set de training si de test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply SMOTE to balance classes in the training set
smote = SMOTE(random_state=42)
X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

# Check class distribution after balancing
unique, counts = np.unique(y_train_balanced, return_counts=True)
print("Distribuția claselor după SMOTE:", dict(zip(unique, counts)))

# Iterate over different max depths to find the best model
results = []
print(f"{'Max Depth':<12}{'Acuratețe':<12}{'Timp (s)':<10}")
print("-" * 35)

best_depth = None
best_accuracy = 0

max_depths = range(1, 15)
for depth in max_depths:
    start_time = time.time()
    
    # Train a Decision Tree with the current max depth
    classifier = DecisionTreeClassifier(criterion="gini", random_state=42, max_depth=depth, min_samples_leaf=5)
    classifier.fit(X_train_balanced, y_train_balanced)
    
    # Predict on the test set
    y_pred = classifier.predict(X_test)
    
    end_time = time.time()
    
    # Calculate accuracy
    acc = accuracy_score(y_test, y_pred)
    elapsed_time = end_time - start_time
    
    # Save results
    results.append({'Max Depth': depth, 'Accuracy': acc, 'Time (s)': elapsed_time})
    print(f"{depth:<12}{acc:<12.4f}{elapsed_time:<10.4f}")
    
    # Update the best model if the current one is better
    if acc > best_accuracy:
        best_accuracy = acc
        best_depth = depth

# Create a DataFrame for results
results_df = pd.DataFrame(results)

# Display the best model
print("\nCel mai bun model:")
print(f"Max Depth: {best_depth}, Accuracy: {best_accuracy:.4f}")

# Retrain the final model with the best depth
final_clf = DecisionTreeClassifier(random_state=42, max_depth=best_depth, min_samples_leaf=5)
final_clf.fit(X_train_balanced, y_train_balanced)

# Predict and evaluate the final model
final_y_pred = final_clf.predict(X_test)
final_accuracy = accuracy_score(y_test, final_y_pred)

# Plot performance
plt.figure(figsize=(10, 6))
plt.plot(results_df['Max Depth'], results_df['Accuracy'], marker='o', label='Acuratețe')
plt.xlabel("Max Depth")
plt.ylabel("Accuracy")
plt.title("Performanța modelului în funcție de Max Depth")
plt.legend()
plt.grid()
plt.show()

# Visualize the decision tree
plt.figure(figsize=(12, 8))
plot_tree(final_clf, 
          feature_names=predictors.columns, 
          class_names=['Low', 'Medium', 'High'], 
          rounded=True, 
          fontsize=12,
          max_depth=5)
plt.title(f"Arborele de decizie cu Max Depth = {best_depth}")
plt.show()


# Confusion matrix and classification report
cm = confusion_matrix(y_test, final_y_pred)
report = classification_report(y_test, final_y_pred, target_names=['Low', 'Medium', 'High'], output_dict=True)
report_df = pd.DataFrame(report).transpose()
print(report_df)
print(cm)
# Confusion matrix heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', xticklabels=['Low', 'Medium', 'High'], yticklabels=['Low', 'Medium', 'High'])
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix')
plt.show()

# Classification report heatmap
plt.figure(figsize=(10, 4))
sns.heatmap(report_df.iloc[:-1, :].T, annot=True, cmap="PRGn", cbar=False, fmt=".2f")
plt.title("Classification Report")
plt.show()



