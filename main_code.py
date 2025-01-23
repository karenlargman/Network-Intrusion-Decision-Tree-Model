import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib  # For model serialization


dataset_path = '/Users/karenl/Downloads/network_intrusion_dataset.csv' 
df = pd.read_csv(dataset_path)


print("First 5 rows of the dataset:")
print(df.head())

import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib  # For model serialization
import numpy as np


dataset_path = '/Users/karenl/Downloads/network_intrusion_dataset.csv' 
df = pd.read_csv(dataset_path)


print("First 5 rows of the dataset:")
print(df.head())


print("Missing values in each column:")
print(df.isnull().sum())


df.ffill(inplace=True)  
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.ffill(inplace=True)


print("Checking for infinite values after replacement:")
print((df == np.inf).sum())
print((df == -np.inf).sum())


print("Data types of each column:")
print(df.dtypes)


X = df.iloc[:, :-1]  
y = df.iloc[:, -1]   


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)


model = DecisionTreeClassifier(random_state=42)


model.fit(X_train, y_train)

joblib.dump(model, 'decision_tree_model.joblib')
print("Model serialized and saved to 'decision_tree_model.joblib'")


loaded_model = joblib.load('decision_tree_model.joblib')
print("Model deserialized and loaded successfully")

y_pred = loaded_model.predict(X_test)

print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

cv_scores = cross_val_score(loaded_model, X_scaled, y, cv=5, scoring='f1_macro')
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean
print("Missing values in each column:")
print(df.isnull().sum())


df.ffill(inplace=True)  
df.replace([float('inf'), float('-inf')], pd.NA, inplace=True)
df.ffill(inplace=True)


print("Checking for infinite values after replacement:")
print((df == float('inf')).sum())
print((df == float('-inf')).sum())


print("Data types of each column:")
print(df.dtypes)


X = df.iloc[:, :-1]  
y = df.iloc[:, -1]   


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


model = DecisionTreeClassifier(random_state=42)


model.fit(X_train, y_train)

joblib.dump(model, 'decision_tree_model.joblib')
print("Model serialized and saved to 'decision_tree_model.joblib'")


loaded_model = joblib.load('decision_tree_model.joblib')
print("Model deserialized and loaded successfully")

y_pred = loaded_model.predict(X_test)

print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

cv_scores = cross_val_score(loaded_model, X_scaled, y, cv=5)
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean cross-validation score: {cv_scores.mean()}")
