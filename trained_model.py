import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report

dataset_path = '/Users/karenl/Downloads/network_intrusion_dataset.csv' 
df = pd.read_csv(dataset_path)

print("First 5 rows of the dataset:")
print(df.head())

print("Missing values in each column:")
print(df.isnull().sum())


print("Checking for infinite values:")
print((df == float('inf')).sum())
print((df == float('-inf')).sum())


df.ffill(inplace=True)  

df.replace([float('inf'), float('-inf')], pd.NA, inplace=True)

df.ffill(inplace=True)

print("Data types of each column:")
print(df.dtypes)

X = df.iloc[:, :-1]  
y = df.iloc[:, -1]   


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


model = DecisionTreeClassifier()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
report = classification_report(y_test, y_pred)
print("Classification Report:\n", report)