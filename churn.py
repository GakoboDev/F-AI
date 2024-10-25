import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import matplotlib.pyplot as plt

# Sample data in a DataFrame
data = {
    'Name': ['Wanjiru', 'Koech', 'Otieno', 'Mwangi', 'Barasa', 'Wambui', 'Ochieng', 'Mutua', 'Omondi', 'Njeri'],
    'Age': [23, 35, 40, 23, 32, 40, 29, 50, 27, 44],
    'Plan': ['Prepaid', 'Postpaid', 'Prepaid', 'Postpaid', 'Prepaid', 'Postpaid', 'Prepaid', 'Postpaid', 'Prepaid', 'Postpaid'],
    'Tenure': [24, 18, 36, 48, 12, 30, 15, 20, 66, 22],
    'Monthly_Bill': [2500, 3000, 3100, 3500, 600, 2800, 800, 2200, 400,  2300],
    'Data_Usage': [12, 15, 20, 5, 8, 10, 3, 9, 5, 7],
    'Customer_Service_Calls': [0, 1, 2, 3, 0, 1, 0, 2, 4, 1],
    'Contract': ['None', 'Annual', 'None', 'Monthly', 'None', 'Monthly', 'None', 'Annual', 'None', 'Monthly'],
    'Churn': ['No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes']
}

df = pd.DataFrame(data)

# Preprocessing: Convert categorical data into numerical values
df['Plan'] = df['Plan'].map({'Prepaid': 0, 'Postpaid': 1})
df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})

# Define features (X) and target (y)
X = df[['Age', 'Plan', 'Tenure', 'Monthly_Bill', 'Data_Usage', 'Customer_Service_Calls']]
y = df['Churn']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Decision Tree Classifier
clf = DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)

# Visualize the Decision Tree
plt.figure(figsize=(10,8))
tree.plot_tree(clf, feature_names=X.columns, class_names=['No', 'Yes'], filled=True)
plt.show()

# Predict on the test set
y_pred = clf.predict(X_test)

# Model accuracy
accuracy = clf.score(X_test, y_test)
print(f"Accuracy: {accuracy * 100:.2f}%")
