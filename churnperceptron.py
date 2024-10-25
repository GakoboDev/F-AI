import pandas as pd
from sklearn.model_selection import train_test_split # type: ignore
from sklearn.linear_model import Perceptron # type: ignore
from sklearn.metrics import accuracy_score # type: ignore

# Sample data in a DataFrame
data = {
    'Name': ['Wanjiru', 'Koech', 'Otieno', 'Mwangi', 'Barasa', 'Wambui', 'Ochieng', 'Mutua', 'Omondi', 'Njeri'],
    'Age': [35, 45, 23, 40, 29, 50, 27, 41, 36, 48],
    'Plan': ['Postpaid', 'Postpaid', 'Prepaid', 'Postpaid', 'Prepaid', 'Postpaid', 'Prepaid', 'Postpaid', 'Postpaid', 'Prepaid'],
    'Tenure': [24, 15, 30, 15, 20, 25, 10, 3, 1, 9],
    'Monthly_Bill': [2500, 3000, 3500, 600, 2800, 800, 2200, 2800, 2200, 3600],
    'Data_Usage': [1, 2, 3, 0, 2, 4, 1, 0, 1, 0],
    'Customer_Service_Calls': [1, 3, 4, 1, 0, 2, 4, 1, 0, 2],
    'Churn': ['Yes', 'No', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes']
}

df = pd.DataFrame(data)

# Preprocessing: Convert categorical data into numerical values
df['Plan'] = df['Plan'].map({'Prepaid': 0, 'Postpaid': 1})
df['Churn'] = df['Churn'].map({'No': 0, 'Yes': 1})

# Define features (X) and target (y)
X = df[['Age', 'Plan', 'Tenure', 'Monthly_Bill', 'Data_Usage', 'Customer_Service_Calls']]
y = df['Churn']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Train a Perceptron model
clf = Perceptron()
clf.fit(X_train, y_train)

# Predict on the test set
y_pred = clf.predict(X_test)

# Model accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")

# Define new data for prediction (make sure it has feature names)
new_data = pd.DataFrame([[35, 1, 24, 3000, 10, 2]], 
                        columns=['Age', 'Plan', 'Tenure', 'Monthly_Bill', 'Data_Usage', 'Customer_Service_Calls'])

# Output next prediction for new data
next_prediction = clf.predict(new_data)

if next_prediction[0] == 0:
    print("Next prediction: No churn")
else:
    print("Next prediction: Yes churn")
