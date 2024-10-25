# decision_tree_regression.py

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

# Sample data: Hours studied vs. Scores obtained
data = {
    'Hours_Studied': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Scores': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
}

# Convert the data into a pandas DataFrame
df = pd.DataFrame(data)

# Define features and target variable
X = df[['Hours_Studied']]  # Feature (independent variable)
y = df['Scores']           # Target (dependent variable)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a Decision Tree regression model
model = DecisionTreeRegressor()

# Train the model
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_test)

# Print the actual vs predicted values
for i in range(len(X_test)):
    print(f"Hours Studied: {X_test.iloc[i][0]}, Actual Score: {y_test.iloc[i]}, Predicted Score: {predictions[i]}")

# Visualize the results
plt.scatter(X, y, color='blue', label='Actual data')
plt.scatter(X_test, predictions, color='red', label='Predicted data', marker='x')
plt.xlabel('Hours Studied')
plt.ylabel('Scores')
plt.title('Decision Tree: Hours Studied vs Scores')
plt.legend()
plt.show()
