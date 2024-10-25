# simple_linear_regression.py

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

data = {
    'Hours_Studied': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Scores': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
}

df = pd.DataFrame(data)

X = df[['Hours_Studied']]  
y = df['Scores']           

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()

model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("Coefficient (slope):", model.coef_[0])
print("Intercept:", model.intercept_)

for i in range(len(X_test)):
    print(f"Hours Studied: {X_test.iloc[i][0]}, Actual Score: {y_test.iloc[i]}, Predicted Score: {predictions[i]}")

plt.scatter(X, y, color='blue', label='Actual data')
plt.plot(X, model.predict(X), color='red', label='Regression line')
plt.xlabel('Hours Studied')
plt.ylabel('Scores')
plt.title('Hours Studied vs Scores')
plt.legend()
plt.show()
