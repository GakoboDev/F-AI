# app.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from models.perceptron import Perceptron

# Step 1: Create the Dataset
data = {
    'Hours_Studied': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Score': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
}
df = pd.DataFrame(data)

# Define a pass/fail threshold
pass_threshold = 80
df['Pass'] = (df['Score'] >= pass_threshold).astype(int)  # 1 for pass, 0 for fail

# Features and labels
X = df[['Hours_Studied']].values
y = df['Pass'].values

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Train the Perceptron
perceptron = Perceptron(learning_rate=0.1, n_iter=1000)
perceptron.fit(X_train, y_train)

# Step 5: Make Predictions
predictions = perceptron.predict(X_test)

# Step 6: Evaluate the Model
accuracy = np.mean(predictions == y_test) * 100
print(f'Accuracy: {accuracy:.2f}%')

# Step 7: Visualize the Decision Boundary
def plot_decision_boundary(X, y, model):
    plt.scatter(X[y == 0][:, 0], y[y == 0], color='red', label='Fail')
    plt.scatter(X[y == 1][:, 0], y[y == 1], color='blue', label='Pass')

    # Create a grid to plot the decision boundary
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    xx = np.linspace(x_min, x_max, 100).reshape(-1, 1)
    yy = model.predict(xx)

    plt.plot(xx, yy, color='green', label='Decision Boundary')
    plt.title("Decision Boundary for Study Hours vs Pass/Fail")
    plt.xlabel("Hours Studied")
    plt.ylabel("Pass/Fail")
    plt.legend()
    plt.show()

plot_decision_boundary(X, y, perceptron)
