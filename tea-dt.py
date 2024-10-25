import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Set random seed for reproducibility
np.random.seed(42)
n_samples = 1000

# Create a sample dataset
data = {
    'Altitude': np.random.randint(1500, 2700, n_samples),
    'Rainfall': np.random.randint(1500, 2500, n_samples),
    'Bush_Age': np.random.randint(5, 40, n_samples),
}

# Generate quality based on realistic conditions
def determine_quality(altitude, rainfall, bush_age):
    if altitude > 2200 and rainfall > 2000 and bush_age < 20:
        return 'Premium'
    elif altitude > 1800 and rainfall > 1800 and bush_age < 30:
        return 'Standard'
    else:
        return 'Basic'

data['Quality'] = [determine_quality(a, r, b) for a, r, b in zip(data['Altitude'], data['Rainfall'], data['Bush_Age'])]

df = pd.DataFrame(data)

# Save to CSV
df.to_csv('kenyan_tea_quality.csv', index=False)

# Load the data (to simulate loading from an external source)
df = pd.read_csv('kenyan_tea_quality.csv')

# Prepare the features and target
X = df[['Altitude', 'Rainfall', 'Bush_Age']]
y = df['Quality']

# Encode the target variable
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Create and train the decision tree
clf = DecisionTreeClassifier(random_state=42, max_depth=3)
clf.fit(X, y_encoded)

# Visualize the decision tree
plt.figure(figsize=(20,10))
plot_tree(clf, feature_names=X.columns, class_names=le.classes_, filled=True, rounded=True)
plt.show()  # Display the tree

# Save the decision tree image
plt.savefig('kenyan_tea_decision_tree.png')
plt.close()

# Function to predict tea quality
def predict_tea_quality(altitude, rainfall, bush_age):
    # Create a DataFrame with the same feature names used during training
    features = pd.DataFrame([[altitude, rainfall, bush_age]], columns=['Altitude', 'Rainfall', 'Bush_Age'])
    prediction = clf.predict(features)
    return le.inverse_transform(prediction)[0]

# Test the model
print("Predictions:")
print("Farm 1 (Altitude: 2300m, Rainfall: 2100mm, Bush Age: 15 years):")
print(predict_tea_quality(2300, 2100, 15))

print("\nFarm 2 (Altitude: 1900m, Rainfall: 1900mm, Bush Age: 25 years):")
print(predict_tea_quality(1900, 1900, 25))

print("\nFarm 3 (Altitude: 1700m, Rainfall: 1700mm, Bush Age: 35 years):")
print(predict_tea_quality(1700, 1700, 35))

# Print feature importances
importances = clf.feature_importances_
for feature, importance in zip(X.columns, importances):
    print(f"\nImportance of {feature}: {importance:.2f}")
