import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import Perceptron

# Step 1: Load the dataset
# For demonstration purposes, we're creating a synthetic dataset
data = {
    'Rainfall': [100, 150, 200, 250, 300, 350, 400, 450, 500, 550],
    'Temperature': [20, 21, 22, 23, 24, 25, 26, 27, 28, 29],
    'Soil_pH': [5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0],
    'Soil_Nitrogen': [50, 60, 70, 80, 90, 100, 110, 120, 130, 140],
    'Crop_Type': ['Maize', 'Maize', 'Wheat', 'Wheat', 'Rice', 'Rice', 'Beans', 'Beans', 'Barley', 'Barley'],
    'Yield': [500, 600, 700, 800, 900, 950, 1000, 1100, 1200, 1300]
}

df = pd.DataFrame(data)

# Step 2: Data Preprocessing
# Encode categorical variables
le = LabelEncoder()
df['Crop_Type'] = le.fit_transform(df['Crop_Type'])

# Split data into features and target variable
X = df[['Rainfall', 'Temperature', 'Soil_pH', 'Soil_Nitrogen', 'Crop_Type']]
y = df['Yield']

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 3: Create the Perceptron Model
model = Perceptron(max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Step 4: Predicting Crop Yield
predictions = model.predict(X_test)

# Display predictions
for i, pred in enumerate(predictions):
    print(f"Predicted yield for test sample {i+1}: {pred} kg/ha")
