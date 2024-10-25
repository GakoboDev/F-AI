import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder

# Generate sample data
np.random.seed(42)
n_samples = 1000

data = {
    'time_of_day': np.random.choice(['Morning', 'Afternoon', 'Evening', 'Night'], n_samples),
    'day_of_week': np.random.choice(['Weekday', 'Weekend'], n_samples),
    'weather': np.random.choice(['Clear', 'Rainy', 'Snowy'], n_samples),
    'road_type': np.random.choice(['Highway', 'Local', 'Intersection'], n_samples),
    'traffic_volume': np.random.randint(0, 1000, n_samples),
    'speed_limit': np.random.choice([30, 40, 50, 60, 70], n_samples),
    'traffic_signals': np.random.choice(['Yes', 'No'], n_samples),
    'near_school': np.random.choice(['Yes', 'No'], n_samples),
    'historical_accidents': np.random.randint(0, 50, n_samples)
}

# Create DataFrame
df = pd.DataFrame(data)

# Create target variable (this is a simplified model for demonstration)
df['risk'] = np.where(
    (df['traffic_volume'] > 500) & 
    (df['speed_limit'] > 50) & 
    (df['historical_accidents'] > 20) |
    ((df['weather'] == 'Rainy') & (df['road_type'] == 'Highway')) |
    ((df['time_of_day'] == 'Night') & (df['traffic_signals'] == 'No')),
    'High', 'Low'
)

# Save to CSV
df.to_csv('traffic_accident_risk.csv', index=False)

# Load the data
df = pd.read_csv('traffic_accident_risk.csv')

# Preprocess the data
le = LabelEncoder()
for column in df.columns:
    if df[column].dtype == 'object':
        df[column] = le.fit_transform(df[column])

# Split features and target
X = df.drop('risk', axis=1)
y = df['risk']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the decision tree model
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)

# Make predictions
y_pred = dt_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.2f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Feature importance
feature_importance = pd.DataFrame({'feature': X.columns, 'importance': dt_model.feature_importances_})
feature_importance = feature_importance.sort_values('importance', ascending=False)
print("\nFeature Importance:")
print(feature_importance)

# Example prediction
example = X.iloc[0].values.reshape(1, -1)
prediction = dt_model.predict(example)
print(f"\nPrediction for the first data point: {prediction[0]}")

