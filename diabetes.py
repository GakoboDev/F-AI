import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import train_test_split as tts
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import seaborn as sns

# Load diabetes dataset
diabetes = datasets.load_diabetes()

# Split dataset
diabetes_train_ftrs, diabetes_test_ftrs, diabetes_train_tgt, diabetes_test_tgt = tts(diabetes.data, diabetes.target, test_size=0.25)

# Convert to DataFrame
diabetes_df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)
diabetes_df['target'] = diabetes.target

# Print the first 5 rows of the dataset
print(diabetes_df.head())

# Create pairplot using Seaborn
sns.pairplot(diabetes_df[['age', 'sex', 'bmi', 'bp', 's1', 'target']], height=1.5, hue='sex', plot_kws={'alpha': .2})

# Show the plot
plt.show()

# Perform K-Nearest Neighbors Regression
knn = KNeighborsRegressor(n_neighbors=3)
knn.fit(diabetes_train_ftrs, diabetes_train_tgt)
preds = knn.predict(diabetes_test_ftrs)

# Evaluate Linear Regression Model
lr = LinearRegression()
fit = lr.fit(diabetes_train_ftrs, diabetes_train_tgt)
preds = lr.predict(diabetes_train_ftrs)
