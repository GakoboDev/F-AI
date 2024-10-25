import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

file_path = 'n2012.csv'
data = pd.read_csv(file_path)

numeric_columns = ['12m Low', '12m High', 'Day Low', 'Day High', 'Day Price', 'Previous', 'Change', 'Volume']
for column in numeric_columns:
    data[column] = pd.to_numeric(data[column].replace({',': ''}, regex=True), errors='coerce')

data_cleaned = data.dropna(subset=numeric_columns)

data_sampled = data_cleaned.sample(n=1000, random_state=42)

subset_columns = ['12m Low', '12m High', 'Day Low', 'Day High', 'Day Price', 'Volume']

plt.figure(figsize=(12, 10))
sns.pairplot(data_sampled[subset_columns])
plt.show()
