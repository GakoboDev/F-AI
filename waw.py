import seaborn as sns
import pandas as pd
from sklearn import datasets

# Load diabetes dataset
diabetes = datasets.load_diabetes()

# Convert to DataFrame
diabetes_df = pd.DataFrame(diabetes.data, columns=diabetes.feature_names)

# Create pairplot
sns.pairplot(diabetes_df[['age', 'sex', 'bmi', 'bp', 's1']],
             height=1.5, hue='sex', plot_kws={'alpha': .2})

# Show the plot
import matplotlib.pyplot as plt
plt.show()
