import pandas as pd
import numpy as np

data = {
    'Outlook': ['Sunny', 'Sunny', 'Overcast', 'Rainy', 'Rainy', 'Rainy', 
                'Overcast', 'Sunny', 'Sunny', 'Rainy', 'Sunny', 
                'Overcast', 'Overcast', 'Rainy'],
    'Temperature': ['Hot', 'Hot', 'Hot', 'Mild', 'Cool', 'Cool', 
                    'Cool', 'Mild', 'Cool', 'Mild', 'Mild', 
                    'Mild', 'Hot', 'Mild'],
    'Humidity': ['High', 'High', 'High', 'High', 'Normal', 'Normal', 
                 'Normal', 'High', 'Normal', 'Normal', 'Normal', 
                 'High', 'Normal', 'High'],
    'Wind': ['Weak', 'Strong', 'Weak', 'Weak', 'Weak', 'Strong', 
             'Strong', 'Weak', 'Weak', 'Strong', 'Strong', 
             'Strong', 'Weak', 'Strong'],
    'Play': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 
             'Yes', 'No', 'Yes', 'Yes', 'Yes', 
             'Yes', 'Yes', 'No']
}

df = pd.DataFrame(data)

def entropy(target_col):
    elements, counts = np.unique(target_col, return_counts=True)
    entropy_value = 0
    for count in counts:
        probability = count / len(target_col)
        entropy_value -= probability * np.log2(probability)
    return entropy_value

def information_gain(data, split_attribute, target_attribute):
    total_entropy = entropy(data[target_attribute])
    values, counts = np.unique(data[split_attribute], return_counts=True)

    weighted_entropy = 0
    for value, count in zip(values, counts):
        subset = data[data[split_attribute] == value]
        weighted_entropy += (count / sum(counts)) * entropy(subset[target_attribute])
    
    return total_entropy - weighted_entropy


attributes = df.columns[:-1] 
target = 'Play'

info_gains = {attr: information_gain(df, attr, target) for attr in attributes}

for attr, gain in info_gains.items():
    print(f"Information Gain for {attr}: {gain:.4f}")

def best_attribute(data, target_attribute):
    attributes = data.columns[:-1]  
    info_gains = {attr: information_gain(data, attr, target_attribute) for attr in attributes}
    return max(info_gains, key=info_gains.get)

best_attr = best_attribute(df, target)
print(f"\nBest attribute to split on: {best_attr}")
