import matplotlib.pyplot as plt

# Data
lags = [3, 4, 5, 6]
methods = ['CART', 'RF', 'SGB', 'XGB', 'ELM']

data = {
    0: [83.48, 82.81, 84.61, 82.96, 83.73],
    1: [84.07, 84.27, 83.91, 82.71, 83.09],
    2: [82.96, 84.92, 84.75, 84.44, 83.00],
    3: [84.08, 82.99, 84.38, 83.19, 84.59]
}

# Accuracy data for each method at each D value
method_data = {
    'CART': {
        0: [83.48, 84.40, 84.47, 83.01],
        1: [84.07, 83.68, 83.21, 84.48],
        2: [82.96, 83.97, 83.98, 84.33],
        3: [84.08, 84.41, 83.11, 83.20]
    },
    'RF': {
        0: [82.81, 83.01, 83.78, 82.93],
        1: [84.27, 84.12, 84.91, 83.67],
        2: [84.92, 82.80, 84.02, 84.90],
        3: [82.99, 83.77, 82.85, 82.79]
    },
    'SGB': {
        0: [84.61, 83.93, 82.92, 84.37],
        1: [83.91, 84.15, 83.60, 82.85],
        2: [84.75, 82.90, 84.56, 83.99],
        3: [84.38, 84.32, 83.79, 82.90]
    },
    'XGB': {
        0: [82.96, 83.50, 84.12, 83.02],
        1: [82.71, 84.77, 82.81, 83.75],
        2: [84.44, 83.29, 84.79, 82.87],
        3: [83.19, 84.10, 84.21, 84.13]
    },
    'ELM': {
        0: [83.73, 84.39, 83.39, 83.33],
        1: [83.09, 84.43, 84.89, 83.90],
        2: [83.00, 83.19, 84.16, 84.36],
        3: [84.59, 82.88, 83.91, 83.45]
    }
}

# Create plots
plt.figure(figsize=(14, 10))

for d in range(4):
    plt.subplot(2, 2, d + 1)
    for method in methods:
        plt.plot(lags, method_data[method][d], marker='o', label=method)
    plt.xlabel('Game Lag')
    plt.ylabel('Accuracy Percentage')
    plt.title(f'Accuracy vs Game Lag for D = {d}')
    plt.xticks(lags)
    plt.ylim(80, 85)
    plt.legend()

plt.tight_layout()
plt.show()
import matplotlib.pyplot as plt

# Data
lags = [3, 4, 5, 6]
methods = ['CART', 'RF', 'SGB', 'XGB', 'ELM']
data = {
    0: [83.48, 82.81, 84.61, 82.96, 83.73],
    1: [84.07, 84.27, 83.91, 82.71, 83.09],
    2: [82.96, 84.92, 84.75, 84.44, 83.00],
    3: [84.08, 82.99, 84.38, 83.19, 84.59]
}

methods_data = {
    'CART': [83.48, 84.40, 84.47, 83.01],
    'RF': [82.81, 83.01, 83.78, 82.93],
    'SGB': [84.61, 83.93, 82.92, 84.37],
    'XGB': [82.96, 83.50, 84.12, 83.02],
    'ELM': [83.73, 84.39, 83.39, 83.33]
}

# Create plots
plt.figure(figsize=(14, 10))

for d in range(4):
    plt.subplot(2, 2, d + 1)
    for method in methods:
        plt.plot(lags, methods_data[method], marker='o', label=method)
    plt.xlabel('Game Lag')
    plt.ylabel('Accuracy Percentage')
    plt.title(f'Accuracy vs Game Lag for D = {d}')
    plt.xticks(lags)
    plt.ylim(80, 85)
    plt.legend()

plt.tight_layout()
plt.show()
import matplotlib.pyplot as plt
import numpy as np

# RMSE values for each model and game-lag for D = 0 (as an example)
models = ['CART', 'RF', 'SGB', 'XGB', 'ELM']
lags = ['Lag 3', 'Lag 4', 'Lag 5', 'Lag 6']
rmse_values = {
    'CART': [10.838, 10.340, 9.628, 9.1251],
    'RF': [10.133, 9.067, 10.995, 10.521],
    'SGB': [9.369, 9.438, 9.838, 10.019],
    'XGB': [10.738, 9.469, 9.955, 8.874],
    'ELM': [9.478, 10.123, 9.9819, 10.834]
}

# Generate bar chart
x = np.arange(len(lags))  # Label locations
width = 0.15  # Bar width

fig, ax = plt.subplots(figsize=(10, 6))

for i, model in enumerate(models):
    ax.bar(x + i*width, rmse_values[model], width, label=model)

# Add labels and title
ax.set_xlabel('Game-lags')
ax.set_ylabel('RMSE')
ax.set_title('RMSE Comparison Across Models for Different Game-lags (D = 0)')
ax.set_xticks(x + width * 2)
ax.set_xticklabels(lags)
ax.legend()

plt.tight_layout()
plt.show()

# Line plot to show RMSE trends across game-lags for each model
fig, ax = plt.subplots(figsize=(10, 6))

for model in models:
    ax.plot(lags, rmse_values[model], label=model, marker='o')

# Add labels and title
ax.set_xlabel('Game-lags')
ax.set_ylabel('RMSE')
ax.set_title('RMSE Trends Across Game-lags (D = 0)')
ax.legend()

plt.tight_layout()
plt.show()

import seaborn as sns
import pandas as pd

# Create a DataFrame with RMSE values
data = {
    'Lag 3': [10.838, 10.133, 9.369, 10.738, 9.478],
    'Lag 4': [10.340, 9.067, 9.438, 9.469, 10.123],
    'Lag 5': [9.628, 10.995, 9.838, 9.955, 9.9819],
    'Lag 6': [9.1251, 10.521, 10.019, 8.874, 10.834]
}
df = pd.DataFrame(data, index=models)

# Generate heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df, annot=True, cmap='coolwarm', linewidths=0.5, fmt=".3f")
plt.title('Heatmap of RMSE Across Models and Game-lags (D = 0)')
plt.xlabel('Game-lags')
plt.ylabel('Models')
plt.show()

# Scatter plot to compare predicted and actual scores
predicted_scores = [101, 102, 110, 98, 115]  # Example predicted scores
actual_scores = [100, 103, 108, 99, 116]     # Example actual scores

plt.figure(figsize=(8, 6))
plt.scatter(actual_scores, predicted_scores, color='b')
plt.plot([min(actual_scores), max(actual_scores)], [min(actual_scores), max(actual_scores)], color='red')  # Line y = x

# Add labels and title
plt.xlabel('Actual Scores')
plt.ylabel('Predicted Scores')
plt.title('Predicted vs Actual Scores')
plt.show()

# Box plot to show RMSE distribution across models
rmse_distribution = {
    'CART': [10.838, 10.340, 9.628, 9.1251],
    'RF': [10.133, 9.067, 10.995, 10.521],
    'SGB': [9.369, 9.438, 9.838, 10.019],
    'XGB': [10.738, 9.469, 9.955, 8.874],
    'ELM': [9.478, 10.123, 9.9819, 10.834]
}
df_box = pd.DataFrame(rmse_distribution)

# Generate box plot
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_box)
plt.title('RMSE Distribution Across Models for Different Game-lags (D = 0)')
plt.ylabel('RMSE')
plt.show()


