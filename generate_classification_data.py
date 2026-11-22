import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Generate 3-class spiral dataset
n = 300 # points per class
classes = 3
np.random.seed(42)

X = np.zeros((n * classes, 2))
y = np.zeros(n * classes, dtype=int)

for j in range(classes):
    ix = range(n * j, n * (j + 1))
    r = np.linspace(0.0, 1, n)  # radius
    t = np.linspace(j * 4, (j + 1) * 4, n) + np.random.randn(n) * 0.2  # angle
    X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
    y[ix] = j

# Add some categorical features just to test the app's categorical handling
colors = np.random.choice(['Red', 'Green', 'Blue'], size=len(y))
df = pd.DataFrame({'Feature_1': X[:, 0], 'Feature_2': X[:, 1], 'Color_Category': colors, 'Target_Class': y})
df.to_csv('classification_example.csv', index=False)

print("✅ Saved 'classification_example.csv'")