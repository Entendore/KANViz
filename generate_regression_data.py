import pandas as pd
import numpy as np

# Generate 1000 samples
n = 1000
np.random.seed(42)

x1 = np.random.uniform(-5, 5, n)
x2 = np.random.uniform(-5, 5, n)
x3 = np.random.uniform(-2, 2, n)

# The true mathematical equation
y = 2 * np.sin(x1) + 0.5 * x2**2 - 1.5 * np.exp(-x3)

# Add a little bit of noise
noise = np.random.normal(0, 0.1, n)
y = y + noise

# Create DataFrame
df = pd.DataFrame({'X1_Sin': x1, 'X2_Square': x2, 'X3_Exp': x3, 'Target_Y': y})
df.to_csv('regression_example.csv', index=False)
print("✅ Saved 'regression_example.csv'")