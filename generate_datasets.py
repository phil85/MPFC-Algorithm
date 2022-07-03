from sklearn.datasets import make_blobs
import numpy as np
import pandas as pd

# Set random seed
np.random.seed(24)

# Generate data
n_points = 100
X, y = make_blobs(n_samples=n_points, centers=10)

# Assign colors to objects
colors = np.zeros(y.shape, dtype=int)
colors[y >= 6] = 1

# Export data to CSV file
df = pd.DataFrame(X, columns=['x1', 'x2'])
df['color'] = colors
df.to_csv('illustrative_example.csv', index=False)
