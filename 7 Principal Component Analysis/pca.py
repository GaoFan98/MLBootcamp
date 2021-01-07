import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

cancer = load_breast_cancer()

df = pd.DataFrame(cancer['data'], columns=cancer['feature_names'])

# SCALE DATA TO REMOVE ALL ENORMOUS INTEGERS
scaler = StandardScaler()
scaler.fit(df)
scaled_data = scaler.transform(df)

# PCA
# CREATING MODEL SET NUMBER OF DIMENSIONS=2
pca = PCA(n_components=2)
# CONVERTING 30 DIMENSIONS TO 2 DIMENSION
pca.fit(scaled_data)
x_pca = scaler.transform(scaled_data)

# VISUALIZE DATA
plt.figure(figsize=(8, 6))
# COLORING DATA BY TARGET COLUMN
plt.scatter(x_pca[:, 0], x_pca[:, 1], c=cancer['target'], cmap='plasma')
plt.xlabel('First principal components')
plt.ylabel('Second principal components')
plt.show()
