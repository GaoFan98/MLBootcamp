import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans

# CREATE ARTIFICAL FAKE DATA
data = make_blobs(n_samples=200, n_features=2, centers=4, cluster_std=1.8, random_state=101)

# CREATE K MEANS CLUSTERING ALGORITHM
kmeans = KMeans(n_clusters=4)
kmeans.fit(data[0])
# FEATURES CENTERS AND LABELS
# print(kmeans.fit(data[0]))
# print(kmeans.cluster_centers_)
# print(kmeans.labels_)

# VIZUALAZING PREDICTED LABELS VS ORIGINAL ONES
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))

ax1.set_title('K Means')
ax1.scatter(data[0][:, 0], data[0][:, 1], c=kmeans.labels_, cmap='rainbow')

ax2.set_title('Original')
ax2.scatter(data[0][:, 0], data[0][:, 1], c=data[1], cmap='rainbow')

plt.show()
