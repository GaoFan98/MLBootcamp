import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

df = pd.read_csv('College_Data', index_col=0)

# DATA VISUALIZATION
# sns.lmplot(x='Room.Board', y='Grad.Rate', data=df, hue='Private',fit_reg=False)
g = sns.FacetGrid(df, hue='Private', palette='coolwarm', height=6)
g = g.map(plt.hist, 'Outstate', bins=20)
# plt.show()

# SOME WRONG DATA FOUND WITH GRADUATION PERCENT MORE THAN 100% WHICH IS IMPOSIBLE
# THIS IS UNIVERSITY WITH GRADUATION RATE ERROR
grad_error = df[df['Grad.Rate'] > 100]
# FIXING THIS ERROR
df['Grad.Rate']['Cazenovia College'] = 100

# CREATE K MEANS CLUSTERING ALGORITHM
kmeans = KMeans(n_clusters=2)
kmeans.fit(df.drop('Private', axis=1))


# CONVERTING PRIVATE COLUMN FROM STRING TO INTEGER VALUES
def converter(private):
    if private == 'Yes':
        return (1)
    else:
        return (0)


# NOW CLUSTER COLUMN CONTAINS INTEGER VALUES OF PRIVATE COLUMN
df['Cluster'] = df['Private'].apply(converter)

# CLASSIFICATION FINAL REPORT
print(classification_report(df['Cluster'], kmeans.labels_))
print(confusion_matrix(df['Cluster'], kmeans.labels_))


