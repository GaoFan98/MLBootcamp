import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

df = pd.read_csv('Classified Data', index_col=0)
# CAUSE A LOT OF DATA FOR KNN COULD INCREASE SCALE DISTANCE BETWEEN OBSERVATION
# WE INCLUDE STANDARD SCALER IN ORDER TO STANDARTIZE EVERYTHING EXCEPT COLUMN THAT WE MUST PREDICT
scaler = StandardScaler()
scaler.fit(df.drop('TARGET CLASS', axis=1))
# PERFORMS STANDARDIZATION BY CENTRING AND SCALING
scaled_features = scaler.transform(df.drop('TARGET CLASS', axis=1))
# CREATE FEATURED DATAFRAME RECREATE EVERYTHING BUT LAST COLUMN THAT WE GONNA PREDICT
df_feat = pd.DataFrame(scaled_features, columns=df.columns[:-1])
# DATA FOR PREDICTION
# GRAB EVERY DATA
X = df_feat
# DATA NEED TO BE PREDICTED
y = df['TARGET CLASS']
# SPLIT DATA FOR TRAIN AND TEST
# TEST SIZE IS PERCENTAGE OF DATA USED IN TEST (in this case 30% of data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
# CREATE MODEL AND SET NEIGHBORS NUMBER
knn = KNeighborsClassifier(n_neighbors=1)
# FIT MODEL ON TRAINING DATA
knn.fit(X_train, y_train)
# PREDICT MODEL
predictions = knn.predict(X_test)
# CLASSIFICATION FINAL REPORT
print('BEFORE FINDING BEST K VALUE')
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))
print('\n')

# FINDING BEST NEIGHBORS NUMBER IN ORDER TO PREDICT MORE PRECISE
error_rate = []

for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    # AVERAGE WHERE PREDICTIONS WERE NOT EQUAL TO ACTUAL TEST VALUES
    error_rate.append(np.mean(pred_i != y_test))

# VISUALIZE OUR DATA
plt.figure(figsize=(10, 6))
plt.plot(range(1, 40), error_rate, color='blue', linestyle='dashed', marker='o', markerfacecolor='red', markersize=10)
plt.title('Error rate vs K value')
plt.xlabel('K')
plt.ylabel('Error rate')
# plt.show()
# WE FOUND OUT THAT K=17 HAS ONE OF THE LOWEST ERROR RATE

# CREATE MODEL AND SET NEIGHBORS NUMBER
knn = KNeighborsClassifier(n_neighbors=17)
# FIT MODEL ON TRAINING DATA
knn.fit(X_train, y_train)
# PREDICT MODEL
predictions = knn.predict(X_test)
# CLASSIFICATION FINAL REPORT
print('AFTER FINDING BEST K VALUE')
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))
