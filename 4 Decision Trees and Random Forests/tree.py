import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier

df = pd.read_csv('kyphosis.csv')

# VISUALIZE DATA
sns.pairplot(df, hue='Kyphosis')
# plt.show(sns)
# DATA FOR PREDICTION
X = df.drop('Kyphosis', axis=1)
# DATA NEED TO BE PREDICTED
y = df['Kyphosis']
# SPLIT DATA FOR TRAIN AND TEST
# TEST SIZE IS PERCENTAGE OF DATA USED IN TEST (in this case 30% of data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
# CREATE MODEL
dtree = DecisionTreeClassifier()
# FIT MODEL ON TRAINING DATA
dtree.fit(X_train, y_train)
# PREDICT MODEL
predictions = dtree.predict(X_test)
# CLASSIFICATION FINAL REPORT
print('DECISION TREE')
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))
print('\n')

# CREATE MODEL
rfc = RandomForestClassifier(n_estimators=200)
# FIT MODEL ON TRAINING DATA
rfc.fit(X_train, y_train)
# PREDICT MODEL
predictions2 = rfc.predict(X_test)
# CLASSIFICATION FINAL REPORT
print('RANDOM FOREST')
print(classification_report(y_test, predictions2))
print(confusion_matrix(y_test, predictions2))
