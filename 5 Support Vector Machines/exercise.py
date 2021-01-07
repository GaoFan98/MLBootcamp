import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GridSearchCV

iris = sns.load_dataset('iris')
# VISUALIZE WHICH FLOWER SPECIES MOST SEPARABLE
sns.pairplot(iris, hue='species', palette='Dark2')
# DATA FOR PREDICTION
X = iris.drop('species', axis=1)
# DATA NEED TO BE PREDICTED
y = iris['species']
# SPLIT DATA FOR TRAIN AND TEST
# TEST SIZE IS PERCENTAGE OF DATA USED IN TEST (in this case 30% of data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
# CREATE MODEL
model = SVC()
# FIT MODEL ON TRAINING DATA
model.fit(X_train, y_train)
# PREDICT MODEL
predictions = model.predict(X_test)
# CLASSIFICATION FINAL REPORT
print('BEFORE ADJUSTING')
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))
print('\n')

# ADJUST SVM MODEL BY ADDING SOME GRID PARAMETERS
param_grid = {'C': [0.1, 1, 10, 100, 1000], 'gamma': [1, 0.1, 0.01, 0.001, 0.0001]}
grid = GridSearchCV(SVC(), param_grid, verbose=2)
# FINDING BEST COMBINATION
grid.fit(X_train, y_train)
# PREDICT GRID MODEL
grid_predictions = grid.predict(X_test)
# CLASSIFICATION FINAL REPORT
print('AFTER GRID ADJUST')
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))
