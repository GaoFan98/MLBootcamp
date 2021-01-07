import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# READING THE CSV FILE
ad_data = pd.read_csv('advertising.csv')
# HISTOGRAM OF AGE COLUMN
# hist = ad_data['Age'].plot.hist(bins=30)
# plt.show(hist)
# AREA INCOME VERSUS AGE
# area_vs_age = sns.jointplot(x='Age', y='Area Income', data=ad_data)
# DAILY TIME SPENT ON WEBSITE VS DAILY INTERNET USAGE
# time_vs_usage = sns.jointplot(x='Daily Time Spent on Site', y='Daily Internet Usage', data=ad_data)
# CLICKED ON AD PAIRPLOT VS EVERY DATA
add_click = sns.pairplot(ad_data, hue='Clicked on Ad')
# plt.show(add_click)
# DATA FOR PREDICTION
X = ad_data[['Daily Time Spent on Site', 'Age', 'Area Income', 'Daily Internet Usage', 'Male']]
y = ad_data['Clicked on Ad']
# SPLIT DATA FOR TRAIN AND TEST
# TEST SIZE IS PERCENTAGE OF DATA USED IN TEST (in this case 30% of data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
# CREATE MODEL
logmodel = LogisticRegression()
# FIT MODEL ON TRAINING DATA
logmodel.fit(X_train, y_train)
# PREDICT MODEL
predictions = logmodel.predict(X_test)
# CLASSIFICATION FINAL REPORT
print(classification_report(y_test, predictions))