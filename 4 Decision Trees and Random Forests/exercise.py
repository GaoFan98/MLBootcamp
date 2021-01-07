import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.ensemble import RandomForestClassifier

loans = pd.read_csv('loan_data.csv')
# CREATING HISTOGRAM OF TWO FICO DISTRIBUTIONS ONE FOR EACH CREDIT POLICY COLUMN
# plt.figure(figsize=(10, 6))
# loans[loans['credit.policy'] == 1]['fico'].hist(bins=35, color='blue', label='Credit Policy 1', alpha=0.6)
# loans[loans['credit.policy'] == 0]['fico'].hist(bins=35, color='red', label='Credit Policy 0', alpha=0.6)
# plt.xlabel('FICO')
# plt.show()
# CREATING HISTOGRAM OF TWO FICO DISTRIBUTIONS ONE FOR EACH NOT FULLY PAID COLUMN
# plt.figure(figsize=(10, 6))
# loans[loans['not.fully.paid'] == 1]['fico'].hist(bins=35, color='blue', label='Not Fully Paid 1', alpha=0.6)
# loans[loans['not.fully.paid'] == 0]['fico'].hist(bins=35, color='red', label='Not Fully Paid 0', alpha=0.6)
# plt.xlabel('FICO')
# plt.show()

# COUNT OF LOANS BY PURPOSE VS NOT FULLY PAID VISUALIZATION
# plt.figure(figsize=(11, 7))
# sns.countplot(x='purpose', hue='not.fully.paid', data=loans,palette='Set1')
# plt.show()
# TREND BETWEEN FICO SCORE AND INTEREST RATE
# sns.jointplot(x='fico', y='int.rate', data=loans, color='purple')
# plt.show()
# TREND BETWEEN NOT FULLY PAID AND CREDIT POLICY
# plt.figure(figsize=(11, 7))
# sns.lmplot(y='int.rate', x='fico', data=loans,hue='credit.policy',col='not.fully.paid',palette='Set1')
# plt.show()

# CLEAN PURPOSE COLUMN DATA FOR ML UNDERSTANDING
cat_feats = ['purpose']
final_data = pd.get_dummies(loans, columns=cat_feats, drop_first=True)

# DATA FOR PREDICTION
X = final_data.drop('not.fully.paid', axis=1)
# DATA NEED TO BE PREDICTED
y = final_data['not.fully.paid']
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
rfc = RandomForestClassifier(n_estimators=300)
# FIT MODEL ON TRAINING DATA
rfc.fit(X_train, y_train)
# PREDICT MODEL
predictions2 = rfc.predict(X_test)
# CLASSIFICATION FINAL REPORT
print('RANDOM FOREST')
print(classification_report(y_test, predictions2))
print(confusion_matrix(y_test, predictions2))
