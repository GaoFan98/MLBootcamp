import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

train = pd.read_csv('titanic_train.csv')
# FIND WHICH COLUMNS CONSIST OF MORE MISSING DATA
# sns.heatmap(train.isnull(),yticklabels=False,cbar=False,cmap='viridis')
# plt.show(sns)
# IN THIS CASE IT WAS MISSING SOME OF AGE DATA AND A LOT OF CABIN DATA

# sns.set_style('whitegrid')

# SURVIVED VS NOT SURVIVED DATA
# sns.countplot(x='Survived',data=train)
# plt.show(sns)
# SURVIVED PEOPLE BY GENDER
# sns.countplot(x='Survived',hue='Sex',data=train,palette='RdBu_r')
# plt.show(sns)
# SURVIVED PEOPLE BY PASSENGER CLASS
# sns.countplot(x='Survived',hue='Pclass',data=train,palette='RdBu_r')

# AMOUNT OF PASSENGERS BY AGE
sns.distplot(train['Age'].dropna(), kde=False, bins=35)


# plt.show(sns)
# GET AVERAGE OF AGE BY CLASS IN ORDER TO REMOVE MISSING DATA FROM AGE CATEGORY
def impute_age(cols):
    age = cols[0]
    pcclass = cols[1]

    if pd.isnull(age):
        if pcclass == 1:
            return 37
        elif pcclass == 2:
            return 29
        else:
            return 24
    else:
        return age


train['Age'] = train[['Age', 'Pclass']].apply(impute_age, axis=1)

# sns.heatmap(train.isnull(), yticklabels=False, cbar=False, cmap='viridis')
# plt.show(sns)

# WE GONNA DROP CABIN COLUMNS CAUSE IT HAS TOO MUCH MISSING DATA
train.drop('Cabin', axis=1, inplace=True)
# DROPPING REMAINED NULL COLUMNS
train.dropna(inplace=True)

# CONVERT MALE/FEMALE TO NUMS 0 AND 1, CAUSE COMPUTER CAN'T USE STRING VALUES
# NEXT STEP IS TO DROP ONE OF COLUMN (MALE OR FEMALE) NOT TO MESS UP EVERYTHING
sex = pd.get_dummies(train['Sex'], drop_first=True)
# DO THE SAME WITH EMBARKED COLUMN
embark = pd.get_dummies(train['Embarked'], drop_first=True)
# DO THE SAME WITH PCLASS COLUMN
# pclass = pd.get_dummies(train['Pclass'], drop_first=True)
# NEXT STEP IS TO CONCAT OUR NEW COLUMNS TO EXISTING ONES
train = pd.concat([train, sex, embark], axis=1)
# DROPPING SEX AND EMBARKED CAUSE WE ALREADY REPLACED THESE COLUMNS WITH NEW ONES
# DROPPING NAME AND TICKET AND PASSENGER ID COLUMNS CAUSE THEY DON'T CONSIST OF USEFUL INFORMATION
train.drop(['Sex', 'Embarked', 'Name', 'Ticket', 'PassengerId'], axis=1, inplace=True)
# print(train)

# DATA FOR PREDICTION
# DROPPING SURVIVED COLUMN, AND GET EACH COLUMN EXCEPT SURVIVED
X = train.drop('Survived', axis=1)
# DATA NEED TO BE PREDICTED
y = train['Survived']

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
# print(confusion_matrix(y_test,predictions))


# 1 TN / True Negative: when a case was negative and predicted negative
# 2 TP / True Positive: when a case was positive and predicted positive
# 3 FN / False Negative: when a case was positive but predicted negative
# 4 FP / False Positive: when a case was negative but predicted positive

# PRECISION
# Precision is the ability of a classifier not to label an instance positive
# that is actually negative. For each class it is defined as the ratio
# of true positives to the sum of true and false positives.
#
# Precision is Accuracy of positive predictions.
# Precision = TP/(TP + FP)

# RECALL
# Recall is the ability of a classifier to find all positive instances.
# For each class it is defined as the ratio of true positives
# to the sum of true positives and false negatives.
#
# Recall: Fraction of positives that were correctly identified.
# Recall = TP/(TP+FN)

# F1 SCORE
# F1 scor is a weighted harmonic mean of precision and recall such that
# the best score is 1.0 and the worst is 0.0. Generally speaking,
# F1 scores are lower than accuracy measures as they embed precision
# and recall into their computation. As a rule of thumb, the weighted
# average of F1 should be used to compare classifier models, not global accuracy.
#
# F1 Score = 2*(Recall * Precision) / (Recall + Precision)
