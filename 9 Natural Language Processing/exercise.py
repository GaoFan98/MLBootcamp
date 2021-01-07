import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline


yelp = pd.read_csv('yelp.csv')

# GET LENGTH OF MESSAGES
yelp['length'] = yelp['text'].apply(len)
# VISUALIZE LENGTH OF MESSAGES BASED ON STARS GIVEN
# g = sns.FaceGrid(yelp, col='stars')
# g.map(plt.hist, 'text length', bins=50)

# MEAN VALUES OF NUMERICAL COLUMNS
stars = yelp.groupby('stars').mean()

# GRABBING REVIEWS WITH ONLY 1 OR 5 STARS
yelp_class = yelp[(yelp['stars'] == 1) | (yelp['stars'] == 5)]

# X and y where X = text column and y = stars column
X = yelp_class['text']
y = yelp_class['stars']

# CREATING COUNT VECTORIZER
# CREATING MODEL
cv = CountVectorizer()
X = cv.fit_transform(X)

# SPLIT DATA FOR TRAIN AND TEST
# TEST SIZE IS PERCENTAGE OF DATA USED IN TEST (in this case 30% of data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# CREATING NAIVE BAYES MODEL
nb = MultinomialNB()
# FITTING MODEL
nb.fit(X_train, y_train)
# PREDICT MODEL
predictions = nb.predict(X_test)
# CLASSIFICATION FINAL REPORT
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))

pipeline = Pipeline([
    ('bow', CountVectorizer()),
    ('tfidf', TfidfTransformer()),
    ('model', MultinomialNB())
])

X = yelp_class['text']
y= yelp_class['stars']

# SPLIT DATA FOR TRAIN AND TEST
# TEST SIZE IS PERCENTAGE OF DATA USED IN TEST (in this case 30% of data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# FIT MODEL ON TRAINING DATA
pipeline.fit(X_train, y_train)
# PREDICT MODEL
predictions = pipeline.predict(X_test)
# CLASSIFICATION FINAL REPORT
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))
