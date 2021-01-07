import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

customers = pd.read_csv('Ecommerce Customers')
# SHOWS EVERYTHING ON GRAPHS
# sns.pairplot(customers)
# plt.show(sns)
############################
# SHOWS BEST APPROPRIATE LINE FOR LENGTH OF MEMBERSHIP COLUMN
# sns.lmplot(x='Length of Membership',y='Yearly Amount Spent',data=customers)
# plt.show(sns)
###########################
# DATA FOR PREDICTION
X = customers[['Avg. Session Length', 'Time on App', 'Time on Website', 'Length of Membership']]
# DATA NEED TO BE PREDICTED
y = customers['Yearly Amount Spent']
# SPLIT DATA FOR TRAIN AND TEST
# TEST SIZE IS PERCENTAGE OF DATA USED IN TEST (in this case 40% of data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
# CREATE MODEL
lm = LinearRegression()
# FIT MODEL ON TRAINING DATA
lm.fit(X_train,y_train)
# CONTAINS PREDICTIONS
predictions = lm.predict(X_test)
# print(predictions)
# CONTAINS REAL DATA
# print(y_test)
# LINEAR REGRESSION METHODS FOR FINDING BEST LINE DRAW IN ORDER TO FIT TO VALUES
mean_absolute_error = metrics.mean_absolute_error(y_test,predictions)
mean_square_error = metrics.mean_squared_error(y_test,predictions)
mean_root_mean_square_error = np.sqrt(mean_square_error)
# variance—in terms of linear regression,
# variance is a measure of how far observed values differ from the average of predicted values,
# i.e., their difference from the predicted value mean.
# The goal is to have a value that is low (it measures between 0 and 100%).
# So if it is 100%, the two variables are perfectly correlated, i.e., with no variance at all.
explained_variance_score = metrics.explained_variance_score(y_test,predictions)
# 0.9890771231889607 is variance score in this example (98%). Great result
# print(mean_absolute_error,mean_square_error,mean_root_mean_square_error,explained_variance_score)

# sns.distplot((y_test,predictions),bins=50)
# plt.show(sns)

# STAY FOCUS ON MOBILE OR WEBSITE
cdf = pd.DataFrame(lm.coef_,X.columns,columns=['Coeff'])
print(cdf)

