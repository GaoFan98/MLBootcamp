import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import metrics

df = pd.read_csv('USA_Housing.csv')

# print(df.head())
# print(df.info())
# print(df.describe())

# sns.pairplot(df)
# sns.distplot(df['Price'])

####TO SHOW GRAPH UNCOMMENT###
# plt.show(sns) ################
##############################

# DATA FOR PREDICTION
X = df[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
        'Avg. Area Number of Bedrooms', 'Area Population']]
# DATA NEED TO BE PREDICTED
y = df['Price']

# SPLIT DATA FOR TRAIN AND TEST
# TEST SIZE IS PERCENTAGE OF DATA USED IN TEST (in this case 40% of data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)
# CREATE MODEL
lm = LinearRegression()
# FIT MODEL ON TRAINING DATA
lm.fit(X_train,y_train)
# PRINTING THE LINE (intercept)
# print(lm.intercept_)
cdf = pd.DataFrame(lm.coef_,X.columns,columns=['Coeff'])
# THIS SHOWS TABLE                   Coeff
# Avg. Area Income                  21.528276
# Avg. Area House Age           164883.282027
# Avg. Area Number of Rooms     122368.678027
# Avg. Area Number of Bedrooms    2233.801864
# Area Population                   15.150420
# Means that increase of one unit of Income increases price for 21$
# Means that increase of one unit of House Age increases price for 164883$
# Means that increase of one unit of Number of Rooms increases price for 122368$ etc...
# print(cdf)

predictions = lm.predict(X_test)
# CONTAINS PREDICTION PRICES OF THE HOUSE
# print(predictions)
# CONTAINS CORRECT PRICES OF THE HOUSE
# print(y_test)

# GRAPH OF CORRECT PRICES AND PREDICTED
# plt.scatter(y_test,predictions)
# plt.show(sns)

# DIFFERENCE BETWEEN ACTUAL PRICES AND PREDICTED
sns.distplot((y_test-predictions))
# plt.show(sns)

# LINEAR REGRESSION METHODS FOR FINDING BEST LINE DRAW IN ORDER TO FIT TO VALUES
mean_absolute_error = metrics.mean_absolute_error(y_test,predictions)
mean_square_error = metrics.mean_squared_error(y_test,predictions)
mean_root_mean_square_error = np.sqrt(mean_square_error)

print(mean_absolute_error,mean_square_error,mean_root_mean_square_error)
