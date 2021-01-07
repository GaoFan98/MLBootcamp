import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_absolute_error, mean_squared_error, explained_variance_score

df = pd.read_csv('kc_house_data.csv')
# START TO EXPLORE DATA
# CHECK IF NULL VALUES
df.isnull().sum()
# VISUALIZATION
plt.figure(figsize=(12, 8))
# # OUTPUT RESULT SHOWS THAT MAJORITY OF HOUSE PRICES ARE BETWEEN 0 AND HALF MILLION
# sns.distplot(df['price'])
# # OUTPUT RESULT SHOWS THAT MAJORITY OF HOUSE BEDROOMS COUNT IS 3
# sns.countplot(df['bedrooms'])
# # PRICE BY LIVING SPACE SQUARE METERS
# sns.scatterplot(x='price', y='sqft_living', data=df)
# # PRICE BY LATITUDE AND LONGITUDE
# sns.scatterplot(x='price', y='lat', data=df)
# sns.scatterplot(x='price', y='long', data=df)
# # HOUSES IN LAT AND LONG
# sns.scatterplot(x='long', y='lat', data=df)
# # HOUSES IN LAT AND LONG BY PRICE
# sns.scatterplot(x='long', y='lat', hue='price', data=df)
# # MAKING DATA MORE PRECISE
# # CREATING MAP BY DROPPING TOP 1% EXPENSIVE HOUSES
# non_top_1_per = df.sort_values('price', ascending=False).iloc[216:]
# sns.scatterplot(x='long', y='lat', hue='price', data=non_top_1_per, palette='RdYlGn')

# DROPPING UNNECESSARY COLUMNS LIKE ID
df = df.drop('id', axis=1)
# CONVERTING DATE STRING FORMAT TO DATE TIME
df['date'] = pd.to_datetime(df['date'])
# TAKING ONLY YEAR AND MONTHS CAUSE DAY IS NOT IMPORTANT
year = df['year'] = df['date'].apply(lambda date: date.year)
month = df['month'] = df['date'].apply(lambda date: date.month)
# PRICE BY MONTH GRAPH
sns.boxplot(x='month', y='price', data=df)
# plt.show()
# LIST OF AVERAGE HOUSE PRICE BY MONTH
av_price_month = df.groupby('month').mean()['price']

# DROPPING DATE,ZIPCODE COLUMNS
df = df.drop('date', axis=1)
df = df.drop('zipcode', axis=1)

# DATA FOR PREDICTION
X = df.drop('price', axis=1).values
# DATA NEED TO BE PREDICTED
y = df['price'].values
# SPLIT DATA FOR TRAIN AND TEST
# TEST SIZE IS PERCENTAGE OF DATA USED IN TEST (in this case 30% of data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)

# NOW WE NEED TO SCALE DATA CAUSE NUMBERS ARE TOO LARGE
# CREATING SCALER INSTANCE
scaler = MinMaxScaler()
# FITTING AND TRANSFORMING ONLY TRAIN DATA
X_train = scaler.fit_transform(X_train)
# TRANSFORMING ONLY TEST DATA
X_test = scaler.transform(X_test)

# CREATING KERAS MODEL
# DENSE IS TYPE OF NEURAL NETWORK WHERE INTEGER IS NUMBER OF NEURONS AND ACTIVATION FUNCTION
# LAYERS
# NEED TO FIND HOW MUCH NEURONS SHOULD WE ADD (19 IN THIS CASE)
neuron_count = X_train.shape
# HERE WE START TO DROP NEURONS AFTER EACH LAYER TO PREVENT OVERFITTING
model = Sequential()
# LAYERS
model.add(Dense(19, activation='relu'))
# INTEGER IN DROPOUT IS PERCENTAGE (IN THIS CASE IS 50%)
# MAIN CASES ARE SMTH BETWEEN 0.2 AND 0.5
model.add(Dropout(0.5))
model.add(Dense(19, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(19, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(19, activation='relu'))
model.add(Dropout(0.5))
# FINAL OUTPUT NODE (LAYER)
model.add(Dense(1))
# COMPILING MODEL
model.compile(optimizer='adam', loss='mse')

# FITTING AND TRAINING MODEL
# SO WE NEED TO STOP MODEL TRAINING BEFORE IT STARTS OVERFIT
# MODES ARE MIN MAX AND AUTO:
# MIN : TRAINING WILL STOP WHEN QUANTITY MONITORED HAS STOPPED DECREASING
# MAX : TRAINING WILL STOP WHEN QUANTITY MONITORED HAS STOPPED INCREASING
# NOTE : IF ACCURACY THEN WE WONNA MAXimize, IF LOSS THEN WE WONNA MINimize
# PATIENCE : HOW MANY EPOCHES WE WAIT BEFORE STOP TRAINING PROCESS
early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)
# NEED TO VALIDATE DATA IN ORDER TO TRACK LOSS EVERY TIME BY PASSING X AND Y TEST DATA
# CHECKING WHETHER OR NOT WE OVERFITTING
# FOCUSING ON SMALLER BATCH SIZE IS BETTER FOR NOT OVERFITTING DATA AND MAKE IT PRECISE
model.fit(x=X_train, y=y_train, validation_data=(X_test, y_test), epochs=600,
          callbacks=[early_stop])
# LOSS
# VISUALIZATION OF LOSS
loss_df = pd.DataFrame(model.history.history)
# SHOWS LOST ON TRAINING DATA AND ON VALIDATION DATA AT THE SAME TIME
# RESULTS SHOWS VALIDATION AND TRAINING ALIGNS GOOD, SO THERE IS NO OVERFITTING
# AND WE COULD CONTINUE TRAINING
print(loss_df)
loss_df.plot()
plt.show()

# PREDICTED PRICES
predictions = model.predict(X_test)
# MEAN ABSOLUTE AND SQUARED ERRORS AND EXPLAINED VARIANCE SCORE
mae = mean_absolute_error(y_test, predictions)
mse = mean_squared_error(y_test, predictions)
evs = explained_variance_score(y_test, predictions)

print('MEAN ABSOLUTE ERROR========================')
print(mae)
print('\n')

print('MEAN SQARED ERROR========================')
print(mse)
print('\n')

print('EXPLAINED VARIANCE SCORE========================')
print(evs)
print('\n')

# PASSING NEW DATA
new_house = df.drop('price', axis=1).iloc[0]
new_house = scaler.transform(new_house.values.reshape(-1, 19))
new_house_predict_price = model.predict(new_house)

print('PRICE OF NEW HOUSE========================')
print(new_house_predict_price)
print('\n')
