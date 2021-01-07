import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import load_model

df = pd.read_csv('fake_reg.csv')

# DATA FOR PREDICTION
X = df[['feature1', 'feature2']].values
# DATA NEED TO BE PREDICTED
y = df['price'].values
# SPLIT DATA FOR TRAIN AND TEST
# TEST SIZE IS PERCENTAGE OF DATA USED IN TEST (in this case 30% of data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# NOW WE NEED TO SCALE DATA CAUSE NUMBERS ARE TOO LARGE
# CREATING SCALER INSTANCE
scaler = MinMaxScaler()
# FITTING DATA
scaler.fit(X_train)
# TRANSFORMING
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

# CREATING KERAS MODEL
# DENSE IS TYPE OF NEURAL NETWORK WHERE INTEGER IS NUMBER OF NEURONS AND ACTIVATION FUNCTION
# model = Sequential([
#     Dense(4, activation='relu'),
#     Dense(2, activation='relu'),
#     Dense(1),
# ])
# ANOTHER WAY TO DO IT
model = Sequential()
# LAYERS
model.add(Dense(4, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(4, activation='relu'))
# FINAL OUTPUT NODE (LAYER)
model.add(Dense(1))
# COMPILING MODEL
model.compile(optimizer='rmsprop', loss='mse')
# FITTING AND TRAINING MODEL
model.fit(x=X_train, y=y_train, epochs=250)
# LOSS
# VISUALIZATION OF LOSS
loss_df = pd.DataFrame(model.history.history)
loss_df.plot()
# plt.show()

# EVALUATING MODEL (HOW WELL MODEL PERFORMS ON UNKNOWN DATA)
# model.evaluate(X_test, y_test)
# PREDICTED PRICES
test_predictions = model.predict(X_test)
# COMPARING PREDICTED PRICES VS REAL
test_predictions = pd.Series(test_predictions.reshape(300, ))
pred_df = pd.DataFrame(y_test, columns=['Test True Y'])
pred_df = pd.concat([pred_df, test_predictions], axis=1)
pred_df.columns = ['Test True Y', 'Model Predictions']

# VISUALIZATION
sns.scatterplot(x='Test True Y', y='Model Predictions', data=pred_df)

# MEAN ABSOLUTE AND SQUARED ERRORS
mae = mean_absolute_error(pred_df['Test True Y'], pred_df['Model Predictions'])
mse = mean_squared_error(pred_df['Test True Y'], pred_df['Model Predictions'])

# PASSING NEW DATA
new_gem = [[998, 1000]]
new_gem = scaler.transform(new_gem)
model.predict(new_gem)

# SAVING MODEL
model.save('my_gem_model.h5')
later_model = load_model('my_gem_model.h5')
