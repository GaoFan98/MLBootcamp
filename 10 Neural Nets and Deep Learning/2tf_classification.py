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
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv('cancer_classification.csv')
# START TO EXPLORE DATA
# CHECK IF NULL VALUES
df.isnull().sum()
# VISUALIZATION
plt.figure(figsize=(12, 8))
# sns.countplot(x='benign_0__mal_1', data=df)

# DATA FOR PREDICTION
X = df.drop('benign_0__mal_1', axis=1).values
# DATA NEED TO BE PREDICTED
y = df['benign_0__mal_1'].values
# SPLIT DATA FOR TRAIN AND TEST
# TEST SIZE IS PERCENTAGE OF DATA USED IN TEST (in this case 25% of data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=101)

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
# NEED TO FIND HOW MUCH NEURONS SHOULD WE ADD (30 IN THIS CASE)
neuron_count = X_train.shape
# HERE WE START TO DROP NEURONS AFTER EACH LAYER TO PREVENT OVERFITTING
model = Sequential()
# LAYERS
model.add(Dense(30, activation='relu'))
# INTEGER IN DROPOUT IS PERCENTAGE (IN THIS CASE IS 50%)
# MAIN CASES ARE SMTH BETWEEN 0.2 AND 0.5
model.add(Dropout(0.5))
model.add(Dense(15, activation='relu'))
model.add(Dropout(0.5))
# FINAL OUTPUT NODE (LAYER)
# BECAUSE IT IS BINARY CLASSIFICATION PROBLEM SO ACTIVATION WOULD BE SIGMOID
model.add(Dense(1, activation='sigmoid'))
# COMPILING MODEL
model.compile(optimizer='adam', loss='binary_crossentropy')
# FITTING AND TRAINING MODEL
# NEED TO VALIDATE DATA IN ORDER TO TRACK LOSS EVERY TIME BY PASSING X AND Y TEST DATA
# CHECKING WHETHER OR NOT WE OVERFITTING
# FOCUSING ON SMALLER BATCH SIZE IS BETTER FOR NOT OVERFITTING DATA AND MAKE IT PRECISE

######################## COMMENTING THIS BETTER VERSION UNDER ###############################
# model.fit(x=X_train, y=y_train, validation_data=(X_test, y_test), epochs=600) #############
#############################################################################################

# LOSS
# VISUALIZATION OF LOSS
# loss_df = pd.DataFrame(model.history.history)
# SHOWS LOST ON TRAINING DATA AND ON VALIDATION DATA AT THE SAME TIME
# RESULTS SHOWS VALIDATION AND TRAINING ALIGNS BAD, MODEL OVERFITTING!!!!!!!!!!!!!!!!!!!!!!!!!!
# print(loss_df)
# loss_df.plot()
# plt.show()

# SO WE NEED TO STOP MODEL TRAINING BEFORE IT STARTS OVERFIT
# MODES ARE MIN MAX AND AUTO:
# MIN : TRAINING WILL STOP WHEN QUANTITY MONITORED HAS STOPPED DECREASING
# MAX : TRAINING WILL STOP WHEN QUANTITY MONITORED HAS STOPPED INCREASING
# NOTE : IF ACCURACY THEN WE WONNA MAXimize, IF LOSS THEN WE WONNA MINimize
# PATIENCE : HOW MANY EPOCHES WE WAIT BEFORE STOP TRAINING PROCESS
early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=25)
# FITTING AND TRAINING MODEL
# NEED TO VALIDATE DATA IN ORDER TO TRACK LOSS EVERY TIME BY PASSING X AND Y TEST DATA
# CHECKING WHETHER OR NOT WE OVERFITTING
# FOCUSING ON SMALLER BATCH SIZE IS BETTER FOR NOT OVERFITTING DATA AND MAKE IT PRECISE
model.fit(x=X_train, y=y_train, validation_data=(X_test, y_test), epochs=600,
          callbacks=[early_stop])

# LOSS
# VISUALIZATION OF LOSS
loss_df = pd.DataFrame(model.history.history)
# SHOWS LOST ON TRAINING DATA AND ON VALIDATION DATA AT THE SAME TIME
# RESULTS SHOWS VALIDATION AND TRAINING ALIGNS BETTER
# MODEL OVERFITTING STARTS AFTER 166th EPOCH
print(loss_df)
loss_df.plot()
plt.show()

# PREDICTED CLASSES CAUSE IT IS BINARY CLASSIFICATION
predictions = model.predict_classes(X_test)
# CLASSIFICATION REPORT
print(classification_report(y_test, predictions))
# CONFUSION MATRIX
print(confusion_matrix(y_test, predictions))
