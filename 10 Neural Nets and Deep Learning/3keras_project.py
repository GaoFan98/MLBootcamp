import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from sklearn.metrics import classification_report, confusion_matrix
import random

# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import MinMaxScaler
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
# from tensorflow.keras.layers import Dropout
# from tensorflow.keras.callbacks import EarlyStopping
# from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv('lending_club_loan_two.csv')

# VISUALIZATION OF LOAN STATUS COLUMN
sns.countplot(x='loan_status', data=df)
# EXPLORE CORRELATION BETWEEN ALL NUMERIC VALUES
df.corr()

# CREATING LOAN_REPAID COLUMN
df['loan_status'].unique()
df['loan_repaid'] = df['loan_status'].map({'Fully Paid': 1, 'Charged Off': 0})

# DATA PREPROCESSING - REMOVING OR FILLING ANY MISSING DATA
null_data = df.isnull().sum()
# MISSING DATA IN TERMS OF PERCENTAGE OF TOTAL DATA
null_per = (null_data / len(df)) * 100

# DROPPING EMP TITLE COLUMN BECAUSE OF USELESS
df = df.drop('emp_title', axis=1)

# VISUALIZATION OF EMPLOYMENT LENGTH ACCORDING TO LOAN STATUS
plt.figure(figsize=(12, 4))
sns.countplot(x='emp_length', data=df, hue='loan_status')
# plt.show()

# PERCENTAGE OF CHARGE OFFS PER CATEGORY
charged = df[df['loan_status'] == 'Charged Off'].groupby('emp_length').count()['loan_status']
paid = df[df['loan_status'] == 'Fully Paid'].groupby('emp_length').count()['loan_status']
# THERE IS NO BIG DIFFERENCE IN RATIO
ratio = charged / (charged + paid)
# SO WE GONNA DROP THAT COLUMN
df = df.drop('emp_length', axis=1)
# ALSO GONNA DROP TITLE COLUMN
df = df.drop('title', axis=1)

# SINCE THERE IS A LOT OF NULL DATA IN MORT_ACC COLUMN WE NEED TO DROP IT
# BUT MORT_ACC DATA CONSIST OF 10% OF ALL DATA SO WE CAN NOT DROP IT
# THAT IS WHY NEED TO REPLACE WITH SOME OTHER DATA
# FIRST NEED TO FIND HIGH CORRELATED DATA TO MORT_ACC COLUMN
# IT SEEMS LIKE TOTAL_ACC CORRELATES WITH MORT_ACC COLUMN
mort_acc_corr = df.corr()['mort_acc'].sort_values()
# SO WE NEED TO GROUP DATA BY TOTAL_ACC AND CALCULATE MEAN VALUE FOR MORT_ACC PER TOTAL_ACC
total_acc_avg = df.groupby('total_acc').mean()['mort_acc']


# IF MORT_ACC COLUMN HAS NULL VALUE WE FILL IT WITH TOTAL_ACC VALUE
def fill_mort_acc(total_acc, mort_acc):
    if np.isnan(mort_acc):
        return total_acc_avg[total_acc]
    else:
        return mort_acc


df['mort_acc'] = df.apply(lambda x: fill_mort_acc(x['total_acc'], x['mort_acc']), axis=1)

# DROPPING REMAINING NULL VALUES CAUSE THERE IS NO IMPORTANT DATA LEFT
df.dropna()

# LISTING ALL NON-NUMERIC COLUMNS
str_cols = df.select_dtypes(['object']).columns

# CONVERTING TERM COLUMN INTO EITHER 36 OR 60 INTEGER NUMERIC DATA TYPE
df['term'] = df['term'].apply(lambda term: int(term[:3]))

# DROPPING GRADE COLUMN CAUSE IT IS PART OF SUBGRADE COLUMN
df = df.drop('grade', axis=1)

# CONVERT SUBGRADE TO DUMMY VARIABLES
# DROP FIRST FOR NOT LOOPING THE SAME DATA AGAIN
# Ex: A/B/C is NOT 0/0/1 , BUT A/B 0/0, SO IF IT IS NOT A OR B COLUMN IT IS C COLUMN
dummies = pd.get_dummies(df['sub_grade'], drop_first=True)
# DROPPING ORIGINAL SUB_GRADE COLUMN SINCE WE REPLACE IT WITH DUMMIES
# AND CONCATENATE DUMMIES TO EXISTING DATAFRAME
df = pd.concat([df.drop('sub_grade', axis=1), dummies], axis=1)

# DO THE SAME WITH VERIFICATION_STATUS,APPLICATION_TYPE,INITIAL_LIST_STATUS,PURPOSE
dummies = pd.get_dummies(df[['verification_status', 'application_type', 'initial_list_status', 'purpose']],
                         drop_first=True)
df = df.drop(['verification_status', 'application_type', 'initial_list_status', 'purpose'], axis=1)
df = pd.concat([df, dummies], axis=1)
# HOME_OWNERSHIP COLUMN CONSIST OF NONE/ANY DATA IN IT, SO WE PUT THESE TWO TO 'OTHER' CATEGORY
df['home_ownership'] = df['home_ownership'].replace(['NONE', 'ANY'], 'OTHER')
# AND CONVERT TO DUMMY VARIABLES
dummies = pd.get_dummies(df['home_ownership'], drop_first=True)
df = pd.concat([df.drop('home_ownership', axis=1), dummies], axis=1)

# GRABBING ONLY ZIPCODES FROM ADDRESS COLUMN
df['zip_code'] = df['address'].apply(lambda address: address[-5:])
# AND CONVERT TO DUMMY VARIABLES
dummies = pd.get_dummies(df['zip_code'], drop_first=True)
df = pd.concat([df.drop('zip_code', axis=1), dummies], axis=1)

# DROPPING ADDRESS SINCE WE DONT NEED THIS COLUMN ANY MORE
df = df.drop('address', axis=1)

# DROPPING ISSUE_DATE SINCE WE DONT NEED THIS COLUMN
df = df.drop('issue_d', axis=1)

# GETTING ONLY YEAR FROM EARLIEST_CR_LINE COLUMN DROPPING MONTH
df['earliest_cr_line'] = df['earliest_cr_line'].apply(lambda date: int(date[-4:]))

# DROPPING LOAN_STATUS COLUMN CAUSE IT IS A DUPLICATE OF LOAR_REPAID COLUMN
df = df.drop('loan_status', axis=1)

# DATA FOR PREDICTION
X = df.drop('loan_repaid', axis=1).values
# DATA NEED TO BE PREDICTED
y = df['loan_repaid'].values
# SPLIT DATA FOR TRAIN AND TEST
# TEST SIZE IS PERCENTAGE OF DATA USED IN TEST (in this case 20% of data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

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
# NEED TO FIND HOW MUCH NEURONS SHOULD WE ADD (78 IN THIS CASE)
neuron_count = X_train.shape
# HERE WE START TO DROP NEURONS AFTER EACH LAYER TO PREVENT OVERFITTING
model = Sequential()
# LAYERS
model.add(Dense(78, activation='relu'))
# INTEGER IN DROPOUT IS PERCENTAGE (IN THIS CASE IS 20%)
# MAIN CASES ARE SMTH BETWEEN 0.2 AND 0.5
model.add(Dropout(0.2))
model.add(Dense(39, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(19, activation='relu'))
model.add(Dropout(0.2))
# FINAL OUTPUT NODE (LAYER)
# BECAUSE IT IS CLASSIFICATION PROBLEM SO ACTIVATION WOULD BE SIGMOID
model.add(Dense(1, activation='sigmoid'))
# COMPILING MODEL
model.compile(optimizer='adam', loss='binary_crossentropy')

# FITTING AND TRAINING MODEL
# NEED TO VALIDATE DATA IN ORDER TO TRACK LOSS EVERY TIME BY PASSING X AND Y TEST DATA
# CHECKING WHETHER OR NOT WE OVERFITTING
# FOCUSING ON SMALLER BATCH SIZE IS BETTER FOR NOT OVERFITTING DATA AND MAKE IT PRECISE
# BECAUSE DATASET IS BIG BETTER TO FITS IN BATCHES
model.fit(x=X_train, y=y_train, batch_size=256, validation_data=(X_test, y_test), epochs=25)

# LOSS
# VISUALIZATION OF LOSS
loss_df = pd.DataFrame(model.history.history)
# SHOWS LOST ON TRAINING DATA AND ON VALIDATION DATA AT THE SAME TIME
# RESULTS SHOWS VALIDATION AND TRAINING ALIGNS BETTER
loss_df.plot()
plt.show()

# PREDICTED CLASSES CAUSE IT IS BINARY CLASSIFICATION
predictions = model.predict_classes(X_test)
# CLASSIFICATION REPORT
print('CLASSIFICATION REPORT===============================')
print(classification_report(y_test, predictions))
print('\n')
# CONFUSION MATRIX
# print('CONFUSION MATRIX===============================')
# print(confusion_matrix(y_test, predictions))
# print('\n')

# WOULD YOU OFFER PERSON LOAN?
random.seed(101)
random_ind = random.randint(0, len(df))

new_customer = df.drop('loan_repaid', axis=1).iloc[random_ind]

new_customer = scaler.transform(new_customer.values.reshape(1, 78))
offer = model.predict_classes(new_customer)
print('OFFER LOAN?')
print(offer)
print('\n')

# DID PERSON ACTUALLY END UP PAYING BACK LOANS
get_back_loan = df.iloc[random_ind]['loan_repaid']
print('PAYING BACK LOANS?')
print(get_back_loan)
print('\n')
