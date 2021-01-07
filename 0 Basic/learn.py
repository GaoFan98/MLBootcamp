from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import numpy as np

x, y = np.arange(10).reshape((5, 2)), range(5)
list(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3)

print(
    'Xtrain', x_train,
    'Ytrain', y_train,
    'Xtest', x_test,
    'Ytest', y_test)
#
model = LinearRegression(normalize=True)
# print(model)

model.fit(x_train,y_train)

predictions = model.predict(x_test)

print('PREDICTION',predictions)
