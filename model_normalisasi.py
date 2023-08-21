import pandas as pd
from tensorflow import keras

df = pd.read_csv('akp_dengan_blok.csv', delimiter=';')

dataset = df.values

X = dataset[:, 0:9]
Y = dataset[:, -1]

from sklearn import preprocessing
min_max_scaler = preprocessing.MinMaxScaler()
X_scale = min_max_scaler.fit_transform(X)
# X_scale = min_max_scaler.fit_transform(X)
# Y_scale = min_max_scaler.fit_transform(Y.reshape(-1, 1))

# X_scale = dataset_scale[:, 0:9]
# Y_scale = dataset_scale[:, -1]

# print(Y_scale)

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(X_scale, Y, test_size=0.2)

from keras.models import Sequential
from keras.layers import Dense
import time

start_time = time.time()

model = Sequential([
    Dense(9, activation='relu', input_shape=(9,)),
    Dense(6, activation='relu'),
    Dense(6, activation='relu'),
    Dense(1),
])

model.compile(optimizer = keras.optimizers.Adam(learning_rate=0.001),
              loss= 'mape',
              metrics= ['mse', 'mape', 'mae'],
              )

history = model.fit(x_train, y_train,
            epochs=500,
            batch_size=32,
            )

model.summary()
end_time = time.time()
waktu = end_time - start_time
print("Waktu yang diperlukan:", waktu)

from matplotlib import pyplot as plt
from sklearn.metrics import r2_score

loss = model.evaluate(x_test, y_test)

# print(loss)

# Make predictions
y_pred = model.predict(x_test)
r2 = r2_score(y_test, y_pred)


# Print the predicted values and actual values
for i in range(len(y_test)):
    print("Actual: {:.2f}, Predicted: {:.2f}".format(y_test[i], y_pred[i][0]))

print("R2: {:.2f}".format(r2))

# Print the loss value
import numpy as np

# print("MAPE: {:.2f}%".format(loss))

# Plot the loss history
plt.plot(history.history['loss'])
plt.title('Model Loss')
plt.ylabel('MAPE')
plt.xlabel('Epoch')
plt.show()

# model.save('model0001_2hl18n_20.h5')
# 1 hidden layer 6 node ulangan 100