# -*- coding: UTF-8 -*-
import numpy as np
import pandas as pd
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
from keras.models import Sequential

l = range(1, 10)
data = pd.DataFrame({'a': l, 'b': l})
data_x = np.array([data.values[:-1], data.values[:-1]])
data_y = np.array([data.values[-1], data.values[-1]])
print data_x.shape
print data_y.shape
print 'Build model...'
model = Sequential()
model.add(LSTM(100, input_shape=(8, 2)))
model.add(Dense(1))
model.add(Activation("linear"))
model.compile(loss="mean_squared_error", optimizer="rmsprop")
model.fit(data_x, data_y, nb_epoch=10)
predicted = model.predict(data_x)
