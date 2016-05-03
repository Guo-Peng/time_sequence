# -*- coding: UTF-8 -*-
import numpy as np
import pandas as pd
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
from keras.models import Sequential

l = range(1, 11)
data = pd.DataFrame({'a': l, 'b': l})
data_x = np.array([data.values[:-1], data.values[:-1]])
data_y = np.array([data.values[-1], data.values[-1]])
print data_x
print 'Build model...'
model = Sequential()
model.add(LSTM(100, input_dim=2, input_length=9, return_sequences=False))
model.add(Dense(2))
model.add(Activation("linear"))
model.compile(loss="mean_squared_error", optimizer="rmsprop")
model.fit(data_x, data_y, nb_epoch=100)

l = range(2, 12)
data = pd.DataFrame({'a': l, 'b': l})
data_x = np.array([data.values[:-1], data.values[:-1]])
data_y = np.array([data.values[-1], data.values[-1]])
print data_x
predicted = model.predict(data_x)
print predicted