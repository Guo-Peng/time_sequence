# -*- coding: UTF-8 -*-
from keras.layers import Lambda, Input
from keras.models import Model

import numpy as np

input = Input(shape=(5,), dtype='int32')
double = Lambda(lambda x: 2 * x)(input)

model = Model(input=[input], output=[double])
model.compile(optimizer='sgd', loss='mse')

data = np.arange(5)
print(model.predict(data))
