# -*- coding: UTF-8 -*-
import numpy as np
import pandas as pd
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
from keras.models import Sequential
from sklearn.cross_validation import KFold


def data_load(path):
    names = xrange(1, 31)
    return pd.read_table(path, names=names, index_col=0)


def scale(data):
    return data.apply(lambda x: (x - x.mean()) / (x.max() - x.min()) if x.sum() != 0 else x, axis=1)


def train_test(data, fold=5):
    index = list(KFold(data.shape[0], n_folds=fold, shuffle=True, random_state=1))
    train = data.ix[index[0][0],]
    test = data.ix[index[0][1],]
    return train, test


def data_transfer(data):
    data_x = []
    data_y = []
    for x in xrange(data.shape[0]):
        line = data.ix[x, :].reshape(-1, 1)
        data_x.append(line[:-1, :])
        data_y.append(line[-1, :])
    return np.array(data_x), np.array(data_y)


def model_build():
    print 'Build model...'
    model = Sequential()
    model.add(LSTM(128, input_dim=1, input_length=29, return_sequences=False))
    model.add(Dense(1))
    model.add(Activation("tanh"))
    model.compile(loss="mean_squared_error", optimizer="rmsprop")
    return model


def time_series(data_path, store_path):
    print 'data loading ...'
    data = data_load(data_path)
    train, test = train_test(data)
    train_norm = scale(train)
    test_norm = scale(test)
    train_x, train_y = data_transfer(train_norm)
    test_x, test_y = data_transfer(test_norm)
    print 'shape of train_x : %s, shape of train_y : %s' % (str(train_x.shape), str(train_y.shape))
    print 'shape of train_x : %s, shape of train_y : %s' % (str(test_x.shape), str(test_y.shape))

    model = model_build()
    print 'fitting ...'
    model.fit(train_x, train_y, nb_epoch=41)
    print 'predicting ...'
    predict = model.predict(test_x)

    print 'saving ...'
    test_norm['predict'] = predict.reshape(-1)
    test_norm['mean'] = test.mean(axis=1)
    test_norm['dis'] = test.max(axis=1) - test.min(axis=1)
    test['predict'] = test_norm.apply(lambda x: x['predict'] * x['dis'] + x['mean'], axis=1)
    test.to_csv(store_path, header=False, sep='\t')


if __name__ == '__main__':
    d_path = '/Users/Peterkwok/Downloads/归档/view count clean increase/vci'
    s_path = '/Users/Peterkwok/Downloads/result'
    time_series(d_path, s_path)
