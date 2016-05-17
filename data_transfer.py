# -*- coding: UTF-8 -*-
import numpy as np
import pandas as pd


def data_load(path):
    names = xrange(1, 31)
    return pd.read_table(path, names=names, index_col=0)


def scaler(data):
    scale = lambda x: (x - x.mean()) / (x.max() - x.min())
    return data.apply(scale, axis=1)


if __name__ == '__main__':
    d = pd.DataFrame({'a': range(4), 'b': range(1, 5), 'c': range(3, 7)})
    print d
    print d.mean(axis=1)
    print d.max(axis=1) - d.min(axis=1)
