# -*- coding: UTF-8 -*-
import numpy as np
import pandas as pd
from sklearn.preprocessing import scale

binary = np.unpackbits(np.array([range(10)], dtype=np.uint8).T, axis=1)
print np.array([[1]]).T
