# -*- coding: UTF-8 -*-
import numpy as np

l = np.arange(16).reshape(4, 4)
print np.c_[l, l]
