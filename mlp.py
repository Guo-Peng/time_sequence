# -*- coding: UTF-8 -*-
import theano
import numpy as np
import theano.tensor as T


class Layer:
    def __init__(self, w, b, activation):
        # w.shape = n_output,input  b.shape = n_output
        # activation 为激活函数
        n_output = w.shape[0]
        self.w = theano.shared(w.astype(theano.config.floatX), name='W', borrow=True)
        self.b = theano.shared(b.reshape(n_output, 1).astype(theano.config.floatX), name='b', borrow=True,
                               broadcastable=(False, True))
        self.activation = activation
        self.params = [self.w, self.b]

    def output(self, x):
        linear_output = T.dot(self.w, x) + self.b
        return linear_output if self.activation is None else self.activation(linear_output)


class MLP:
    def __init__(self, w_init, b_init, activations):
        self.layers = []
        self.params = []
        for w, b, activation in zip(w_init, b_init, activations):
            self.layers.append(Layer(w, b, activation))
            # deference
            self.params.append([w, b])

    def output(self, x):
        for layer in self.layers:
            x = layer.output(x)
        return x

    def square_error(self, x, y):
        return T.sum((self.output(x) - y) ** 2)
