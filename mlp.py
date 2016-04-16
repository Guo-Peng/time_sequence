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
        for layer in self.layers:
            self.params += layer.params

    def output(self, x):
        for layer in self.layers:
            x = layer.output(x)
        return x

    def square_error(self, x, y):
        return T.sum((self.output(x) - y) ** 2)


def gradient_updates_momentum(cost, params, learning_rate, momentum):
    updates = []
    for param in params:
        param_update = theano.shared(param.get_value() * 0.0, broadcastable=param.broadcastable)
        updates.append((param, param - learning_rate * param_update))
        updates.append((param_update, momentum * param_update + (1 - momentum) * T.grad(cost, param)))
    return updates


np.random.seed(0)
N = 1000
y = np.random.random_integers(0, 1, N)
means = np.array([[-1, 1], [-1, 1]])
covariances = np.random.random_sample((2, 2)) + 1

# 以数组为下标表示以每个元素为下标取值
X = np.vstack([np.random.randn(N) * covariances[0, y] + means[0, y],
               np.random.randn(N) * covariances[1, y] + means[1, y]]).astype(theano.config.floatX)
y = y.astype(theano.config.floatX)
# Plot the data
# plt.figure(figsize=(8, 8))
# plt.scatter(X[0, :], X[1, :], c=y, lw=.3, s=3, cmap=plt.cm.cool)
# plt.axis([-6, 6, -6, 6])
# plt.show()

layer_size = [X.shape[0], X.shape[0] * 2, 1]
W_init = []
B_init = []
Activations = []
for n_input, n_output in zip(layer_size[:-1], layer_size[1:]):
    W_init.append(np.random.randn(n_output, n_input))
    B_init.append(np.ones(n_output))
    Activations.append(T.nnet.sigmoid)
mlp = MLP(W_init, B_init, Activations)
mlp_input = T.matrix('mlp_input')
mlp_output = T.vector('mlp_output')
cost = mlp.square_error(mlp_input, mlp_output)
learning_rate = 0.01
momentum = 0.9
train = theano.function([mlp_input, mlp_output], cost,
                        updates=gradient_updates_momentum(cost, mlp.params, learning_rate, momentum))
mlp_output = theano.function([mlp_input], mlp.output(mlp_input))

iteration = 0
max_iteration = 20
while iteration < max_iteration:
    current_cost = train(X, y)
    current_output = mlp_output(X)
    accuracy = np.mean((current_output > 0.5) == y)
    print 'iteration :  %d ,accuracy : %f ,cost: %f' % (iteration, accuracy, current_cost)
    iteration += 1
