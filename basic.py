# -*- coding: UTF-8 -*-
import theano
import theano.tensor as T
import numpy as np

# 变量定义
foo = T.scalar('foo')
bar = foo ** 2
f = theano.function([foo], bar)


# 共享变量
shared_var = theano.shared(np.array([[1, 2], [2, 3]], dtype=float))
shared_square = shared_var ** 2
function_1 = theano.function([], shared_square)
print function_1()

subtract = T.matrix('subtract')
function_2 = theano.function([subtract], shared_var, updates={shared_var: shared_var - subtract})
print shared_var.get_value()
print function_2(np.array([[1, 1], [1, 1]]))
print shared_var.get_value()
print function_1()


# 梯度计算
fun_grad = T.grad(bar, foo)
print fun_grad.eval({foo: 20})


A = T.matrix('A')
x = T.vector('x')
b = T.vector('b')
y = T.dot(A, x) + b
# 也可以为hessian
y_j = theano.gradient.jacobian(y, x)
y_j_f = theano.function([A, x, b], y_j)
print y_j_f(np.array([[9, 8, 7], [4, 5, 6]], dtype=theano.config.floatX),  # A
            np.array([1, 2, 3], dtype=theano.config.floatX),  # x
            np.array([4, 5], dtype=theano.config.floatX))


# test value
theano.config.compute_test_value = 'warn'
A = T.matrix('A')
B = T.matrix('B')
A.tag.test_value = np.random.random((3, 4)).astype(theano.config.floatX)
B.tag.test_value = np.random.random((5, 6)).astype(theano.config.floatX)
C = T.dot(A, B)


# debug mode 不允许nan inf值的出现
num = T.scalar('num')
den = T.scalar('den')
f = theano.function([num, den], num / den, mode='DebugMode')
print f(5, 2)
print f(0, 0)


# Using the CPU vs GPU
print theano.config.device
print theano.config.floatX

theano.config.floatX = 'float32'
print theano.config.floatX
# 默认使用numpy的类型
var = theano.shared(np.array([1.0, 2, 3]))
print var.type # float64
var = theano.shared(np.array([1.0, 2, 3],dtype=theano.config.floatX))
print var.type # float32


# 使用scan进行迭代
# 每次使用x的第一维的一个元素进行运算
#  tanh(x(t).dot(w) + b)
x = T.matrix('x')
w = T.matrix('w')
b = T.vector('b')
results, updates = theano.scan(lambda t: T.tanh((T.dot(t, w) + b)), sequences=x)
f = theano.function([x, w, b], results)
x = np.eye(2, dtype=theano.config.floatX)
w = np.ones((2, 2), dtype=theano.config.floatX)
b = np.ones((2), dtype=theano.config.floatX)
b[1] = 2

print(f(x, w, b))

print(np.tanh(x.dot(w) + b))

# 使用过去状态进行迭代
# x(t) = tanh(x(t - 1).dot(W) + y(t).dot(U) + p(T - t).dot(V))
# function input 的顺序跟scan中变量的顺序一致
X = T.vector("X")
W = T.matrix("W")
U = T.matrix("U")
Y = T.matrix("Y")
V = T.matrix("V")
P = T.matrix("P")
results, updates = theano.scan(lambda y, p, x_temp: T.tanh(T.dot(x_temp, W) + T.dot(y, U) + T.dot(p, V)),
                               sequences=[Y, P[::-1]], outputs_info=[X])
f = theano.function([X, W, Y, U, P, V], outputs=results)
x = np.zeros((2), dtype=theano.config.floatX)
x[1] = 1
print x
w = np.ones((2, 2), dtype=theano.config.floatX)
y = np.ones((5, 2), dtype=theano.config.floatX)
y[0, :] = -3
u = np.ones((2, 2), dtype=theano.config.floatX)
p = np.ones((5, 2), dtype=theano.config.floatX)
p[0, :] = 3
v = np.ones((2, 2), dtype=theano.config.floatX)
print f(x, w, y, u, p, v)

x_res = np.zeros((5, 2), dtype=theano.config.floatX)
x_res[0] = np.tanh(x.dot(w) + y[0].dot(u) + p[4].dot(v))
for i in range(1, 5):
    x_res[i] = np.tanh(x_res[i - 1].dot(w) + y[i].dot(u) + p[4 - i].dot(v))
print x_res
