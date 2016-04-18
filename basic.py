# -*- coding: UTF-8 -*-
import theano
import theano.tensor as T
import numpy as np

# # 变量定义
# foo = T.scalar('foo')
# bar = foo ** 2
# f = theano.function([foo], bar)
#
#
# # 共享变量
# shared_var = theano.shared(np.array([[1, 2], [2, 3]], dtype=float))
# shared_square = shared_var ** 2
# function_1 = theano.function([], shared_square)
# print function_1()
#
# subtract = T.matrix('subtract')
# function_2 = theano.function([subtract], shared_var, updates={shared_var: shared_var - subtract})
# print shared_var.get_value()
# print function_2(np.array([[1, 1], [1, 1]]))
# print shared_var.get_value()
# print function_1()
#
#
# # 梯度计算
# fun_grad = T.grad(bar, foo)
# print fun_grad.eval({foo: 20})
#
#
# A = T.matrix('A')
# x = T.vector('x')
# b = T.vector('b')
# y = T.dot(A, x) + b
# # 也可以为hessian
# y_j = theano.gradient.jacobian(y, x)
# y_j_f = theano.function([A, x, b], y_j)
# print y_j_f(np.array([[9, 8, 7], [4, 5, 6]], dtype=theano.config.floatX),  # A
#             np.array([1, 2, 3], dtype=theano.config.floatX),  # x
#             np.array([4, 5], dtype=theano.config.floatX))
#
#
# # test value
# theano.config.compute_test_value = 'warn'
# A = T.matrix('A')
# B = T.matrix('B')
# A.tag.test_value = np.random.random((3, 4)).astype(theano.config.floatX)
# B.tag.test_value = np.random.random((5, 6)).astype(theano.config.floatX)
# C = T.dot(A, B)
#
#
# # debug mode 不允许nan inf值的出现
# num = T.scalar('num')
# den = T.scalar('den')
# f = theano.function([num, den], num / den, mode='DebugMode')
# print f(5, 2)
# print f(0, 0)
#
#
# # Using the CPU vs GPU
# print theano.config.device
# print theano.config.floatX
#
# theano.config.floatX = 'float32'
# print theano.config.floatX
# # 默认使用numpy的类型
# var = theano.shared(np.array([1.0, 2, 3]))
# print var.type # float64
# var = theano.shared(np.array([1.0, 2, 3],dtype=theano.config.floatX))
# print var.type # float32


# 使用scan进行迭代
# 每次使用x的第一维的一个元素进行运算
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