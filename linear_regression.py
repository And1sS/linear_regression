import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import time

SGD_costs = []
FGD_costs = []
MBGD_costs = []

"""
input:
    [
     [feature_1_1, ..., feature_1_n]
     ...
     [feature_m_1, ..., feature_m_n]
    ]
desired:
    [desired_1, ..., desired_n]
"""


def linear_prediction(inputs, coef):
    return np.dot(coef, inputs)


def cost(inputs, desired, coef):
    return np.sum((linear_prediction(inputs, coef) - desired) ** 2) / inputs.shape[1]


def grad(inputs, desired, coef):
    return 2 / inputs.shape[1] * np.dot(inputs, linear_prediction(inputs, coef) - desired)


def sgrad(input, desired, coef):
    return 2 * np.dot(input, linear_prediction(input, coef) - desired)


def sgd(inputs, desired, coef, epoches, alpha):
    for i in range(epoches):
        index = i % len(desired)
        dcoef = sgrad(inputs[:, index], desired[index], coef)
        coef -= np.multiply(dcoef, np.asarray(alpha))
        SGD_costs.append(cost(inputs, desired, coef))
    return coef


def mbgd(inputs, desired, coef, epoches, minibatch_size, alpha):
    for i in range(epoches):
        indexes = [j % len(desired) for j in range(i * minibatch_size, (i + 1) * minibatch_size)]
        inputs_batch = np.array([inputs[:, i] for i in indexes]).transpose()
        desired_batch = np.array([desired[i] for i in indexes])
        dcoef = grad(inputs_batch, desired_batch, coef)
        coef -= np.multiply(dcoef, np.asarray(alpha))
        MBGD_costs.append(cost(inputs, desired, coef))
    return coef


def fgd(inputs, desired, coef, epoches, alpha):
    for i in range(epoches):
        dcoef = grad(inputs, desired, coef)
        coef -= np.multiply(dcoef, np.asarray(alpha))
        FGD_costs.append(cost(inputs, desired, coef))
    return coef

# getting and normalising dataset
data = pd.read_csv("ex1data1.csv")
mean = data['population'].mean()
std = data['population'].std()
data['population'] = (data['population'] - mean) / std

inputs = np.vstack([data['population'].values, [1 for i in range(len(data['population'].values))]])
desired = data['profit'].values

# hyperparameters
fgd_epochs = 10000
mbgd_epochs = 10000
sgd_epochs = 100000

fgd_lr = 0.001
mbgd_lr = 0.001
sgd_lr = 0.0001

# getting coefficients
sgd_st = time.time()
sgd_coef = sgd(inputs, desired, np.zeros(inputs.shape[0]), sgd_epochs, sgd_lr)
sgd_time = time.time() - sgd_st
fgd_st = time.time()
fgd_coef = fgd(inputs, desired, np.zeros(inputs.shape[0]), fgd_epochs, fgd_lr)
fgd_time = time.time() - fgd_st
mbgd_st = time.time()
mbgd_coef = mbgd(inputs, desired, np.zeros(inputs.shape[0]), mbgd_epochs, 32, mbgd_lr)
mbgd_time = time.time() - mbgd_st

# printing results of learning
print("SGD result: {0}, mserror = {1}, execution time: {2}".format(sgd_coef, cost(inputs, desired, sgd_coef), sgd_time))
print("FGD result: {0}, mserror = {1}, execution time: {2}".format(fgd_coef, cost(inputs, desired, fgd_coef), fgd_time))
print("MBGD result: {0}, mserror = {1}, execution time: {2}".format(mbgd_coef, cost(inputs, desired, mbgd_coef), mbgd_time))

# plotting errors
plt.plot(range(fgd_epochs), FGD_costs, color="black", label='FGD')
plt.xlabel("epochs (learning rate: " + str(fgd_lr) + \
           ")\n (red = SGD, black = FGD, green = MBGD)" )
plt.ylabel("mserror")
plt.legend()
plt.show()
plt.plot(range(sgd_epochs), SGD_costs, color="red", label='SGD')
plt.xlabel("epochs (learning rate: " + str(sgd_lr) + \
           ")\n (red = SGD, black = FGD, green = MBGD)" )
plt.ylabel("mserror")
plt.legend()
plt.show()
plt.plot(range(mbgd_epochs), MBGD_costs, color="green", label='MBGD')
plt.xlabel("epochs (learning rate: " + str(mbgd_lr) + \
           ")\n (red = SGD, black = FGD, green = MBGD)" )
plt.ylabel("mserror")
plt.legend()
plt.show()

# graphing predicted & actual
plt.scatter(data['population'].tolist(), data['profit'])

range_x = range(-1, 5, 1)
test = np.array([range_x, [1 for x in range_x]])

actual = [linear_prediction(test[:, x], sgd_coef) for x in range(len(range_x))]
plt.plot(range_x, actual, color="red", label='SGD')

actual = [linear_prediction(test[:, x], mbgd_coef) for x in range(len(range_x))]
plt.plot(range_x, actual, color="green", label='MBGD')

actual = [linear_prediction(test[:, x], fgd_coef) for x in range(len(range_x))]
plt.plot(range_x, actual, color="black", label='FGD')

plt.ylabel("profit")
plt.xlabel("population(red = SGD, black = FGD, green = MBGD)")
plt.legend()
plt.show()

# для SGD при learning_rate = 0.001 наблюдаются большие флуктуации, при
# learning_rate = 0.0001 они уже не так сильно выражены. По графику ошибки от эпохи обучения видно,
# что на 100000й эпохе ошибка практически не уменьшается, а значит можно остановить обучение на 100000 эпохах, в
# угоду ускорению работы алгоритма. если же уменьшать эпохи еще до, например, 50000, то выйдет, что ошибка еще
# уменьшается, хоть и незначительно, а значит можно выиграть еще немного точности.
# SGD result: [4.6173002  5.83199527], mserror = 8.953993885971, execution time: 1.841611623764038
#
# для MBGD при learning_rate = 0.001 флуктуации уже не так сильно выражены. при 10000 эпохах ошибка
# перестает уменьшаться, дальнейшее обучение бесполезно. до 10000 еще есть смысл пытаться уменьшить
# ошибку,  10000 эпох при learning_rate = 0.001 - оптимальный вариант
# MBGD result: [4.61693524 5.8370017 ], mserror = 8.953947304292086, execution time: 0.532395601272583
#
# для FGD при learning_rate = 0.001 не наблюдается сильно выраженых флуктуаций.
# при 1000 эпохах ошибка практически перестает уменьшаться, но все еще можно выиграть
# немного точности. При 10000 эпохах точность уже совсем не увеличивается - дальнейшее обучение
# бесполезно. 10000 эпох при learning_rate = 0.001 - оптимальный вариант.
# FGD result: [4.61690124 5.83913504], mserror = 8.953942751950358, execution time: 0.1591358184814453
#
# в результате имеем, что FGD самый точный из всех алгоритмов, и работает быстрее остальных,
# итоговый результат работы FGD не уступает результату linear_model.LinearRegression.



