from sklearn import linear_model
import pandas as pd
import numpy as np

data = pd.read_csv("ex1data1.csv")
mean = data['population'].mean()
std = data['population'].std()
data['population'] = (data['population'] - mean) / std

reg = linear_model.LinearRegression(copy_X=True, fit_intercept=True,
                                    n_jobs=None, normalize=False)
reg.fit(data['population'].values.reshape(-1, 1), data['profit'].values.reshape(-1, 1))
print(reg.coef_)
print(reg.intercept_)

k = reg.coef_[0][0]
b = reg.intercept_[0]


def linear_prediction(x):
    return k * x + b


def cost(inputs, desired):
    return np.sum((linear_prediction(inputs) - desired) ** 2) \
        / (1 if type(inputs) != np.ndarray else len(inputs))


print(cost(data['population'].values, data['profit'].values))

#final result of model & mserror
#[[4.61690125]]
#[5.83913505]
#8.953942751950358