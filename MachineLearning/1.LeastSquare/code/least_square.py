import numpy as np
import matplotlib.pyplot as plt

plt.figure(figsize=(8,8), dpi=80)
plt.subplots_adjust(top = 0.94, bottom = 0.06, right = 0.94, left = 0.06, hspace = 0.1, wspace = 0.1)

# %matplotlib inline
x = np.array([142.08, 177.30, 204.68, 242.88, 316.24, 332.69, 341.99, 389.29, 453.40])
y = np.array([3.93,   5.96,   7.85,   9.82,   12.50,  15.79,  15.55,  16.39,  18.45])

# 通过代数方法求解
# numpy的协方差默认是样本方差，无偏的，自由度n-1，因此要加bias = True
beta_0 = np.cov(x, y, bias = True)[0,1] / np.var(x)  
beta_1 = np.sum(y) / 9 - np.sum(x) / 9 * beta_0
print(beta_0, beta_1)

# 通过公式计算，与上面相同
# a = np.sum(np.multiply(x, y)) - np.sum(x) * np.sum(y) / 9
# b = np.sum(np.multiply(x, x)) - np.sum(x) * np.sum(x) / 9

# 通过矩阵法实现最小二乘法
def least_sqaure(X, Y):
    return (X.T * X).I * X.T * Y

# 生成多项式
def polynomial(x, n):
    X = np.mat(x)
    X = np.append(np.ones((1, 9)), X, axis = 0)
    for i in range(1, n):
        X = np.append(X, np.mat(x ** (i + 1)), axis = 0)
    return X.T

Y = np.mat(y).T

# 线性拟合
X = polynomial(x, 1)
beta = np.array(least_sqaure(X, Y)).flatten()[::-1]
print('beta:', beta)
plt.subplot(221)
plt.plot(x, y, 'bo', label='noise')
plt.plot(x, np.poly1d(beta)(x), label='fitted curve')

# 二次拟合
X = polynomial(x, 2)
beta = np.array(least_sqaure(X, Y)).flatten()[::-1]
print('beta:', beta)
plt.subplot(222)
plt.plot(x, y, 'bo', label='noise')
plt.plot(x, np.poly1d(beta)(x), label='fitted curve')

# 三次拟合
X = polynomial(x, 3)
beta = np.array(least_sqaure(X, Y)).flatten()[::-1]
print('beta:', beta)
plt.subplot(223)
plt.plot(x, y, 'bo', label='noise')
plt.plot(x, np.poly1d(beta)(x), label='fitted curve')

# 六次拟合
X = polynomial(x, 6)
beta = np.array(least_sqaure(X, Y)).flatten()[::-1]
print('beta:', beta)
plt.subplot(224)
plt.plot(x, y, 'bo', label='noise')
plt.plot(x, np.poly1d(beta)(x), label='fitted curve')

# plt.show()
plt.savefig('least_square.png')