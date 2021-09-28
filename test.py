import matplotlib.pyplot as plt
import numpy as np
x = np.array([142.08, 177.30, 204.68, 242.88, 316.24, 332.69, 341.99, 389.29, 453.40])
y = np.array([3.93,   5.96,   7.85,   9.82,   12.50,  15.79,  15.55,  16.39,  18.45])


# %%
# 通过代数方法求解
# numpy的协方差默认是样本方差，无偏的，自由度n-1，因此要加bias = True
beta_0 = np.cov(x, y, bias = True)[0,1] / np.var(x)  
beta_1 = np.sum(y) / 9 - np.sum(x) / 9 * beta_0
beta_0, beta_1

# 通过公式计算，与上面相同
# a = np.sum(np.multiply(x, y)) - np.sum(x) * np.sum(y) / 9
# b = np.sum(np.multiply(x, x)) - np.sum(x) * np.sum(x) / 9

# %% [markdown]
# 同时也可以通过矩阵法求解，将多个方程看做一个整体进行求解。
# 
# $$
# {\displaystyle 
#     {\begin{pmatrix}
#     1& x_{11}& \cdots & x_{1j}\cdots & x_{1q}\\
#     1& x_{21}& \cdots & x_{2j}\cdots & x_{2q}\\
#     \vdots \\
#     1& x_{i1}& \cdots & x_{ij}\cdots & x_{iq}\\
#     \vdots \\
#     1& x_{n1}& \cdots & x_{nj}\cdots & x_{nq}
#     \end{pmatrix}} 
#     \cdot 
#     {\begin{pmatrix}\beta_{0}\\\beta_{1}\\\beta_{2}\\\vdots \\\beta_{j}\\\vdots \\\beta_{q}\end{pmatrix}}=
#     {\begin{pmatrix}y_{1}\\y_{2}\\\vdots \\y_{i}\\\vdots \\y_{n}\end{pmatrix}}}
# $$
# 
# 矩阵表达式为：
# $$
#     Q=min{||Xw-y||}^2
# $$
# 
# 求 $w$ 的最小二乘估计，即求 $\frac{\partial Q}{\partial w}$ 的零点。其中 $y$ 是 $m\times 1$ 列向量，$X$ 是 $m\times n$ 矩阵，$w$是$n\times 1$列向量，$Q$是标量。
# 
# 将向量模平方改写成向量与自身的内积：
# $$Q=(Xw-y)^T(Xw-y)$$
# 
# 求微分：
# $$
# \begin{aligned}
#     dQ&=(Xdw)^T(Xw-y)+(Xw-y)^T(Xdw)\\
#     &=2(Xw-y)^T(Xdw)
# \end{aligned}
# $$
# 这里是因为两个向量的内积满足$u^Tv=v^Tu$。
# 
# 导数与微分的关系式
# $$dQ={\frac{\partial Q}{\partial w}}^Tdw$$
# 得到
# $${\frac{\partial Q}{\partial w}}=2(Xw-y)(X)^T=0$$
# 求解可得
# $$
# \begin{aligned}
#     X^TXw&=X^Ty\\
#     w&=(X^TX)^{-1}X^Ty
# \end{aligned}
# $$
# 

# %%
# 通过矩阵法实现最小二乘法
def least_sqaure(X, Y):
    return (X.T * X).I * X.T * Y

X = np.mat(x)    
X = np.append(np.ones((1, 9)), X, axis = 0).T
Y = np.mat(y).T
beta = np.array(least_sqaure(X, Y)).flatten()[::-1]
print('beta:', beta)

plt.plot(x, y, 'bo', label='noise')
plt.plot(x, np.poly1d(beta)(x), label='fitted curve')
plt.show()


# %%
# 二次拟合

X = np.mat(x)
X = np.append(np.ones((1, 9)), X, axis = 0)
X = np.append(X, np.mat(x*x), axis = 0).T
Y = np.mat(y).T
beta = np.array(least_sqaure(X, Y)).flatten()[::-1]
print('beta:', beta)

plt.plot(x, y, 'bo', label='noise')
plt.plot(x, np.poly1d(beta)(x), label='fitted curve')
plt.show()

