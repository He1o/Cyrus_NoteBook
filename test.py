import numpy as np
import matplotlib.pyplot as plt

# y = lambda x: 1 /(np.exp(((x - 5000)/2000)) + 1)
y = lambda x: (x - 2)**2 * (1/4)
# y = lambda x: np.log(50*x)/np.log(0.9)
# y = lambda x: 1 / np.exp(x - 0.05)
# y = lambda x: np.power(x, -0.5)
# y = lambda x: 1/  x - 1/0.02
print(y(0.005),y(0.03),y(0.01))
X = []
Y = []
n = 0
for x in np.linspace(-1, 3, 1000):
    X.append(x)
    Y.append(y(x))
    n += 1
    if n % 10 == 0:
        print(x, y(x))

plt.plot(X, Y)
plt.show()
        


[time_stamp, user_id, function = "total_used:" || function = "total_used_algorithm:" || function = "tile_used:", time, index]
