import numpy as np
import matplotlib.pyplot as plt
import random
def drawminkowski(sampling =10001, p = [0.1,0.5,1,2,3,10,100]):
 
    X = np.linspace(-1, 1, sampling)
 
    plt.figure(figsize=(8,8), dpi=80)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.xlim(-1,1)
    plt.ylim(-1,1)
    plt.title("minkowski distance, distance = 1, p = " + str(p))
 
    plt.plot(np.linspace(-1, 1, 100), np.zeros(100), "k,", linewidth = 2)
    plt.plot(np.zeros(100), np.linspace(-1, 1, 100), "k,", linewidth = 2)
 
    for pi in p:
        Y = np.power(1- np.power(np.abs(X), pi), 1/float(pi))
        
        color = random.randint(1, 0xffffff)
        plt.plot(X,Y, label ="p=" + str(pi), color = "#" + format(color, "06x"), linewidth = 1)
        plt.plot(X,-Y, color = "#" + format(color, "06x"), linewidth = 1)
 
    plt.legend()
    plt.show()
drawminkowski()