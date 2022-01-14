import numpy as np
import matplotlib.pyplot as plt
import random
def drawminkowski(sampling =10001, p = [0.1,0.5,1,2,3,10,100]):
 
    X = np.linspace(-1, 1, sampling)
    # plt.style.use('Solarize_Light2')
    # plt.figure().add_subplot(111).set_aspect('equal')
    plt.figure(figsize=(6,6), dpi=300)
    # plt.axis('off')
    plt.gca().xaxis.set_major_locator(plt.NullLocator())
    plt.gca().yaxis.set_major_locator(plt.NullLocator())
    plt.subplots_adjust(top = 0.93, bottom = 0.07, right = 0.93, left = 0.07, hspace = 0, wspace = 0)
    # plt.margins(10,10)
    
    
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
    # plt.show()
    plt.savefig('distance.svg')
drawminkowski()