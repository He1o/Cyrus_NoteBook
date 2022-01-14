import matplotlib.pyplot as plt

class Figure():
    def __init__(self):
        self.fig, self.axes = plt.subplots()
        self.fig.set_size_inches(8, 8)
        self.fig.subplots_adjust(top = 0.94, bottom = 0.06, right = 0.94, left = 0.06, hspace = 0, wspace = 0)
        plt.style.use('Solarize_Light2')
    def tmp(self):    
        return plt