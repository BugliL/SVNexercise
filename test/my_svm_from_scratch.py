import matplotlib
matplotlib.use('ps')


import matplotlib.pyplot as plt
from matplotlib import style
import numpy as np

import pylab


style.use('ggplot')

class SVM(object):

    def _setup_visualization(self):
        self.colors = {1: 'r', -1: 'b'}
        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(1, 1, 1)

    def __init__(self):
        self.b = None
        self.W = None
        self._setup_visualization()

    def fit(self, data):
        pass

    def predict(self, dot):
        pass

    def classify(self, X):
        # sign( <X,w> + b )
        return np.sign(np.dot(np.array(X), self.W) + self.b)

    def visualize(self):
        pass

