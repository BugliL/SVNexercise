import logging
import random

import matplotlib.pyplot as plt
import numpy as np
import sys


FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(format=FORMAT, level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logger.propagate = True


class SGD_SVM(object):
    n = 0

    def __init__(self):
        self.w = None
        np.set_printoptions(precision=3)

    def calc_w(self, x, y, g=0.1, l=0.00001):
        return self._calc_w(self.w, x, y, g, l)

    def _calc_w(self, w, x_, y_, g=0.1, l=0.00001):
        v = y_ * np.dot(w, x_)
        return w - (g * l) * w + (0 if v > 1 else y_ * x_)

    def Qsvm(self, x_, y_, l=0.00001):
        return self._Qsvm(self.w, x_, y_, l)

    def _Qsvm(self, w, x_, y_, l=0.0001):
        v = y_ * np.dot(w, x_)
        loss = l * np.linalg.norm(w)**2 + max(0, 1 - v)
        grad = 0 if v > 1 else -y_ * x_
        return loss, grad

    def fw(self, x):
        return self._fw(self.w, x)[0]

    def _fw(self, w, x):
        value = np.dot(x, w)
        return np.sign(value), value

    def score(self, matrix, categories):
        n = 0; m = 0; v = []; s = []
        for x_, y_ in zip(matrix, categories):
            l, v_ = self._fw(self.w, x_)
            v.append(v_), s.append(y_)
            if l == y_:
                m += 1
            else:
                pass
                # logger.info("{} invece era {}".format(l, y_))
            n += 1

        return m / float(n), s, v

    def fit(self, matrix, categories):
        i = random.choice(range(matrix.shape[0]))
        self.w = self.calculate_w(matrix, categories, matrix[i].copy())
        return self.w

    def calculate_empirical_risk(self, x, y, w, g, l):
        loss = 0.0; n = len(y)
        for x_, y_ in zip(x, y):
            loss += l * np.linalg.norm(w) ** 2 + max(0, 1 - y_ * np.dot(w, x_))
        loss /= n
        return loss

    def calculate_w(self, x, y, w):

        l = 0.001
        g0 = 0.5; g = 0.0
        n = len(y); t = 1.0; i = 0

        old_p = int(100 * (t / n)); p = 0.9
        delta = np.inf; loss0 = np.inf; loss = 0

        # logger.info("{:>20}{:>20}{:>20}{:>20}".format("LOSS", "DELTA", "ACTUAL LOSS", "DELTA 2"))
        data = []
        while t < n:  # loss0 == np.inf or delta > 0.000001 or t < n:
            i = random.choice(range(n))
            x_, y_ = x[i], y[i]

            if (old_p != p and p % 10 == 0):
                logger.info("Completamento {}%".format(p))
                old_p = p
                #actual_q = self._Qsvm(w, x_, y_, l)[0]
                #loss = self.calculate_empirical_risk(x, y, w, g, l)
                #delta2 = np.abs(actual_q - loss)
                #delta = np.abs(loss0 - loss)
                #loss0 = loss
                #data.append( (loss, delta, actual_q, delta2) )
            p = int(100 * (t / n))


            g = g0 / (1 + l * g0 * t)
            # g = 1 / t
            grad = 0 if (y_ * np.dot(w, x_) > 1) else y_ * x_
            w = w - (g * l) * w + g * grad
            t += 1

            # v = y_ * np.dot(w, x_)
            # actual_q = self._Qsvm(w, x_, y_, l)[0]
            # loss = self.calculate_empirical_risk(x, y, w, g, l)
            # delta = np.abs(loss0 - loss)
            # delta2 = np.abs(actual_q - loss)
            # loss0 = loss
            # logger.info("{:>20}{:>20}{:>20}{:>20}".format(loss, delta, actual_q, delta2))

        #loss = self.calculate_empirical_risk(x, y, w, g, l)
        #delta2 = np.abs(actual_q - loss)
        #delta = np.abs(loss0 - loss)
        #logger.info("{:>20}{:>20}{:>20}{:>20}".format("LOSS", "DELTA", "ACTUAL LOSS", "DELTA 2"))
        #for d in data:
        #    logger.info("{:>20}{:>20}{:>20}{:>20}".format(d[0], d[1], d[2], d[3]))
        return w


if __name__ == '__main__':
    pass
