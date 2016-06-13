import logging
import matplotlib.pyplot as plt
import numpy as np
import sys


FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(format=FORMAT, level=logging.INFO)
logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)
logger.propagate = False

class SGD_SVM():
    n = 0

    def __init__(self):
        np.set_printoptions(precision=3)

    def fit(self, matrix, categories, n, step=0.1):
        self.w = self.grad_descent(matrix, categories, np.array((0,) * n), step=step)

    def score(self, matrix, categories):
        n = 0;
        m = 0
        for x_,y_ in zip(matrix,categories):
            l = self.predict(x_)
            if l == y_:
                m += 1
            n += 1
        return m / float(n)

    def predict(self,x):
        # sign( <X,w> + b )
        return np.sign(np.dot(np.array(x), self.w))

    def grad_descent(self, x, y, w, step, e=0.1):
        """
        Calcola W attraverso il metodo descend gratient
        :param x: np vector vettore dei punti
        :param y: np vector vettore delle categorie
        :param w: np vector vettore w dell'iperpiano
        :param step: float step di progressione
        :param e: float scarto dal risultato
        :return: np array, w modificato
        """
        logging.disable(logging.CRITICAL)

        grad = np.inf; n = len(y)
        delta = np.inf; loss0 = np.inf

        # help(type(x))
        logger.info("{} : n = {}".format(type(n), n))
        # print "{} : x = {}".format(type(x), x[0].toarray())
        logger.info("{} : y = {}".format(type(y), y))

        # ws = np.zeros((2, 0))
        ws = np.zeros((len(w), 0))
        # ws = np.hstack((ws, w.reshape(2, 1)))
        ws = np.hstack((ws, w.reshape(len(w), 1)))
        t = 1.0

        # | delta | >  e
        while np.abs(delta) > e:
            loss, grad = self.hinge_loss(w, x, y)
            logger.info(("loss={}\ngrad={}".format(loss, grad)))
            delta = loss0 - loss
            loss0 = loss
            grad_dir = grad / np.linalg.norm(grad)
            logger.info(("grad_dir = {}".format(grad_dir)))
            w = w - step * grad_dir / t
            logger.info(("w = {}".format(w)))
            ws = np.hstack((ws, w.reshape((len(w), 1))))
            logger.info(("ws = {}".format(ws)))
            t += 1

        logging.disable(logging.NOTSET)
        return np.sum(ws, 1) / np.size(ws, 1)

    def hinge_loss(self, w, x, y):
        """
        Calcola la funzione costo e il gradiente di W
        :param w: np vector vettore della classificazione
        :param x: np vector vettore di punti
        :param y: np vector vettore delle categorie
        :return: (float, float)
        """
        loss, grad = 0, 0
        logger.info(("loss = {} ; grad = {}".format(loss, grad)))
        for (x_, y_) in zip(x, y):
            logger.info(("\n{} : xi = {}".format(type(x_), x_)))
            logger.info(("{} : yi = {}".format(type(y_), y_)))

            # v = yi<w,xi>
            v = y_ * np.dot(w, x_)
            logger.info(("<w,x> = {}".format(np.dot(w, x_))))

            # loss = loss + max(0, 1 - yi<w,xi>)
            logger.info(" v : = {}".format(v))
            loss += max(0, 1 - v)
            # raw_input("$")

            # if yi<w,xi> < 1:
            #   grad = grad - yixi
            grad += 0 if v > 1 else -y_ * x_
        return loss, grad

    def test1(self):
        # sample data points
        x1 = np.array((0, 1, 3, 4, 1))
        x2 = np.array((1, 2, 0, 1, 1))
        x = np.vstack((x1, x2)).T
        logger.info("{} : x1= {}".format(type(x1),x1))
        logger.info("{} : x2= {}".format(type(x2),x2))
        logger.info("\n{} : x= {}".format(type(x),x))

        # sample labels
        y = np.array((1, 1, -1, -1, -1))
        logger.info("{} : y= {}".format(type(y),y))

        w = self.grad_descent(x, y, np.array((0, 0)), 0.1)
        logger.info("\n{} : w= {}".format(type(w),w))

        loss, grad = self.hinge_loss(w, x, y)
        logger.info("{} : loss= {}".format(type(loss),loss))
        logger.info("{} : grad= {}".format(type(grad), grad))

        logger.info("-"*80)
        logger.info("{}".format(w))
        logger.info("-"*80)
        self.plot_test(x,y,w)

    def plot_test(self, x, y, w):
        plt.figure()
        x1, x2 = x[:, 0], x[:, 1]
        x1_min, x1_max = np.min(x1) * .7, np.max(x1) * 1.3
        x2_min, x2_max = np.min(x2) * .7, np.max(x2) * 1.3
        gridpoints = 2000
        x1s = np.linspace(x1_min, x1_max, gridpoints)
        x2s = np.linspace(x2_min, x2_max, gridpoints)
        gridx1, gridx2 = np.meshgrid(x1s, x2s)
        grid_pts = np.c_[gridx1.ravel(), gridx2.ravel()]
        predictions = np.array([np.sign(np.dot(w, x_)+0.5) for x_ in grid_pts]).reshape((gridpoints, gridpoints))
        plt.contourf(gridx1, gridx2, predictions, cmap=plt.cm.Paired)
        plt.scatter(x[:, 0], x[:, 1], c=y, cmap=plt.cm.Paired)
        plt.title('total hinge loss: %g' % self.hinge_loss(w, x, y)[0])
        plt.show()
        plt.savefig('test.png')


if __name__ == '__main__':
    svm = SGD_SVM()
    svm.test1()
