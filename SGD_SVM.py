import numpy as np
import sys


class SGD_SVM():
    n = 0

    def __init__(self):
        pass # self.n = n
        np.set_printoptions(precision=3)

    def fit(self, matrix, categories, n):
        self.w = self.grad_descent(matrix, categories, np.array((0,) * n), step=0.001)

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

    def grad_descent(self, x, y, w, step, e=0.001):
        """
        Calcola W attraverso il metodo descend gratient
        :param x: np vector vettore dei punti
        :param y: np vector vettore delle categorie
        :param w: np vector vettore w dell'iperpiano
        :param step: float step di progressione
        :param e: float scarto dal risultato
        :return: np array, w modificato
        """

        grad = np.inf; n = len(y)
        delta = np.inf; loss0 = np.inf

        # help(type(x))
        print "{} : n = {}".format(type(n), n)
        # print "{} : x = {}".format(type(x), x[0].toarray())
        print "{} : y = {}".format(type(y), y)

        # ws = np.zeros((2, 0))
        ws = np.zeros((len(w), 0))
        # ws = np.hstack((ws, w.reshape(2, 1)))
        ws = np.hstack((ws, w.reshape(len(w), 1)))
        t = 1.0

        # | delta | >  e
        while np.abs(delta) > e:
            loss, grad = self.hinge_loss(w, x, y)
            print ("loss={}\ngrad={}".format(loss, grad))
            delta = loss0 - loss
            loss0 = loss
            grad_dir = grad / np.linalg.norm(grad)
            print ("grad_dir = {}".format(grad_dir))
            w = w - step * grad_dir / t
            print ("w = {}".format(w))
            ws = np.hstack((ws, w.reshape((len(w), 1))))
            print ("ws = {}".format(ws))
            t += 1

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
        print ("loss = {} ; grad = {}".format(loss, grad))
        for (x_, y_) in zip(x, y):
            print ("\n{} : xi = {}".format(type(x_), x_))
            print ("{} : yi = {}".format(type(y_), y_))

            # v = yi<w,xi>
            v = y_ * np.dot(w, x_)
            print ("<w,x> = {}".format(np.dot(w, x_)))

            # loss = loss + max(0, 1 - yi<w,xi>)
            print " v : = {}".format(v)
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
        print "{} : x1= {}".format(type(x1),x1)
        print "{} : x2= {}".format(type(x2),x2)
        print "\n{} : x= {}".format(type(x),x)

        # sample labels
        y = np.array((1, 1, -1, -1, -1))
        print "{} : y= {}".format(type(y),y)

        w = self.grad_descent(x, y, np.array((0, 0)), 0.1)
        print "\n{} : w= {}".format(type(w),w)

        loss, grad = self.hinge_loss(w, x, y)
        print "{} : loss= {}".format(type(loss),loss)
        print "{} : grad= {}".format(type(grad), grad)

        print "-"*80
        print "{}".format(w)
        print "-"*80


if __name__ == '__main__':
    svm = SGD_SVM(2)
    svm.test1()

    x1 = np.array((0, 1, 3, 4, 1))
    x2 = np.array((1, 2, 0, 1, 1))
    x = np.vstack((x1, x2)).T
    print x
    x = np.vstack((x1, x2))
    print x