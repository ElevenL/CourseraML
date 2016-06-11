import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
import pandas as pd
import scipy.io #Used to load the OCTAVE *.mat files
import scipy.misc #Used to show matrix as an image
import matplotlib.cm as cm #Used to display images in a specific colormap
import random #To pick random images to display
from scipy.special import expit #Vectorized sigmoid function

class LogisticRegression:
    def __init__(self):
        pass

    def plotData(self, pos, neg):
        plt.plot(pos[:, 1], pos[:, 2], 'k+', label='y=1')
        plt.plot(neg[:, 1], neg[:, 2], 'yo', label='y=0')
        plt.xlabel('Microchip Test 1')
        plt.ylabel('Microchip Test 2')
        plt.legend()
        plt.grid(True)
        # plt.show()

    # This code I took from someone else (the OCTAVE equivalent was provided in the HW)
    def mapFeature(self, x1col, x2col, degrees = 6):
        """
        Function that takes in a column of n- x1's, a column of n- x2s, and builds
        a n- x 28-dim matrix of featuers as described in the homework assignment
        """
        out = np.ones((x1col.shape[0], 1))

        for i in range(1, degrees + 1):
            for j in range(0, i + 1):
                term1 = x1col ** (i - j)
                term2 = x2col ** (j)
                term = (term1 * term2).reshape(term1.shape[0], 1)
                out = np.hstack((out, term))
        return out

    # Hypothesis function and cost function for logistic regression
    def h(self, theta, X):  # Logistic hypothesis function
        return expit(np.dot(X, theta))

    # Cost function, default lambda (regularization) 0
    def computeCost(self, theta, myX, myy, mylambda=0.):
        """
        theta_start is an n- dimensional vector of initial theta guess
        X is matrix with n- columns and m- rows
        y is a matrix with m- rows and 1 column
        Note this includes regularization, if you set mylambda to nonzero
        For the first part of the homework, the default 0. is used for mylambda
        """
        # note to self: *.shape is (rows, columns)
        m = myX.shape[0]
        myh = self.h(theta, myX)
        term1 = np.dot(-np.array(myy).T, np.log(myh))
        term2 = np.dot((1 - np.array(myy)).T, np.log(1 - myh))
        regterm = (mylambda / 2) * np.sum(np.dot(theta[1:].T, theta[1:]))  # Skip theta0
        return float((1. / m) * (np.sum(term1 - term2) + regterm))

    # An alternative to OCTAVE's 'fmincg' we'll use some scipy.optimize function, "fmin_cg"
    # This is more efficient with large number of parameters.
    # In the previous homework, I didn't have to compute the cost gradient because
    # the scipy.optimize function did it for me with some kind of interpolation...
    # However, fmin_cg needs the gradient handed do it, so I'll implement that here
    def costGradient(self, mytheta, myX, myy, mylambda=0.):
        m = myX.shape[0]
        # Tranpose y here because it makes the units work out in dot products later
        # (with the way I've written them, anyway)
        beta = self.h(mytheta, myX) - myy.T  # shape: (5000,5000)

        # regularization skips the first element in theta
        regterm = mytheta[1:] * (mylambda / m)  # shape: (400,1)

        grad = (1. / m) * np.dot(myX.T, beta)  # shape: (401, 5000)
        # regularization skips the first element in theta
        grad[1:] = grad[1:] + regterm
        return grad  # shape: (401, 5000)

    def optimizeTheta(self, mytheta, myX, myy, mylambda=0.):
        result = optimize.fmin_cg(self.computeCost, fprime=self.costGradient, x0=mytheta, \
                                  args=(myX, myy, mylambda), maxiter=50, disp=False, \
                                  full_output=True)
        return result[0], result[1]    ## fit_theta, mincost

    def optimizeRegularizedTheta(self, mytheta, myX, myy, mylambda=0.):
        result = optimize.minimize(self.computeCost, mytheta, args=(myX, myy, mylambda), method='BFGS',
                                   options={"maxiter": 500, "disp": False})
        return np.array([result.x]), result.fun    ## fit_theta, mincost

    # Note: I spent a LONG time trying to optimize everything. Initially training 10 classes
    # took about 5 minutes. Now I've got it down to taking ~5 seconds total!
    def buildTheta(self, X, y, class_count, mylambda=0.):
        """
        Function that determines an optimized theta for each class
        and returns a Theta function where each row corresponds
        to the learned logistic regression params for one class
        """
        initial_theta = np.zeros((X.shape[1], 1)).reshape(-1)
        Theta = np.zeros((class_count, X.shape[1]))
        for i in xrange(class_count):
            iclass = i if i else 10  # class "10" corresponds to handwritten zero
            print "Optimizing for handwritten number %d..." % i
            logic_Y = np.array([1 if x == iclass else 0 for x in y])  # .reshape((X.shape[0],1))
            itheta, imincost = self.optimizeTheta(initial_theta, X, logic_Y, mylambda)
            Theta[i, :] = itheta
        print "Done!"
        return Theta

    def predictOneVsAll(self, myTheta, myrow):
        """
        Function that computes a hypothesis for an individual image (row in X)
        and returns the predicted integer corresponding to the handwritten image
        """
        classes = [10] + range(1, 10)
        hypots = [0] * len(classes)
        # Compute a hypothesis for each possible outcome
        # Choose the maximum hypothesis to find result
        for i in xrange(len(classes)):
            hypots[i] = self.h(myTheta[i], myrow)
        return classes[np.argmax(np.array(hypots))]

    def plotBoundary(self, mytheta, myX, myy, mylambda=0.):
        """
        Function to plot the decision boundary for arbitrary theta, X, y, lambda value
        Inside of this function is feature mapping, and the minimization routine.
        It works by making a grid of x1 ("xvals") and x2 ("yvals") points,
        And for each, computing whether the hypothesis classifies that point as
        True or False. Then, a contour is drawn with a built-in pyplot function.
        """
        theta, mincost = self.optimizeRegularizedTheta(mytheta, myX, myy, mylambda)
        xvals = np.linspace(-1, 1.5, 50)
        yvals = np.linspace(-1, 1.5, 50)
        zvals = np.zeros((len(xvals), len(yvals)))
        for i in xrange(len(xvals)):
            for j in xrange(len(yvals)):
                myfeaturesij = self.mapFeature(np.array([xvals[i]]), np.array([yvals[j]]))
                zvals[i][j] = np.dot(theta, myfeaturesij.T)
        zvals = zvals.transpose()

        u, v = np.meshgrid(xvals, yvals)
        mycontour = plt.contour(xvals, yvals, zvals, [0])
        # Kind of a hacky way to display a text on top of the decision boundary
        myfmt = {0: 'Lambda = %d' % mylambda}
        plt.clabel(mycontour, inline=1, fontsize=15, fmt=myfmt)
        plt.title("Decision Boundary")
        # plt.show()

if __name__ == '__main__':
    datafile = 'ex2/data/ex2data2.txt'
    # !head $datafile
    cols = np.loadtxt(datafile, delimiter=',', usecols=(0, 1, 2), unpack=True)  # Read in comma separated data
    ##Form the usual "X" matrix and "y" vector
    X = np.transpose(np.array(cols[:-1]))
    y = np.transpose(np.array(cols[-1:]))
    m = y.size  # number of training examples
    ##Insert the usual column of 1's into the "X" matrix
    X = np.insert(X, 0, 1, axis=1)

    # Divide the sample into two: ones with positive classification, one with null classification
    pos = np.array([X[i] for i in xrange(X.shape[0]) if y[i] == 1])
    neg = np.array([X[i] for i in xrange(X.shape[0]) if y[i] == 0])
    # Check to make sure I included all entries
    # print "Included everything? ",(len(pos)+len(neg) == X.shape[0])

    LR = LogisticRegression()

    mappedX = LR.mapFeature(X[:, 1], X[:, 2])
    initial_theta = np.zeros((mappedX.shape[1], 1))
    costJ = LR.computeCost(initial_theta, mappedX, y)
    print costJ

    plt.figure(figsize=(12, 10))
    plt.subplot(221)
    LR.plotData(pos, neg)
    LR.plotBoundary(initial_theta, mappedX, y, 0.)

    plt.subplot(222)
    LR.plotData(pos, neg)
    LR.plotBoundary(initial_theta, mappedX, y, 1.)

    plt.subplot(223)
    LR.plotData(pos, neg)
    LR.plotBoundary(initial_theta, mappedX, y, 10.)

    plt.subplot(224)
    LR.plotData(pos, neg)
    LR.plotBoundary(initial_theta, mappedX, y, 100.)

    plt.show()
