import numpy as np
import matplotlib.pyplot as plt
import scipy.io #Used to load the OCTAVE *.mat files
import scipy.optimize #fmin_cg to train the linear regression

import warnings
warnings.filterwarnings('ignore')

class LinearRegression():
    def __init__(self, X, y):
        self.X = np.insert(X, 0, 1, axis=1)
        self.y = y
        self.mylambda = 0.
        self.d = 0

    def setVal(self, Xval, yval):
        self.Xval = np.insert(Xval, 0, 1, axis=1)
        self.yval = yval

    def plotData(self, X, y):
        plt.figure(figsize=(8, 5))
        plt.ylabel('Water flowing out of the dam (y)')
        plt.xlabel('Change in water level (x)')
        plt.plot(X[:, 1], y, 'rx')
        plt.grid(True)

    def h(self, theta, X):  # Linear hypothesis function
        return np.dot(X, theta)

    def computeCost(self, mytheta, myX, myy, mylambda=0.):  # Cost function
        """
        theta_start is an n- dimensional vector of initial theta guess
        X is matrix with n- columns and m- rows
        y is a matrix with m- rows and 1 column
        """
        m = myX.shape[0]
        myh = self.h(mytheta, myX).reshape((m, 1))
        mycost = float((1. / (2 * m)) * np.dot((myh - myy).T, (myh - myy)))
        regterm = (float(mylambda) / (2 * m)) * float(mytheta[1:].T.dot(mytheta[1:]))
        return mycost + regterm

    def computeGradient(self, mytheta, myX, myy, mylambda=0.):
        mytheta = mytheta.reshape((mytheta.shape[0], 1))
        m = myX.shape[0]
        # grad has same shape as myTheta (2x1)
        myh = self.h(mytheta, myX).reshape((m, 1))
        grad = (1. / float(m)) * myX.T.dot(myh - myy)
        regterm = (float(mylambda) / m) * mytheta
        regterm[0] = 0  # don't regulate bias term
        regterm.reshape((grad.shape[0], 1))
        return grad + regterm

    # Here's a wrapper for computeGradient that flattens the output
    # This is for the minimization routine that wants everything flattened
    def computeGradientFlattened(self, mytheta, myX, myy, mylambda=0.):
        return self.computeGradient(mytheta, myX, myy, mylambda).flatten()

    def optimizeTheta(self,myTheta_initial, myX, myy, mylambda=0., print_output=True):
        fit_theta = scipy.optimize.fmin_cg(self.computeCost, x0=myTheta_initial, \
                                           fprime=self.computeGradientFlattened, \
                                           args=(myX, myy, mylambda), \
                                           disp=False, \
                                           epsilon=1.49e-12, \
                                           maxiter=1000)
        fit_theta = fit_theta.reshape((myTheta_initial.shape[0], 1))
        return fit_theta

    def plotLearningCurve(self, X, y, Xval, yval, mylambda):
        """
        Loop over first training point, then first 2 training points, then first 3 ...
        and use each training-set-subset to find trained parameters.
        With those parameters, compute the cost on that subset (Jtrain)
        remembering that for Jtrain, lambda = 0 (even if you are using regularization).
        Then, use the trained parameters to compute Jval on the entire validation set
        again forcing lambda = 0 even if using regularization.
        Store the computed errors, error_train and error_val and plot them.
        """
        initial_theta = np.array([[1.], [1.]])
        mym, error_train, error_val = [], [], []
        for x in xrange(1, X.shape[0], 1):
            train_subset = X[:x, :]
            y_subset = y[:x]
            mym.append(y_subset.shape[0])
            fit_theta = self.optimizeTheta(initial_theta, train_subset, y_subset, mylambda, print_output=False)
            error_train.append(self.computeCost(fit_theta, train_subset, y_subset, mylambda))
            error_val.append(self.computeCost(fit_theta, Xval, yval, mylambda))

        plt.figure(figsize=(8, 5))
        plt.plot(mym, error_train, label='Train')
        plt.plot(mym, error_val, label='Cross Validation')
        plt.legend()
        plt.title('Learning curve for linear regression')
        plt.xlabel('Number of training examples')
        plt.ylabel('Error')
        plt.grid(True)
        plt.show()

    def plotPolyLearningCurve(self, X, y, Xval, yval, d, mylambda):

        initial_theta = np.ones((d + 2, 1))
        mym, error_train, error_val = [], [], []
        myXval, dummy1, dummy2 = self.featureNormalize(self.genPolyFeatures(Xval, d))

        for x in xrange(1, X.shape[0], 1):
            train_subset = X[:x, :]
            y_subset = y[:x]
            mym.append(y_subset.shape[0])
            train_subset = self.genPolyFeatures(train_subset, d)
            train_subset, dummy1, dummy2 = self.featureNormalize(train_subset)
            fit_theta = self.optimizeTheta(initial_theta, train_subset, y_subset, mylambda=mylambda, print_output=False)
            error_train.append(self.computeCost(fit_theta, train_subset, y_subset, mylambda=mylambda))
            error_val.append(self.computeCost(fit_theta, myXval, yval, mylambda=mylambda))

        plt.figure(figsize=(8, 5))
        plt.plot(mym, error_train, label='Train')
        plt.plot(mym, error_val, label='Cross Validation')
        plt.legend()
        plt.title('Polynomial Regression Learning Curve (lambda = %d)' % mylambda)
        plt.xlabel('Number of training examples')
        plt.ylabel('Error')
        plt.ylim([0, 100])
        plt.grid(True)
        plt.show()

    def genPolyFeatures(self, myX, p):
        """
        Function takes in the X matrix (with bias term already included as the first column)
        and returns an X matrix with "p" additional columns.
        The first additional column will be the 2nd column (first non-bias column) squared,
        the next additional column will be the 2nd column cubed, etc.
        """
        newX = myX.copy()
        for i in xrange(p):
            dim = i + 2
            newX = np.insert(newX, newX.shape[1], np.power(newX[:, 1], dim), axis=1)
        return newX

    def featureNormalize(self, myX):
        """
        Takes as input the X array (with bias "1" first column), does
        feature normalizing on the columns (subtract mean, divide by standard deviation).
        Returns the feature-normalized X, and feature means and stds in a list
        Note this is different than my implementation in assignment 1...
        I didn't realize you should subtract the means, THEN compute std of the
        mean-subtracted columns.
        Doesn't make a huge difference, I've found
        """

        Xnorm = myX.copy()
        stored_feature_means = np.mean(Xnorm, axis=0)  # column-by-column
        Xnorm[:, 1:] = Xnorm[:, 1:] - stored_feature_means[1:]
        stored_feature_stds = np.std(Xnorm, axis=0, ddof=1)
        Xnorm[:, 1:] = Xnorm[:, 1:] / stored_feature_stds[1:]
        return Xnorm, stored_feature_means, stored_feature_stds

    def plotFit(self, X, y, fit_theta,  means, stds):
        """
        Function that takes in some learned fit values (on feature-normalized data)
        It sets x-points as a linspace, constructs an appropriate X matrix,
        un-does previous feature normalization, computes the hypothesis values,
        and plots on top of data
        """
        n_points_to_plot = 50
        xvals = np.linspace(-1.1*n_points_to_plot, 1.1*n_points_to_plot, n_points_to_plot)
        xmat = np.ones((n_points_to_plot, 1))

        xmat = np.insert(xmat, xmat.shape[1], xvals.T, axis=1)
        xmat = self.genPolyFeatures(xmat, len(fit_theta) - 2)
        # This is undoing feature normalization
        xmat[:, 1:] = xmat[:, 1:] - means[1:]
        xmat[:, 1:] = xmat[:, 1:] / stds[1:]
        self.plotData(X, y)
        plt.plot(xvals, self.h(fit_theta, xmat), 'b--')
        plt.show()

    def errorWithLambda(self, X, y, Xval, yval, global_d, start, end, number):
        lambdas = np.linspace(start, end, number)
        errors_train, errors_val = [], []
        for mylambda in lambdas:
            newXtrain = self.genPolyFeatures(X, global_d)
            newXtrain_norm, dummy1, dummy2 = self.featureNormalize(newXtrain)
            newXval = self.genPolyFeatures(Xval, global_d)
            newXval_norm, dummy1, dummy2 = self.featureNormalize(newXval)
            init_theta = np.ones((newXtrain_norm.shape[1], 1))
            fit_theta = self.optimizeTheta(init_theta, newXtrain_norm, y, mylambda, False)
            errors_train.append(self.computeCost(fit_theta, newXtrain_norm, y, mylambda=mylambda))
            errors_val.append(self.computeCost(fit_theta, newXval_norm, yval, mylambda=mylambda))
        return lambdas, errors_train, errors_val

    def plotErrorWithLambda(self, lambdas, errors_train, errors_val):
        plt.figure(figsize=(8, 5))
        plt.plot(lambdas, errors_train, label='Train')
        plt.plot(lambdas, errors_val, label='Cross Validation')
        plt.legend()
        plt.title('Errors with different lambda')
        plt.xlabel('lambda')
        plt.ylabel('Error')
        plt.grid(True)
        plt.show()

    def traingLinearRegression(self, X, y, Xval, yval, d, mylambda=1.):
        newX = self.genPolyFeatures(X, d)
        newX_norm, stored_means, stored_stds = self.featureNormalize(newX)
        initial_theta = np.ones((newX_norm.shape[1], 1))
        fit_theta = self.optimizeTheta(initial_theta, newX_norm, y, mylambda)
        self.plotFit(X, y, fit_theta, stored_means, stored_stds)
        self.plotPolyLearningCurve(X, y, Xval, yval, d, mylambda=mylambda)

        lambdas, errors_train, errors_val = self.errorWithLambda(X, y, Xval, yval, d, 0, 5, 20)
        self.plotErrorWithLambda(lambdas, errors_train, errors_val)




if __name__ == '__main__':
    datafile = 'ex5/data/ex5data1.mat'
    mat = scipy.io.loadmat(datafile)
    # Training set
    X, y = mat['X'], mat['y']
    # Cross validation set
    Xval, yval = mat['Xval'], mat['yval']
    # Test set
    Xtest, ytest = mat['Xtest'], mat['ytest']
    # Insert a column of 1's to all of the X's, as usual
    X = np.insert(X, 0, 1, axis=1)
    Xval = np.insert(Xval, 0, 1, axis=1)
    Xtest = np.insert(Xtest, 0, 1, axis=1)
    LR = LinearRegression(X, y)
    # LR.plotData(X,y)

    global_d = 5
    mylambda = 1.0

    LR.traingLinearRegression(X, y, Xval, yval, 5, mylambda=mylambda)
    newX = LR.genPolyFeatures(X, global_d)
    newX_norm, stored_means, stored_stds = LR.featureNormalize(newX)
    mytheta = np.ones((newX_norm.shape[1], 1))
    fit_theta = LR.optimizeTheta(mytheta, newX_norm, y, mylambda)
    costJ = LR.computeCost(fit_theta, newX_norm, y, mylambda=mylambda)
    print fit_theta
    print costJ

