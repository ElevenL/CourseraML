import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io #Used to load the OCTAVE *.mat files
import scipy.misc #Used to show matrix as an image
import matplotlib.cm as cm #Used to display images in a specific colormap
import random #To pick random images to display
import scipy.optimize #fmin_cg to train neural network
import itertools
from scipy.special import expit #Vectorized sigmoid function


class NN:
    def __init__(self, X, y):
        self.X = X
        self.y = y

        self.input_layer_size = 400
        self.hidden_layer_size = 25
        self.output_layer_size = 10
        self.n_training_samples = self.X.shape[0]

    # Some utility functions. There are lot of flattening and
    # reshaping of theta matrices, the input X matrix, etc...
    # Nicely shaped matrices make the linear algebra easier when developing,
    # but the minimization routine (fmin_cg) requires that all inputs

    def flattenParams(self, thetas_list):
        """
        Hand this function a list of theta matrices, and it will flatten it
        into one long (n,1) shaped numpy array
        """
        flattened_list = [mytheta.flatten() for mytheta in thetas_list]
        combined = list(itertools.chain.from_iterable(flattened_list))
        assert len(combined) == (self.input_layer_size + 1) * self.hidden_layer_size + \
                                (self.hidden_layer_size + 1) * self.output_layer_size
        return np.array(combined).reshape((len(combined), 1))


    def reshapeParams(self, flattened_array):
        theta1 = flattened_array[:(self.input_layer_size + 1) * self.hidden_layer_size] \
            .reshape((self.hidden_layer_size, self.input_layer_size + 1))
        theta2 = flattened_array[(self.input_layer_size + 1) * self.hidden_layer_size:] \
            .reshape((self.output_layer_size, self.hidden_layer_size + 1))

        return [theta1, theta2]

    def flattenX(self, myX):
        return np.array(myX.flatten()).reshape((self.n_training_samples * (self.input_layer_size + 1), 1))

    def reshapeX(self, flattenedX):
        return np.array(flattenedX).reshape((self.n_training_samples, self.input_layer_size + 1))

    def propagateForward(self, row, Thetas):
        """
        Function that given a list of Thetas (NOT flattened), propagates the
        row of features forwards, assuming the features ALREADY
        include the bias unit in the input layer, and the
        Thetas also include the bias unit

        The output is a vector with element [0] for the hidden layer,
        and element [1] for the output layer
            -- Each element is a tuple of (zs, as)
            -- where "zs" and "as" have shape (# of units in that layer, 1)

        ***The 'activations' are the same as "h", but this works for many layers
        (hence a vector of thetas, not just one theta)
        Also, "h" is vectorized to do all rows at once...
        this function takes in one row at a time***
        """

        features = row
        zs_as_per_layer = []
        for i in xrange(len(Thetas)):
            Theta = Thetas[i]
            # Theta is (25,401), features are (401, 1)
            # so "z" comes out to be (25, 1)
            # this is one "z" value for each unit in the hidden layer
            # not counting the bias unit
            z = Theta.dot(features).reshape((Theta.shape[0], 1))
            a = expit(z)
            zs_as_per_layer.append((z, a))
            if i == len(Thetas) - 1:
                return np.array(zs_as_per_layer)
            a = np.insert(a, 0, 1)  # Add the bias unit
            features = a

    def computeCost(self, mythetas_flattened, myX_flattened, myy, mylambda=0.):
        """
        This function takes in:
            1) a flattened vector of theta parameters (each theta would go from one
               NN layer to the next), the thetas include the bias unit.
            2) the flattened training set matrix X, which contains the bias unit first column
            3) the label vector y, which has one column
        It loops over training points (recommended by the professor, as the linear
        algebra version is "quite complicated") and:
            1) constructs a new "y" vector, with 10 rows and 1 column,
                with one non-zero entry corresponding to that iteration
            2) computes the cost given that y- vector and that training point
            3) accumulates all of the costs
            4) computes a regularization term (after the loop over training points)
        """

        # First unroll the parameters
        mythetas = self.reshapeParams(mythetas_flattened)

        # Now unroll X
        myX = self.reshapeX(myX_flattened)

        # This is what will accumulate the total cost
        total_cost = 0.

        m = self.n_training_samples

        # Loop over the training points (rows in myX, already contain bias unit)
        for irow in xrange(m):
            myrow = myX[irow]

            # First compute the hypothesis (this is a (10,1) vector
            # of the hypothesis for each possible y-value)
            # propagateForward returns (zs, activations) for each layer
            # so propagateforward[-1][1] means "activation for -1st (last) layer"
            myhs = self.propagateForward(myrow, mythetas)[-1][1]

            # Construct a 10x1 "y" vector with all zeros and only one "1" entry
            # note here if the hand-written digit is "0", then that corresponds
            # to a y- vector with 1 in the 10th spot (different from what the
            # homework suggests)
            tmpy = np.zeros((10, 1))
            tmpy[myy[irow] - 1] = 1

            # Compute the cost for this point and y-vector
            mycost = -tmpy.T.dot(np.log(myhs)) - (1 - tmpy.T).dot(np.log(1 - myhs))

            # Accumulate the total cost
            total_cost += mycost

        # Normalize the total_cost, cast as float
        total_cost = float(total_cost) / m

        # Compute the regularization term
        total_reg = 0.
        for mytheta in mythetas:
            total_reg += np.sum(mytheta * mytheta)  # element-wise multiplication
        total_reg *= float(mylambda) / (2 * m)

        return total_cost + total_reg

    def sigmoidGradient(self, z):
        dummy = expit(z)
        return dummy * (1 - dummy)

    def genRandThetas(self):
        epsilon_init = 0.12
        theta1_shape = (self.hidden_layer_size, self.input_layer_size + 1)
        theta2_shape = (self.output_layer_size, self.hidden_layer_size + 1)
        rand_thetas = [np.random.rand(*theta1_shape) * 2 * epsilon_init - epsilon_init, \
                       np.random.rand(*theta2_shape) * 2 * epsilon_init - epsilon_init]
        return rand_thetas

    def backPropagate(self, mythetas_flattened, myX_flattened, myy, mylambda=0.):

        # First unroll the parameters
        mythetas = self.reshapeParams(mythetas_flattened)

        # Now unroll X
        myX = self.reshapeX(myX_flattened)

        # Note: the Delta matrices should include the bias unit
        # The Delta matrices have the same shape as the theta matrices
        Delta1 = np.zeros((self.hidden_layer_size, self.input_layer_size + 1))
        Delta2 = np.zeros((self.output_layer_size, self.hidden_layer_size + 1))

        # Loop over the training points (rows in myX, already contain bias unit)
        m = self.n_training_samples
        for irow in xrange(m):
            myrow = myX[irow]
            a1 = myrow.reshape((self.input_layer_size + 1, 1))
            # propagateForward returns (zs, activations) for each layer excluding the input layer
            temp = self.propagateForward(myrow, mythetas)
            z2 = temp[0][0]
            a2 = temp[0][1]
            z3 = temp[1][0]
            a3 = temp[1][1]
            tmpy = np.zeros((10, 1))
            tmpy[myy[irow] - 1] = 1
            delta3 = a3 - tmpy
            delta2 = mythetas[1].T[1:, :].dot(delta3) * self.sigmoidGradient(z2)  # remove 0th element
            a2 = np.insert(a2, 0, 1, axis=0)
            Delta1 += delta2.dot(a1.T)  # (25,1)x(1,401) = (25,401) (correct)
            Delta2 += delta3.dot(a2.T)  # (10,1)x(1,25) = (10,25) (should be 10,26)

        D1 = Delta1 / float(m)
        D2 = Delta2 / float(m)

        # Regularization:
        D1[:, 1:] = D1[:, 1:] + (float(mylambda) / m) * mythetas[0][:, 1:]
        D2[:, 1:] = D2[:, 1:] + (float(mylambda) / m) * mythetas[1][:, 1:]

        return self.flattenParams([D1, D2]).flatten()

    def checkGradient(self, mythetas, myDs, myX, myy, mylambda=0.):
        myeps = 0.0001
        flattened = self.flattenParams(mythetas)
        flattenedDs = self.flattenParams(myDs)
        myX_flattened = self.flattenX(myX)
        n_elems = len(flattened)
        # Pick ten random elements, compute numerical gradient, compare to respective D's
        for i in xrange(10):
            x = int(np.random.rand() * n_elems)
            epsvec = np.zeros((n_elems, 1))
            epsvec[x] = myeps
            cost_high = self.computeCost(flattened + epsvec, myX_flattened, myy, mylambda)
            cost_low = self.computeCost(flattened - epsvec, myX_flattened, myy, mylambda)
            mygrad = (cost_high - cost_low) / float(2 * myeps)
            print "Element: %d. Numerical Gradient = %f. BackProp Gradient = %f." % (x, mygrad, flattenedDs[x])

    def trainNN(self, mylambda=0.):
        """
        Function that generates random initial theta matrices, optimizes them,
        and returns a list of two re-shaped theta matrices
        """

        randomThetas_unrolled = self.flattenParams(self.genRandThetas())
        result = scipy.optimize.fmin_cg(self.computeCost, x0=randomThetas_unrolled, fprime=self.backPropagate, \
                                        args=(self.flattenX(self.X), self.y, mylambda), maxiter=50, disp=True, full_output=True)
        return self.reshapeParams(result[0])

    def predictNN(self, row, Thetas):
        """
        Function that takes a row of features, propagates them through the
        NN, and returns the predicted integer that was hand written
        """
        classes = range(1, 10) + [10]
        output = self.propagateForward(row, Thetas)
        # -1 means last layer, 1 means "a" instead of "z"
        return classes[np.argmax(output[-1][1])]

    def computeAccuracy(self, myX, myThetas, myy):
        """
        Function that loops over all of the rows in X (all of the handwritten images)
        and predicts what digit is written given the thetas. Check if it's correct, and
        compute an efficiency.
        """
        n_correct, n_total = 0, myX.shape[0]
        for irow in xrange(n_total):
            if int(self.predictNN(myX[irow], myThetas)) == int(myy[irow]):
                n_correct += 1
        print "Training set accuracy: %0.1f%%" % (100 * (float(n_correct) / n_total))


if __name__ == '__main__':
    datafile = 'ex3/data/ex3data1.mat'
    mat = scipy.io.loadmat(datafile)
    X, y = mat['X'], mat['y']
    # Insert a column of 1's to X as usual
    X = np.insert(X, 0, 1, axis=1)
    print "'y' shape: %s. Unique elements in y: %s" % (mat['y'].shape, np.unique(mat['y']))
    print "'X' shape: %s. X[0] shape: %s" % (X.shape, X[0].shape)

    datafile = 'ex3/data/ex3weights.mat'
    mat = scipy.io.loadmat(datafile)
    Theta1, Theta2 = mat['Theta1'], mat['Theta2']

    # nn = NN(X, y)
    #
    # learned_Thetas = nn.trainNN()
    # nn.computeAccuracy(nn.X, learned_Thetas, nn.y)