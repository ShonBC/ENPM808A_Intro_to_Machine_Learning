"""
The perceptron learning algorithm works like this:  In each iteration t, pick a
random x(t),y(t) and compute the signal s(t) = wT(t)x(t).  If y(t).s(t)≤0,update w 
by 
        w(t+ 1)←w(t) +y(t).x(t); 
One  may  argue  that  this  algorithm  does  not  take the  ’closeness’  
between s(t) and y(t) into consideration.  Let’s look at another 
perceptron learning algorithm:  In each iteration, pick a random (x(t),y(t)) and 
compute s(t).  If y(t).s(t)≤1, update w by 
    w(t+ 1)←w(t) +η.(y(t)−s(t)).x(t); 
Where η is a constant.  That is, if s(t) agrees with y(t) well (their product is > 1),  
the  algorithm  does  nothing.   On  the  other  hand,  if s(t)  is  further from y(t), 
the algorithm changes w(t) more.  In this problem, you are asked to implement this 
algorithm and study its performance.

(a)  Genrate a training data set of size 100 similar to that used in Exercise1.4.  
Generate a test data set of size 10,000 from the same process.  To get g, run the 
algorithm above with η= 100 on the training data set,until a maximum of 1,000 
updates has been reached.  Plot the training data set, the target function f, and 
the final hypothesis g on the same figure.  Report the error on the test set.
(b)  Use the data set in (a) and redo everything with η= 1.
(c)  Use the data set in (a) and redo everything with η= 0.01.
(d)  Use the data set in (a) and redo everything with η= 0.0001.
(e)  Compare the results that you get from (a) to (d).

The algorithm above is a variant of the so-called Adaline (Adaptive LinearNeuron) algorithm 
for perceptron learning.
"""

import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg, random

class Perceptron:
    """Perceptron Algorithm class
    """

    def __init__(self):
        """Init with random weight vector 'w' and n
        """

        self.w = random.rand(2)
        self.n = 100

    def Output(self, x):
        """Calculates the output for a given input data value.

        Args:
            x (list): input data (x, y) points

        Returns:
            [int]: 1 if the point is properly classified, -1 is point is misclassified
        """

        y = x[0] * self.w[0] + x[1] * self.w[1]

        if y >= 0:
            return 1
        else:
            return -1

    def Weight(self, x, err):
        """Recalculates the weight vector based off the new error.

        Args:
            x (list): input data (x, y) points
            err (int): difference in predicted vs actual output (y(t)−s(t))
        """

        self.w[0] += self.n * err * x[0]
        self.w[1] += self.n * err * x[1]

    def Train(self, data):
        """Train the model no more than 1000 times

        Args:
            data (list): List of (x, y) pairs for input and output data
        """

        it = 0 
        max_updates = 1000
        iterating = True

        while iterating:

            total_err = 0
            for x in data:

                out = self.Output(x)

                if x[2] != out:

                    err = x[2] - out
                    self.Weight(x, err)
                    total_err += abs(err)

            it += 1

            if total_err == 0 or it >= max_updates:
                print(f'Iterated {it} iterations.')
                iterating = False   
        

    def Data(total_samples):
        """Generates random data points to either test or train with.

        Args:
            total_samples (int): Total number of samples to generate

        Returns:
            [list]: List of data points (x,y)
        """

        total_samples = total_samples / 2
        total_samples = int(total_samples)
        x1 = random.rand(total_samples)
        y1 = random.rand(total_samples)
        x2 = random.rand(total_samples)
        y2 = random.rand(total_samples)

        data_set = []

        for i in range(len(x1)):

            data_set.append([x1[i], y1[i], 1])
            data_set.append([x2[i], y2[i], -1])
        
        return data_set

    def main(n = 100):
        """Main function that trains data then tests the model with random test data.

        Args:
            n (int, optional): Weight adjustment parameter. Defaults to 100.
        """

        train_data = Perceptron.Data(100)
        test_data = Perceptron.Data(10000)
        perceptron = Perceptron()
        perceptron.n = n
        perceptron.Train(train_data)

        for x in test_data:

            err = perceptron.Output(x) 

            if err == 1: # Properly classified

                plt.plot(x[0], x[1], 'go')

            else: # Incorrectly classified

                plt.plot(x[0], x[1], 'ro')

        n = linalg.norm(perceptron.w)
        ww = perceptron.w / n
        ww1 = [ww[1], -ww[0]]
        ww2 = [-ww[1], ww[0]]
        plt.plot([ww1[0], ww2[0]], [ww1[1], ww2[1]], '-k')
        plt.show()

if __name__ == '__main__':

    # Part (a)
    Perceptron.main()

    # Part (b)
    Perceptron.main(1)

    # Part (c)
    Perceptron.main(.01)

    # Part (d)
    Perceptron.main(.0001)
