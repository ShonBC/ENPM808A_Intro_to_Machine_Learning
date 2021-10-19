""" For the double-semi-circle task in Problem 3.1, vary sep
in the range {0.2, 0.4,...,5}/ Generate 2000 examples and run 
the PLA starting with w = 0. Record the number of iterations 
PLA takes to converge.

Plot sep versus the number of iterations taken for PLA to converge.
Explain your observations.
"""

# Add lib input sys.path
import os
import sys
nb_dir = os.path.split(os.getcwd())[0]
if nb_dir not in sys.path:
    sys.path.append(nb_dir)

import libs.linear_models as lm
import libs.data_util as data

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':

    N = 2000
    max_v = 10000
    rad, thk = 10, 5
    eta = 1
    use_adaline=False
    maxit = 1000
    dim = 2
    seps = np.arange(0.2, 5.2, 0.2)

    radiuses, radians = data.generate_random_ring(N, rad, rad+thk, max_v)
    its, threoticals_ts = [], []
    for sep in seps:
        xs, ys, signs = data.move_bottom_ring_and_assign(radiuses, radians, rad + thk/2.0, -sep)
        df = pd.DataFrame({'x1':xs.flatten(), 'x2':ys.flatten(), 'y':signs.flatten()})
        df['x0'] = 1
        df = df[['x0','x1','x2','y']]
        positives = df.loc[df['y']==1]
        negatives = df.loc[df['y']==-1]
        norm_g, num_its, theoretical_t = lm.perceptron(df.values, dim, maxit, use_adaline, eta, 
                                                    randomize=False, print_out = True)
        its.append(num_its)
        threoticals_ts.append(theoretical_t)

    plt.plot(seps, its, marker='.')
    plt.show()
    