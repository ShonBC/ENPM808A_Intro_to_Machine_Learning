
import matplotlib.pyplot as plt
import pandas as pd
import math
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score
from tqdm import tqdm

def generate_random_numbers01(N, dim, max_v = 10000):
    """
    Generate random numbers between 0 and 1
    max_v: maximum value used to generate random integers
    """
    random_ints = np.random.randint(max_v, size=(N, dim))
    init_lb = 0
    return (random_ints - init_lb)/(max_v - 1 - init_lb)

def generate_random_numbers(N, dim, max_v, lb, ub):
    """
    Generate random numbers between 'lb' and 'ub'
    """
    zero_to_one_points = generate_random_numbers01(N, dim, max_v)
    res = lb + (ub - lb)*zero_to_one_points
    return res

def generate_random_ring(N, r1, r2, max_v):
    """Generate random numbers in a ring between r1 and r2
    """
    radiuses = generate_random_numbers(N, 1, max_v, r1, r2)
    radians = generate_random_numbers(N, 1, max_v, 0, 2.0*math.pi)
    return radiuses, radians
def move_bottom_ring_and_assign(radiuses, radians, diffx, diffy):
    """
    Give the points within a ring, move the bottom half 'diffx' and 'diffy' along
    x and y directions respectively. Assign the bottom points to have sign -1
    """
    xs = radiuses * np.cos(radians)
    ys = radiuses * np.sin(radians)
    signs = np.ones(len(xs))

    for idx, r in enumerate(radiuses):
        rad = radians[idx]
        xi, yi = xs[idx], ys[idx]
        if rad > math.pi and rad < 2*math.pi:
            xs[idx] = xi + diffx
            ys[idx] = yi +  diffy
            signs[idx] = -1
    return xs, ys, signs


def perceptron(points, dim, max_it=100, use_adaline=False,
               eta=1, randomize=False, print_out=True):
    w = np.zeros(dim + 1)
    xs, ys = points[:, :dim + 1], points[:, dim + 1]
    num_points = points.shape[0]
    for it in range(max_it):
        correctly_predicted_ids = set()
        idxs = np.arange(num_points)
        if randomize:
            idxs = np.random.choice(np.arange(num_points), num_points, replace=False)
        for idx in idxs:
            x, y = xs[idx], ys[idx]
            st = np.dot(w.T, x)
            prod = st * y  # np.dot(w.T, x)*y
            if prod < -100:  # avoid out of bound error
                st = -100
            threshold = 1 if use_adaline else 0
            st = st if use_adaline else 0
            if prod <= threshold:
                w = w + eta * (y - st) * x
                break  # PLA picks one example at each iteration
            else:
                correctly_predicted_ids.add(idx)
        if len(correctly_predicted_ids) == num_points:
            break

    rou = math.inf
    R = 0
    c = 0
    for x, y in zip(xs, ys):
        prod = np.dot(w.T, x) * y
        if prod > 0:
            c += 1
        if prod < rou:
            rou = prod
        abs_x = np.linalg.norm(x)
        if abs_x > R:
            R = abs_x
    theoretical_t = (R ** 2) * (np.linalg.norm(w) ** 2) / rou / rou  # LFD problem 1.3
    # w = w/w[-1]
    if print_out:
        print('Final correctness: ', c, '. Total iteration: ', it)
        print('Final w:', w)
    return w, it, theoretical_t

def linear_regression(X, y):
    XT = np.transpose(X)
    x_pseudo_inv = np.matmul(np.linalg.inv(np.matmul(XT,X)), XT)
    w = np.matmul(x_pseudo_inv,y)
    return w

def PLA(points, dim, max_it, use_adaline=False, eta=1, randomize=False, print_out=True):
    w = np.zeros(dim + 1)
    xs, ys = points[:, :dim + 1], points[:, dim + 1]
    num_points = points.shape[0]
    for it in range(max_it):
        correctly_predicted_ids = set()
        idxs = np.arange(num_points)
        if randomize:
            idxs = np.random.choice(np.arange(num_points), num_points, replace=False)
        for idx in idxs:
            x, y = xs[idx], ys[idx]
            st = np.dot(w.T, x)
            prod = st * y  # np.dot(w.T, x)*y
            if prod < -100:  # avoid out of bound error
                st = -100
            threshold = 1 if use_adaline else 0
            st = st if use_adaline else 0
            if prod <= threshold:
                w = w + eta * (y - st) * x
                break  # PLA picks one example at each iteration
            else:
                correctly_predicted_ids.add(idx)
        if len(correctly_predicted_ids) == num_points:
            break

    rou = math.inf
    R = 0
    c = 0
    for x, y in zip(xs, ys):
        prod = np.dot(w.T, x) * y
        if prod > 0:
            c += 1
        if prod < rou:
            rou = prod
        abs_x = np.linalg.norm(x)
        if abs_x > R:
            R = abs_x
    theoretical_t = (R ** 2) * (np.linalg.norm(w) ** 2) / rou / rou  # LFD problem 1.3
    # w = w/w[-1]
    if print_out:
        print('Final correctness: ', c, '. Total iteration: ', it)
        print('Final w:', w)
    return w, it, theoretical_t
