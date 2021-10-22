"""
Using the image data uploaded on ELMS, perform classification using the
following algorithms for non-separable data:
(a) Linear Regression for classification followed by pocket for improvement.
(b) Logistic regression for classification using gradient descent.
Use your chosen algorithm to find the best separator you can using the train-
ing data only (you can create your own features). The output is +1 if the
example is a 1 and -1 for a 5.
(a) Give separate plots of the training and test data, together with the sep-
arators.
(b) Compute Ein on your training data and Etest, the test error on the test
data.
(c) Obtain a bound on the true out-of-sample error. You should get two
bounds, one based on Ein and one based on Etest. Use a tolerance δ =
0.05. Which is the better bound?
(d) Now repeat using a 3rd order polynomial transform.
(e) As your final deliverable to a customer, would you use the linear model
with or without the 3rd order polynomial transform? Explain.
"""

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib
import matplotlib.pyplot as plt
import sklearn.linear_model
import csv

# data_path = "MNIST_data/"
# train_data = np.loadtxt(data_path + "mnist_train_binary.csv", 
#                         delimiter=",")
# test_data = np.loadtxt(data_path + "mnist_test_binary.csv", 
#                        delimiter=",") 

# # Read data
# train_data = pd.read_csv('MNIST_data/mnist_test_binary.csv',thousands=',')
# train_data.info()

def load_data(filepath, delimiter=",", dtype=float):
    """Load a numerical numpy array from a file."""

    print(f"Loading {filepath}...")
    with open(filepath, "r") as f:
        data_iterator = csv.reader(f, delimiter=delimiter)
        next(data_iterator) # Skip first row header information
        data_list = list(data_iterator)
    data = np.asarray(data_list, dtype=dtype)
    print("Done.")
    return data

def GetMatrix(data):
    a=[]
    for i in range(1, len(train_data[0])): # Row, col
        if train_data[-1][i] > 100:
            a.append(1)
        else:
            a.append(0)
    return a

def Intensity(data):
    inten = []
    for i in range(len(data)):
        a = GetMatrix(data)
        inten.append(np.mean(a))
    return inten

def Symmetry(data):
    sym = []

    for i in range(len(data)):
        a = GetMatrix(data)
        sym.append(np.mean(np.abs(a - np.flip(a))))
    return sym

def Labels(data):
    label = []

    for i in range(0, len(data)): # Row, col
        label.append(data[i][0]) 
    return label

def ShowData(x, y, label):
    x_range = []
    y_range = []
    for i in range(len(label)):
        if label[i] == 1:
            plt.plot(x[i], y[i], 'bo')
        else:
            plt.plot(x[i], y[i], 'co')
    
    plt.xlabel("Intensity")
    plt.ylabel("Symmetry")

if __name__ == "__main__":

    train_data = load_data("MNIST_data/mnist_train_binary.csv", ",", float)
    test_data = load_data("MNIST_data/mnist_test_binary.csv", ",", float)

    # a=[]
    # for j in range(1, len(train_data[0])): # Row, col
    #     if train_data[-1][j] > 100:
    #         a.append(1)
    #     else:
    #         a.append(0)

    # a = np.reshape(a, (28, 28))
    # print(np.mean(a))
    # print(a)

    inten = Intensity(train_data)
    sym = Symmetry(train_data)
    label = Labels(train_data)
    # print(label)
    # plt.plot(inten, sym)
    # plt.show()

    # label = []
    # # print(len(train_data))
    # for i in range(0, len(train_data)): # Row, col
    #     label.append(train_data[i][0])    
    # print(len(label))
    # print(len(inten))
    # print(len(sym))

    # ShowData(inten, sym, label)
    # for i in range(len(label)):
    #     if label[i] == 1:
    #         plt.plot(x[i], y[i], 'bo')
    # for i in range(len(label)):
    plt.scatter(inten, sym)
    plt.xlabel("Intensity")
    plt.ylabel("Symmetry")
    plt.show()