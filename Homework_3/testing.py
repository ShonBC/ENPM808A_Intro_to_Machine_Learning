import matplotlib
import os
import libs.libs
import numpy as np
import pandas as pd
from matplotlib import style
import matplotlib.pyplot as plt
import sklearn.linear_model
import sklearn.metrics

style.use("ggplot")


def show_ima(X):
    # get single digit graphical data
    # to display it, we need to convert single line of 784 values to 28x28 square
    digit_image = X.reshape(28, 28)

    plt.imshow(digit_image, cmap=matplotlib.cm.binary, interpolation="nearest")
    plt.axis("off")
    plt.show()

def intensity(x):
    x = np.mean(x, axis=1)
    return x/255

def symmetry(x):
    new = x-np.flip(x, axis=1)
    new = np.mean(np.abs(new), axis=1)
    return -new/255

def LinearReg(x1, x, x_test):
    """Show plot of the train data. Train the linear model using Linear Regression and use test data to test the model.

    Args:
        x1 ([type]): Target output data
        x ([type]): Input data
        x_test ([list]): Input data to test the model
    """

    # Select a linear model
    model = sklearn.linear_model.LinearRegression()
    
    # # Train the model
    x = x.reshape(len(x),1)
    x1 = x1.reshape(len(x1),1)
    model.fit(x, x1)

    # Make a prediction
    TestModel(x_test, model)


def TestModel(x_test, model):
    """Tests the given model using the input test data.

    Args:
        x_test ([list]): Input data to test model
        model ([type]): Trained model to test
    """

    y_test = x_test['label']
    X_test = x_test.drop(['label'], axis=1)

    X_test = X_test.values
    y_test = y_test.values

    inten_test = intensity(X_test)
    sym_test = symmetry(X_test)
    inten_test = inten_test.reshape(len(inten_test),1)
    sym_test = sym_test.reshape(len(sym_test),1)
    # print(model.score(X_test, y_test))
    predict_test = model.predict(inten_test)
    print(f'MSE: [{sklearn.metrics.mean_squared_error(sym_test, predict_test)*100}%]')

def Error(x, y, w, num_points):

    err = 0
    for i in range(num_points):
        err += ((y[i] * x[i])/ (1 + np.exp((y[i] * w[1]) * (x[i] * w[0]))))
    return err     

def LogGradDescent(x, y, itterations = 100):
    w_0 = [0, 0]
    n = 0.1 # Step size
    w = w_0
    num_data_points = len(x)
    for i in range(itterations):
        g = -(1/num_data_points) * Error(x, y, w, num_data_points)
        v = -g
        w = w + (n * v)
    return w
    ## h(x,y) = w0*x + w1*y
    # h = x * w[0] + y * w[1]
    

print(os.listdir("MNIST_data"))

# get data
train = pd.read_csv("MNIST_data/mnist_train_binary.csv")
y = train['label']
X = train.drop(['label'], axis=1)

#X = X.values.reshape(-1,28,28)
X = X.values
y = y.values

# get data
test_data = pd.read_csv("MNIST_data/mnist_test_binary.csv")

# delete train to gain some space
del train

print("Shape of X:{0}".format(X.shape))
print("Shape of y:{0}".format(y.shape))

inten = intensity(X)
sym = symmetry(X)

df = pd.DataFrame({'Intensity':inten.flatten(), 'Symmetry':sym.flatten(), 'y':y.flatten()})
df['x0'] = 1
df = df[['x0','Intensity','Symmetry','y']]
ones = df.loc[df['y']==1]
fives = df.loc[df['y']==5]

# Data

eta = 1
use_adaline = False
maxit = 1000
dim = 2

figsize = plt.figaspect(1)
fig, ax1 = plt.subplots(1, 1, figsize=figsize)
ps = ax1.scatter(ones[['Intensity']].values, ones[['Symmetry']].values, marker='+', c= 'b', label='+1 labels')
ns = ax1.scatter(fives[['Intensity']].values, fives[['Symmetry']].values, marker=r'$-$', c= 'r', label='-1 labels')

norm_g, num_its, _ = libs.libs.PLA(df.values, dim, maxit, use_adaline, eta, randomize=False, print_out = True)
# hypothesis = ax1.plot(inten, (norm_g[0]+norm_g[1]*inten), c = 'g', label='Final Hypothesis')
LinearReg(sym, inten, test_data)
# plt.show()

print(LogGradDescent(inten, sym))
