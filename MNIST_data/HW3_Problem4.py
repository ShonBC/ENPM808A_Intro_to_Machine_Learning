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

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
import pandas
import matplotlib.pyplot as plt

def intensity(x):
    intn = np.mean(x)
    return intn

def symmetry(x):
    rev = np.flip(x) 
    symm = - np.mean(np.abs(x - rev))
    return symm

def LinReg(feature, label, show_plot=False):
    lin_reg = LinearRegression()
    lin_reg.fit(feature, label)
    # w_lin = lin_reg.coef_

    w_lin = np.matmul(np.matmul(np.linalg.inv(np.matmul(feature.T,feature)),feature.T),label)

    if show_plot:
        x_plot = np.linspace(0, 0.4, 100)
        y_plot = - w_lin[1]/w_lin[2]*x_plot - w_lin[0]/w_lin[2]
        plt.scatter(feature[:6742,1],feature[:6742,2],c='b',label='1',marker='+')
        plt.scatter(feature[6742:,1],feature[6742:,2],c='r',label='5',marker='o')
        plt.plot(x_plot,y_plot,'-') 

    return w_lin

def Pocket(feature, label, weight, show_plot=False):
    iteration = 0
    weights = weight
    label[label==5] = -1
    while(iteration<100):
        iteration +=1
        w = weights
        misClassifications=0
        for i in range(0,len(feature)):
            currentX = feature[i].reshape(-1,feature.shape[1])
            currentY = label[i]
            if currentY != np.sign(np.dot(currentX, w.T)):
                w = w + currentY*currentX
                misClassifications=1
            # print(misClassifications)
            if misClassifications==1:
                break
        Ein_w = 0
        Ein_weights = 0
        for i in range(0,len(feature)):
            currentX = feature[i].reshape(-1,feature.shape[1])
            currentY = label[i]    
            if currentY != np.sign(np.dot(currentX, w.T)):
                Ein_w +=1
            if currentY != np.sign(np.dot(currentX, weights.T)):
                Ein_weights +=1
        if Ein_w < Ein_weights:
            weights = w
            
    w_poc = weights

    Eout = 0
    for i in range(0,len(feature)):
        currentX = feature[i].reshape(-1,feature.shape[1])
        currentY = label[i]    
        if currentY != np.sign(np.dot(currentX, w_poc.T)):
            Eout +=1

    if show_plot:
        x_plot = np.linspace(0, 0.4, 100)
        y_plot = - w_poc[1]/w_poc[2]*x_plot - w_poc[0]/w_poc[2]
        plt.scatter(feature[:6742,1],feature[:6742,2],c='b',label='1',marker='+')
        plt.scatter(feature[6742:,1],feature[6742:,2],c='r',label='5',marker='o')
        plt.plot(x_plot,y_plot,'-') 
        
    return w_poc, Eout

def LogRegression(feature, label, show_plot=False):
    # Logistic Regression
    log_reg = LogisticRegression()
    log_reg.fit(feature, label)

    """ What is this for?"""
    # w_log = lin_reg.coef_

    # y_plot = - w_log[1]/w_log[2]*x_plot - w_log[0]/w_log[2]


    x1 = feature[:,1]
    x2 = feature[:,2]
    train_poly = np.column_stack((feature,x1**2,x1*x2,x2**2,x1**3,x1**2*x2,x1*x2**2,x2**3))
    w_poly = np.matmul(np.matmul(np.linalg.inv(np.matmul(train_poly.T,train_poly)),train_poly.T),label)

    if show_plot:
        x = np.linspace(0,0.4,100)
        y = np.linspace(-0.3,0,100)
        x,y = np.meshgrid(x,y)
        z = 1*w_poly[0] + x*w_poly[1] + y*w_poly[2] + x**2*w_poly[3] + x*y*w_poly[4] + y**2*w_poly[5] + x**3*w_poly[6] + x**2*y*w_poly[7] + x*y**2*w_poly[8] + y**3*w_poly[9]
        levels = np.array([0])
        plt.scatter(feature[:6742,1],feature[:6742,2],c='b',label='1',marker='+')
        plt.scatter(feature[6742:,1],feature[6742:,2],c='r',label='5',marker='o')
        cs = plt.contour(x,y,z,levels)
    
    return w_poly 

def main():
    # Define Training Data
    data = pandas.read_csv('MNIST_data/mnist_train_binary.csv')
    data = (data.to_numpy())
    train_Y  = (data.T[0]).T
    # Set all 5 labels equal to -1
    train_Y[train_Y==5] = -1
    # Scale data to be between 0 and 1
    train_X  = (data.T[1:]).T/255

    # Define Testing data
    data = pandas.read_csv('MNIST_data/mnist_test_binary.csv')
    data = (data.to_numpy())
    test_Y  = (data.T[0]).T
    # Set all 5 labels equal to -1
    test_Y[test_Y==5] = -1
    # Scale data to be between 0 and 1
    test_X  = (data.T[1:]).T/255

    # Define training features vector (X_train)
    train_feature = []
    for curX in train_X:
        cur_intn = intensity(curX.reshape((28, 28)))
        cur_symm = symmetry(curX.reshape((28, 28)))
        train_feature = np.append(train_feature,(1,cur_intn,cur_symm))
    train_feature = np.reshape(train_feature,(12163,3))

    # Define testing features vector (X_test)
    test_feature = []
    for curX in test_X:
        cur_intn = intensity(curX.reshape((28, 28)))
        cur_symm = symmetry(curX.reshape((28, 28)))
        test_feature = np.append(test_feature,(1,cur_intn,cur_symm))
    test_feature = np.reshape(test_feature,(2027,3))

    w_lin = LinReg(train_feature, train_Y, True)
    w_poc, Eout_poc = Pocket(train_feature, train_Y, w_lin, True)
    w_reg = LogRegression(train_feature, train_Y, True)
    plt.show()

if __name__ == '__main__':

    main()

    # # Define Training Data
    # data = pandas.read_csv('MNIST_data/mnist_train_binary.csv')
    # data = (data.to_numpy())
    # train_Y  = (data.T[0]).T
    # # Set all 5 labels equal to -1
    # train_Y[train_Y==5] = -1
    # # Scale data to be between 0 and 1
    # train_X  = (data.T[1:]).T/255

    # # Define Testing data
    # data = pandas.read_csv('MNIST_data/mnist_test_binary.csv')
    # data = (data.to_numpy())
    # test_Y  = (data.T[0]).T
    # # Set all 5 labels equal to -1
    # test_Y[test_Y==5] = -1
    # # Scale data to be between 0 and 1
    # test_X  = (data.T[1:]).T/255

    # # Define training features vector (X_train)
    # train_feature = []
    # for curX in train_X:
    #     cur_intn = intensity(curX.reshape((28, 28)))
    #     cur_symm = symmetry(curX.reshape((28, 28)))
    #     train_feature = np.append(train_feature,(1,cur_intn,cur_symm))
    # train_feature = np.reshape(train_feature,(12163,3))
    # # plt.figure()
    # # plt.scatter(train_feature[:6742,1],train_feature[:6742,2],c='b',label='1',marker='+')
    # # plt.scatter(train_feature[6742:,1],train_feature[6742:,2],c='r',label='5',marker='o')
    # # plt.show()

    # # Define testing features vector (X_test)
    # test_feature = []
    # for curX in test_X:
    #     cur_intn = intensity(curX.reshape((28, 28)))
    #     cur_symm = symmetry(curX.reshape((28, 28)))
    #     test_feature = np.append(test_feature,(1,cur_intn,cur_symm))
    # test_feature = np.reshape(test_feature,(2027,3))
    # # plt.figure()
    # # plt.scatter(test_feature[:1135,1],test_feature[:1135,2],c='b',label='1',marker='+')
    # # plt.scatter(test_feature[1135:,1],test_feature[1135:,2],c='r',label='5',marker='o')
    # # plt.show()

    # # Linear Regression
    # lin_reg = LinearRegression()
    # lin_reg.fit(train_feature, train_Y)
    # # w_lin = lin_reg.coef_

    # w_lin = np.matmul(np.matmul(np.linalg.inv(np.matmul(train_feature.T,train_feature)),train_feature.T),train_Y)

    # # maxiter = 1000
    # # for t in range range(maxiter):
    # #     for i in range(len(train_feature)):

    # # Pocket Algorithm on Linear Regression Weights      
    # iteration = 0
    # weights = w_lin
    # train_Y[train_Y==5] = -1
    # while(iteration<100):
    #     iteration +=1
    #     w = weights
    #     misClassifications=0
    #     for i in range(0,len(train_feature)):
    #         currentX = train_feature[i].reshape(-1,train_feature.shape[1])
    #         currentY = train_Y[i]
    #         if currentY != np.sign(np.dot(currentX, w.T)):
    #             w = w + currentY*currentX
    #             misClassifications=1
    #         # print(misClassifications)
    #         if misClassifications==1:
    #             break
    #     Ein_w = 0
    #     Ein_weights = 0
    #     for i in range(0,len(train_feature)):
    #         currentX = train_feature[i].reshape(-1,train_feature.shape[1])
    #         currentY = train_Y[i]    
    #         if currentY != np.sign(np.dot(currentX, w.T)):
    #             Ein_w +=1
    #         if currentY != np.sign(np.dot(currentX, weights.T)):
    #             Ein_weights +=1
    #     if Ein_w < Ein_weights:
    #         weights = w
            
    # w_poc = weights

    # x_plot = np.linspace(0, 0.4, 100)
    # y_plot = - w_poc[1]/w_poc[2]*x_plot - w_poc[0]/w_poc[2]
    # plt.scatter(train_feature[:6742,1],train_feature[:6742,2],c='b',label='1',marker='+')
    # plt.scatter(train_feature[6742:,1],train_feature[6742:,2],c='r',label='5',marker='o')
    # plt.plot(x_plot,y_plot,'-') 

    # plt.scatter(test_feature[:1135,1],test_feature[:1135,2],c='b',label='1',marker='+')
    # plt.scatter(test_feature[1135:,1],test_feature[1135:,2],c='r',label='5',marker='o')
    # plt.plot(x_plot,y_plot,'-') 

    # # Error of Pocket algorithm on Linear Regression
    # Eout = 0
    # for i in range(0,len(test_feature)):
    #     currentX = test_feature[i].reshape(-1,test_feature.shape[1])
    #     currentY = test_Y[i]    
    #     if currentY != np.sign(np.dot(currentX, w_poc.T)):
    #         Eout +=1

    # # Logistic Regression
    # log_reg = LogisticRegression()
    # log_reg.fit(train_feature, train_Y)
    # w_log = lin_reg.coef_

    # y_plot = - w_log[1]/w_log[2]*x_plot - w_log[0]/w_log[2]


    # x1 = train_feature[:,1]
    # x2 = train_feature[:,2]
    # train_poly = np.column_stack((train_feature,x1**2,x1*x2,x2**2,x1**3,x1**2*x2,x1*x2**2,x2**3))
    # w_poly = np.matmul(np.matmul(np.linalg.inv(np.matmul(train_poly.T,train_poly)),train_poly.T),train_Y)


    # x = np.linspace(0,0.4,100)
    # y = np.linspace(-0.3,0,100)
    # x,y = np.meshgrid(x,y)
    # z = 1*w_poly[0] + x*w_poly[1] + y*w_poly[2] + x**2*w_poly[3] + x*y*w_poly[4] + y**2*w_poly[5] + x**3*w_poly[6] + x**2*y*w_poly[7] + x*y**2*w_poly[8] + y**3*w_poly[9]
    # levels = np.array([0])
    # plt.scatter(train_feature[:6742,1],train_feature[:6742,2],c='b',label='1',marker='+')
    # plt.scatter(train_feature[6742:,1],train_feature[6742:,2],c='r',label='5',marker='o')
    # cs = plt.contour(x,y,z,levels)

    # plt.show()
    