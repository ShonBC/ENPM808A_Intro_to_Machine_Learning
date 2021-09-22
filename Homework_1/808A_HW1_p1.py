"""
For  this  problem,  please  use  the  dataset  in  the  link  below.   
Please  use  thefirst column as the sales data.  Others are predictors.  
Use the first twenty entries to create a regression model using the code 
snippet included in the class  discussion.   Then  use  the  remaining  
points  to  judge  the  performance of the predictor.  The code snippet 
also an available resource with the book(Hands on ML, HML Book)
https://college.cengage.com/mathematics/brase/understandable_statistics/7e/students/datasets/mlr/excel/mlr05.xls

"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy.core.records import array
import pandas as pd
import sklearn.linear_model

def ShowData(x, y, color):
    """Plots data

    Args:
        x ([type]): Input data values
        y ([type]): Output data values
        color ([str]): Color of data points
    """

    plt.plot(x, y, color)
    plt.xlabel("Target Values")
    plt.ylabel("Features")

def SepTestData(x_test):
    """Removes the 'nan' type data from test data list

    Args:
        x_test ([list]): Test data to train model

    Returns:
        [list]: Test data with 'nan' data values removed
    """

    test_data = []
    i = 0

    while i <= len(list(x_test.T)) - 1:

        if not np.isnan(x_test.T.values[0][i]):
            
            test_data.append(x_test.T.values[0][i])
            
        else:

            pass            

        i += 1
    
    return test_data

def TestModel(x_test, model):
    """Tests the given model using the input test data.

    Args:
        x_test ([list]): Input data to test model
        model ([type]): Trained model to test
    """

    for data in x_test:

        x_new = [[data]]
        print(f'Test Data Value: {data} Model Prediction: {model.predict(x_new)}')

def LinearReg(x1, x, x_test, color):
    """Show plot of the train data. Train the linear model using Linear Regression and use test data to test the model.

    Args:
        x1 ([type]): Target output data
        x ([type]): Input data
        x_test ([list]): Input data to test the model
        color ([str]): color to use for plotting model
    """
    # Visualize the data
    ShowData(x2, x1, color)

    # Select a linear model
    model = sklearn.linear_model.LinearRegression()
    
    # # Train the model
    model.fit(x, x1)

    # Make a prediction
    x_test = SepTestData(x_test)
    TestModel(x_test, model)
    plt.show()

if __name__ == '__main__':
   
    # Load and prepare the data
    sales_data = pd.read_excel('Homework_1/mlr05.xls', 'Mlr05')
    x1 = pd.DataFrame(sales_data, columns=['X1'])
    x2 = pd.DataFrame(sales_data, columns=['X2'])
    x2_test = pd.DataFrame(sales_data, columns=['X2_test'])
    x3 = pd.DataFrame(sales_data, columns=['X3'])
    x3_test = pd.DataFrame(sales_data, columns=['X3_test'])
    x4 = pd.DataFrame(sales_data, columns=['X4'])
    x4_test = pd.DataFrame(sales_data, columns=['X4_test'])
    x5 = pd.DataFrame(sales_data, columns=['X5'])
    x5_test = pd.DataFrame(sales_data, columns=['X5_test'])
    x6 = pd.DataFrame(sales_data, columns=['X6'])
    x6_test = pd.DataFrame(sales_data, columns=['X6_test'])

    print("Model for 'X2' data:")
    LinearReg(x1, x2, x2_test, 'ro')
    print("Model for 'X3' data:")
    LinearReg(x1, x3, x3_test, 'bo')
    print("Model for 'X4' data:")
    LinearReg(x1, x4, x4_test, 'go')
    print("Model for 'X5' data:")
    LinearReg(x1, x5, x5_test, 'co')
    print("Model for 'X6' data:")
    LinearReg(x1, x6, x6_test, 'mo')    
