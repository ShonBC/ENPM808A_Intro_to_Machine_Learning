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
import pandas as pd
import sklearn.linear_model

def ShowData(x, y, color):

    plt.plot(x, y, color)
    plt.xlabel("Target Values")
    plt.ylabel("Features")

def ImportData(sales_data, col_title):

    x = pd.DataFrame(sales_data, columns=[col_title])
    test_data =[]
    i = 20

    while i <= len(x):

        test_data.append(x.pop(i))
    
    return x, test_data


if __name__ == '__main__':
    # Load the data
    sales_data = pd.read_excel('Homework_1/mlr05.xls', 'Mlr05')
    x1 = pd.DataFrame(sales_data, columns=['X1'])
    x2 = pd.DataFrame(sales_data, columns=['X2'])
    x3 = pd.DataFrame(sales_data, columns=['X3'])
    x4 = pd.DataFrame(sales_data, columns=['X4'])
    x5 = pd.DataFrame(sales_data, columns=['X5'])
    x6 = pd.DataFrame(sales_data, columns=['X6'])
    # print(x1)
    # print(x2)

    # Prepare the data


    # Visualize the data
    ShowData(x2, x1, 'ro')
    # ShowData(x3, x1, 'bo')
    # ShowData(x4, x1, 'go')
    # ShowData(x5, x1, 'co')
    # ShowData(x6, x1, 'mo')
    plt.show()

    # Select a linear model
    model2 = sklearn.linear_model.LinearRegression()
    model3 = sklearn.linear_model.LinearRegression()
    model4 = sklearn.linear_model.LinearRegression()
    model5 = sklearn.linear_model.LinearRegression()
    model6 = sklearn.linear_model.LinearRegression()

    # # Train the model
    model2.fit(x2,x1)
    model3.fit(x3,x1)
    model4.fit(x4,x1)
    model5.fit(x5,x1)
    model6.fit(x6,x1)

    # Make a prediction
    X_new = [[4]]
    print(model2.predict(X_new))
