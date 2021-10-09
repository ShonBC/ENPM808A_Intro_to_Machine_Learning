import numpy as np

def get_a_hat_list(K = 10000):
    """Returns a list of a_hats based on the
       number of random samples K
       
       Args: 
           K: number of random samples
       Returns:
           a_hat_list: an array all the a_hats
    """
    # a_hat_list = np.zeros(K)
    
    x1 = np.random.uniform(-1, 1, K) # random 1st point for K trials
    y1 = np.sin(np.pi * x1)
    x2 = np.random.uniform(-1, 1, K) # random 2nd point for K trials
    y2 = np.sin(np.pi * x2)

    a_hat_list_1 = h_1(x1, x2, y1, y2)
    a_hat_list_2 = h_2(x1, x2, y1, y2)
    a_hat_list_3 = h_3(x1, x2, y1, y2)
    
    return a_hat_list_1, a_hat_list_2, a_hat_list_3

def h_1(x1, x2, y1, y2):
    """
    h = ax + b
    """
    return (y2 - y1) / (x2 - x1) # K a_hats

def h_2(x1, x2, y1, y2):
    """
    h = ax
    """

    return (x1 * y1 + x2 * y2) / (x1 ** 2 + x2 ** 2) # K a_hats

def h_3(x1, x2, y1, y2):
    '''
    h = b
    '''
    return (y1 + y2) / 2 # K a_hats



def bias_variance(g, a_hat_list, a_bar):

    x_range = np.linspace(-1, 1, 1000)
    bias = np.mean((g * x_range - np.sin(np.pi * x_range))**2)
    # print("bias = {}".format(np.round(bias, 2)))

    variance = np.mean((np.outer(a_hat_list, x_range) - a_bar * x_range)**2)
    # print("variance = {}".format(np.round(variance, 2)))

    return np.round(bias, 2), np.round(variance, 2)



a_hat_list_1, a_hat_list_2, a_hat_list_3 = get_a_hat_list() # save for a_bar, g_bar, bias and variance
a_bar_1 = np.mean(a_hat_list_1)
a_bar_2 = np.mean(a_hat_list_2)
a_bar_3 = np.mean(a_hat_list_3)

g1 = round(a_bar_1, 5)
g2 = round(a_bar_2, 5)
g3 = round(a_bar_3, 5)

print(f"h = ax + b --> g_bar(x) = {g1}x bias, variance = {bias_variance(g1, a_hat_list_1, a_bar_1)}")
print(f"h = ax --> g_bar(x) = {g2}x bias, variance = {bias_variance(g2, a_hat_list_2, a_bar_2)}")
print(f"h = b --> g_bar(x) = {g3}x bias, variance = {bias_variance(g3, a_hat_list_3, a_bar_3)}")
