import numpy as np
from scipy.optimize import minimize
from functools import partial

def calc_E_with_length(theta, norm):
    '''Compute the E value with fixed length of (du, dv)
    '''
    u, v = norm*np.sin(theta), norm*np.cos(theta)
    return np.exp(u) + np.exp(2.0*v) + np.exp(u*v) + u**2 - 3*u*v + 4*v**2 - 3*u - 5*v

calc_E_half = partial(calc_E_with_length, norm=0.5)
x0 = 0
res = minimize(calc_E_half, x0, method='nelder-mead', options={'xatol': 1e-8, 'disp': True})
print('Optimal direction by minimizing E(u+du,v+dv): ', 0.5*np.sin(res.x), 0.5*np.cos(res.x))
print('Minimal E(u+du, v+dv): ', res.fun)