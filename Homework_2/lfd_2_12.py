import numpy as np

def get_N(dvc=10, delta=0.05, epsilon=0.05, initial_N=1000, tolerance = 1):
    """Uses recursion to iterate N until it converges within a tolerance
       
       Args: dvc = VC dimension
             delta = 1 - %confidence
             epsilon = generalization error
             initial_N = initial guess of sample size 
             tolerance = constraint at which to stop the recursion and state convergence
             
        Returns: N = Number of samples required
    
    """
    
    new_N = 8 / epsilon**2 * np.log((4 * ((2 * initial_N) ** dvc + 1)) / delta) # formula to generate new N
    
    if abs(new_N - initial_N) < tolerance: # Did it converge within a specific tolerance?
        return new_N
          
    else: # If so return N
        return get_N(dvc, delta, epsilon, new_N, tolerance) # If not iterate with new N

print("The closest numerical approximation of the minimum sample "\
      f"size that the VC generalization bound predicts is {get_N()}")

