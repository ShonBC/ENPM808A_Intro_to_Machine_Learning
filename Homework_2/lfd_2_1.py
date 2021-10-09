import numpy as np

def NumExamples(m):
    """Calculates the minimum number of examples required given an m M-factor.

    Args:
        m (int): M-Factor to be used in the generalization bound equation.
    """

    e = 0.05
    d = 0.03

    # num_examples = (8 / (e**2)) * np.log((2 * m) / d)
    num_examples = (1 / (2 * e**2)) * np.log((2 * m) / d)


    print(f"M = {m} requires {num_examples} examples.")

NumExamples(1)
NumExamples(100)
NumExamples(10000)