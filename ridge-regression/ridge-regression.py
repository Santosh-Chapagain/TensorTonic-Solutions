import numpy as np

def ridge_regression(X: list, y: list, lam: float) -> list:
    """
    Returns the ridge-regression weight vector.
    """
    # Write code here
    a= np.array(X)
    b= np.array(y)

    trans_a= a.T
    first_sum= np.linalg.inv(np.dot(trans_a, a) + lam * np.identity(a.shape[1]))
    second_part= np.dot(trans_a, b) 
    return np.dot(first_sum, second_part)