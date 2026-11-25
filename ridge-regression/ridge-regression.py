import numpy as np

def ridge_regression(X: list, y: list, lam: float) -> list:
    """
    Returns the ridge-regression weight vector.
    """
    # Write code here
    a= np.array(X)
    b= np.array(y)

    trans_a= a.T
    first_sum= np.dot(trans_a, a) + lam * np.identity(a.shape[1])
    sum_inv= np.linalg.inv(first_sum)
    second_part= np.dot(trans_a, b) 
    return np.dot(sum_inv, second_part)