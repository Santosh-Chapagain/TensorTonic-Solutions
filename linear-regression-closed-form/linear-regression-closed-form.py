import numpy as np

def linear_regression_closed_form(X: list, y: list) -> list:
    """
    Returns the optimal weight vector as a list.
    """
    # Write code here
    a= np.array(X)
    b= np.array(y)
    trans_a= a.T 
    first_part= np.linalg.inv(np.dot(trans_a, a))
    second_part= np.dot(trans_a, b)
    return np.dot(first_part, second_part)