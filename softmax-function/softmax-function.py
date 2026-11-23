import numpy as np

def softmax(x: list) -> np.ndarray:
    """
    Returns stable softmax probabilities as a NumPy array matching the shape of x.
    """
    # Write code here
    a= np.array(x)
    if a.ndim==1:
        shifted= a-np.max(a)
        exponent= np.exp(shifted)
        return exponent/np.sum(exponent)

    shifted= a-np.max(a, axis= 1, keepdims= True)
    exponent= np.exp(shifted)
    return exponent/np.sum(exponent, axis= 1, keepdims= True)