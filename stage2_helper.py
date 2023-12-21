import numpy as np

def polar2cmplx(r, theta):
    """
    Creates a complex number in the format a+bj when radius and angle in polar form are given.
    :param r: absolute value/modulus/magnitude of the complex number
    :param theta: argument/angle of the complex number

    Returns the corresponding complex number (in the format a+bj). 
    """
    return r * np.exp((0+1j) * theta)