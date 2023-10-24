import os, sys
import time
import numpy as np
from scipy.io import wavfile
from scipy.fft import fft, rfft, fftshift, fftfreq
from scipy.signal import convolve, freqz
import matplotlib.pyplot as plt

# ================================================ STAGE 1 ================================================

def to_min_size_int_array(arr):
    """
    Convert the provided numpy array to a compatible numpy integer array with the minimum size possible

    :param arr: numpy array of floating point type
    """

    if np.min(arr) >= np.iinfo(np.int8).min and np.max(arr) <= np.iinfo(np.int8).max:
        print(f"converting from {arr.dtype} to {np.int8}; array ranges from min:  {np.min(arr)} (>={np.iinfo(np.int8).min})  to max: {np.max(arr)} (<={np.iinfo(np.int8).max})")
        return arr.astype(np.int8)
    if np.min(arr) >= np.iinfo(np.int16).min and np.max(arr) <= np.iinfo(np.int16).max:
        print(f"converting from {arr.dtype} to {np.int16}; array ranges from min: {np.min(arr)} (>={np.iinfo(np.int16).min}) to max: {np.max(arr)} (<={np.iinfo(np.int16).max})")
        return arr.astype(np.int16)
    if np.min(arr) >= np.iinfo(np.int32).min and np.max(arr) <= np.iinfo(np.int32).max:
        print(f"converting from {arr.dtype} to {np.int32}; array ranges from min: {np.min(arr)} (>={np.iinfo(np.int32).min}) to max: {np.max(arr)} (<={np.iinfo(np.int32).max})")
        return arr.astype(np.int32)
    
    # if np array is out of the ranges of np.int8, int16, and int32 integer data types, then raise an error
    print(f"array ranges from min: {np.min(arr)} to max: {np.max(arr)}")
    raise Exception(f"the array elements are too big to handle; {arr.dtype}")

def Spectrum(x, sampling_space: int=1, type: str='magnitude'):
    """
    Returns the spectrum of a given real-valued signal, 'x'. 

    :param x: the signal to find the spectrum 
    :param sampling_space: sampling period (the reciprocal of sampling frequency); defaults to 1
    :param type: specifies the output representation of the spectrum; defaults to 'magnitude'
        options:
        - 'magnitude': magnitude spectrum
        - 'phase':     phase spectrum (in radians)
        - 'complex':   complex-valued spectrum

    """
    spectrum = rfft(x) # note here, we use 'rfft' instead of 'fft'
    freq_bins = fftshift(fftfreq(len(x), sampling_space))
    freq_bins = freq_bins[-len(spectrum):] # get the frequency bins for half of the spectrum

    if type == 'magnitude':
        spectrum = np.abs(spectrum)
    elif type == 'phase':
        spectrum = np.angle(spectrum) # in radians
    elif type == 'complex':
        pass
    else:
        raise KeyError(f"{type} is not a valid option for the param 'type'.")

    return freq_bins, spectrum

def power(signal):
    """
    Return the power of the signal (arr) without normalization (i.e. 1/n is not performed).
    This function can be used to compare the power of a signal both in time and frequency domains (see Parseval's Theorem).
    """
    # signals are typically given in np.int16 datatype, but their square values do not usually fit into this same datatype. 
    # so, we have to adjust the datatype of the input signal arrays into np.int32 to avoid any possible overflows and unexpected values for the signal power. 
    return np.sum(np.abs(signal.astype(np.int32)) ** 2)

def unpackbits(sample_array):
    "unpack a numpy array of samples into a stram of bits (str) according to the datatype/precision of the array elements"
    if   sample_array.dtype == np.int32: precision = 32
    elif sample_array.dtype == np.int16: precision = 16
    elif sample_array.dtype == np.uint8: precision =  8
    else:
        raise TypeError("the numpy datatype of the provided array is not compatible.")

    # format_ = lambda sample: np.array(list((format(sample, f'0>{precision}b')))).astype(np.uint8)
    format_ = lambda sample: np.array(list(f"{sample & ((1 << precision) - 1):0{precision}b}")).astype(np.int8)
    # some references:
    # - python format function: 'http://www.trytoprogram.com/python-programming/python-built-in-functions/format/'
    # - using format with numpy arrays - 'https://stackoverflow.com/a/52824395/21146493'
    # - implementing two's complement in python - 'https://stackoverflow.com/a/63274998/21146493'

    return np.array(list(map(format_, sample_array)))

def biterrorcount(target_bit_stream, received_bit_stream):
    "calculates the bit error count of the received bit stream compared to the target bit stream"
    return np.count_nonzero(np.bitwise_xor(target_bit_stream, received_bit_stream))


# ================================================ STAGE 2 ================================================

def LPF(N: int, cut_off_freq: int|float, sampling_freq: int|float = 1):
    """
    Implements a Low Pass Filter with the specified cut-off frequency and the filter length using the method of direct truncation of Fourier Series. 
    (refer to the section 9.3 of the book 'Digital Signal Processing, signals systems and filters')

    :param N: filter length; must be an odd integer; if an even integer is assigned, an error will be raised. 
    :param cut_off_freq: cut-off frequency of the filter; must be in the same units as 'sampling_freq'.
    :paramm sampling_freq: sampling frequency of the signal; defaults to 1. 
    """

    if N%2==0:
        # if N is even, raise an error
        raise Exception("the filter length must be odd.")
    
    n = np.arange(-(N-1)/2, (N-1)/2+1)
    filter = 2*cut_off_freq/sampling_freq * np.sinc(2*cut_off_freq/sampling_freq * n)

    return n, filter

def apply_filter(filter, signal):
    """
    Applies the given filter on the provided signal and returns the filtered signal.
    Filtered signal would have the same length as the input signal; make sure that the provided signal is larger than the filter. 

    :param filter: an array containing the coefficients of the FIR filter
    :param signal: an array corresponding to the signal sequence 
    """

    return convolve(signal, filter, mode='same')

def mean_L1_dist(sig1, sig2):
    """
    Returns the L1 distance (Manhattan distance) of the two signals provided: sig1 and sig2.
    The two signals must of the same length; if not, a broadcasting error will be raised. 

    :param sig1: a numpy array representing the first signal
    :param sig2: a numpy array representing the second signal 
    """
    return np.sum(np.abs(sig1 - sig2)) / len(sig1)

def mean_L2_dist(sig1, sig2):
    """
    Returns the L2 distance (Euclidean) of the two signals provided: sig1 and sig2.
    The two signals must of the same length; if not, a broadcasting error will be raised. 

    :param sig1: a numpy array representing the first signal
    :param sig2: a numpy array representing the second signal 
    """
    return np.sqrt(np.sum((sig1 - sig2)**2) / len(sig1)) # RMS

def mean_Lp_dist(sig1, sig2, p: int):
    """
    Returns the Lp distance of the two signals provided: sig1 and sig2.
    The two signals must of the same length; if not, a broadcasting error will be raised. 

    :param sig1: a numpy array representing the first signal
    :param sig2: a numpy array representing the second signal 
    """
    return (np.sum(np.abs(sig1 - sig2)**p) / len(sig1))**(1/p)
