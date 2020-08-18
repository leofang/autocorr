import numpy as np


def fftautocorr(signal):
    """Autocorrelation of a signal using FFT.

    Parameters
    ----------
    signal: 2-D array
        input signal, autocorrelation is calculated along `1-axis`

    Returns:
    --------
    autocorrelations: numpy.ndarray
        should be self-explanatory

    """
    # 1-D data hack
    if signal.ndim == 1:
        N = signal.shape[0]
        a = signal[np.newaxis, :].copy()
    elif signal.ndim == 2:
        # copy data a local array
        n, N = signal.shape
        a = signal.copy()
    elif signal.ndim > 2:
        raise ValueError(
            'Flatten dimensions before passing to autocorrelate.')
    a = np.pad(a, ((0, 0), (0, N)), 'constant', constant_values=0)

    # calculate autocorrelations for all the lag-times
    #ntimes = np.arange(N)
    g2 = np.fft.ifft(np.abs(np.fft.fft(a, axis=1))**2, axis=1).real
    #g2 = g2[:, :N] #/ np.arange(N, 0, -1)
 
    #norm = np.array([
    #    np.mean(signal[:, :N - i], axis=1) * np.mean(signal[:, i:], axis=1)
    #    for i in range(N)
    #]).T
    temp = np.zeros_like(g2)
    #for i in range(N):
    #    temp[:,i] = np.mean(signal[:,i], axis=1)
    total_sum = np.sum(signal, axis=1)

    norm = np.zeros_like(g2)
    left = total_sum
    right = total_sum.copy()
    norm[:,N-1] = total_sum * total_sum / (N - 1)
    for i in range(N-1):
        #breakpoint()
        left -= signal[:, N-1-i]
        right -= signal[:, i]
        norm[:, i] = left * right / (N - i)
    #norm = np.zeros_like(g2)
    #for i in range(N):
    #    norm[:, i] = np.mean(signal[:, :N - i], axis=1) * np.mean(signal[:, i:], axis=1) * (N - i)
    
    return g2 / norm  #, ntimes
