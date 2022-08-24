import numpy as np
import detect_waves as dw
from itertools import tee
from types import SimpleNamespace as box


def pairwise(iterable):
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


def sliceup(data, start, ending, ID, AVC, df_q_wave_times, point):
    # take the time(t), the strain(s) and the ecg(e) of each patient
    t, s, e = data[0], data[1], data[2]

    # locate the peaks of the ECG signal. "Point" indicates if the peak or the start of the peak wave will be used.
    peaks = dw.select_peaks(e, start, ending, int(ID), AVC, t, df_q_wave_times, point)

    t, s, e = t[start:ending + 1], s[start:ending + 1], e[start:ending + 1]
    peaks = peaks[start < peaks].astype(int)
    peaks = peaks - start
    peaks = peaks[peaks < len(t)].astype(int)

    peaks = np.insert(peaks, 0, 0)
    peaks = np.append(peaks, len(e)-1)
    slices = [(t[p1:p2+1], s[p1:p2+1], e[p1:p2+1]) for p1, p2 in pairwise(peaks)]
    return box(t=t, s=s, e=e, peaks=peaks, slices=slices)


def mima(a):
    return a.min(), a.max()


def stretch(data, ref):
    # this functions takes the slices as indicated by the peaks variable
    # and performs interpolation so that the markers of the signal to be aligned,
    # are exactly at the same point of the respective marker in the reference signal
    while len(data.slices) > len(ref.slices):
        position = np.argmin([len(data.slices[i][0]) for i in range(len(data.slices))])
        del data.slices[position]

    data.rescaled_slices = [None] * len(data.slices)
    for i, (d, r) in enumerate(zip(data.slices, ref.slices)):
        td = np.interp(d[0], mima(d[0]), mima(r[0]))
        data.rescaled_slices[i] = (td, d[1], d[2])
    return box(t=data.t, s=data.s, e=data.e, peaks=data.peaks, slices=data.slices, rescaled_slices=data.rescaled_slices)
