from scipy import signal
import numpy as np


def data_interpolation(ecg, strain, time, index):
    # normalize the time axis
    norm_time = []
    for i in range(len(time)):
        norm_time.append([(float(j) - min(time[i])) / (max(time[i]) - min(time[i]))
                          for j in time[i]])

    # resample the data to have the same number of samples
    ecg_interp = []
    strain_interp = []
    time_interp = []
    for i in range(len(time)):
        # ECG
        interp = signal.resample(ecg[i], 115)
        ecg_interp.append(interp)

        # Strain
        interp = signal.resample(strain[i], 115)
        strain_interp.append(interp)

        # Time
        time_interp = np.linspace(0, 1, 115, endpoint=False)
        # time_interp.append(interp)

    return ecg_interp, strain_interp, index, norm_time, time_interp
