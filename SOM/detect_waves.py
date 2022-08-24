from scipy.signal import find_peaks
from scipy.stats import linregress
import numpy as np


def work_with_peaks(ecg, start, ending):
    
    # find all the peaks in the ECG signal.
    # It may contains wrong data points
    peaks, _ = find_peaks(ecg)

    # select the peak that is closer to markers
    # markers are always closer to the peak of the R-waves
    absolute_diff = np.abs(peaks-start)
    smallest_diff_ind = np.argmin(absolute_diff)
    first_r_peak = peaks[smallest_diff_ind]

    # if the first peak that the function has located is the first R-wave
    # then there is no p-wave at the beginning of the signal and thus it set to
    # None
    # If it isn't there might be a possibility of having a P-wave at the beginning
    # of the signal or having a noisy sample. If it's noise the magnitude of the
    # peak should be smaller than the first values 
    if smallest_diff_ind-1 >= 0:
        first_p_wave_ind = np.argmax(ecg[peaks[0:smallest_diff_ind]])
        first_p_wave = peaks[first_p_wave_ind]
        if ecg[first_p_wave] < ecg[0]:
            first_p_wave = None
    else:
        first_p_wave = None

    # to find the second R-wave begin from the highest peaks
    # the first highest peak that it is in the second half of the signal,
    # close to the ending marker indicated at the txt file
    # and has considerably high value (above 0.4) is the R-peak
    # in any other case there isn't any R-wave
    sorted_indexes = np.argsort(ecg[peaks])
    second_r_peak = None
    for i in range(len(sorted_indexes)-1, -1, -1):
        if peaks[sorted_indexes[i]] > len(ecg)/2 and abs(peaks[sorted_indexes[i]]-ending) < 5 and \
                ecg[peaks[sorted_indexes[i]]] > 0.4:
            second_r_peak = peaks[sorted_indexes[i]]
            break
            
    # for the p-wave, if there isn't any R-wave it set as the highest peak in the second half of the ecg signal
    # second_R_peak!=None, the p-wave is set as the highest peak in the second half of the ecg signal that occurs
    # before the second_R_peak
    second_p_wave = None
    for i in range(len(sorted_indexes)-1, -1, -1):
        if second_r_peak is None:
            if peaks[sorted_indexes[i]] > len(ecg)/2:
                second_p_wave = peaks[sorted_indexes[i]]
                break
        else:
            if len(ecg)/2 < peaks[sorted_indexes[i]] < second_r_peak:
                second_p_wave = peaks[sorted_indexes[i]]
                break

    peaks = np.array([first_p_wave, first_r_peak, second_p_wave, second_r_peak])
    return peaks


def work_with_minima(ecg):
    begin = int(0.2*len(ecg))
    stop = int(len(ecg)/2)

    # find the peak of the T-wave. This it should be around the middle of the first half of the ecg signal
    peaks, _ = find_peaks(ecg[begin:stop])
    peaks = peaks + begin
    
    max_peak_index = np.argmax(ecg[peaks])
    peaks = peaks[max_peak_index]

    # starting from the peak of T-wave, find the point where the slope of the curve changes
    # normally the slope changes from negative to either zero or positive
    start_point = peaks
    slope = [linregress([i+1, i], [ecg[i+1], ecg[i]]).slope for i in range(start_point, stop)]
    assign = np.sign(slope)

    try:
        inverse_t = np.array([start_point + np.where(assign > 0)[0][0]])
    except:
        # if the slope is not clear and the algorithm fails to assign a point
        # set it manually and arbitrarily to middle of the interval of T-wave's peak
        # and the middle of the entire ecg signal
        inverse_t = start_point + int((stop-start_point)/2)
    return inverse_t


def find_inverse_t(patient, start, AVC, time, ecg):
    # read the xlsx file with the times of the AVC. If the data for the patient are not available
    # estimate the AVC point according to work_with_minima function
    try:
        avc_time = (AVC['AVC (ms)'][patient]/1000) + time[start]
        time_diff = np.abs(time-avc_time)
        avc_point = np.argmin(time_diff)
    except:
        avc_point = work_with_minima(ecg)
    return avc_point 


def start_of_p_wave(maxima, ecg):
    
    # from the p-wave peak and going backwards find the point
    # that the slope changes sign
    start_point = maxima[-2]
    slope = [linregress([i, i-1], [ecg[i], ecg[i-1]]).slope for i in range(start_point, int(len(ecg)/2), -1)]
    assign = np.sign(slope)
    try:
        p_wave_start = start_point - np.where(assign <= 0)[0][0]
        # if the slope doesn't change so clear set manually and arbitrarily as the sample immediately before the peak
        # in such cases perhaps manual annotation using the patient's file is needed
    except:
        p_wave_start = start_point - 1

    return p_wave_start


def find_second_q_wave(ecg, maxima):
    begin = int(len(ecg)/2)
    stop = len(ecg)

    local_minima, _ = find_peaks(-ecg[begin:stop])
    local_minima += begin

    absolute_diff = maxima[-1] - local_minima
    smallest_diff_ind = np.where(absolute_diff == min([i for i in absolute_diff if i > 0]))

    return local_minima[smallest_diff_ind]


def select_peaks(ecg, starting, ending, patient, AVC, time, df_q_wave_times, demand):
    # maxima contains the points where the peaks of P- and R-wave are located
    maxima = work_with_peaks(ecg, starting, ending)

    # q-wave before the second R-wave.
    if patient in df_q_wave_times['ID'].values:
        q_wave = find_second_q_wave(ecg, maxima)
        q_wave_time = time[q_wave]
        p_wave_start_sec = q_wave_time - \
                           df_q_wave_times.loc[df_q_wave_times['ID'] == patient, 'PQ interval (ms)'].iloc[0]/1000

        absolute_diff = np.abs(time - p_wave_start_sec)
        p_wave_start = np.argmin(absolute_diff)

        if not np.isnan(df_q_wave_times.loc[df_q_wave_times['ID'] == patient, 'P duration (ms)'].iloc[0]):
            p_wave_time = p_wave_start_sec + \
                          df_q_wave_times.loc[df_q_wave_times['ID'] == patient, 'P duration (ms)'].iloc[0]/2000

            absolute_diff = np.abs(time - p_wave_time)
            maxima[-2] = np.argmin(absolute_diff)
            # maxima[-2] = smallest_diff_ind
    else:
        # "p_wave_start" refers to the starting point of the P-wave before the second
        # R-wave
        p_wave_start = start_of_p_wave(maxima, ecg)

    # inverse_t refers to the point where the Aortic Valve Closure occurs
    # the values are obtained from an annotated file
    inverse_t = find_inverse_t(patient, starting, AVC, time, ecg)

    # determine if you want to use the peak of the P-wave or its starting point
    peaks = [None]*5
    if demand == "peak":
        peaks = np.hstack((maxima[0:2], inverse_t, maxima[-2:len(maxima)]))
    elif demand == "start":
        peaks = np.hstack((maxima[0:2], inverse_t, p_wave_start, maxima[-1]))
    else:
        print('Wrong Input. Please use "peak" or "start" as keywords')
        
    # combine the points into one numpy array.
    peaks = peaks[peaks != np.array(None)].astype(int)
    peaks = np.sort(peaks)
    return peaks
