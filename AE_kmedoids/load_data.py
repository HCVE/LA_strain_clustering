import os
import numpy as np
import pandas as pd

"""
This module contains functions to read the strain, ECG and time data available in txt format. 
The functions "read_avc_time" and "read_p_wave_data" are called optionally in case of very noisy recordings and a manual
annotation from an expert is necessary. 
"""


# Read the patient id and the intervals indicated at the .txt files available for each patient
def read_info(path):
    patient_id = [file.split('_')[0] for file in os.listdir(path)]

    interval = []
    for file in os.listdir(path):
        with open(os.path.join(path, file)) as f:
            head = [next(f) for i in range(3)]
            split_line = head[2].split("Time=")
            interval.append((float(split_line[1][0:6]), float(split_line[2][0:6])))
    return patient_id, interval


# Read the .xlsx files (if available) which annotate the aortic valve closure times for each patient,
# as recorded from echopath.
def read_avc_time(path, files):
    filenames = [os.path.join(path, file) for file in files]
    df = pd.DataFrame()
    for f in filenames:
        df = df.append(pd.read_excel(f, usecols=(0, 1)), ignore_index=True)

    duplicates = df[df.duplicated(['id'], keep='last')]
    df = df.drop(duplicates['id'].index.tolist())

    exclude_patients = df[df['AVC (ms)'].isnull()].index.tolist()
    exclude_patients_id = df.iloc[exclude_patients]["id"]
    df = df.drop(exclude_patients)
    avc = df.set_index('id').to_dict()
    return exclude_patients_id.values, avc


# Read the .xlsx file (if available) which annotate the duration of the PQ complex
# and the duration of P wave, as recorded from echopath.
# They are used to correctly assign the start and the peak of the P-wave in ECGs with noise or poor quality.
def read_p_wave_data(path, files):
    filenames = [os.path.join(path, file) for file in files]
    df = pd.DataFrame()
    for f in filenames:
        df = df.append(pd.read_excel(f, usecols=(0, 1, 2)), ignore_index=True)
    exclude_patients = df[df['PQ interval (ms)'].isnull()].index.tolist()
    exclude_patients_id = df.iloc[exclude_patients]["ID"]
    df = df.drop(exclude_patients)
    return exclude_patients_id.values, df


# Read the .txt files containing the strain and ECG measurements for each patient
def read_data(path):
    patients = [os.path.join(path, file) for file in os.listdir(path)]
    data = [np.loadtxt(f, delimiter="\t", skiprows=4, usecols=(0, -2, -1)).T for f in patients]

    patient_id, interval = read_info(path)
    filtered_data = []
    for i, d in enumerate(data):
        start_index = np.where(d[0] == interval[i][0])[0]
        end_index = np.where(d[0] == interval[i][1])[0]
        filtered_data.append(d[:][:, start_index[0]:end_index[0] + 1])

    return data, filtered_data, patient_id, interval
