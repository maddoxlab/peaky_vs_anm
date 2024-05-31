#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PREDICT EEG SIGNAL WITH KERNELS

Created on Tue Nov 23 09:49:29 2021

@author: tong
"""

import numpy as np
import scipy.signal as signal
from numpy.fft import irfft, rfft, fft, ifft
import matplotlib.pyplot as plt
import h5py
import scipy.stats as stats

from expyfun.io import read_wav, read_tab
from expyfun.io import write_hdf5, read_hdf5
import mne


import matplotlib.pyplot as plt

# %% Function

def butter_highpass(cutoff, fs, order=1):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a


def butter_highpass_filter(data, cutoff, fs, order=1, axis=-1):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.lfilter(b, a, data, axis)
    return y


def butter_lowpass(cutoff, fs, order=1):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=1, axis=-1):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = signal.lfilter(b, a, data, axis)
    return y


def butter_bandpass(lowcut, highcut, fs, order=1):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=1, axis=-1):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = signal.lfilter(b, a, data, axis)
    return y

def get_abr_range(abr_response, time_lags, time_range):
    """
    input:
        abr_response: derived abr response
        time_lags: time vector
        time_range: in which time range to find the peaks [start, end] in ms
    output:
        time_vec: time vector
        response: response in the specific range
    """

    abr_response = abr_response
    start_time = time_range[0]
    end_time = time_range[1]
    ind = np.where((time_lags >= start_time) & (time_lags <=end_time))[0]
    time_vec = time_lags[ind]
    response = abr_response[ind]
    return time_vec, response

# %% input subject and experiment information
n_jobs = 'cuda'
subject_list = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20,
            21, 22, 23, 24]
# NOTE: s1 only had 1st40 rpt (pilot), s15 not completed (equipment problems)
eeg_2files = [2, 4]  # when there are 2 eeg files (had to change battery)
subject_num = len(subject_list)
# saving options
overwrite_file = True

# paths and root filenames
opath = "/media/tshan@urmc-sh.rochester.edu/Elements/AMPLab/peaky_vs_ANM/exp1/"
# NOTE: combine all trials into the same folder of each narrator
stim_path = opath + 'present_files/Exp1_Stimuli_{}_Narrator/'
stim_root = '{}Narrator{:03}_bands.hdf5'
eeg_path = opath + 'Exp1_Peaky-vs-Unaltered_Speech_Dataset/sub-{:02}/eeg/'
eeg_root = 'sub-{:02}_task-{}'

save_path = opath + 'subject_response/'
#ANM_path = opath + '/present_files/ANM/'
eeg_root = 'sub-{:02}_task-{}'


# %%
n_stim = 2 # "unaltered", "broadband"
len_stim = 64.
eeg_fs = 10e3
n_epoch=40
task = 'peakyvsunaltered'

t_start = -0.2
t_stop = 0.6
lags = np.arange(start=t_start*1000, stop=t_stop*1000, step=1e3/eeg_fs)
len_eeg = int(len_stim*eeg_fs)

shift = 3.4 # or 2.75
window = 0.2
# %% ANM
##### LOADING KERNELS #####
regressor_list = ['rect','pulse','ANM']

for regressor in regressor_list:
    print("==="+regressor+"===")
    
    if regressor == 'ANM':
        data = read_hdf5(opath + 'subject_response/' + regressor + '_response_all_subjects.hdf5')
    else:
        data = read_hdf5(opath + 'subject_response/' + regressor + '_response_all_subjects_new.hdf5')
        
    # Unaltered 
    unaltered_response = data[regressor+'_unaltered_response_all']
    if regressor == 'ANM':
        # shift back 2.75 ms
        unaltered_response = np.roll(unaltered_response, int(-shift*eeg_fs/1000), axis=1)
    unaltered_kernel = np.zeros((subject_num, int(window*eeg_fs)))
    for si in range(subject_num):
        time_vec, unaltered_kernel[si,:] = get_abr_range(unaltered_response[si], lags, [0, int(window*1000)])
    
    # Broadband    
    broadband_response = data[regressor+'_broadband_response_all']
    if regressor == 'ANM':
        # shift back 2.75 ms
        broadband_response = np.roll(broadband_response, int(-shift*eeg_fs/1000), axis=1)
    broadband_kernel = np.zeros((subject_num, int(window*eeg_fs)))
    for si in range(subject_num):
        time_vec, broadband_kernel[si,:] = get_abr_range(broadband_response[si], lags, [0, int(window*1000)])
    
    ##### LOADING REGRESSORS #####
    regressor_path = opath + 'present_files/' + regressor + "/"
    t_start = 0
    t_stop = window
    
    # Unaltered
    print("===Unaltered===")
    out_unaltered_pos = np.zeros((subject_num, n_epoch, len_eeg))
    out_unaltered_neg = np.zeros((subject_num, n_epoch, len_eeg))
    out_unaltered_predicted = np.zeros((subject_num, n_epoch, len_eeg))
    
    data = read_hdf5(regressor_path + 'unaltered_x_in.hdf5')
    # Load x_in
    if regressor == 'pulse':
        x_in_pos = data['x_in_unaltered']
        x_in_neg = data['x_in_unaltered']
    else:
        x_in_pos = data['x_in_unaltered_pos']
        x_in_neg = data['x_in_unaltered_neg']
    
    for si in range(subject_num):
        print(si)
        #### fft ####
        # zero pad kernels
        unaltered_kernel_sub = np.zeros(len_eeg)
        unaltered_kernel_sub[int(t_start*eeg_fs):int(t_stop*eeg_fs)] = unaltered_kernel[si,:]
        x_out_pos = np.zeros(x_in_pos.shape)
        x_out_neg = np.zeros(x_in_neg.shape)
        x_predict = np.zeros(x_in_neg.shape)
        # fft multiplication
        for ei in range(n_epoch):
            print("unaltered {}".format(ei))
            x_out_pos[ei, :] = np.fft.ifft(np.fft.fft(x_in_pos[ei,:])*np.fft.fft(unaltered_kernel_sub)).real
            x_out_neg[ei, :] = np.fft.ifft(np.fft.fft(x_in_neg[ei,:])*np.fft.fft(unaltered_kernel_sub)).real
            x_predict[ei, :] = (x_out_pos[ei, :] + x_out_neg[ei, :]) / 2
        out_unaltered_pos[si,:,:] = x_out_pos
        out_unaltered_neg[si,:,:] = x_out_neg
        out_unaltered_predicted[si,:,:] = x_predict
    
    # Broadband
    print("===Unaltered===")
    out_broadband_pos = np.zeros((subject_num, n_epoch, len_eeg))
    out_broadband_neg = np.zeros((subject_num, n_epoch, len_eeg))
    out_broadband_predicted = np.zeros((subject_num, n_epoch, len_eeg))
    
    data = read_hdf5(regressor_path + 'broadband_x_in.hdf5')
    # Load x_in
    if regressor == 'pulse':
        x_in_pos = data['x_in_broadband']
        x_in_neg = data['x_in_broadband']
    else:
        x_in_pos = data['x_in_broadband_pos']
        x_in_neg = data['x_in_broadband_neg']
    
    for si in range(subject_num):
        print(si)
        #### fft ####
        # zero pad kernels
        broadband_kernel_sub = np.zeros(len_eeg)
        broadband_kernel_sub[int(t_start*eeg_fs):int(t_stop*eeg_fs)] = broadband_kernel[si,:]
        x_out_pos = np.zeros(x_in_pos.shape)
        x_out_neg = np.zeros(x_in_neg.shape)
        x_predict = np.zeros(x_in_neg.shape)
        # fft multiplication
        for ei in range(n_epoch):
            print("broadband {}".format(ei))
            x_out_pos[ei, :] = np.fft.ifft(np.fft.fft(x_in_pos[ei,:])*np.fft.fft(broadband_kernel_sub)).real
            x_out_neg[ei, :] = np.fft.ifft(np.fft.fft(x_in_neg[ei,:])*np.fft.fft(broadband_kernel_sub)).real
            x_predict[ei, :] = (x_out_pos[ei, :] + x_out_neg[ei, :]) / 2
        out_broadband_pos[si,:,:] = x_out_pos
        out_broadband_neg[si,:,:] = x_out_neg
        out_broadband_predicted[si,:,:] = x_predict
    
    #  Save file
    predicted_eeg_path = opath + '/predicted_eeg/' 
    # write_hdf5(predicted_eeg_path + regressor + '_predict_x_out_new.hdf5',
    #            dict(out_unaltered_pos=out_unaltered_pos, out_unaltered_neg=out_unaltered_neg, 
    #                 out_unaltered_predicted=out_unaltered_predicted,
    #                 out_broadband_pos=out_broadband_pos, out_broadband_neg=out_broadband_neg,
    #                 out_broadband_predicted=out_broadband_predicted),
    #                 overwrite=True)
    write_hdf5(predicted_eeg_path + regressor + '_predict_x_out_new_200ms.hdf5',
               dict(out_unaltered_predicted=out_unaltered_predicted,
                    out_broadband_predicted=out_broadband_predicted),
                    overwrite=True)


# %% Rectified
##### LOADING KERNELS #####
regressor = 'rect'
# Unaltered 
rect_data = read_hdf5(opath + 'subject_response/' + regressor + '_response_all_subjects_new.hdf5')
rect_unaltered_response_ave = rect_data['rect_unaltered_response_ave']
time_vec, rect_unaltered_kernel = get_abr_range(rect_unaltered_response_ave, lags, [0, 250])

# Broadband    
rect_broadband_response_ave = rect_data['rect_broadband_response_ave']
time_vec, rect_broadband_kernel = get_abr_range(rect_broadband_response_ave, lags, [0, 250])

##### LOADING REGRESSORS #####
rect_path = opath + 'present_files/' + regressor + "/"
t_start = 0
t_stop = 0.25
# Unaltered
out_unaltered_pos = np.zeros((n_epoch, len_eeg))
out_unaltered_neg = np.zeros((n_epoch, len_eeg))
out_unaltered_predicted = np.zeros((n_epoch, len_eeg))

data = read_hdf5(rect_path + 'unaltered_x_in.hdf5')
# Load x_in
x_in_pos = data['x_in_unaltered_pos']
x_in_neg = data['x_in_unaltered_neg']

#### fft ####
# zero pad kernels
unaltered_kernel = np.zeros(len_eeg)
unaltered_kernel[int(t_start*eeg_fs):int(t_stop*eeg_fs)] = rect_unaltered_kernel
x_out_pos = np.zeros(x_in_pos.shape)
x_out_neg = np.zeros(x_in_neg.shape)
x_predict = np.zeros(x_in_neg.shape)
# fft multiplication
for ei in range(n_epoch):
    print("unaltered {}".format(ei))
    x_out_pos[ei, :] = np.fft.ifft(np.fft.fft(x_in_pos[ei,:])*np.fft.fft(unaltered_kernel)).real
    x_out_neg[ei, :] = np.fft.ifft(np.fft.fft(x_in_neg[ei,:])*np.fft.fft(unaltered_kernel)).real
    x_predict[ei, :] = (x_out_pos[ei, :] + x_out_neg[ei, :]) / 2
out_unaltered_pos = x_out_pos
out_unaltered_neg = x_out_neg
out_unaltered_predicted = x_predict

# Broadband
out_broadband_pos = np.zeros((n_epoch, len_eeg))
out_broadband_neg = np.zeros((n_epoch, len_eeg))
out_broadband_predicted = np.zeros((n_epoch, len_eeg))

data = read_hdf5(rect_path + 'broadband_x_in.hdf5')
# Load x_in
x_in_pos = data['x_in_broadband_pos']
x_in_neg = data['x_in_broadband_neg']

#### fft ####
# zero pad kernels
broadband_kernel = np.zeros(len_eeg)
broadband_kernel[int(t_start*eeg_fs):int(t_stop*eeg_fs)] = rect_broadband_kernel
x_out_pos = np.zeros(x_in_pos.shape)
x_out_neg = np.zeros(x_in_neg.shape)
x_predict = np.zeros(x_in_neg.shape)
# fft multiplication
for ei in range(n_epoch):
    print("broadband {}".format(ei))
    x_out_pos[ei, :] = np.fft.ifft(np.fft.fft(x_in_pos[ei,:])*np.fft.fft(broadband_kernel)).real
    x_out_neg[ei, :] = np.fft.ifft(np.fft.fft(x_in_neg[ei,:])*np.fft.fft(broadband_kernel)).real
    x_predict[ei, :] = (x_out_pos[ei, :] + x_out_neg[ei, :]) / 2
out_broadband_pos = x_out_pos
out_broadband_neg = x_out_neg
out_broadband_predicted = x_predict

#  Save file
predicted_eeg_path = opath + '/predicted_eeg/' 
write_hdf5(predicted_eeg_path + regressor + '_predict_x_out_new.hdf5',
           dict(out_unaltered_pos=out_unaltered_pos, out_unaltered_neg=out_unaltered_neg, 
                out_unaltered_predicted=out_unaltered_predicted,
                out_broadband_pos=out_broadband_pos, out_broadband_neg=out_broadband_neg,
                out_broadband_predicted=out_broadband_predicted),
                overwrite=True)
# %% Pulse
##### LOADING KERNELS #####
regressor = 'pulse'
# Unaltered 
pulse_data = read_hdf5(opath + 'subject_response/' + regressor + '_response_all_subjects_new.hdf5')
pulse_unaltered_response_ave = pulse_data['pulse_unaltered_response_ave']
time_vec, pulse_unaltered_kernel = get_abr_range(pulse_unaltered_response_ave, lags, [0, 250])

# Broadband    
pulse_broadband_response_ave = pulse_data['pulse_broadband_response_ave']
time_vec, pulse_broadband_kernel = get_abr_range(pulse_broadband_response_ave, lags, [0, 250])

##### LOADING REGRESSORS #####
pulse_path = opath + 'present_files/' + regressor + "/"
t_start = 0
t_stop = 0.25
# Unaltered
out_unaltered_pos = np.zeros((n_epoch, len_eeg))
out_unaltered_neg = np.zeros((n_epoch, len_eeg))
out_unaltered_predicted = np.zeros((n_epoch, len_eeg))

data = read_hdf5(pulse_path + 'unaltered_x_in.hdf5')
# Load x_in
x_in_pos = data['x_in_unaltered']
x_in_neg = data['x_in_unaltered']

#### fft ####
# zero pad kernels
unaltered_kernel = np.zeros(len_eeg)
unaltered_kernel[int(t_start*eeg_fs):int(t_stop*eeg_fs)] = pulse_unaltered_kernel
x_out_pos = np.zeros(x_in_pos.shape)
x_out_neg = np.zeros(x_in_neg.shape)
x_predict = np.zeros(x_in_neg.shape)
# fft multiplication
for ei in range(n_epoch):
    print("unaltered {}".format(ei))
    x_out_pos[ei, :] = np.fft.ifft(np.fft.fft(x_in_pos[ei,:])*np.fft.fft(unaltered_kernel)).real
    x_out_neg[ei, :] = np.fft.ifft(np.fft.fft(x_in_neg[ei,:])*np.fft.fft(unaltered_kernel)).real
    x_predict[ei, :] = (x_out_pos[ei, :] + x_out_neg[ei, :]) / 2
out_unaltered_pos = x_out_pos
out_unaltered_neg = x_out_neg
out_unaltered_predicted = x_predict

# Broadband
out_broadband_pos = np.zeros((n_epoch, len_eeg))
out_broadband_neg = np.zeros((n_epoch, len_eeg))
out_broadband_predicted = np.zeros((n_epoch, len_eeg))

data = read_hdf5(pulse_path + 'broadband_x_in.hdf5')
# Load x_in
x_in_pos = data['x_in_broadband']
x_in_neg = data['x_in_broadband']

#### fft ####
# zero pad kernels
broadband_kernel = np.zeros(len_eeg)
broadband_kernel[int(t_start*eeg_fs):int(t_stop*eeg_fs)] = pulse_broadband_kernel
x_out_pos = np.zeros(x_in_pos.shape)
x_out_neg = np.zeros(x_in_neg.shape)
x_predict = np.zeros(x_in_neg.shape)
# fft multiplication
for ei in range(n_epoch):
    print("broadband {}".format(ei))
    x_out_pos[ei, :] = np.fft.ifft(np.fft.fft(x_in_pos[ei,:])*np.fft.fft(broadband_kernel)).real
    x_out_neg[ei, :] = np.fft.ifft(np.fft.fft(x_in_neg[ei,:])*np.fft.fft(broadband_kernel)).real
    x_predict[ei, :] = (x_out_pos[ei, :] + x_out_neg[ei, :]) / 2
out_broadband_pos = x_out_pos
out_broadband_neg = x_out_neg
out_broadband_predicted = x_predict

#  Save file
predicted_eeg_path = opath + '/predicted_eeg/' 
write_hdf5(predicted_eeg_path + regressor + '_predict_x_out_new.hdf5',
           dict(out_unaltered_pos=out_unaltered_pos, out_unaltered_neg=out_unaltered_neg, 
                out_unaltered_predicted=out_unaltered_predicted,
                out_broadband_pos=out_broadband_pos, out_broadband_neg=out_broadband_neg,
                out_broadband_predicted=out_broadband_predicted),
                overwrite=True)