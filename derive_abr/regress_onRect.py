#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 11:38:06 2022

@author: tong
"""

import numpy as np
import scipy.signal as signal
from numpy.fft import irfft, rfft, fft, ifft
import matplotlib.pyplot as plt
import h5py

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


def butter_highpass_filter(data, cutoff, fs, order=1):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = signal.lfilter(b, a, data)
    return y


def butter_lowpass(cutoff, fs, order=1):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = signal.butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=1):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = signal.lfilter(b, a, data)
    return y


def butter_bandpass(lowcut, highcut, fs, order=1):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = signal.butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=1):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = signal.lfilter(b, a, data)
    return y

# %% input subject and experiment information
n_jobs = 'cuda'
subject_list = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20,
            21, 22, 23, 24]
# NOTE: s1 only had 1st40 rpt (pilot), s15 not completed (equipment problems)
eeg_2files = [2, 4]  # when there are 2 eeg files (had to change battery)

# saving options
overwrite_file = True

# paths and root filenames
opath = "/media/tong/Elements/AMPLab/peaky_vs_ANM/exp1/"
# NOTE: combine all trials into the same folder of each narrator
stim_path = opath + 'present_files/Exp1_Stimuli_{}_Narrator/'
stim_root = '{}Narrator{:03}_bands.hdf5'
eeg_path = opath + 'Exp1_Peaky-vs-Unaltered_Speech_Dataset/sub-{:02}/eeg/'
eeg_root = 'sub-{:02}_task-{}'

save_path = opath + 'subject_response/'
rect_path = opath + '/present_files/rect/'
eeg_root = 'sub-{:02}_task-{}'

# %%
n_stim = 2 # "unaltered", "broadband"
len_stim = 64.
eeg_fs = 10e3
n_epoch=40
task = 'peakyvsunaltered'
Bayesian = True
# %%
for subject in subject_list:
    # %% +/- ANM | IHC regressor
    len_eeg = int(len_stim*eeg_fs)

    t_start = -0.2
    t_stop = 0.6
    lags = np.arange(start=t_start*1000, stop=t_stop*1000, step=1e3/eeg_fs)
    w_rect = np.zeros((n_stim, len_eeg))
    abr_rect = np.zeros((n_stim, 8000))
    ##### For unaltered response
    print("unaltered sub{:02}".format(subject))


    stim_data = read_hdf5(rect_path + 'unaltered_x_in.hdf5')
    # Load x_in
    x_in_pos = stim_data['x_in_unaltered_pos']
    x_in_neg = stim_data['x_in_unaltered_neg']
    del stim_data
    # Load x_out
    eeg_data = read_hdf5(opath + 'eeg_data/' + eeg_root.format(subject, task) +'_data_eeg.hdf5')
    clock_adjusts = eeg_data['clock_adjusts']
    x_out = eeg_data['eeg'][0]
    x_out = np.mean(x_out, axis=1)
    # Adjust x_in

    # x_in fft
    x_in_pos_fft = fft(x_in_pos)
    x_in_neg_fft = fft(x_in_neg)
    # x_out fft
    x_out_fft = fft(x_out)
    if Bayesian:
        ivar = 1 / np.var(x_out, axis=1)
        weight = ivar/np.nansum(ivar)
    # TRF
    denom_pos = np.mean(x_in_pos_fft * np.conj(x_in_pos_fft), axis=0)
    denom_neg = np.mean(x_in_neg_fft * np.conj(x_in_neg_fft), axis=0)
    w_pos = []
    w_neg = []
    for ei in range(n_epoch):
        w_i_pos = (weight[ei] * np.conj(x_in_pos_fft[ei, :]) *
                   x_out_fft[ei, :]) / denom_pos
        w_i_neg = (weight[ei] * np.conj(x_in_neg_fft[ei, :]) *
                   x_out_fft[ei, :]) / denom_neg
        w_pos += [w_i_pos]
        w_neg += [w_i_neg]
    w_rect[0] = (ifft(np.array(w_pos).sum(0)).real +
                   ifft(np.array(w_neg).sum(0)).real) / 2

    abr_rect[0] = np.concatenate((w_rect[0][int(t_start*eeg_fs):],
                            w_rect[0][0:int(t_stop*eeg_fs)]))
    
    ##### For broadband peaky response
    print("broadband peaky sub{:02}".format(subject))
    w_broadband = np.zeros(len_eeg)
    abr_broadband = np.zeros(8000)
    
    stim_data = read_hdf5(rect_path + 'broadband_x_in.hdf5')
    # Load x_in
    x_in_pos = stim_data['x_in_broadband_pos']
    x_in_neg = stim_data['x_in_broadband_neg']
    # Load x_out
    x_out = eeg_data['eeg'][1]
    x_out = np.mean(x_out, axis=1)
    # x_in fft
    x_in_pos_fft = fft(x_in_pos)
    x_in_neg_fft = fft(x_in_neg)
    # x_out fft
    x_out_fft = fft(x_out)
    if Bayesian:
        ivar = 1 / np.var(x_out, axis=1)
        weight = ivar/np.nansum(ivar)
    # TRF
    denom_pos = np.mean(x_in_pos_fft * np.conj(x_in_pos_fft), axis=0)
    denom_neg = np.mean(x_in_neg_fft * np.conj(x_in_neg_fft), axis=0)
    w_pos = []
    w_neg = []
    for ei in range(n_epoch):
        w_i_pos = (weight[ei] * np.conj(x_in_pos_fft[ei, :]) *
                   x_out_fft[ei, :]) / denom_pos
        w_i_neg = (weight[ei] * np.conj(x_in_neg_fft[ei, :]) *
                   x_out_fft[ei, :]) / denom_neg
        w_pos += [w_i_pos]
        w_neg += [w_i_neg]
    w_rect[1] = (ifft(np.array(w_pos).sum(0)).real +
                    ifft(np.array(w_neg).sum(0)).real) / 2

    abr_rect[1] = np.concatenate((w_rect[1][int(t_start*eeg_fs):],
                             w_rect[1][0:int(t_stop*eeg_fs)]))
    
    write_hdf5(save_path + 'rect_response/' + eeg_root.format(subject, task) +
               '_rect_responses.hdf5',
               dict(w_rect=w_rect,
                    abr_rect=abr_rect,
                    lags=lags))
