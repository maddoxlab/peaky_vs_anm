#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spectral coherence

Created on Sun Dec 12 21:18:02 2021

@author: tong
"""

import numpy as np
import scipy.signal as sig
from numpy.fft import irfft, rfft, fft, ifft
import matplotlib.pyplot as plt
import h5py
from itertools import combinations
import scipy.stats as stats

from expyfun.io import read_wav, read_tab
from expyfun.io import write_hdf5, read_hdf5
import mne


import matplotlib.pyplot as plt

# %% Function

def butter_highpass(cutoff, fs, order=1):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = sig.butter(order, normal_cutoff, btype='high', analog=False)
    return b, a


def butter_highpass_filter(data, cutoff, fs, order=1, axis=-1):
    b, a = butter_highpass(cutoff, fs, order=order)
    y = sig.lfilter(b, a, data, axis)
    return y


def butter_lowpass(cutoff, fs, order=1):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = sig.butter(order, normal_cutoff, btype='low', analog=False)
    return b, a


def butter_lowpass_filter(data, cutoff, fs, order=1, axis=-1):
    b, a = butter_lowpass(cutoff, fs, order=order)
    y = sig.lfilter(b, a, data, axis)
    return y


def butter_bandpass(lowcut, highcut, fs, order=1):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = sig.butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=1, axis=-1):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = sig.lfilter(b, a, data, axis)
    return y

def coherence(x, y, fs, window_size=0.5, absolute=True):
    '''
    import scipy.signal
    Inputs:
        x, y: input signals
        window_size: time chunk size, default 0.5s
        fs: sampling frequency
        magnitude: if True, return absolute real values, otherwise return complex values
    Outputs:
        coh: coherence vector
        freq: frequency vector
    '''
    f, t, Zxx = sig.stft(x, fs=fs, window='boxcar', nperseg=window_size*fs, noverlap=0, return_onesided=True, boundary=None)
    f, t, Zyy = sig.stft(y, fs=fs, window='boxcar', nperseg=window_size*fs, noverlap=0, return_onesided=True, boundary=None)
    n_slice = len(x) / (window_size*fs) 
    Num = np.sum(np.conjugate(Zxx) * Zyy, axis=-1) / n_slice
    Den = np.sqrt((np.sum(np.conjugate(Zxx) * Zxx, axis=-1) / n_slice)*(np.sum(np.conjugate(Zyy) * Zyy, axis=-1) / n_slice))
    if absolute:
        coh = abs(Num) / Den
    else:
        coh = Num / Den
    freq = f
    return coh, freq


# %% Parameters
n_stim = 2 # "unaltered", "broadband"
len_stim = 64.
eeg_fs = 10e3
n_epoch=40
len_eeg = int(len_stim*eeg_fs)
task = 'peakyvsunaltered'
# %% input subject and experiment information
n_jobs = 'cuda'
subject_list = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20,
            21, 22, 23, 24]
# NOTE: s1 only had 1st40 rpt (pilot), s15 not completed (equipment problems)
subject_num = len(subject_list)
# saving options
overwrite_file = True

# paths and root filenames
opath = "/media/tshan@urmc-sh.rochester.edu/Elements/AMPLab/peaky_vs_ANM/exp1/"

# NOTE: combine all trials into the same folder of each narrator
stim_path = opath + 'present_files/Exp1_Stimuli_{}_Narrator/'
stim_root = '{}Narrator{:03}_bands.hdf5'
eeg_path = opath + 'eeg_data'
eeg_root = 'sub-{:02}_task-{}'
predicted_eeg_path = opath + 'predicted_eeg/'

task = 'peakyvsunaltered'
save_path = opath + 'subject_response/'
eeg_root = 'sub-{:02}_task-{}'

regressors = ['rect', 'pulse', 'ANM']
regressors = ['rect']
# %% Coherence params
dur_slice = 0.2
n_slices = int(len_stim / dur_slice)
len_slice = int(dur_slice * eeg_fs)
n_bands = int((eeg_fs / 2) * dur_slice + 1)
high_pass = 40
mismatch = True
# %% Compute coherence
for regressor in regressors:
    print(regressor)
    coh_unaltered = np.zeros((subject_num, n_bands), dtype='complex_')
    coh_broadband = np.zeros((subject_num, n_bands), dtype='complex_')
    corr_unaltered = np.zeros((subject_num, 2))
    corr_broadband = np.zeros((subject_num, 2))
    corr_unaltered_hp = np.zeros((subject_num, 2))
    corr_broadband_hp = np.zeros((subject_num, 2))
    predicted_data = read_hdf5(predicted_eeg_path + regressor + '_predict_x_out_new_15ms.hdf5')
    
    for subject in subject_list:
        si = subject_list.index(subject)
        ###### TRUE EEG #####
        print(regressor+str(subject))
        si = subject_list.index(subject)
        ###### Unaltered #####
        print("###### Unaltered #####")
        ###### TRUE EEG #####
        eeg_data = read_hdf5(opath + 'eeg_data/' + eeg_root.format(subject, task) +'_data_eeg.hdf5')
        x_out = eeg_data['eeg'][0]
        x_out = np.mean(x_out, axis=1)
        if mismatch:
            shift_size = np.random.randint(1,41)
            x_out = np.roll(x_out, shift_size, axis=0)

        x_out_all = np.reshape(x_out, -1)

        ##### PREDICTED EEG #####
        out_unaltered_predicted = predicted_data['out_unaltered_predicted']
        out_unaltered_predicted_all = np.reshape(out_unaltered_predicted[si], -1)

        #out_unaltered_predicted_all = np.roll(out_unaltered_predicted_all, int(len_stim*eeg_fs)) # for noise floor computation
        coh_unaltered[si, :], freq = coherence(x_out_all, out_unaltered_predicted_all, fs=eeg_fs, window_size=dur_slice, absolute=False)
        corr_unaltered[si,:] = stats.pearsonr(x_out_all, out_unaltered_predicted_all)
        if high_pass:
            x_out_all_hp = butter_highpass_filter(x_out_all, high_pass, eeg_fs)
            out_unaltered_predicted_all_hp = butter_highpass_filter(out_unaltered_predicted_all, high_pass, eeg_fs)
            corr_unaltered_hp[si,:] = stats.pearsonr(x_out_all_hp, out_unaltered_predicted_all_hp)
               
        ###### Broadband #####
        print("###### Broadband #####")
        ###### TRUE EEG #####
        eeg_data = read_hdf5(opath + 'eeg_data/' + eeg_root.format(subject, task) +'_data_eeg.hdf5')
        x_out = eeg_data['eeg'][1]
        x_out = np.mean(x_out, axis=1)
        if mismatch:
            shift_size = np.random.randint(1,41)
            x_out = np.roll(x_out, shift_size, axis=0)
        x_out_all = np.reshape(x_out, -1)
            
        ##### PREDICTED EEG #####
        out_broadband_predicted = predicted_data['out_broadband_predicted']
        out_broadband_predicted_all = np.reshape(out_broadband_predicted[si], -1)
        #out_broadband_predicted_all = np.roll(out_broadband_predicted_all, int(len_stim*eeg_fs))
        coh_broadband[si, :], freq = coherence(x_out_all, out_broadband_predicted_all, fs=eeg_fs, window_size=dur_slice, absolute=False)
        corr_broadband[si,:] = stats.pearsonr(x_out_all, out_broadband_predicted_all)
        if high_pass:
            x_out_all_hp = butter_highpass_filter(x_out_all, high_pass, eeg_fs)
            out_broadband_predicted_all_hp = butter_highpass_filter(out_broadband_predicted_all, high_pass, eeg_fs)
            corr_broadband_hp[si,:] = stats.pearsonr(x_out_all_hp, out_broadband_predicted_all_hp)
        
    
    # write_hdf5(predicted_eeg_path + regressor + '_coherence_slice_200ms_kernel_15ms.hdf5',
    #            dict(coh_unaltered=coh_unaltered,
    #                 coh_broadband=coh_broadband,
    #                 corr_unaltered=corr_unaltered,
    #                 corr_broadband=corr_broadband,
    #                 corr_unaltered_hp=corr_unaltered_hp,
    #                 corr_broadband_hp=corr_broadband_hp,
    #                 dur_slice=dur_slice,
    #                 freq=freq), overwrite=True)

    write_hdf5(predicted_eeg_path + regressor + '_coherence_slice_200ms_kernel_15ms_mismatch.hdf5',
               dict(coh_unaltered=coh_unaltered,
                    coh_broadband=coh_broadband,
                    corr_unaltered=corr_unaltered,
                    corr_broadband=corr_broadband,
                    corr_unaltered_hp=corr_unaltered_hp,
                    corr_broadband_hp=corr_broadband_hp,
                    dur_slice=dur_slice,
                    freq=freq), overwrite=True)

# %% Noise floor calculation

