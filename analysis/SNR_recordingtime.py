#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 19 16:58:03 2022

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
import time

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

def abr_SNR(abr_data, lags, abr_time_window=15, noise_range=[-200, -20]):
    """
    abr_time_window: time range in ms define as abr, 15 ms by default
    noise_range: prestim time range to calculate noise level, [-200, -20] ms by default
    """    
    ind_abr = np.where((lags>=0) & (lags<abr_time_window))
    abr_var = np.var(abr_data[ind_abr])
    noise_seg_num = int((noise_range[1]-noise_range[0]) / abr_time_window)
    noise_var = 0
    for i in range(noise_seg_num):
        ind_noise = np.where((lags>=(noise_range[0]+abr_time_window*i)) & (lags<(noise_range[0]+abr_time_window*(i+1))))
        noise_var += np.var(abr_data[ind_noise])
    noise_var = noise_var / noise_seg_num # averaging the var of noise
    SNR = 10*np.log10((abr_var - noise_var)/noise_var)
    return SNR

# %% input subject and experiment information
n_jobs = 'cuda'
subject_list = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20,
            21, 22, 23, 24]
# NOTE: s1 only had 1st40 rpt (pilot), s15 not completed (equipment problems)
eeg_2files = [2, 4]  # when there are 2 eeg files (had to change battery)

# saving options
overwrite_file = True

# paths and root filenames
opath = "/media/tshan@urmc-sh.rochester.edu/Elements/AMPLab/peaky_vs_ANM/exp1/"
# NOTE: combine all trials into the same folder of each narrator
stim_path = opath + 'present_files/'
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
Bayesian = True
len_eeg = int(len_stim*eeg_fs)
subject_num = len(subject_list)

regressor='pulse'

snr_unaltered = np.zeros((subject_num, n_epoch))
snr_unaltered_bp = np.zeros((subject_num, n_epoch))
snr_broadband = np.zeros((subject_num, n_epoch))
snr_broadband_bp = np.zeros((subject_num, n_epoch))
#%% START LOOPING SUBJECTS
for subject in subject_list:
    start_time = time.time()
    si = subject_list.index(subject)
    #%% DATA NEED TO COMPUTE
    abr_unaltered = np.zeros((n_epoch, 8000))
    abr_unaltered_bp = np.zeros((n_epoch, 8000))

    abr_broadband = np.zeros((n_epoch, 8000))
    abr_broadband_bp = np.zeros((n_epoch, 8000))
    # %% Loading and filtering TRUE EEG data
    eeg_data = read_hdf5(opath + 'eeg_data/' + eeg_root.format(subject, task) +'_data_eeg.hdf5')
    clock_adjusts = eeg_data['clock_adjusts']

    # %% Deriving ABR for different number of epochs
    x_in_path = stim_path + regressor + '/'
    t_start = -0.2
    t_stop = 0.6
    lags = np.arange(start=t_start*1000, stop=t_stop*1000, step=1e3/eeg_fs)
    unaltered_data = read_hdf5(x_in_path + 'unaltered_x_in.hdf5')
    broadband_data = read_hdf5(x_in_path + 'broadband_x_in.hdf5')
    # Looping from 1 epoch average to 240 epoch average
    for n in range(n_epoch):
        n_epoch_run = n+1
        ######## UNALTERED ABR #########
        print("subject" + str(subject) + ": epoch run number " + str(n_epoch_run) + ": Unaltered")
        # x in
        if regressor=='pulse':
            x_in_unaltered_pos = unaltered_data['x_in_unaltered'][:n_epoch_run,:]
            x_in_unaltered_neg = unaltered_data['x_in_unaltered'][:n_epoch_run,:]
        else:
            x_in_unaltered_pos = unaltered_data['x_in_unaltered_pos'][:n_epoch_run,:]
            x_in_unaltered_neg = unaltered_data['x_in_unaltered_neg'][:n_epoch_run,:]
        # x out
        x_out_unaltered = eeg_data['eeg'][0][:n_epoch_run,:]
        x_out_unaltered = np.mean(x_out_unaltered, axis=1)
        # x_in fft
        x_in_unaltered_pos_fft = fft(x_in_unaltered_pos)
        x_in_unaltered_neg_fft = fft(x_in_unaltered_neg)
        # x_out fft
        x_out_unaltered_fft = fft(x_out_unaltered)
        if Bayesian:
            ivar = 1 / np.var(x_out_unaltered, axis=1)
            weight = ivar/np.nansum(ivar)
        # TRF
        denom_pos = np.mean(x_in_unaltered_pos_fft * np.conj(x_in_unaltered_pos_fft), axis=0)
        denom_neg = np.mean(x_in_unaltered_neg_fft * np.conj(x_in_unaltered_neg_fft), axis=0)
        w_pos = []
        w_neg = []
        for ei in range(n_epoch_run):
            w_i_pos = (weight[ei] * np.conj(x_in_unaltered_pos_fft[ei, :]) *
                       x_out_unaltered_fft[ei, :]) / denom_pos
            w_i_neg = (weight[ei] * np.conj(x_in_unaltered_neg_fft[ei, :]) *
                       x_out_unaltered_fft[ei, :]) / denom_neg
            w_pos += [w_i_pos]
            w_neg += [w_i_neg]
        w_unaltered = (ifft(np.array(w_pos).sum(0)).real + ifft(np.array(w_neg).sum(0)).real) / 2
        abr_unaltered[n,:] = np.roll(np.concatenate((w_unaltered[int(t_start*eeg_fs):],
                            w_unaltered[0:int(t_stop*eeg_fs)])),int(3.4*eeg_fs/1000))
        abr_unaltered_bp[n,:] = butter_highpass_filter(abr_unaltered[n,:], 150, eeg_fs, order=1)
        snr_unaltered[si,n] = abr_SNR(abr_unaltered[n,:], lags, abr_time_window=15, noise_range=[-200, -20])
        snr_unaltered_bp[si,n] = abr_SNR(abr_unaltered_bp[n,:], lags, abr_time_window=15, noise_range=[-200, -20])
        
        ######## BROADBAND ABR #########
        print("subject" + str(subject) + ": epoch run number " + str(n_epoch_run) + ": Broadband")
        # x in
        if regressor=='pulse':
            x_in_broadband_pos = broadband_data['x_in_broadband'][:n_epoch_run,:]
            x_in_broadband_neg = broadband_data['x_in_broadband'][:n_epoch_run,:]
        else:
            x_in_broadband_pos = broadband_data['x_in_broadband_pos'][:n_epoch_run,:]
            x_in_broadband_neg = broadband_data['x_in_broadband_neg'][:n_epoch_run,:]
        # x out
        x_out_broadband = eeg_data['eeg'][1][:n_epoch_run,:]
        x_out_broadband = np.mean(x_out_broadband, axis=1)
        # x_in fft
        x_in_broadband_pos_fft = fft(x_in_broadband_pos)
        x_in_broadband_neg_fft = fft(x_in_broadband_neg)
        # x_out fft
        x_out_broadband_fft = fft(x_out_broadband)
        if Bayesian:
            ivar = 1 / np.var(x_out_broadband, axis=1)
            weight = ivar/np.nansum(ivar)
        # TRF
        denom_pos = np.mean(x_in_broadband_pos_fft * np.conj(x_in_broadband_pos_fft), axis=0)
        denom_neg = np.mean(x_in_broadband_neg_fft * np.conj(x_in_broadband_neg_fft), axis=0)
        w_pos = []
        w_neg = []
        for ei in range(n_epoch_run):
            w_i_pos = (weight[ei] * np.conj(x_in_broadband_pos_fft[ei, :]) *
                       x_out_broadband_fft[ei, :]) / denom_pos
            w_i_neg = (weight[ei] * np.conj(x_in_broadband_neg_fft[ei, :]) *
                       x_out_broadband_fft[ei, :]) / denom_neg
            w_pos += [w_i_pos]
            w_neg += [w_i_neg]
        w_broadband = (ifft(np.array(w_pos).sum(0)).real + ifft(np.array(w_neg).sum(0)).real) / 2
        abr_broadband[n,:] = np.roll(np.concatenate((w_broadband[int(t_start*eeg_fs):],
                            w_broadband[0:int(t_stop*eeg_fs)])),int(3.4*eeg_fs/1000))
        abr_broadband_bp[n,:] = butter_highpass_filter(abr_broadband[n,:], 150, eeg_fs, order=1)
        snr_broadband[si,n] = abr_SNR(abr_broadband[n,:], lags, abr_time_window=15, noise_range=[-200, -20])
        snr_broadband_bp[si,n] = abr_SNR(abr_broadband_bp[n,:], lags, abr_time_window=15, noise_range=[-200, -20])
        
        print("RUN--- %s seconds ---" % (time.time() - start_time))

    write_hdf5(save_path + regressor + '_response/' + eeg_root.format(subject, task) +
               '_' + regressor+'_responses_by_numEpoch.hdf5',
               dict(abr_unaltered=abr_unaltered, abr_unaltered_bp=abr_unaltered_bp, 
                    abr_broadband=abr_broadband, abr_broadband_bp=abr_broadband_bp,lags=lags), overwrite=True)
    
    print("SUBJECT TIME--- %s seconds ---" % (time.time() - start_time))
    
write_hdf5(save_path+regressor+"_by_numEpoch.hdf5",
           dict(snr_unaltered=snr_unaltered, snr_unaltered_bp=snr_unaltered_bp,
                snr_broadband=snr_broadband, snr_broadband_bp=snr_broadband_bp), overwrite=True)

