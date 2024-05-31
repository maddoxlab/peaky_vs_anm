#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 27 11:35:35 2022

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
from mne.filter import resample


import matplotlib.pyplot as plt

# %%
opath = "/media/tong/Elements/AMPLab/peaky_vs_ANM/exp1/"
n_epoch=40
data = read_hdf5(opath + 'present_files/Exp1_stimuli.hdf5')
stim_fs=44100
eeg_fs=10e3
len_stim = 64.
n_jobs = 'cuda'
#%% rectified
unaltered = data['audio'][0]
x_in_unaltered_pos = resample(np.maximum(0, unaltered), eeg_fs, stim_fs, npad='auto', n_jobs=n_jobs, verbose=False)
x_in_unaltered_neg = resample(np.maximum(0, -unaltered), eeg_fs, stim_fs, npad='auto', n_jobs=n_jobs, verbose=False)
write_hdf5(opath + 'present_files/rect/unaltered_x_in.hdf5',
           dict(x_in_unaltered_pos=x_in_unaltered_pos,
                x_in_unaltered_neg=x_in_unaltered_neg,
                fs=eeg_fs))

broadband = data['audio'][1]
x_in_broadband_pos = resample(np.maximum(0, broadband), eeg_fs, stim_fs, npad='auto', n_jobs=n_jobs, verbose=False)
x_in_broadband_neg = resample(np.maximum(0, -broadband), eeg_fs, stim_fs, npad='auto', n_jobs=n_jobs, verbose=False)
write_hdf5(opath + 'present_files/rect/broadband_x_in.hdf5',
           dict(x_in_broadband_pos=x_in_broadband_pos,
                x_in_broadband_neg=x_in_broadband_neg,
                fs=eeg_fs))

# %% pulse
unaltered_pulse_inds = data['pulseinds'][0]
unaltered_pulses_all = []
for pulse_ind in unaltered_pulse_inds:
    pulses = np.zeros((len(pulse_ind), int(eeg_fs * len_stim)))
    pinds = [(pi * float(eeg_fs) / (stim_fs)).astype(int)
             for pi in pulse_ind]
    for bi in range(len(pinds)):
        pulses[bi, pinds[bi]] = 1.
    unaltered_pulses_all += [pulses]
x_in_unaltered = np.array(unaltered_pulses_all).reshape(n_epoch, -1)

write_hdf5(opath + 'present_files/pulse/unaltered_x_in.hdf5',
           dict(x_in_unaltered=x_in_unaltered,
                fs=eeg_fs))

broadband_pulse_inds = data['pulseinds'][1]
broadband_pulses_all = []
for pulse_ind in broadband_pulse_inds:
    pulses = np.zeros((len(pulse_ind), int(eeg_fs * len_stim)))
    pinds = [(pi * float(eeg_fs) / (stim_fs)).astype(int)
             for pi in pulse_ind]
    for bi in range(len(pinds)):
        pulses[bi, pinds[bi]] = 1.
    broadband_pulses_all += [pulses]
x_in_broadband = np.array(broadband_pulses_all).reshape(n_epoch, -1)

write_hdf5(opath + 'present_files/pulse/broadband_x_in.hdf5',
           dict(x_in_broadband=x_in_broadband,
                fs=eeg_fs))
