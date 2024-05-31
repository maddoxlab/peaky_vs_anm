#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 21 12:05:14 2024

@author: tshan
"""


import numpy as np
import pandas as pd
import scipy.signal as signal
from numpy.fft import irfft, rfft, fft, ifft
import matplotlib.pyplot as plt
import h5py
from scipy.stats import pearsonr, shapiro, wilcoxon
import statsmodels
import pingouin
import statsmodels.api as sm
import statsmodels.formula.api as smf


from expyfun.io import read_wav, read_tab
from expyfun.io import write_hdf5, read_hdf5
import mne

import seaborn as sns

import matplotlib.pyplot as plt
from cycler import cycler

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

#%% FIG SETTING
dpi = 300
opath = "/media/tshan@urmc-sh.rochester.edu/Elements/AMPLab/peaky_vs_ANM/exp1/"
figure_path = opath + 'paper_figures/'

plt.rc('axes', titlesize=10, labelsize=10)
plt.rc('xtick', labelsize=10)
plt.rc('ytick', labelsize=10)
plt.rc('axes', prop_cycle=cycler(color=["#4477AA","#66CCEE","#228833","#CCBB44","#EE6677","#AA3377","#BBBBBB"]))
plt.rcParams["font.family"] = "Arial"
# %% Parametersrue
# param
n_stim = 2 # "unaltered", "broadband"
len_stim = 64.
eeg_fs = 10e3
n_epoch=40
task = 'peakyvsunaltered'
subject_list = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20,
            21, 22, 23, 24]
subject_num = len(subject_list)
eeg_root = 'sub-{:02}_task-{}'
# %% Rectified
t_start = -0.2
t_stop = 0.6
lags = np.arange(start=t_start*1000, stop=t_stop*1000, step=1e3/eeg_fs)
rect_unaltered_response_all = np.zeros((subject_num, 8000))
for subject in subject_list:
    subi = subject_list.index(subject)
    rect_response_data = read_hdf5(opath + "subject_response/rect_pulse_response/" + eeg_root.format(subject,task) + '_data_responses.hdf5')
    temp = rect_response_data['w_audio'][0]
    rect_unaltered_response_all[subi] = np.concatenate((temp[int(t_start*eeg_fs):], temp[0:int(t_stop*eeg_fs)]))
    
rect_unaltered_response_ave = np.sum(rect_unaltered_response_all, axis=0) / subject_num
rect_unaltered_response_err = np.std(rect_unaltered_response_all, axis=0) / np.sqrt(subject_num)

rect_broadband_response_all = np.zeros((subject_num, 8000))
for subject in subject_list:
    subi = subject_list.index(subject)
    rect_response_data = read_hdf5(opath + "subject_response/rect_pulse_response/" + eeg_root.format(subject,task) + '_data_responses.hdf5')
    temp = rect_response_data['w_audio'][1]
    rect_broadband_response_all[subi] = np.concatenate((temp[int(t_start*eeg_fs):], temp[0:int(t_stop*eeg_fs)]))
    
rect_broadband_response_ave = np.sum(rect_broadband_response_all, axis=0) / subject_num
rect_broadband_response_err = np.std(rect_broadband_response_all, axis=0) / np.sqrt(subject_num)

write_hdf5(opath + 'subject_response/' + 'rect_response_all_subjects.hdf5',
           dict(rect_unaltered_response_all=rect_unaltered_response_all,
                rect_unaltered_response_ave=rect_unaltered_response_ave,
                rect_unaltered_response_err=rect_unaltered_response_err,
                rect_broadband_response_all=rect_broadband_response_all,
                rect_broadband_response_ave=rect_broadband_response_ave,
                rect_broadband_response_err=rect_broadband_response_err))

#ABR
fig = plt.figure(dpi=dpi)
fig.set_size_inches(8, 6)
plt.plot(lags, rect_unaltered_response_ave, c='C0', linewidth=1.2, label='Unaltered (rectified)')
plt.fill_between(lags, rect_unaltered_response_ave-rect_unaltered_response_err, rect_unaltered_response_ave+rect_unaltered_response_err, alpha=0.6, color='C0', linewidth=0)
plt.plot(lags, rect_broadband_response_ave, c='C1', linewidth=1.2, linestyle='--', label='Peaky (rectified)')
plt.fill_between(lags, rect_broadband_response_ave-rect_broadband_response_err, rect_broadband_response_ave+rect_broadband_response_err, alpha=0.6, color='C1', ec=None, linewidth=0)
plt.xlim(-10, 30)
plt.ylim(-5e-7,10e-7)
plt.ylabel('Magnitude (AU)')
plt.xlabel('Time (ms)')
plt.grid(alpha=0.5)
plt.vlines(0,-5e-7,10e-7, linestyles='solid', color='k', linewidth=2, alpha=0.5)
plt.hlines(0,-10,30, linestyles='solid', color='k', linewidth=2, alpha=0.5)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
#plt.title("Averaged Response Regress on ANM (ABR) " + "(num=" + str(subject_num) + ')')
plt.legend(fontsize=8)
plt.tight_layout()

#Cortical
fig = plt.figure(dpi=dpi)
fig.set_size_inches(8, 6)
plt.plot(lags, rect_unaltered_response_ave, c='C0', linewidth=1.2, label='Unaltered (rectified)')
plt.fill_between(lags, rect_unaltered_response_ave-rect_unaltered_response_err, rect_unaltered_response_ave+rect_unaltered_response_err, alpha=0.6, color='C0', linewidth=0)
plt.plot(lags, rect_broadband_response_ave, c='C1', linewidth=1.2, linestyle='--', label='Peaky (rectified)')
plt.fill_between(lags, rect_broadband_response_ave-rect_broadband_response_err, rect_broadband_response_ave+rect_broadband_response_err, alpha=0.6, color='C1', ec=None, linewidth=0)
plt.xlim(-50, 300)
plt.ylim(-5e-7,10e-7)
plt.ylabel('Magnitude (AU)')
plt.xlabel('Time (ms)')
plt.grid(alpha=0.5)
plt.vlines(0,-5e-7,10e-7, linestyles='solid', color='k', linewidth=2, alpha=0.5)
plt.hlines(0,-50,300, linestyles='solid', color='k', linewidth=2, alpha=0.5)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
#plt.title("Averaged Response Regress on ANM (ABR) " + "(num=" + str(subject_num) + ')')
plt.legend(fontsize=8)
plt.tight_layout()

# %% Rectified - new responses
rect_unaltered_response_all = np.zeros((subject_num, 8000))
for subject in subject_list:
    subi = subject_list.index(subject)
    rect_response_data = read_hdf5(opath + "subject_response/rect_response/" + eeg_root.format(subject,task) + '_rect_responses.hdf5')
    rect_unaltered_response_all[subi] = rect_response_data['abr_rect'][0]
    lags = rect_response_data['lags']
    
rect_unaltered_response_ave = np.sum(rect_unaltered_response_all, axis=0) / subject_num
rect_unaltered_response_err = np.std(rect_unaltered_response_all, axis=0) / np.sqrt(subject_num)

rect_broadband_response_all = np.zeros((subject_num, 8000))
for subject in subject_list:
    subi = subject_list.index(subject)
    rect_response_data = read_hdf5(opath + "subject_response/rect_response/" + eeg_root.format(subject,task) + '_rect_responses.hdf5')
    rect_broadband_response_all[subi] = rect_response_data['abr_rect'][1]
    
rect_broadband_response_ave = np.sum(rect_broadband_response_all, axis=0) / subject_num
rect_broadband_response_err = np.std(rect_broadband_response_all, axis=0) / np.sqrt(subject_num)

write_hdf5(opath + 'subject_response/' + 'rect_response_all_subjects_new.hdf5',
           dict(rect_unaltered_response_all=rect_unaltered_response_all,
                rect_unaltered_response_ave=rect_unaltered_response_ave,
                rect_unaltered_response_err=rect_unaltered_response_err,
                rect_broadband_response_all=rect_broadband_response_all,
                rect_broadband_response_ave=rect_broadband_response_ave,
                rect_broadband_response_err=rect_broadband_response_err),overwrite=True)

#ABR
fig = plt.figure(dpi=dpi)
fig.set_size_inches(3.5, 3)
plt.plot(lags, rect_unaltered_response_ave, c='C0', linewidth=1, label='Unaltered')
plt.fill_between(lags, rect_unaltered_response_ave-rect_unaltered_response_err, rect_unaltered_response_ave+rect_unaltered_response_err, alpha=0.6, color='C0', linewidth=0)
plt.plot(lags, rect_broadband_response_ave, c='C1', linewidth=1, label='Peaky')
plt.fill_between(lags, rect_broadband_response_ave-rect_broadband_response_err, rect_broadband_response_ave+rect_broadband_response_err, alpha=0.6, color='C1', ec=None, linewidth=0)
plt.xlim(-10, 15)
plt.ylim(-5e-7,10e-7)
plt.ylabel('Magnitude (AU)')
plt.xlabel('Time (ms)')
plt.grid(alpha=0.5)
plt.vlines(0,-5e-7,10e-7, linestyles='solid', color='k', linewidth=2, alpha=0.5)
plt.hlines(0,-10,30, linestyles='solid', color='k', linewidth=2, alpha=0.5)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
#plt.title("Averaged Response Regress on ANM (ABR) " + "(num=" + str(subject_num) + ')')
plt.legend(fontsize=10,loc='upper left')
plt.title("HWR")
plt.tight_layout()
plt.savefig(figure_path+'waveforms_HWR_ABR.svg', dpi=dpi, format='svg')

#Cortical
fig = plt.figure(dpi=dpi)
fig.set_size_inches(3.5, 3)
plt.plot(lags, rect_unaltered_response_ave, c='C0', linewidth=1, label='Unaltered')
plt.fill_between(lags, rect_unaltered_response_ave-rect_unaltered_response_err, rect_unaltered_response_ave+rect_unaltered_response_err, alpha=0.6, color='C0', linewidth=0)
plt.plot(lags, rect_broadband_response_ave, c='C1', linewidth=1, label='Peaky')
plt.fill_between(lags, rect_broadband_response_ave-rect_broadband_response_err, rect_broadband_response_ave+rect_broadband_response_err, alpha=0.6, color='C1', ec=None, linewidth=0)
plt.xlim(-50, 300)
plt.ylim(-6e-7,10e-7)
plt.ylabel('Magnitude (AU)')
plt.xlabel('Time (ms)')
plt.grid(alpha=0.5)
plt.vlines(0,-6e-7,10e-7, linestyles='solid', color='k', linewidth=2, alpha=0.5)
plt.hlines(0,-50,300, linestyles='solid', color='k', linewidth=2, alpha=0.5)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
#plt.title("Averaged Response Regress on ANM (ABR) " + "(num=" + str(subject_num) + ')')
plt.legend(fontsize=10)
plt.title("HWR")
plt.tight_layout()
plt.savefig(figure_path+'waveforms_HWR_cortical.svg', dpi=dpi, format='svg')

# %% Rerified high-pass 150 Hz
rect_unaltered_response_all = np.zeros((subject_num, 8000))
for subject in subject_list:
    subi = subject_list.index(subject)
    rect_response_data = read_hdf5(opath + "subject_response/rect_response/" + eeg_root.format(subject,task) + '_rect_responses.hdf5')
    temp = butter_highpass_filter(rect_response_data['abr_rect'][0], 150, eeg_fs)
    rect_unaltered_response_all[subi] = temp
    lags = rect_response_data['lags']
    
rect_unaltered_response_ave = np.sum(rect_unaltered_response_all, axis=0) / subject_num
rect_unaltered_response_err = np.std(rect_unaltered_response_all, axis=0) / np.sqrt(subject_num)

rect_broadband_response_all = np.zeros((subject_num, 8000))
for subject in subject_list:
    subi = subject_list.index(subject)
    rect_response_data = read_hdf5(opath + "subject_response/rect_response/" + eeg_root.format(subject,task) + '_rect_responses.hdf5')
    temp = butter_highpass_filter(rect_response_data['abr_rect'][1], 150, eeg_fs)
    rect_broadband_response_all[subi] = temp
    
rect_broadband_response_ave = np.sum(rect_broadband_response_all, axis=0) / subject_num
rect_broadband_response_err = np.std(rect_broadband_response_all, axis=0) / np.sqrt(subject_num)

#ABR
fig = plt.figure(dpi=dpi)
fig.set_size_inches(3.5, 3)
plt.plot(lags, rect_unaltered_response_ave, c='C0', linewidth=1, label='Unaltered')
plt.fill_between(lags, rect_unaltered_response_ave-rect_unaltered_response_err, rect_unaltered_response_ave+rect_unaltered_response_err, alpha=0.6, color='C0', linewidth=0)
plt.plot(lags, rect_broadband_response_ave, c='C1', linewidth=1, label='Peaky')
plt.fill_between(lags, rect_broadband_response_ave-rect_broadband_response_err, rect_broadband_response_ave+rect_broadband_response_err, alpha=0.6, color='C1', ec=None, linewidth=0)
plt.xlim(-10, 15)
plt.ylim(-3e-7,3e-7)
plt.ylabel('Magnitude (AU)')
plt.xlabel('Time (ms)')
plt.grid(alpha=0.5)
plt.vlines(0,-3e-7,3e-7, linestyles='solid', color='k', linewidth=2, alpha=0.5)
plt.hlines(0,-10,30, linestyles='solid', color='k', linewidth=2, alpha=0.5)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
#plt.title("Averaged Response Regress on ANM (ABR) " + "(num=" + str(subject_num) + ')')
plt.legend(fontsize=10,loc='upper left')
plt.title("HWR")
plt.tight_layout()
plt.savefig(figure_path+'waveforms_HWR_ABR_hp150.svg', dpi=dpi, format='svg')

# %% Rectified Whited- new responses
do_HP = True
rect_unaltered_response_all = np.zeros((subject_num, 14000))
for subject in subject_list:
    subi = subject_list.index(subject)
    rect_response_data = read_hdf5(opath + "subject_response/rect_response/" + eeg_root.format(subject,task) + '_rect_whited_responses.hdf5')
    temp = rect_response_data['abr_rect'][0] 
    if do_HP:
        rect_unaltered_response_all[subi] = butter_highpass_filter(temp, 150, fs=eeg_fs)# HP @150Hz
    else:
        rect_unaltered_response_all[subi] = temp
    lags = rect_response_data['lags']
    
rect_unaltered_response_ave = np.sum(rect_unaltered_response_all*1e6, axis=0) / subject_num
rect_unaltered_response_err = np.std(rect_unaltered_response_all*1e6, axis=0) / np.sqrt(subject_num)

rect_broadband_response_all = np.zeros((subject_num, 14000))
for subject in subject_list:
    subi = subject_list.index(subject)
    rect_response_data = read_hdf5(opath + "subject_response/rect_response/" + eeg_root.format(subject,task) + '_rect_whited_responses.hdf5')
    temp = rect_response_data['abr_rect'][1]
    if do_HP:
        rect_broadband_response_all[subi] = butter_highpass_filter(temp, 150, fs=eeg_fs)
    else:
        rect_broadband_response_all[subi] = temp
        
rect_broadband_response_ave = np.sum(rect_broadband_response_all*1e6, axis=0) / subject_num # *1e6 to get meaningful unit micro volts
rect_broadband_response_err = np.std(rect_broadband_response_all*1e6, axis=0) / np.sqrt(subject_num)

write_hdf5(opath + 'subject_response/' + 'rect_whited_response_all_subjects_new.hdf5',
           dict(rect_unaltered_response_all=rect_unaltered_response_all,
                rect_unaltered_response_ave=rect_unaltered_response_ave,
                rect_unaltered_response_err=rect_unaltered_response_err,
                rect_broadband_response_all=rect_broadband_response_all,
                rect_broadband_response_ave=rect_broadband_response_ave,
                rect_broadband_response_err=rect_broadband_response_err),overwrite=True)

#ABR
fig = plt.figure(dpi=dpi)
fig.set_size_inches(3.5, 3)
plt.plot(lags, rect_unaltered_response_ave, c='C0', linewidth=1, label='Unaltered')
plt.fill_between(lags, rect_unaltered_response_ave-rect_unaltered_response_err, rect_unaltered_response_ave+rect_unaltered_response_err, alpha=0.6, color='C0', linewidth=0)
plt.plot(lags, rect_broadband_response_ave, c='C1', linewidth=1, label='Peaky')
plt.fill_between(lags, rect_broadband_response_ave-rect_broadband_response_err, rect_broadband_response_ave+rect_broadband_response_err, alpha=0.6, color='C1', ec=None, linewidth=0)
plt.xlim(-10, 15)
plt.ylim(-7,17)
#plt.ylim(-0.9e-5,0.7e-5)
plt.ylabel('Potential (μV)')
plt.xlabel('Time (ms)')
plt.grid(alpha=0.5)
plt.vlines(0,-7,17, linestyles='solid', color='k', linewidth=2, alpha=0.5)
plt.hlines(0,-10,15, linestyles='solid', color='k', linewidth=2, alpha=0.5)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
#plt.title("Averaged Response Regress on ANM (ABR) " + "(num=" + str(subject_num) + ')')
plt.legend(fontsize=10, loc="upper left")
plt.title("HWR (Phase-only)")
plt.tight_layout()
plt.savefig(figure_path+'waveforms_whited_HWR_ABR.svg', dpi=dpi, format='svg')

#Cortical
fig = plt.figure(dpi=dpi)
fig.set_size_inches(3.5, 3)
plt.plot(lags, rect_unaltered_response_ave, c='C0', linewidth=1.2, label='Unaltered')
plt.fill_between(lags, rect_unaltered_response_ave-rect_unaltered_response_err, rect_unaltered_response_ave+rect_unaltered_response_err, alpha=0.6, color='C0', linewidth=0)
plt.plot(lags, rect_broadband_response_ave, c='C1', linewidth=1.2, linestyle='--', label='Peaky')
plt.fill_between(lags, rect_broadband_response_ave-rect_broadband_response_err, rect_broadband_response_ave+rect_broadband_response_err, alpha=0.6, color='C1', ec=None, linewidth=0)
plt.xlim(-50, 300)
plt.ylim(-1e-5,1e-5)
plt.ylabel('Potential (μV)')
plt.xlabel('Time (ms)')
plt.grid(alpha=0.5)
plt.vlines(0,-1e-5,1e-5, linestyles='solid', color='k', linewidth=2, alpha=0.5)
plt.hlines(0,-50,300, linestyles='solid', color='k', linewidth=2, alpha=0.5)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
#plt.title("Averaged Response Regress on ANM (ABR) " + "(num=" + str(subject_num) + ')')
plt.legend(fontsize=10)
plt.tight_layout()
plt.title("HWR (Phase-only)")
plt.savefig(figure_path+'waveforms_whited_HWR_cortical_hp.svg', dpi=dpi, format='svg')
# %% Pulse train
t_start = -0.2
t_stop = 0.6
lags = np.arange(start=t_start*1000, stop=t_stop*1000, step=1e3/eeg_fs)
pulse_unaltered_response_all = np.zeros((subject_num, 8000))
for subject in subject_list:
    subi = subject_list.index(subject)
    pulse_response_data = read_hdf5(opath + "subject_response/rect_pulse_response/" + eeg_root.format(subject,task) + '_data_responses.hdf5')
    temp = pulse_response_data['w_pulses'][0]
    pulse_unaltered_response_all[subi] = np.concatenate((temp[int(t_start*eeg_fs):], temp[0:int(t_stop*eeg_fs)]))
    
pulse_unaltered_response_ave = np.sum(pulse_unaltered_response_all, axis=0) / subject_num
pulse_unaltered_response_err = np.std(pulse_unaltered_response_all, axis=0) / np.sqrt(subject_num)

pulse_broadband_response_all = np.zeros((subject_num, 8000))
for subject in subject_list:
    subi = subject_list.index(subject)
    pulse_response_data = read_hdf5(opath + "subject_response/rect_pulse_response/" + eeg_root.format(subject,task) + '_data_responses.hdf5')
    temp = pulse_response_data['w_pulses'][1]
    pulse_broadband_response_all[subi] = np.concatenate((temp[int(t_start*eeg_fs):], temp[0:int(t_stop*eeg_fs)]))
    
pulse_broadband_response_ave = np.sum(pulse_broadband_response_all, axis=0) / subject_num
pulse_broadband_response_err = np.std(pulse_broadband_response_all, axis=0) / np.sqrt(subject_num)

write_hdf5(opath + 'subject_response/' + 'pulse_response_all_subjects.hdf5',
           dict(pulse_unaltered_response_all=pulse_unaltered_response_all,
                pulse_unaltered_response_ave=pulse_unaltered_response_ave,
                pulse_unaltered_response_err=pulse_unaltered_response_err,
                pulse_broadband_response_all=pulse_broadband_response_all,
                pulse_broadband_response_ave=pulse_broadband_response_ave,
                pulse_broadband_response_err=pulse_broadband_response_err), overwrite=True)

#ABR
fig = plt.figure(dpi=dpi)
fig.set_size_inches(8, 6)
plt.plot(lags, pulse_unaltered_response_ave, c='C2', linewidth=1.2, label='Unaltered')
plt.fill_between(lags, pulse_unaltered_response_ave-pulse_unaltered_response_err, pulse_unaltered_response_ave+pulse_unaltered_response_err, alpha=0.6, color='C2', linewidth=0)
plt.plot(lags, pulse_broadband_response_ave, c='C3', linewidth=1.2, linestyle='--', label='Peaky')
plt.fill_between(lags, pulse_broadband_response_ave-pulse_broadband_response_err, pulse_broadband_response_ave+pulse_broadband_response_err, alpha=0.6, color='C3', linewidth=0)
plt.xlim(-10, 30)
plt.ylim(-1e-7,3e-7)
plt.ylabel('Magnitude (AU)')
plt.xlabel('Time (ms)')
plt.grid(alpha=0.5)
plt.vlines(0,-1e-7,3e-7, linestyles='solid', color='k', linewidth=2, alpha=0.5)
plt.hlines(0,-10,30, linestyles='solid', color='k', linewidth=2, alpha=0.5)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.legend(fontsize=10)
plt.tight_layout()

#cortical
fig = plt.figure(dpi=dpi)
fig.set_size_inches(8, 6)
plt.plot(lags, pulse_unaltered_response_ave, c='C2', linewidth=1.2, label='Unaltered (pulse)')
plt.fill_between(lags, pulse_unaltered_response_ave-pulse_unaltered_response_err, pulse_unaltered_response_ave+pulse_unaltered_response_err, alpha=0.6, color='C2', linewidth=0)
plt.plot(lags, pulse_broadband_response_ave, c='C3', linewidth=1.2, linestyle='--', label='Peaky (pulse)')
plt.fill_between(lags, pulse_broadband_response_ave-pulse_broadband_response_err, pulse_broadband_response_ave+pulse_broadband_response_err, alpha=0.6, color='C3', linewidth=0)
plt.xlim(-50, 300)
plt.ylim(-1e-7,3e-7)
plt.ylabel('Magnitude (AU)')
plt.xlabel('Time (ms)')
plt.grid(alpha=0.5)
plt.vlines(0,-1e-7,3e-7, linestyles='solid', color='k', linewidth=2, alpha=0.5)
plt.hlines(0,-50,300, linestyles='solid', color='k', linewidth=2, alpha=0.5)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.legend(fontsize=15)
plt.tight_layout()


# %% Pulse train - new responses
pulse_unaltered_response_all = np.zeros((subject_num, 8000))
for subject in subject_list:
    subi = subject_list.index(subject)
    pulse_response_data = read_hdf5(opath + "subject_response/pulse_response/" + eeg_root.format(subject,task) + '_pulse_responses.hdf5')
    pulse_unaltered_response_all[subi] = pulse_response_data['abr_pulse'][0]
    lags = pulse_response_data['lags']
    
pulse_unaltered_response_ave = np.sum(pulse_unaltered_response_all*1e6, axis=0) / subject_num
pulse_unaltered_response_err = np.std(pulse_unaltered_response_all*1e6, axis=0) / np.sqrt(subject_num)

pulse_broadband_response_all = np.zeros((subject_num, 8000))
for subject in subject_list:
    subi = subject_list.index(subject)
    pulse_response_data = read_hdf5(opath + "subject_response/pulse_response/" + eeg_root.format(subject,task) + '_pulse_responses.hdf5')
    pulse_broadband_response_all[subi] = pulse_response_data['abr_pulse'][1]
    
pulse_broadband_response_ave = np.sum(pulse_broadband_response_all*1e6, axis=0) / subject_num
pulse_broadband_response_err = np.std(pulse_broadband_response_all*1e6, axis=0) / np.sqrt(subject_num)

write_hdf5(opath + 'subject_response/' + 'pulse_response_all_subjects_new.hdf5',
           dict(pulse_unaltered_response_all=pulse_unaltered_response_all,
                pulse_unaltered_response_ave=pulse_unaltered_response_ave,
                pulse_unaltered_response_err=pulse_unaltered_response_err,
                pulse_broadband_response_all=pulse_broadband_response_all,
                pulse_broadband_response_ave=pulse_broadband_response_ave,
                pulse_broadband_response_err=pulse_broadband_response_err),overwrite=True)

#ABR
fig = plt.figure(dpi=dpi)
fig.set_size_inches(3.5, 3)
plt.plot(lags, pulse_unaltered_response_ave, c='C2', linewidth=1, label='Unaltered')
plt.fill_between(lags, pulse_unaltered_response_ave-pulse_unaltered_response_err, pulse_unaltered_response_ave+pulse_unaltered_response_err, alpha=0.6, color='C2', linewidth=0)
plt.plot(lags, pulse_broadband_response_ave, c='C3', linewidth=1, label='Peaky')
plt.fill_between(lags, pulse_broadband_response_ave-pulse_broadband_response_err, pulse_broadband_response_ave+pulse_broadband_response_err, alpha=0.6, color='C3', linewidth=0)
plt.xlim(-10, 15)
plt.ylim(-0.1,0.3)
plt.ylabel('Potential (μV)')
plt.xlabel('Time (ms)')
plt.grid(alpha=0.5)
plt.vlines(0,-0.1,0.3, linestyles='solid', color='k', linewidth=2, alpha=0.5)
plt.hlines(0,-10,30, linestyles='solid', color='k', linewidth=2, alpha=0.5)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.legend(fontsize=10)
plt.tight_layout()
plt.title("GP")
plt.savefig(figure_path+'waveforms_pulse_ABR.svg', dpi=dpi, format='svg')

#cortical
fig = plt.figure(dpi=dpi)
fig.set_size_inches(3.5, 3)
plt.plot(lags, pulse_unaltered_response_ave, c='C2', linewidth=1, label='Unaltered')
plt.fill_between(lags, pulse_unaltered_response_ave-pulse_unaltered_response_err, pulse_unaltered_response_ave+pulse_unaltered_response_err, alpha=0.6, color='C2', linewidth=0)
plt.plot(lags, pulse_broadband_response_ave, c='C3', linewidth=1, label='Peaky')
plt.fill_between(lags, pulse_broadband_response_ave-pulse_broadband_response_err, pulse_broadband_response_ave+pulse_broadband_response_err, alpha=0.6, color='C3', linewidth=0)
plt.xlim(-50, 300)
plt.ylim(-0.2,0.3)
plt.ylabel('Potential (μV)')
plt.xlabel('Time (ms)')
plt.grid(alpha=0.5)
plt.vlines(0,-0.2,0.3, linestyles='solid', color='k', linewidth=2, alpha=0.5)
plt.hlines(0,-50,300, linestyles='solid', color='k', linewidth=2, alpha=0.5)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.legend(fontsize=10)
plt.tight_layout()
plt.title("GP")
plt.savefig(figure_path+'waveforms_pulse_cortical.svg', dpi=dpi, format='svg')

# %% Pulse train - highpass 150 Hz
pulse_unaltered_response_all = np.zeros((subject_num, 8000))
for subject in subject_list:
    subi = subject_list.index(subject)
    pulse_response_data = read_hdf5(opath + "subject_response/pulse_response/" + eeg_root.format(subject,task) + '_pulse_responses.hdf5')
    temp = butter_highpass_filter(pulse_response_data['abr_pulse'][0], 150, eeg_fs)
    pulse_unaltered_response_all[subi] = temp
    lags = pulse_response_data['lags']
    
pulse_unaltered_response_ave = np.sum(pulse_unaltered_response_all*1e6, axis=0) / subject_num
pulse_unaltered_response_err = np.std(pulse_unaltered_response_all*1e6, axis=0) / np.sqrt(subject_num)

pulse_broadband_response_all = np.zeros((subject_num, 8000))
for subject in subject_list:
    subi = subject_list.index(subject)
    pulse_response_data = read_hdf5(opath + "subject_response/pulse_response/" + eeg_root.format(subject,task) + '_pulse_responses.hdf5')
    temp = butter_highpass_filter(pulse_response_data['abr_pulse'][1], 150, eeg_fs)
    pulse_broadband_response_all[subi] = temp
    
pulse_broadband_response_ave = np.sum(pulse_broadband_response_all*1e6, axis=0) / subject_num
pulse_broadband_response_err = np.std(pulse_broadband_response_all*1e6, axis=0) / np.sqrt(subject_num)

#ABR
fig = plt.figure(dpi=dpi)
fig.set_size_inches(3.5, 3)
plt.plot(lags, pulse_unaltered_response_ave, c='C2', linewidth=1, label='Unaltered')
plt.fill_between(lags, pulse_unaltered_response_ave-pulse_unaltered_response_err, pulse_unaltered_response_ave+pulse_unaltered_response_err, alpha=0.6, color='C2', linewidth=0)
plt.plot(lags, pulse_broadband_response_ave, c='C3', linewidth=1, label='Peaky')
plt.fill_between(lags, pulse_broadband_response_ave-pulse_broadband_response_err, pulse_broadband_response_ave+pulse_broadband_response_err, alpha=0.6, color='C3', linewidth=0)
plt.xlim(-10, 15)
plt.ylim(-0.15,0.15)
plt.ylabel('Potential (μV)')
plt.xlabel('Time (ms)')
plt.grid(alpha=0.5)
plt.vlines(0,-0.15,0.15, linestyles='solid', color='k', linewidth=2, alpha=0.5)
plt.hlines(0,-10,30, linestyles='solid', color='k', linewidth=2, alpha=0.5)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.legend(fontsize=10)
plt.tight_layout()
plt.title("GP")
plt.savefig(figure_path+'waveforms_pulse_ABR_hp150.svg', dpi=dpi, format='svg')

# %% Pulse train Whited- new responses
do_HP = True
pulse_unaltered_response_all = np.zeros((subject_num, 14000))
for subject in subject_list:
    subi = subject_list.index(subject)
    pulse_response_data = read_hdf5(opath + "subject_response/pulse_response/" + eeg_root.format(subject,task) + '_pulse_whited_responses.hdf5')
    temp = pulse_response_data['abr_pulse'][0]
    if do_HP:
        pulse_unaltered_response_all[subi] = butter_highpass_filter(temp, 150, fs=eeg_fs)
    else:
        pulse_unaltered_response_all[subi] = temp
    lags = pulse_response_data['lags']
    
pulse_unaltered_response_ave = np.sum(pulse_unaltered_response_all*1e6, axis=0) / subject_num
pulse_unaltered_response_err = np.std(pulse_unaltered_response_all*1e6, axis=0) / np.sqrt(subject_num)

pulse_broadband_response_all = np.zeros((subject_num, 14000))
for subject in subject_list:
    subi = subject_list.index(subject)
    pulse_response_data = read_hdf5(opath + "subject_response/pulse_response/" + eeg_root.format(subject,task) + '_pulse_whited_responses.hdf5')
    temp = pulse_response_data['abr_pulse'][1]
    if do_HP:
        pulse_broadband_response_all[subi] = butter_highpass_filter(temp, 150, fs=eeg_fs)
    else:
        pulse_broadband_response_all[subi] = temp
    
pulse_broadband_response_ave = np.sum(pulse_broadband_response_all*1e6, axis=0) / subject_num
pulse_broadband_response_err = np.std(pulse_broadband_response_all*1e6, axis=0) / np.sqrt(subject_num)

write_hdf5(opath + 'subject_response/' + 'pulse_whited_response_all_subjects_new.hdf5',
           dict(pulse_unaltered_response_all=pulse_unaltered_response_all,
                pulse_unaltered_response_ave=pulse_unaltered_response_ave,
                pulse_unaltered_response_err=pulse_unaltered_response_err,
                pulse_broadband_response_all=pulse_broadband_response_all,
                pulse_broadband_response_ave=pulse_broadband_response_ave,
                pulse_broadband_response_err=pulse_broadband_response_err),overwrite=True)

#ABR
fig = plt.figure(dpi=dpi)
fig.set_size_inches(3.5, 3)
plt.plot(lags, pulse_unaltered_response_ave, c='C2', linewidth=1, label='Unaltered')
plt.fill_between(lags, pulse_unaltered_response_ave-pulse_unaltered_response_err, pulse_unaltered_response_ave+pulse_unaltered_response_err, alpha=0.6, color='C2', linewidth=0)
plt.plot(lags, pulse_broadband_response_ave, c='C3', linewidth=1, label='Peaky')
plt.fill_between(lags, pulse_broadband_response_ave-pulse_broadband_response_err, pulse_broadband_response_ave+pulse_broadband_response_err, alpha=0.6, color='C3', linewidth=0)
plt.xlim(-10, 15)
plt.ylim(-7,17)
#plt.ylim(-9,7)
plt.ylabel('Magnitude (AU)')
plt.xlabel('Time (ms)')
plt.grid(alpha=0.5)
plt.vlines(0,-9,17, linestyles='solid', color='k', linewidth=2, alpha=0.5)
plt.hlines(0,-10,30, linestyles='solid', color='k', linewidth=2, alpha=0.5)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.legend(fontsize=10)
plt.title("GP (Phase-only)")
plt.tight_layout()
plt.savefig(figure_path+'waveforms_whited_pulse_ABR.svg', dpi=dpi, format='svg')

#cortical
fig = plt.figure(dpi=dpi)
fig.set_size_inches(8, 6)
plt.plot(lags, pulse_unaltered_response_ave, c='C2', linewidth=1.2, label='Unaltered')
plt.fill_between(lags, pulse_unaltered_response_ave-pulse_unaltered_response_err, pulse_unaltered_response_ave+pulse_unaltered_response_err, alpha=0.6, color='C2', linewidth=0)
plt.plot(lags, pulse_broadband_response_ave, c='C3', linewidth=1.2, linestyle='--', label='Peaky')
plt.fill_between(lags, pulse_broadband_response_ave-pulse_broadband_response_err, pulse_broadband_response_ave+pulse_broadband_response_err, alpha=0.6, color='C3', linewidth=0)
plt.xlim(-50, 300)
plt.ylim(-1e-5,1.7e-5)
plt.ylabel('Magnitude (AU)')
plt.xlabel('Time (ms)')
plt.grid(alpha=0.5)
plt.vlines(0,-1e-5,1.7e-5, linestyles='solid', color='k', linewidth=2, alpha=0.5)
plt.hlines(0,-50,300, linestyles='solid', color='k', linewidth=2, alpha=0.5)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.legend(fontsize=10)
plt.tight_layout()
plt.title("GP (Phase-only)")
plt.savefig(figure_path+'waveforms_whited_pulse_cortical_hp.svg', dpi=dpi, format='svg')
# %% ANM
ANM_unaltered_response_all = np.zeros((subject_num, 8000))
for subject in subject_list:
    subi = subject_list.index(subject)
    anm_response_data = read_hdf5(opath + "subject_response/ANM_response/" + eeg_root.format(subject,task) + '_ANM_responses.hdf5')
    ANM_unaltered_response_all[subi] = anm_response_data['abr_anm'][0]
    lags = anm_response_data['lags']
    
ANM_unaltered_response_ave = np.sum(ANM_unaltered_response_all, axis=0) / subject_num
ANM_unaltered_response_err = np.std(ANM_unaltered_response_all, axis=0) / np.sqrt(subject_num)

ANM_broadband_response_all = np.zeros((subject_num, 8000))
for subject in subject_list:
    subi = subject_list.index(subject)
    anm_response_data = read_hdf5(opath + "subject_response/ANM_response/" + eeg_root.format(subject,task) + '_ANM_responses.hdf5')
    ANM_broadband_response_all[subi] = anm_response_data['abr_anm'][1]
    
ANM_broadband_response_ave = np.sum(ANM_broadband_response_all, axis=0) / subject_num
ANM_broadband_response_err = np.std(ANM_broadband_response_all, axis=0) / np.sqrt(subject_num)

write_hdf5(opath + 'subject_response/' + 'ANM_response_all_subjects.hdf5',
           dict(ANM_unaltered_response_all=ANM_unaltered_response_all,
                ANM_unaltered_response_ave=ANM_unaltered_response_ave,
                ANM_unaltered_response_err=ANM_unaltered_response_err,
                ANM_broadband_response_all=ANM_broadband_response_all,
                ANM_broadband_response_ave=ANM_broadband_response_ave,
                ANM_broadband_response_err=ANM_broadband_response_err),overwrite=True)

#ABR
fig = plt.figure(dpi=dpi)
fig.set_size_inches(3.5, 3)
plt.plot(lags, ANM_unaltered_response_ave, c='C4', linewidth=1, label='Unaltered')
plt.fill_between(lags, ANM_unaltered_response_ave-ANM_unaltered_response_err, ANM_unaltered_response_ave+ANM_unaltered_response_err, alpha=0.6, color='C4', linewidth=0)
plt.plot(lags, ANM_broadband_response_ave, c='C5', linewidth=1,label='Peaky')
plt.fill_between(lags, ANM_broadband_response_ave-ANM_broadband_response_err, ANM_broadband_response_ave+ANM_broadband_response_err, alpha=0.6, color='C5', ec=None, linewidth=0)
plt.xlim(-10, 15)
plt.ylim(-50,80)
plt.ylabel('Magnitude (AU)')
plt.xlabel('Time (ms)')
plt.grid(alpha=0.5)
plt.vlines(0,-50,80, linestyles='solid', color='k', linewidth=2, alpha=0.5)
plt.hlines(0,-10,30, linestyles='solid', color='k', linewidth=2, alpha=0.5)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
#plt.title("Averaged Response Regress on ANM (ABR) " + "(num=" + str(subject_num) + ')')
plt.legend(fontsize=10)
plt.title("ANM")
plt.tight_layout()
plt.savefig(figure_path+'waveforms_ANM_ABR.svg', dpi=dpi, format='svg')


#cortical
fig = plt.figure(dpi=dpi)
fig.set_size_inches(3.5, 3)
plt.plot(lags, ANM_unaltered_response_ave, c='C4', linewidth=1, label='Unaltered')
plt.fill_between(lags, ANM_unaltered_response_ave-ANM_unaltered_response_err, ANM_unaltered_response_ave+ANM_unaltered_response_err, alpha=0.6, color='C4', linewidth=0)
plt.plot(lags, ANM_broadband_response_ave, c='C5', linewidth=1, label='Peaky')
plt.fill_between(lags, ANM_broadband_response_ave-ANM_broadband_response_err, ANM_broadband_response_ave+ANM_broadband_response_err, alpha=0.6, color='C5', ec=None, linewidth=0)
plt.xlim(-50, 300)
plt.ylim(-60,80)
plt.ylabel('Magnitude (AU)')
plt.xlabel('Time (ms)')
plt.grid(alpha=0.5)
plt.vlines(0,-60,80, linestyles='solid', color='k', linewidth=2, alpha=0.5)
plt.hlines(0,-50,300, linestyles='solid', color='k', linewidth=2, alpha=0.5)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
#plt.title("Averaged Response Regress on ANM (ABR) " + "(num=" + str(subject_num) + ')')
plt.legend(fontsize=10)
plt.title("ANM")
plt.tight_layout()
plt.savefig(figure_path+'waveforms_ANM_cortical.svg', dpi=dpi, format='svg')

# %% ANM high-pass 150 Hz
ANM_unaltered_response_all = np.zeros((subject_num, 8000))
for subject in subject_list:
    subi = subject_list.index(subject)
    anm_response_data = read_hdf5(opath + "subject_response/ANM_response/" + eeg_root.format(subject,task) + '_ANM_responses.hdf5')
    temp = butter_highpass_filter(anm_response_data['abr_anm'][0], 150, eeg_fs)
    ANM_unaltered_response_all[subi] = temp
    lags = anm_response_data['lags']
    
ANM_unaltered_response_ave = np.sum(ANM_unaltered_response_all, axis=0) / subject_num
ANM_unaltered_response_err = np.std(ANM_unaltered_response_all, axis=0) / np.sqrt(subject_num)

ANM_broadband_response_all = np.zeros((subject_num, 8000))
for subject in subject_list:
    subi = subject_list.index(subject)
    anm_response_data = read_hdf5(opath + "subject_response/ANM_response/" + eeg_root.format(subject,task) + '_ANM_responses.hdf5')
    temp = butter_highpass_filter(anm_response_data['abr_anm'][1], 150, eeg_fs)
    ANM_broadband_response_all[subi] = temp
    
ANM_broadband_response_ave = np.sum(ANM_broadband_response_all, axis=0) / subject_num
ANM_broadband_response_err = np.std(ANM_broadband_response_all, axis=0) / np.sqrt(subject_num)

#ABR
fig = plt.figure(dpi=dpi)
fig.set_size_inches(3.5, 3)
plt.plot(lags, ANM_unaltered_response_ave, c='C4', linewidth=1, label='Unaltered')
plt.fill_between(lags, ANM_unaltered_response_ave-ANM_unaltered_response_err, ANM_unaltered_response_ave+ANM_unaltered_response_err, alpha=0.6, color='C4', linewidth=0)
plt.plot(lags, ANM_broadband_response_ave, c='C5', linewidth=1,label='Peaky')
plt.fill_between(lags, ANM_broadband_response_ave-ANM_broadband_response_err, ANM_broadband_response_ave+ANM_broadband_response_err, alpha=0.6, color='C5', ec=None, linewidth=0)
plt.xlim(-10, 15)
plt.ylim(-60,50)
plt.ylabel('Magnitude (AU)')
plt.xlabel('Time (ms)')
plt.grid(alpha=0.5)
plt.vlines(0,-60,50, linestyles='solid', color='k', linewidth=2, alpha=0.5)
plt.hlines(0,-10,30, linestyles='solid', color='k', linewidth=2, alpha=0.5)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
#plt.title("Averaged Response Regress on ANM (ABR) " + "(num=" + str(subject_num) + ')')
plt.legend(fontsize=10)
plt.title("ANM")
plt.tight_layout()
plt.savefig(figure_path+'waveforms_ANM_ABR_hp150.svg', dpi=dpi, format='svg')

# %% ANM Whited
do_HP = False
ANM_unaltered_response_all = np.zeros((subject_num, 14000))
for subject in subject_list:
    subi = subject_list.index(subject)
    anm_response_data = read_hdf5(opath + "subject_response/ANM_response/" + eeg_root.format(subject,task) + '_ANM_whited_responses.hdf5')
    temp = anm_response_data['abr_anm'][0]
    if do_HP:
        ANM_unaltered_response_all[subi] = butter_highpass_filter(temp, 150, fs=eeg_fs)
    else:
        ANM_unaltered_response_all[subi] = temp
    lags = anm_response_data['lags']
    
ANM_unaltered_response_ave = np.sum(ANM_unaltered_response_all*1e6, axis=0) / subject_num
ANM_unaltered_response_err = np.std(ANM_unaltered_response_all*1e6, axis=0) / np.sqrt(subject_num)

ANM_broadband_response_all = np.zeros((subject_num, 14000))
for subject in subject_list:
    subi = subject_list.index(subject)
    anm_response_data = read_hdf5(opath + "subject_response/ANM_response/" + eeg_root.format(subject,task) + '_ANM_whited_responses.hdf5')
    temp = anm_response_data['abr_anm'][1]
    if do_HP:
        ANM_broadband_response_all[subi] = butter_highpass_filter(temp, 150, fs=eeg_fs)
    else:
        ANM_broadband_response_all[subi] = temp
    
ANM_broadband_response_ave = np.sum(ANM_broadband_response_all*1e6, axis=0) / subject_num
ANM_broadband_response_err = np.std(ANM_broadband_response_all*1e6, axis=0) / np.sqrt(subject_num)

write_hdf5(opath + 'subject_response/' + 'ANM_whited_response_all_subjects.hdf5',
           dict(ANM_unaltered_response_all=ANM_unaltered_response_all,
                ANM_unaltered_response_ave=ANM_unaltered_response_ave,
                ANM_unaltered_response_err=ANM_unaltered_response_err,
                ANM_broadband_response_all=ANM_broadband_response_all,
                ANM_broadband_response_ave=ANM_broadband_response_ave,
                ANM_broadband_response_err=ANM_broadband_response_err), overwrite=True)

#ABR
fig = plt.figure(dpi=dpi)
fig.set_size_inches(3.5, 3)
plt.plot(lags, ANM_unaltered_response_ave, c='C4', linewidth=1, label='Unaltered')
plt.fill_between(lags, ANM_unaltered_response_ave-ANM_unaltered_response_err, ANM_unaltered_response_ave+ANM_unaltered_response_err, alpha=0.6, color='C4', linewidth=0)
plt.plot(lags, ANM_broadband_response_ave, c='C5', linewidth=1, label='Peaky')
plt.fill_between(lags, ANM_broadband_response_ave-ANM_broadband_response_err, ANM_broadband_response_ave+ANM_broadband_response_err, alpha=0.6, color='C5', ec=None, linewidth=0)
plt.xlim(-10, 15)
plt.ylim(-7,17)
#plt.ylim(-0.9e-5,0.7e-5)
plt.ylabel('Potential (μV)')
plt.xlabel('Time (ms)')
plt.grid(alpha=0.5)
plt.vlines(0,-9,17, linestyles='solid', color='k', linewidth=2, alpha=0.5)
plt.hlines(0,-10,30, linestyles='solid', color='k', linewidth=2, alpha=0.5)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
#plt.title("Averaged Response Regress on ANM (ABR) " + "(num=" + str(subject_num) + ')')
plt.legend(fontsize=10)
plt.title("ANM (Phase-only)")
plt.tight_layout()
plt.savefig(figure_path+'waveforms_whited_ANM_ABR.svg', dpi=dpi, format='svg')

#cortical
fig = plt.figure(dpi=dpi)
fig.set_size_inches(3.5, 3)
plt.plot(lags, ANM_unaltered_response_ave, c='C4', linewidth=1.2, label='Unaltered')
plt.fill_between(lags, ANM_unaltered_response_ave-ANM_unaltered_response_err, ANM_unaltered_response_ave+ANM_unaltered_response_err, alpha=0.6, color='C4', linewidth=0)
plt.plot(lags, ANM_broadband_response_ave, c='C5', linewidth=1.2, linestyle='--', label='Peaky')
plt.fill_between(lags, ANM_broadband_response_ave-ANM_broadband_response_err, ANM_broadband_response_ave+ANM_broadband_response_err, alpha=0.6, color='C5', ec=None, linewidth=0)
plt.xlim(-50, 300)
plt.ylim(-0.8e-5,1.7e-5)
plt.ylabel('Magnitude (AU)')
plt.xlabel('Time (ms)')
plt.grid(alpha=0.5)
plt.vlines(0,-0.8e-5,1.7e-5, linestyles='solid', color='k', linewidth=2, alpha=0.5)
plt.hlines(0,-50,300, linestyles='solid', color='k', linewidth=2, alpha=0.5)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
#plt.title("Averaged Response Regress on ANM (ABR) " + "(num=" + str(subject_num) + ')')
plt.legend(fontsize=10)
plt.tight_layout()
plt.title("ANM (Phase-only)")
plt.savefig(figure_path+'waveforms_whited_ANM_cortical.svg', dpi=dpi, format='svg')

# %% Whited responses Peaky vs ANM
#ABR
fig = plt.figure(dpi=dpi)
fig.set_size_inches(5, 4)
plt.plot(lags, rect_unaltered_response_ave, c='C0', linewidth=1, label='Unaltered (HWR)')
plt.fill_between(lags, rect_unaltered_response_ave-rect_unaltered_response_err, rect_unaltered_response_ave+rect_unaltered_response_err, alpha=0.6, color='C0', linewidth=0)
plt.plot(lags, rect_broadband_response_ave, c='C1', linewidth=1, label='Peaky (HWR)')
plt.fill_between(lags, rect_broadband_response_ave-rect_broadband_response_err, rect_broadband_response_ave+rect_broadband_response_err, alpha=0.6, color='C1', ec=None, linewidth=0)
plt.plot(lags, pulse_unaltered_response_ave, c='C2', linewidth=1, label='Unaltered (pulse)')
plt.fill_between(lags, pulse_unaltered_response_ave-pulse_unaltered_response_err, pulse_unaltered_response_ave+pulse_unaltered_response_err, alpha=0.6, color='C2', linewidth=0)
plt.plot(lags, pulse_broadband_response_ave, c='C3', linewidth=1, label='Peaky (pulse)')
plt.fill_between(lags, pulse_broadband_response_ave-pulse_broadband_response_err, pulse_broadband_response_ave+pulse_broadband_response_err, alpha=0.6, color='C3', linewidth=0)
plt.plot(lags, ANM_unaltered_response_ave, c='C4', linewidth=1, label='Unaltered (ANM)')
plt.fill_between(lags, ANM_unaltered_response_ave-ANM_unaltered_response_err, ANM_unaltered_response_ave+ANM_unaltered_response_err, alpha=0.6, color='C4', linewidth=0)
plt.plot(lags, ANM_broadband_response_ave, c='C5', linewidth=1, label='Peaky (ANM)')
plt.fill_between(lags, ANM_broadband_response_ave-ANM_broadband_response_err, ANM_broadband_response_ave+ANM_broadband_response_err, alpha=0.6, color='C5', ec=None, linewidth=0)
plt.xlim(-10, 30)
plt.ylim(-7,17)
plt.ylabel('Potential (μV)')
plt.xlabel('Time (ms)')
plt.grid(alpha=0.5)
plt.vlines(0,-7,17, linestyles='solid', color='k', linewidth=2, alpha=0.5)
plt.hlines(0,-10,30, linestyles='solid', color='k', linewidth=2, alpha=0.5)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
#plt.title("Averaged Response Regress on ANM (ABR) " + "(num=" + str(subject_num) + ')')
lg = plt.legend(fontsize=10,bbox_to_anchor=(1.01,0.55))
plt.savefig(figure_path + 'waveforms_whited_regressors_ABR.svg',  dpi=dpi, format='svg', bbox_extra_artists=(lg,), bbox_inches='tight')

#Cortical
fig = plt.figure(dpi=dpi)
fig.set_size_inches(3.5, 3)
plt.plot(lags, rect_unaltered_response_ave, c='C0', linewidth=1.2, label='Unaltered (HWR)')
plt.fill_between(lags, rect_unaltered_response_ave-rect_unaltered_response_err, rect_unaltered_response_ave+rect_unaltered_response_err, alpha=0.6, color='C0', linewidth=0)
plt.plot(lags, rect_broadband_response_ave, c='C1', linewidth=1.2, linestyle='--', label='Peaky (HWR)')
plt.fill_between(lags, rect_broadband_response_ave-rect_broadband_response_err, rect_broadband_response_ave+rect_broadband_response_err, alpha=0.6, color='C1', ec=None, linewidth=0)
plt.plot(lags, pulse_unaltered_response_ave, c='C2', linewidth=1.2, label='Unaltered (GP)')
plt.fill_between(lags, pulse_unaltered_response_ave-pulse_unaltered_response_err, pulse_unaltered_response_ave+pulse_unaltered_response_err, alpha=0.6, color='C2', linewidth=0)
plt.plot(lags, pulse_broadband_response_ave, c='C3', linewidth=1.2, linestyle='--', label='Peaky (GP)')
plt.fill_between(lags, pulse_broadband_response_ave-pulse_broadband_response_err, pulse_broadband_response_ave+pulse_broadband_response_err, alpha=0.6, color='C3', linewidth=0)
plt.plot(lags, ANM_unaltered_response_ave, c='C4', linewidth=1.2, label='Unaltered (ANM)')
plt.fill_between(lags, ANM_unaltered_response_ave-ANM_unaltered_response_err, ANM_unaltered_response_ave+ANM_unaltered_response_err, alpha=0.6, color='C4', linewidth=0)
plt.plot(lags, ANM_broadband_response_ave, c='C5', linewidth=1.2, linestyle='--', label='Peaky (ANM)')
plt.fill_between(lags, ANM_broadband_response_ave-ANM_broadband_response_err, ANM_broadband_response_ave+ANM_broadband_response_err, alpha=0.6, color='C5', ec=None, linewidth=0)
plt.xlim(-30, 50)
plt.ylim(-10,17)
plt.ylabel('Potential (μV)')
plt.xlabel('Time (ms)')
plt.grid(alpha=0.5)
plt.vlines(0,-10,17, linestyles='solid', color='k', linewidth=2, alpha=0.5)
plt.hlines(0,-50, 1200, linestyles='solid', color='k', linewidth=2, alpha=0.5)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
#plt.title("Averaged Response Regress on ANM (ABR) " + "(num=" + str(subject_num) + ')')
plt.legend(fontsize=10)
plt.tight_layout()
plt.savefig(figure_path + 'waveforms_whited_regressors_cortical.png',  dpi=dpi, format='png')

# For high passed
#ABR
fig = plt.figure(dpi=dpi)
fig.set_size_inches(5, 4)
plt.plot(lags, rect_unaltered_response_ave, c='C0', linewidth=1, label='Unaltered (HWR)')
plt.fill_between(lags, rect_unaltered_response_ave-rect_unaltered_response_err, rect_unaltered_response_ave+rect_unaltered_response_err, alpha=0.6, color='C0', linewidth=0)
plt.plot(lags, rect_broadband_response_ave, c='C1', linewidth=1,label='Peaky (HWR)')
plt.fill_between(lags, rect_broadband_response_ave-rect_broadband_response_err, rect_broadband_response_ave+rect_broadband_response_err, alpha=0.6, color='C1', ec=None, linewidth=0)
plt.plot(lags, pulse_unaltered_response_ave, c='C2', linewidth=1, label='Unaltered (pulse)')
plt.fill_between(lags, pulse_unaltered_response_ave-pulse_unaltered_response_err, pulse_unaltered_response_ave+pulse_unaltered_response_err, alpha=0.6, color='C2', linewidth=0)
plt.plot(lags, pulse_broadband_response_ave, c='C3', linewidth=1, label='Peaky (pulse)')
plt.fill_between(lags, pulse_broadband_response_ave-pulse_broadband_response_err, pulse_broadband_response_ave+pulse_broadband_response_err, alpha=0.6, color='C3', linewidth=0)
plt.plot(lags, ANM_unaltered_response_ave, c='C4', linewidth=1, label='Unaltered (ANM)')
plt.fill_between(lags, ANM_unaltered_response_ave-ANM_unaltered_response_err, ANM_unaltered_response_ave+ANM_unaltered_response_err, alpha=0.6, color='C4', linewidth=0)
plt.plot(lags, ANM_broadband_response_ave, c='C5', linewidth=1, label='Peaky (ANM)')
plt.fill_between(lags, ANM_broadband_response_ave-ANM_broadband_response_err, ANM_broadband_response_ave+ANM_broadband_response_err, alpha=0.6, color='C5', ec=None, linewidth=0)
plt.xlim(-10, 15)
plt.ylim(-9,7)
plt.ylabel('Potential (μV)')
plt.xlabel('Time (ms)')
plt.grid(alpha=0.5)
plt.vlines(0,-9,7, linestyles='solid', color='k', linewidth=2, alpha=0.5)
plt.hlines(0,-10,30, linestyles='solid', color='k', linewidth=2, alpha=0.5)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
#plt.title("Averaged Response Regress on ANM (ABR) " + "(num=" + str(subject_num) + ')')
lg = plt.legend(fontsize=10,bbox_to_anchor=(1.01,0.55))
#plt.tight_layout()
plt.savefig(figure_path + 'waveforms_whited_regressors_ABR_HP150.svg', dpi=dpi, format='svg', bbox_extra_artists=(lg,), bbox_inches='tight')

# %% SNR analysis
# %% getting SNR dataframe
# Rectified
abr_time_window = [15, 30, 90]
SNR_rect_unaltered = np.zeros((len(abr_time_window), subject_num))
SNR_rect_broadband = np.zeros((len(abr_time_window), subject_num))
SNR_rect_unaltered_ave = np.zeros(len(abr_time_window),)
SNR_rect_broadband_ave = np.zeros(len(abr_time_window),)
for window in abr_time_window:
    wi = abr_time_window.index(window)
    for si in range(subject_num):
        SNR_rect_unaltered[wi, si] = abr_SNR(rect_unaltered_response_all[si], lags, abr_time_window=window, noise_range=[-200, -20])
        SNR_rect_broadband[wi, si] = abr_SNR(rect_broadband_response_all[si], lags, abr_time_window=window, noise_range=[-200, -20])

    SNR_rect_unaltered_ave[wi] = abr_SNR(rect_unaltered_response_ave, lags, abr_time_window=window, noise_range=[-200, -20])
    SNR_rect_broadband_ave[wi] = abr_SNR(rect_broadband_response_ave, lags, abr_time_window=window, noise_range=[-200, -20])

def abr_SNR_freq(abr_data, lags, abr_time_window=15, noise_range=[-200, -20]):
    """
    abr_time_window: time range in ms define as abr, 15 ms by default
    noise_range: prestim time range to calculate noise level, [-200, -20] ms by default
    """    
    ind_abr = np.where((lags>=0) & (lags<abr_time_window))
    freq_vec = np.arange(0, eeg_fs, step=(eeg_fs/len(ind_abr[0])))
    freq_vec = freq_vec[:int(len(freq_vec)/2)]
    abr_var = abs(fft(abr_data[ind_abr]))**2
    noise_seg_num = int((noise_range[1]-noise_range[0]) / abr_time_window)
    noise_var = np.zeros((abr_var.shape[-1]))
    for i in range(noise_seg_num):
        ind_noise = np.where((lags>(noise_range[0]+abr_time_window*i)) & (lags<=(noise_range[0]+abr_time_window*(i+1))))
        noise_var += abs(fft(abr_data[ind_noise]))**2
    noise_var = noise_var / noise_seg_num # averaging the var of noise
    SNR = 10*np.log10((abr_var - noise_var)/noise_var)
    SNR = SNR[:int((len(SNR)/2))]
    return SNR, freq_vec

SNR_rect_unaltered_ave_freq, freq_vec = abr_SNR_freq(rect_unaltered_response_ave, lags, 15, [-200, -20])
SNR_rect_broadband_ave_freq, freq_vec = abr_SNR_freq(rect_broadband_response_ave, lags, 15, [-200, -20])

# Pulse
abr_time_window = [15, 30, 90]
SNR_pulse_unaltered = np.zeros((len(abr_time_window), subject_num))
SNR_pulse_broadband = np.zeros((len(abr_time_window), subject_num))
SNR_pulse_unaltered_ave = np.zeros(len(abr_time_window),)
SNR_pulse_broadband_ave = np.zeros(len(abr_time_window),)
for window in abr_time_window:
    wi = abr_time_window.index(window)
    for si in range(subject_num):
        SNR_pulse_unaltered[wi, si] = abr_SNR(pulse_unaltered_response_all[si], lags, abr_time_window=window, noise_range=[-200, -20])
        SNR_pulse_broadband[wi, si] = abr_SNR(pulse_broadband_response_all[si], lags, abr_time_window=window, noise_range=[-200, -20])

    SNR_pulse_unaltered_ave[wi] = abr_SNR(pulse_unaltered_response_ave, lags, abr_time_window=window, noise_range=[-200, -20])
    SNR_pulse_broadband_ave[wi] = abr_SNR(pulse_broadband_response_ave, lags, abr_time_window=window, noise_range=[-200, -20])

SNR_pulse_unaltered_ave_freq, freq_vec = abr_SNR_freq(pulse_unaltered_response_ave, lags, 15, [-200, -20])
SNR_pulse_broadband_ave_freq, freq_vec = abr_SNR_freq(pulse_broadband_response_ave, lags, 15, [-200, -20])

# ANM
abr_time_window = [15, 30, 90]
SNR_ANM_unaltered = np.zeros((len(abr_time_window), subject_num))
SNR_ANM_broadband = np.zeros((len(abr_time_window), subject_num))
SNR_ANM_unaltered_ave = np.zeros(len(abr_time_window),)
SNR_ANM_broadband_ave = np.zeros(len(abr_time_window),)
for window in abr_time_window:
    wi = abr_time_window.index(window)
    for si in range(subject_num):
        SNR_ANM_unaltered[wi, si] = abr_SNR(ANM_unaltered_response_all[si], lags, abr_time_window=window, noise_range=[-200, -20])
        SNR_ANM_broadband[wi, si] = abr_SNR(ANM_broadband_response_all[si], lags, abr_time_window=window, noise_range=[-200, -20])

    SNR_ANM_unaltered_ave[wi] = abr_SNR(ANM_unaltered_response_ave, lags, abr_time_window=window, noise_range=[-200, -20])
    SNR_ANM_broadband_ave[wi] = abr_SNR(ANM_broadband_response_ave, lags, abr_time_window=window, noise_range=[-200, -20])

SNR_ANM_unaltered_ave_freq, freq_vec = abr_SNR_freq(ANM_unaltered_response_ave, lags, 15, [-200, -20])
SNR_ANM_broadband_ave_freq, freq_vec = abr_SNR_freq(ANM_broadband_response_ave, lags, 15, [-200, -20])


################## Statistics ####################

## Shapiro Normal dist test
shapiro(SNR_rect_unaltered[wi, :])
shapiro(SNR_pulse_unaltered[wi, :])
shapiro(SNR_ANM_unaltered[wi, :])
shapiro(SNR_rect_broadband[wi, :])
shapiro(SNR_pulse_broadband[wi, :])
shapiro(SNR_ANM_broadband[wi, :])
# #makeing dataframe
SNR_df = pd.DataFrame(columns=["subject","regressor","stim","window","SNR"])

SNR_df = pd.DataFrame({'subject':list(np.arange(22)),
               "regressor":list(np.repeat("HWR", 22)),
               "stim":list(np.repeat("unaltered", 22)),
               "window":list(np.repeat("15", 22)),
               "SNR":list(SNR_rect_unaltered[0,:])})

SNR_df = pd.concat([SNR_df, pd.DataFrame({'subject':list(np.arange(22)),
                                         "regressor":list(np.repeat("HWR", 22)),
                                         "stim":list(np.repeat("broadband", 22)),
                                         "window":list(np.repeat("15", 22)),
                                         "SNR":list(SNR_rect_broadband[0,:])})])

SNR_df = pd.concat([SNR_df, pd.DataFrame({'subject':list(np.arange(22)),
                                         "regressor":list(np.repeat("GPT", 22)),
                                         "stim":list(np.repeat("unaltered", 22)),
                                         "window":list(np.repeat("15", 22)),
                                         "SNR":list(SNR_pulse_unaltered[0,:])})])

SNR_df = pd.concat([SNR_df, pd.DataFrame({'subject':list(np.arange(22)),
                                         "regressor":list(np.repeat("GPT", 22)),
                                         "stim":list(np.repeat("broadband", 22)),
                                         "window":list(np.repeat("15", 22)),
                                         "SNR":list(SNR_pulse_broadband[0,:])})])

SNR_df = pd.concat([SNR_df, pd.DataFrame({'subject':list(np.arange(22)),
                                         "regressor":list(np.repeat("ANM", 22)),
                                         "stim":list(np.repeat("unaltered", 22)),
                                         "window":list(np.repeat("15", 22)),
                                         "SNR":list(SNR_ANM_unaltered[0,:])})])

SNR_df = pd.concat([SNR_df, pd.DataFrame({'subject':list(np.arange(22)),
                                         "regressor":list(np.repeat("ANM", 22)),
                                         "stim":list(np.repeat("broadband", 22)),
                                         "window":list(np.repeat("15", 22)),
                                         "SNR":list(SNR_ANM_broadband[0,:])})])

SNR_df = pd.concat([SNR_df, pd.DataFrame({'subject':list(np.arange(22)),
                                         "regressor":list(np.repeat("HWR", 22)),
                                         "stim":list(np.repeat("unaltered", 22)),
                                         "window":list(np.repeat("30", 22)),
                                         "SNR":list(SNR_rect_unaltered[1,:])})])

SNR_df = pd.concat([SNR_df, pd.DataFrame({'subject':list(np.arange(22)),
                                         "regressor":list(np.repeat("HWR", 22)),
                                         "stim":list(np.repeat("broadband", 22)),
                                         "window":list(np.repeat("30", 22)),
                                         "SNR":list(SNR_rect_broadband[1,:])})])

SNR_df = pd.concat([SNR_df, pd.DataFrame({'subject':list(np.arange(22)),
                                         "regressor":list(np.repeat("GPT", 22)),
                                         "stim":list(np.repeat("unaltered", 22)),
                                         "window":list(np.repeat("30", 22)),
                                         "SNR":list(SNR_pulse_unaltered[1,:])})])

SNR_df = pd.concat([SNR_df, pd.DataFrame({'subject':list(np.arange(22)),
                                         "regressor":list(np.repeat("GPT", 22)),
                                         "stim":list(np.repeat("broadband", 22)),
                                         "window":list(np.repeat("30", 22)),
                                         "SNR":list(SNR_pulse_broadband[1,:])})])

SNR_df = pd.concat([SNR_df, pd.DataFrame({'subject':list(np.arange(22)),
                                         "regressor":list(np.repeat("ANM", 22)),
                                         "stim":list(np.repeat("unaltered", 22)),
                                         "window":list(np.repeat("30", 22)),
                                         "SNR":list(SNR_ANM_unaltered[1,:])})])

SNR_df = pd.concat([SNR_df, pd.DataFrame({'subject':list(np.arange(22)),
                                         "regressor":list(np.repeat("ANM", 22)),
                                         "stim":list(np.repeat("broadband", 22)),
                                         "window":list(np.repeat("30", 22)),
                                         "SNR":list(SNR_ANM_broadband[1,:])})])

SNR_df.to_csv(opath+"ABR_phase-only_SNR_df.csv")

# %% SNR plotting
import pingouin
import scipy.stats as stats


SNR_df = pd.read_csv(opath+"ABR_SNR_df.csv")
#### Statistics ####
# For 15 ms SNR
SNR_df_15_unaltered = SNR_df[(SNR_df["window"]==15) & (SNR_df["stim"]=="unaltered")]
SNR_df_15_broadband = SNR_df[(SNR_df["window"]==15) & (SNR_df["stim"]=="broadband")]
SNR_df_15 = SNR_df[(SNR_df["window"]==15)].dropna()

# Random effect Linear regression
SNR_lm = smf.mixedlm("SNR ~ regressor + stim + regressor*stim", SNR_df_15, groups=SNR_df_15["subject"])
SNR_lmf = SNR_lm.fit()
print(SNR_lmf.summary())
# RM ANOVA
anova_unaltered = pingouin.rm_anova(data=SNR_df_15_unaltered, dv="SNR", subject="subject", within=["regressor"], detailed=True)
anova_unaltered.round(3)
anova_broadband = pingouin.rm_anova(data=SNR_df_15_broadband, dv="SNR", subject="subject", within=["regressor"], detailed=True)
anova_broadband.round(3)
# Pairwise T-TEST
pairwise_unaltered = pingouin.pairwise_ttests(data=SNR_df_15_unaltered, dv="SNR", subject="subject", within=["regressor"], parametric=True, alpha=0.05, padjust='holm', nan_policy='pairwise')
pairwise_unaltered.round(3)
pairwise_broadband = pingouin.pairwise_ttests(data=SNR_df_15_broadband, dv="SNR", subject="subject", within=["regressor"], parametric=True, alpha=0.05, padjust='holm', nan_policy='pairwise')
pairwise_broadband.round(3)
pairwise_SNR_15_all = pingouin.pairwise_ttests(data=SNR_df_15, dv="SNR", subject="subject", within=["regressor","stim"], parametric=True, alpha=0.05, padjust='holm', nan_policy='pairwise')

# statistic estimate
SNR_15_unaltered_ave = [SNR_df_15_unaltered[SNR_df_15_unaltered['regressor']=="HWR"]["SNR"].mean(),
                        SNR_df_15_unaltered[SNR_df_15_unaltered['regressor']=="GPT"]["SNR"].mean(),
                        SNR_df_15_unaltered[SNR_df_15_unaltered['regressor']=="ANM"]["SNR"].mean()]

SNR_15_unaltered_sem = [SNR_df_15_unaltered[SNR_df_15_unaltered['regressor']=="HWR"]["SNR"].sem(),
                        SNR_df_15_unaltered[SNR_df_15_unaltered['regressor']=="GPT"]["SNR"].sem(),
                        SNR_df_15_unaltered[SNR_df_15_unaltered['regressor']=="ANM"]["SNR"].sem()]

SNR_15_broadband_ave = [SNR_df_15_broadband[SNR_df_15_broadband['regressor']=="HWR"]["SNR"].mean(),
                        SNR_df_15_broadband[SNR_df_15_broadband['regressor']=="GPT"]["SNR"].mean(),
                        SNR_df_15_broadband[SNR_df_15_broadband['regressor']=="ANM"]["SNR"].mean()]

SNR_15_broadband_sem = [SNR_df_15_broadband[SNR_df_15_broadband['regressor']=="HWR"]["SNR"].sem(),
                        SNR_df_15_broadband[SNR_df_15_broadband['regressor']=="GPT"]["SNR"].sem(),
                        SNR_df_15_broadband[SNR_df_15_broadband['regressor']=="ANM"]["SNR"].sem()]

# Plotting
y=[SNR_15_unaltered_ave[0],SNR_15_broadband_ave[0],SNR_15_unaltered_ave[1],SNR_15_broadband_ave[1],SNR_15_unaltered_ave[2],SNR_15_broadband_ave[2]]
yerr = [SNR_15_unaltered_sem[0], SNR_15_broadband_sem[0], SNR_15_unaltered_sem[1], SNR_15_broadband_sem[1],SNR_15_unaltered_sem[2],SNR_15_broadband_sem[2]]
barWidth = 0.25
br1 = np.arange(2)
br2 = [x + barWidth for x in br1]
br3 = [x + barWidth for x in br2]
fig = plt.figure(dpi=dpi)
fig.set_size_inches(6, 3)
plt.bar(br1, [SNR_15_unaltered_ave[0],SNR_15_broadband_ave[0]], color='C0', width=barWidth, label='HWR')
plt.bar(br2, [SNR_15_unaltered_ave[1],SNR_15_broadband_ave[1]], color='C2', width=barWidth, label='GP')
plt.bar(br3, [SNR_15_unaltered_ave[2],SNR_15_broadband_ave[2]], color='C4', width=barWidth, label='ANM')
plt.errorbar([br1[0],br1[1],br2[0],br2[1],br3[0],br3[1]], y=y, yerr=yerr,linestyle="", capsize=3.5, color="k",zorder=4,linewidth=1)
plt.xlabel('Stimulus')
plt.ylabel('SNR (dB)')
plt.ylim(-15, 30)
plt.xticks([r + barWidth for r in range(2)], ['Unaltered', 'Peaky'])
plt.hlines(0, -0.125, 1.625, linestyles='solid', linewidth=1, alpha=1, color='k')
for si in range(subject_num):
    plt.plot([br1[0],br2[0], br3[0]],[SNR_df_15_unaltered[SNR_df_15_unaltered['regressor']=="HWR"].iloc[si]["SNR"],
                                      SNR_df_15_unaltered[SNR_df_15_unaltered['regressor']=="GPT"].iloc[si]["SNR"],
                                      SNR_df_15_unaltered[SNR_df_15_unaltered['regressor']=="ANM"].iloc[si]["SNR"]],
             ".-", linewidth=0.5, c='gray')
    plt.plot([br1[1],br2[1], br3[1]],[SNR_df_15_broadband[SNR_df_15_broadband['regressor']=="HWR"].iloc[si]["SNR"],
                                      SNR_df_15_broadband[SNR_df_15_broadband['regressor']=="GPT"].iloc[si]["SNR"],
                                      SNR_df_15_broadband[SNR_df_15_broadband['regressor']=="ANM"].iloc[si]["SNR"]], 
             ".-", linewidth=0.5, c='gray')
lg = plt.legend(fontsize=10, bbox_to_anchor=(1.03, 1.0), loc='upper left')
plt.grid(alpha=0.5)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.savefig(figure_path + 'SNR_comparison-{}ms.svg'.format(15), dpi=dpi, format='svg', bbox_extra_artists=(lg,), bbox_inches='tight')


# For 30 ms SNR
SNR_df_30_unaltered = SNR_df[(SNR_df["window"]==30) & (SNR_df["stim"]=="unaltered")]
SNR_df_30_broadband = SNR_df[(SNR_df["window"]==30) & (SNR_df["stim"]=="broadband")]
SNR_df_30 = SNR_df[(SNR_df["window"]==30)].dropna()

# Random effect Linear regression
SNR_lm = smf.mixedlm("SNR ~ regressor + stim + regressor*stim", SNR_df_30, groups=SNR_df_30["subject"])
SNR_lmf = SNR_lm.fit()
print(SNR_lmf.summary())
# RM ANOVA
anova_unaltered = pingouin.rm_anova(data=SNR_df_30_unaltered, dv="SNR", subject="subject", within=["regressor"], detailed=True)
anova_unaltered.round(3)
anova_broadband = pingouin.rm_anova(data=SNR_df_30_broadband, dv="SNR", subject="subject", within=["regressor"], detailed=True)
anova_broadband.round(3)
# Pairwise T-TEST
pairwise_unaltered = pingouin.pairwise_ttests(data=SNR_df_30_unaltered, dv="SNR", subject="subject", within=["regressor"], parametric=True, alpha=0.05, padjust='holm', nan_policy='pairwise')
pairwise_unaltered.round(3)
pairwise_broadband = pingouin.pairwise_ttests(data=SNR_df_30_broadband, dv="SNR", subject="subject", within=["regressor"], parametric=True, alpha=0.05, padjust='holm', nan_policy='pairwise')
pairwise_broadband.round(3)
pairwise_SNR_30_all = pingouin.pairwise_tests(data=SNR_df_30, dv="SNR", subject="subject", within=["regressor","stim"], parametric=True, alpha=0.05, padjust='holm', nan_policy='pairwise')

# statistic estimate
SNR_30_unaltered_ave = [SNR_df_30_unaltered[SNR_df_30_unaltered['regressor']=="HWR"]["SNR"].mean(),
                        SNR_df_30_unaltered[SNR_df_30_unaltered['regressor']=="GPT"]["SNR"].mean(),
                        SNR_df_30_unaltered[SNR_df_30_unaltered['regressor']=="ANM"]["SNR"].mean()]

SNR_30_unaltered_sem = [SNR_df_30_unaltered[SNR_df_30_unaltered['regressor']=="HWR"]["SNR"].sem(),
                        SNR_df_30_unaltered[SNR_df_30_unaltered['regressor']=="GPT"]["SNR"].sem(),
                        SNR_df_30_unaltered[SNR_df_30_unaltered['regressor']=="ANM"]["SNR"].sem()]

SNR_30_broadband_ave = [SNR_df_30_broadband[SNR_df_30_broadband['regressor']=="HWR"]["SNR"].mean(),
                        SNR_df_30_broadband[SNR_df_30_broadband['regressor']=="GPT"]["SNR"].mean(),
                        SNR_df_30_broadband[SNR_df_30_broadband['regressor']=="ANM"]["SNR"].mean()]

SNR_30_broadband_sem = [SNR_df_30_broadband[SNR_df_30_broadband['regressor']=="HWR"]["SNR"].sem(),
                        SNR_df_30_broadband[SNR_df_30_broadband['regressor']=="GPT"]["SNR"].sem(),
                        SNR_df_30_broadband[SNR_df_30_broadband['regressor']=="ANM"]["SNR"].sem()]

# Plotting
y=[SNR_30_unaltered_ave[0],SNR_30_broadband_ave[0],SNR_30_unaltered_ave[1],SNR_30_broadband_ave[1],SNR_30_unaltered_ave[2],SNR_30_broadband_ave[2]]
yerr = [SNR_30_unaltered_sem[0], SNR_30_broadband_sem[0], SNR_30_unaltered_sem[1], SNR_30_broadband_sem[1],SNR_30_unaltered_sem[2],SNR_30_broadband_sem[2]]
barWidth = 0.25
br1 = np.arange(2)
br2 = [x + barWidth for x in br1]
br3 = [x + barWidth for x in br2]
fig = plt.figure(dpi=dpi)
fig.set_size_inches(6, 3)
plt.bar(br1, [SNR_30_unaltered_ave[0],SNR_30_broadband_ave[0]], color='C0', width=barWidth, label='HWR')
plt.bar(br2, [SNR_30_unaltered_ave[1],SNR_30_broadband_ave[1]], color='C2', width=barWidth, label='GP')
plt.bar(br3, [SNR_30_unaltered_ave[2],SNR_30_broadband_ave[2]], color='C4', width=barWidth, label='ANM')
plt.errorbar([br1[0],br1[1],br2[0],br2[1],br3[0],br3[1]], y=y, yerr=yerr,linestyle="", capsize=3.5, color="k",zorder=4,linewidth=1)
plt.xlabel('Stimulus')
plt.ylabel('SNR (dB)')
plt.ylim(-20, 25)
plt.xticks([r + barWidth for r in range(2)], ['Unaltered', 'Peaky'])
plt.hlines(0, -0.125, 1.625, linestyles='solid', linewidth=1, alpha=1, color='k')
for si in range(subject_num):
    plt.plot([br1[0],br2[0], br3[0]],[SNR_df_30_unaltered[SNR_df_30_unaltered['regressor']=="HWR"].iloc[si]["SNR"],
                                      SNR_df_30_unaltered[SNR_df_30_unaltered['regressor']=="GPT"].iloc[si]["SNR"],
                                      SNR_df_30_unaltered[SNR_df_30_unaltered['regressor']=="ANM"].iloc[si]["SNR"]],
             ".-", linewidth=0.5, c='gray')
    plt.plot([br1[1],br2[1], br3[1]],[SNR_df_30_broadband[SNR_df_30_broadband['regressor']=="HWR"].iloc[si]["SNR"],
                                      SNR_df_30_broadband[SNR_df_30_broadband['regressor']=="GPT"].iloc[si]["SNR"],
                                      SNR_df_30_broadband[SNR_df_30_broadband['regressor']=="ANM"].iloc[si]["SNR"]], 
             ".-", linewidth=0.5, c='gray')
lg = plt.legend(fontsize=10, bbox_to_anchor=(1.03, 1.0), loc='upper left')
plt.grid(alpha=0.5)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.savefig(figure_path + 'SNR_comparison-{}ms.svg'.format(30), dpi=dpi, format='svg', bbox_extra_artists=(lg,), bbox_inches='tight')


# # Plotting SNR corrected
# snr_corrected = 10*np.log10(subject_num)
# barWidth = 0.25
# br1 = np.arange(2)
# br2 = [x + barWidth for x in br1]
# br3 = [x + barWidth for x in br2]
# fig = plt.figure(dpi=dpi)
# fig.set_size_inches(8, 3)
# plt.bar(br1, [SNR_rect_unaltered_ave[wi]-snr_corrected,SNR_rect_broadband_ave[wi]-snr_corrected], color='C0', width=barWidth, label='HWR')
# plt.bar(br2, [SNR_pulse_unaltered_ave[wi]-snr_corrected,SNR_pulse_broadband_ave[wi]-snr_corrected], color='C2', width=barWidth, label='Pulse')
# plt.bar(br3, [SNR_ANM_unaltered_ave[wi]-snr_corrected,SNR_ANM_broadband_ave[wi]-snr_corrected], color='C4', width=barWidth, label='ANM')
# plt.xlabel('Stimulus', fontsize = 20)
# plt.ylabel('SNR (dB)', fontsize = 20)
# plt.ylim(-13, 25)
# plt.xticks([r + barWidth for r in range(2)], ['unaltered', 'peaky'])
# plt.hlines(0, -0.125, 1.625, linestyles='solid', linewidth=3, alpha=1, color='k')
# for si in range(subject_num):
#     plt.plot([br1[0],br2[0], br3[0]],[SNR_rect_unaltered[wi,si],SNR_pulse_unaltered[wi,si], SNR_ANM_unaltered[wi,si]], ".-", linewidth=0.5, c='k', alpha=0.5)
#     plt.plot([br1[1],br2[1], br3[1]],[SNR_rect_broadband[wi,si],SNR_pulse_broadband[wi,si], SNR_ANM_broadband[wi,si]], ".-", linewidth=0.5, c='k', alpha=0.5)
# plt.grid(alpha=0.5)
# plt.gca().spines['top'].set_visible(False)
# plt.gca().spines['right'].set_visible(False)
# lg = plt.legend(fontsize=15, bbox_to_anchor=(1.03, 1.0), loc='upper left')
# plt.savefig(figure_path + 'whited_SNR_comparison_corrected-{}ms_HP150.svg'.format(window), dpi=dpi, format='svg', bbox_extra_artists=(lg,), bbox_inches='tight')

# Plotting SNR per freq
fig = plt.figure(dpi=dpi)
fig.set_size_inches(8, 6)
plt.rc('axes', titlesize=20, labelsize=20)
plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=15)
plt.plot(freq_vec, SNR_rect_unaltered_ave_freq-snr_corrected, linewidth=1.2, label="Unaltered (HWR)")
plt.plot(freq_vec, SNR_rect_broadband_ave_freq-snr_corrected, linewidth=1.2, linestyle="--", label="Peaky (HWR)")
plt.plot(freq_vec, SNR_pulse_unaltered_ave_freq-snr_corrected, linewidth=1.2, label="Unaltered (Pulse)")
plt.plot(freq_vec, SNR_pulse_broadband_ave_freq-snr_corrected, linewidth=1.2, linestyle="--", label="Peaky (Pulse)")
plt.plot(freq_vec, SNR_ANM_unaltered_ave_freq-snr_corrected, linewidth=1.2, label="Unaltered (ANM)")
plt.plot(freq_vec, SNR_ANM_broadband_ave_freq-snr_corrected, linewidth=1.2, linestyle="--", label="Peaky (ANM)")
plt.xlim([1,1000])
plt.xlabel('Frequency (Hz)', fontsize = 15)
plt.ylabel('SNR (dB)', fontsize = 15)
plt.grid(alpha=0.5)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.legend(fontsize=12, loc="upper right")
plt.tight_layout()
plt.savefig(figure_path + 'SNR_per_freq_comparison_corrected-15ms.png', dpi=dpi, format='png')

# %% SNR Phase-only regressor plotting
import pingouin
import scipy.stats as stats
SNR_df = pd.read_csv(opath+"ABR_phase-only_SNR_df.csv")
#### Statistics ####
# For 15 ms SNR
SNR_df_15_unaltered = SNR_df[(SNR_df["window"]==15) & (SNR_df["stim"]=="unaltered")]
SNR_df_15_broadband = SNR_df[(SNR_df["window"]==15) & (SNR_df["stim"]=="broadband")]
SNR_df_15 = SNR_df[(SNR_df["window"]==15)].dropna()
# Random effect Linear regression
SNR_lm = smf.mixedlm("SNR ~ regressor + stim + regressor*stim", SNR_df_15, groups=SNR_df_15["subject"])
SNR_lmf = SNR_lm.fit()
print(SNR_lmf.summary())
# RM ANOVA
anova_unaltered = pingouin.rm_anova(data=SNR_df_15_unaltered, dv="SNR", subject="subject", within=["regressor"], detailed=True)
anova_unaltered.round(3)
anova_broadband = pingouin.rm_anova(data=SNR_df_15_broadband, dv="SNR", subject="subject", within=["regressor"], detailed=True)
anova_broadband.round(3)
# Pairwise T-TEST
pairwise_unaltered = pingouin.pairwise_ttests(data=SNR_df_15_unaltered, dv="SNR", subject="subject", within=["regressor"], parametric=True, alpha=0.05, padjust='holm', nan_policy='pairwise')
pairwise_unaltered.round(3)
pairwise_broadband = pingouin.pairwise_ttests(data=SNR_df_15_broadband, dv="SNR", subject="subject", within=["regressor"], parametric=True, alpha=0.05, padjust='holm', nan_policy='pairwise')
pairwise_broadband.round(3)
pairwise_SNR_15_all = pingouin.pairwise_ttests(data=SNR_df_15, dv="SNR", subject="subject", within=["regressor","stim"], parametric=True, alpha=0.05, padjust='holm', nan_policy='pairwise')

# statistic estimate
SNR_15_unaltered_ave = [SNR_df_15_unaltered[SNR_df_15_unaltered['regressor']=="HWR"]["SNR"].mean(),
                        SNR_df_15_unaltered[SNR_df_15_unaltered['regressor']=="GPT"]["SNR"].mean(),
                        SNR_df_15_unaltered[SNR_df_15_unaltered['regressor']=="ANM"]["SNR"].mean()]

SNR_15_unaltered_sem = [SNR_df_15_unaltered[SNR_df_15_unaltered['regressor']=="HWR"]["SNR"].sem(),
                        SNR_df_15_unaltered[SNR_df_15_unaltered['regressor']=="GPT"]["SNR"].sem(),
                        SNR_df_15_unaltered[SNR_df_15_unaltered['regressor']=="ANM"]["SNR"].sem()]

SNR_15_broadband_ave = [SNR_df_15_broadband[SNR_df_15_broadband['regressor']=="HWR"]["SNR"].mean(),
                        SNR_df_15_broadband[SNR_df_15_broadband['regressor']=="GPT"]["SNR"].mean(),
                        SNR_df_15_broadband[SNR_df_15_broadband['regressor']=="ANM"]["SNR"].mean()]

SNR_15_broadband_sem = [SNR_df_15_broadband[SNR_df_15_broadband['regressor']=="HWR"]["SNR"].sem(),
                        SNR_df_15_broadband[SNR_df_15_broadband['regressor']=="GPT"]["SNR"].sem(),
                        SNR_df_15_broadband[SNR_df_15_broadband['regressor']=="ANM"]["SNR"].sem()]

# Plotting
y=[SNR_15_unaltered_ave[0],SNR_15_broadband_ave[0],SNR_15_unaltered_ave[1],SNR_15_broadband_ave[1],SNR_15_unaltered_ave[2],SNR_15_broadband_ave[2]]
yerr = [SNR_15_unaltered_sem[0], SNR_15_broadband_sem[0], SNR_15_unaltered_sem[1], SNR_15_broadband_sem[1],SNR_15_unaltered_sem[2],SNR_15_broadband_sem[2]]
barWidth = 0.25
br1 = np.arange(2)
br2 = [x + barWidth for x in br1]
br3 = [x + barWidth for x in br2]
fig = plt.figure(dpi=dpi)
fig.set_size_inches(6, 3)
plt.bar(br1, [SNR_15_unaltered_ave[0],SNR_15_broadband_ave[0]], color='C0', width=barWidth, label='HWR')
plt.bar(br2, [SNR_15_unaltered_ave[1],SNR_15_broadband_ave[1]], color='C2', width=barWidth, label='GP')
plt.bar(br3, [SNR_15_unaltered_ave[2],SNR_15_broadband_ave[2]], color='C4', width=barWidth, label='ANM')
plt.errorbar([br1[0],br1[1],br2[0],br2[1],br3[0],br3[1]], y=y, yerr=yerr,linestyle="", capsize=3.5, color="k",zorder=4,linewidth=1)
plt.xlabel('Stimulus')
plt.ylabel('SNR (dB)')
plt.ylim(-15, 30)
plt.xticks([r + barWidth for r in range(2)], ['Unaltered', 'Peaky'])
plt.hlines(0, -0.125, 1.625, linestyles='solid', linewidth=1, alpha=1, color='k')
for si in range(subject_num):
    plt.plot([br1[0],br2[0], br3[0]],[SNR_df_15_unaltered[SNR_df_15_unaltered['regressor']=="HWR"].iloc[si]["SNR"],
                                      SNR_df_15_unaltered[SNR_df_15_unaltered['regressor']=="GPT"].iloc[si]["SNR"],
                                      SNR_df_15_unaltered[SNR_df_15_unaltered['regressor']=="ANM"].iloc[si]["SNR"]],
             ".-", linewidth=0.5, c='gray')
    plt.plot([br1[1],br2[1], br3[1]],[SNR_df_15_broadband[SNR_df_15_broadband['regressor']=="HWR"].iloc[si]["SNR"],
                                      SNR_df_15_broadband[SNR_df_15_broadband['regressor']=="GPT"].iloc[si]["SNR"],
                                      SNR_df_15_broadband[SNR_df_15_broadband['regressor']=="ANM"].iloc[si]["SNR"]], 
             ".-", linewidth=0.5, c='gray')
lg = plt.legend(fontsize=10, bbox_to_anchor=(1.03, 1.0), loc='upper left')
plt.grid(alpha=0.5)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.savefig(figure_path + 'phase-only_SNR_comparison-{}ms.svg'.format(15), dpi=dpi, format='svg', bbox_extra_artists=(lg,), bbox_inches='tight')


# %% SNR by number of Eooch rect, pulse, ANM
rect_snr_data = read_hdf5(opath + 'subject_response/'+"rect_by_numEpoch.hdf5")
rect_snr_unaltered = rect_snr_data["snr_unaltered"]
rect_snr_unaltered_bp = rect_snr_data["snr_unaltered_bp"]
rect_snr_broadband = rect_snr_data["snr_broadband"]
rect_snr_broadband_bp = rect_snr_data["snr_broadband_bp"]

pulse_snr_data = read_hdf5(opath + 'subject_response/'+"pulse_by_numEpoch.hdf5")
pulse_snr_unaltered = pulse_snr_data["snr_unaltered"]
pulse_snr_unaltered_bp = pulse_snr_data["snr_unaltered_bp"]
pulse_snr_broadband = pulse_snr_data["snr_broadband"]
pulse_snr_broadband_bp = pulse_snr_data["snr_broadband_bp"]

ANM_snr_data = read_hdf5( opath + 'subject_response/'+"ANM_by_numEpoch.hdf5")
ANM_snr_unaltered = ANM_snr_data["snr_unaltered"]
ANM_snr_unaltered_bp = ANM_snr_data["snr_unaltered_bp"]
ANM_snr_broadband = ANM_snr_data["snr_broadband"]
ANM_snr_broadband_bp = ANM_snr_data["snr_broadband_bp"]

epoch_num_vec = np.arange(1, n_epoch+1, 1)
min_num_vec = np.arange(len_stim, len_stim*(n_epoch+1), len_stim)/60

# Portion of subject >= 0 dB SNR rect
rect_unaltered_portion = np.zeros(n_epoch)
rect_broadband_portion = np.zeros(n_epoch)
rect_unaltered_sub_list = []
rect_broadband_sub_list = []
for i in range(n_epoch):
    curr_sub = [ind for ind, snr in enumerate(rect_snr_unaltered[:, i]) if snr >= 0]
    for sub in curr_sub:
        if sub not in rect_unaltered_sub_list:
            rect_unaltered_sub_list.append(sub)
            print(sub)
    rect_unaltered_portion[i] = len(rect_unaltered_sub_list) / subject_num
    print(rect_unaltered_sub_list)
    curr_sub = [ind for ind, snr in enumerate(rect_snr_broadband[:, i]) if snr >= 0]
    for sub in curr_sub:
        if sub not in rect_broadband_sub_list:
            rect_broadband_sub_list.append(sub)
    rect_broadband_portion[i] = len(rect_broadband_sub_list) / subject_num
    
# Portion of subject >= 0 dB SNR pulse
pulse_unaltered_portion = np.zeros(n_epoch)
pulse_broadband_portion = np.zeros(n_epoch)
pulse_unaltered_sub_list = []
pulse_broadband_sub_list = []
for i in range(n_epoch):
    curr_sub = [ind for ind, snr in enumerate(pulse_snr_unaltered[:, i]) if snr >= 0]
    for sub in curr_sub:
        if sub not in pulse_unaltered_sub_list:
            pulse_unaltered_sub_list.append(sub)
            print(sub)
    pulse_unaltered_portion[i] = len(pulse_unaltered_sub_list) / subject_num
    print(pulse_unaltered_sub_list)
    curr_sub = [ind for ind, snr in enumerate(pulse_snr_broadband[:, i]) if snr >= 0]
    for sub in curr_sub:
        if sub not in pulse_broadband_sub_list:
            pulse_broadband_sub_list.append(sub)
    pulse_broadband_portion[i] = len(pulse_broadband_sub_list) / subject_num

# Portion of subject >= 0 dB SNR ANM
ANM_unaltered_portion = np.zeros(n_epoch)
ANM_broadband_portion = np.zeros(n_epoch)
ANM_unaltered_sub_list = []
ANM_broadband_sub_list = []
for i in range(n_epoch):
    curr_sub = [ind for ind, snr in enumerate(ANM_snr_unaltered[:, i]) if snr >= 0]
    for sub in curr_sub:
        if sub not in ANM_unaltered_sub_list:
            ANM_unaltered_sub_list.append(sub)
            print(sub)
    ANM_unaltered_portion[i] = len(ANM_unaltered_sub_list) / subject_num
    print(ANM_unaltered_sub_list)
    curr_sub = [ind for ind, snr in enumerate(ANM_snr_broadband[:, i]) if snr >= 0]
    for sub in curr_sub:
        if sub not in ANM_broadband_sub_list:
            ANM_broadband_sub_list.append(sub)
    ANM_broadband_portion[i] = len(ANM_broadband_sub_list) / subject_num

# By minutes
fig = plt.figure(dpi=300)
fig.set_size_inches(8, 4)
plt.xlim([0,25])
plt.xticks(np.arange(0,26,1))
plt.yticks(np.arange(0,1.1,0.1))
plt.step(min_num_vec, rect_unaltered_portion, label="Unaltered (HWR)", color="C0")
plt.step(min_num_vec, rect_broadband_portion, label="Peaky (HWR)", linestyle="--", color="C1")
plt.step(min_num_vec, pulse_unaltered_portion, label="Unaltered (GP)", color="C2")
plt.step(min_num_vec, pulse_broadband_portion, label="Peaky (GP)",linestyle="--", color="C3")
plt.step(min_num_vec, ANM_unaltered_portion, label="Unaltered (ANM)", color="C4")
plt.step(min_num_vec, ANM_broadband_portion, label="Peaky (ANM)",linestyle="--", color="C5")
plt.xlabel('Recording time (min)', fontsize = 10)
plt.ylabel('Proportion of subjects >= 0 dB SNR', fontsize = 10)
plt.grid(alpha=0.5)
lg = plt.legend(fontsize=10)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.savefig(figure_path+'abr_snr_byMin_portion.svg', dpi=dpi, format='svg', bbox_extra_artists=(lg,), bbox_inches='tight')

# %% Show correlation pearson's r
regressors = ['rect', 'pulse', 'ANM']
predicted_eeg_path = opath + 'predicted_eeg/'
cc_unaltered = np.zeros((len(regressors), subject_num))
cc_broadband = np.zeros((len(regressors), subject_num))
for regressor in regressors:
    ri = regressors.index(regressor)
    data = read_hdf5(predicted_eeg_path + regressor + '_coherence_slice_200ms_kernel_200ms.hdf5')
    cc_unaltered[ri] = data['corr_unaltered'][:,0]
    cc_broadband[ri] = data['corr_broadband'][:,0]

cc_unaltered_ave = np.average(cc_unaltered, axis=1)
cc_broadband_ave = np.average(cc_broadband, axis=1)

#  #makeing dataframe

cc_df = pd.DataFrame({'subject':list(np.arange(22)),
               "regressor":list(np.repeat("HWR", 22)),
               "stim":list(np.repeat("unaltered", 22)),
               "kernel":list(np.repeat("200", 22)),
               "HP":list(np.repeat(False, 22)),
               "cc":list(cc_unaltered[0])})
cc_df = pd.concat([cc_df, pd.DataFrame({'subject':list(np.arange(22)),
                                        "regressor":list(np.repeat("GPT", 22)),
                                        "stim":list(np.repeat("unaltered", 22)),
                                        "kernel":list(np.repeat("200", 22)),
                                        "HP":list(np.repeat(False, 22)),
                                        "cc":list(cc_unaltered[1])})])
cc_df = pd.concat([cc_df,pd.DataFrame({'subject':list(np.arange(22)),
                                       "regressor":list(np.repeat("ANM", 22)),
                                       "stim":list(np.repeat("unaltered", 22)),
                                       "kernel":list(np.repeat("200", 22)),
                                       "HP":list(np.repeat(False, 22)),
                                       "cc":list(cc_unaltered[2])})])
cc_df = pd.concat([cc_df,pd.DataFrame({'subject':list(np.arange(22)),
                                       "regressor":list(np.repeat("HWR", 22)),
                                       "stim":list(np.repeat("broadband", 22)),
                                       "kernel":list(np.repeat("200", 22)),
                                       "HP":list(np.repeat(False, 22)),
                                       "cc":list(cc_broadband[0])})])
cc_df = pd.concat([cc_df,pd.DataFrame({'subject':list(np.arange(22)),
                                       "regressor":list(np.repeat("GPT", 22)),
                                       "stim":list(np.repeat("broadband", 22)),
                                       "kernel":list(np.repeat("200", 22)),
                                       "HP":list(np.repeat(False, 22)),
                                       "cc":list(cc_broadband[1])})])
cc_df = pd.concat([cc_df,pd.DataFrame({'subject':list(np.arange(22)),
                                       "regressor":list(np.repeat("ANM", 22)),
                                       "stim":list(np.repeat("broadband", 22)),
                                       "kernel":list(np.repeat("200", 22)),
                                       "HP":list(np.repeat(False, 22)),
                                       "cc":list(cc_broadband[2])})])

regressors = ['rect', 'pulse', 'ANM']
predicted_eeg_path = opath + 'predicted_eeg/'
cc_unaltered = np.zeros((len(regressors), subject_num))
cc_broadband = np.zeros((len(regressors), subject_num))
cc_unaltered_hp = np.zeros((len(regressors), subject_num))
cc_broadband_hp = np.zeros((len(regressors), subject_num))

for regressor in regressors:
    ri = regressors.index(regressor)
    data = read_hdf5(predicted_eeg_path + regressor + '_coherence_slice_200ms_kernel_15ms.hdf5')
    cc_unaltered[ri] = data['corr_unaltered'][:,0]
    cc_unaltered_hp[ri] = data['corr_unaltered_hp'][:,0]
    cc_broadband[ri] = data['corr_broadband'][:,0]
    cc_broadband_hp[ri] = data['corr_broadband_hp'][:,0]


cc_df = pd.concat([cc_df,pd.DataFrame({'subject':list(np.arange(22)),
                                       "regressor":list(np.repeat("HWR", 22)),
                                       "stim":list(np.repeat("unaltered", 22)),
                                       "kernel":list(np.repeat("15", 22)),
                                       "HP":list(np.repeat(False, 22)),
                                       "cc":list(cc_unaltered[0])})])
cc_df = pd.concat([cc_df,pd.DataFrame({'subject':list(np.arange(22)),
                                       "regressor":list(np.repeat("GPT", 22)),
                                       "stim":list(np.repeat("unaltered", 22)),
                                       "kernel":list(np.repeat("15", 22)),
                                       "HP":list(np.repeat(False, 22)),
                                       "cc":list(cc_unaltered[1])})])
cc_df = pd.concat([cc_df,pd.DataFrame({'subject':list(np.arange(22)),
                                       "regressor":list(np.repeat("ANM", 22)),
                                       "stim":list(np.repeat("unaltered", 22)),
                                       "kernel":list(np.repeat("15", 22)),
                                       "HP":list(np.repeat(False, 22)),
                                       "cc":list(cc_unaltered[2])})])
cc_df = pd.concat([cc_df,pd.DataFrame({'subject':list(np.arange(22)),
                                       "regressor":list(np.repeat("HWR", 22)),
                                       "stim":list(np.repeat("broadband", 22)),
                                       "kernel":list(np.repeat("15", 22)),
                                       "HP":list(np.repeat(False, 22)),
                                       "cc":list(cc_broadband[0])})])
cc_df = pd.concat([cc_df,pd.DataFrame({'subject':list(np.arange(22)),
                                       "regressor":list(np.repeat("GPT", 22)),
                                       "stim":list(np.repeat("broadband", 22)),
                                       "kernel":list(np.repeat("15", 22)),
                                       "HP":list(np.repeat(False, 22)),
                                       "cc":list(cc_broadband[1])})])
cc_df = pd.concat([cc_df,pd.DataFrame({'subject':list(np.arange(22)),
                                       "regressor":list(np.repeat("ANM", 22)),
                                       "stim":list(np.repeat("broadband", 22)),
                                       "kernel":list(np.repeat("15", 22)),
                                       "HP":list(np.repeat(False, 22)),
                                       "cc":list(cc_broadband[2])})])
# HP
cc_df = pd.concat([cc_df,pd.DataFrame({'subject':list(np.arange(22)),
                                       "regressor":list(np.repeat("HWR", 22)),
                                       "stim":list(np.repeat("unaltered", 22)),
                                       "kernel":list(np.repeat("15", 22)),
                                       "HP":list(np.repeat(True, 22)),
                                       "cc":list(cc_unaltered_hp[0])})])
cc_df = pd.concat([cc_df,pd.DataFrame({'subject':list(np.arange(22)),
                                       "regressor":list(np.repeat("GPT", 22)),
                                       "stim":list(np.repeat("unaltered", 22)),
                                       "kernel":list(np.repeat("15", 22)),
                                       "HP":list(np.repeat(True, 22)),
                                       "cc":list(cc_unaltered_hp[1])})])
cc_df = pd.concat([cc_df,pd.DataFrame({'subject':list(np.arange(22)),
                                       "regressor":list(np.repeat("ANM", 22)),
                                       "stim":list(np.repeat("unaltered", 22)),
                                       "kernel":list(np.repeat("15", 22)),
                                       "HP":list(np.repeat(True, 22)),
                                       "cc":list(cc_unaltered_hp[2])})])
cc_df = pd.concat([cc_df,pd.DataFrame({'subject':list(np.arange(22)),
                                       "regressor":list(np.repeat("HWR", 22)),
                                       "stim":list(np.repeat("broadband", 22)),
                                       "kernel":list(np.repeat("15", 22)),
                                       "HP":list(np.repeat(True, 22)),
                                       "cc":list(cc_broadband_hp[0])})])
cc_df = pd.concat([cc_df,pd.DataFrame({'subject':list(np.arange(22)),
                                       "regressor":list(np.repeat("GPT", 22)),
                                       "stim":list(np.repeat("broadband", 22)),
                                       "kernel":list(np.repeat("15", 22)),
                                       "HP":list(np.repeat(True, 22)),
                                       "cc":list(cc_broadband_hp[1])})])
cc_df = pd.concat([cc_df,pd.DataFrame({'subject':list(np.arange(22)),
                                       "regressor":list(np.repeat("ANM", 22)),
                                       "stim":list(np.repeat("broadband", 22)),
                                       "kernel":list(np.repeat("15", 22)),
                                       "HP":list(np.repeat(True, 22)),
                                       "cc":list(cc_broadband_hp[2])})])
cc_df.to_csv(opath+'prediction_cc.csv')

# %% Correlation coefficient statistics
cc_df = pd.read_csv(opath+"prediction_cc.csv")

# For 200 ms kernel cc
cc_df_200_unaltered = cc_df[(cc_df["kernel"]==200) & (cc_df["stim"]=="unaltered")]
cc_df_200_broadband = cc_df[(cc_df["kernel"]==200) & (cc_df["stim"]=="broadband")]
cc_df_200 = cc_df[(cc_df["kernel"]==200)]
# Random effect Linear regression
cc_200_lm = smf.mixedlm("cc ~ regressor + stim + regressor*stim", cc_df_200, groups=cc_df_200["subject"])
cc_200_lmf = cc_200_lm.fit()
print(cc_200_lmf.summary())      
# RM ANOVA
anova_unaltered = pingouin.rm_anova(data=cc_df_200_unaltered, dv="cc", subject="subject", within=["regressor"], detailed=True)
anova_unaltered.round(3)
anova_broadband = pingouin.rm_anova(data=cc_df_200_broadband, dv="cc", subject="subject", within=["regressor"], detailed=True)
anova_broadband.round(3)
pairwise_all = pingouin.pairwise_tests(data=cc_df_200, dv="cc", subject="subject", within=["regressor","stim"], parametric=True, alpha=0.05, padjust='holm', nan_policy='pairwise')


# For 15 ms kernel cc
cc_df_15_unaltered = cc_df[(cc_df["kernel"]==15) & (cc_df["stim"]=="unaltered") & (cc_df["HP"]==True)]
cc_df_15_broadband = cc_df[(cc_df["kernel"]==15) & (cc_df["stim"]=="broadband") & (cc_df["HP"]==True)]
cc_df_15 = cc_df[(cc_df["kernel"]==15)]
# Random effect Linear regression
cc_15_lm = smf.mixedlm("cc ~ regressor + stim + regressor*stim", cc_df_15, groups=cc_df_15["subject"])
cc_15_lmf = cc_15_lm.fit()
print(cc_15_lmf.summary())   
# RM ANOVA
anova_unaltered = pingouin.rm_anova(data=cc_df_15_unaltered, dv="cc", subject="subject", within=["regressor"], detailed=True)
anova_unaltered.round(3)
anova_broadband = pingouin.rm_anova(data=cc_df_15_broadband, dv="cc", subject="subject", within=["regressor"], detailed=True)
anova_broadband.round(3)

# Pairwise T-TEST
pairwise_unaltered = pingouin.pairwise_ttests(data=cc_df_15_unaltered, dv="cc", subject="subject", within=["regressor"], parametric=True, alpha=0.05, padjust='holm', nan_policy='pairwise')
pairwise_unaltered.round(3)
pairwise_broadband = pingouin.pairwise_ttests(data=cc_df_15_broadband, dv="cc", subject="subject", within=["regressor"], parametric=True, alpha=0.05, padjust='holm', nan_policy='pairwise')
pairwise_broadband.round(3)
pairwise_all = pingouin.pairwise_tests(data=cc_df_15, dv="cc", subject="subject", within=["regressor","stim"], parametric=True, alpha=0.05, padjust='holm', nan_policy='pairwise')
# %% Plot Correlation coefficient

regressors = ['rect', 'pulse', 'ANM']
predicted_eeg_path = opath + 'predicted_eeg/'
cc_unaltered = np.zeros((len(regressors), subject_num))
cc_broadband = np.zeros((len(regressors), subject_num))
for regressor in regressors:
    ri = regressors.index(regressor)
    data = read_hdf5(predicted_eeg_path + regressor + '_coherence_slice_200ms_kernel_200ms.hdf5')
    cc_unaltered[ri] = data['corr_unaltered'][:,0]
    cc_broadband[ri] = data['corr_broadband'][:,0]

cc_unaltered_ave = np.average(cc_unaltered, axis=1)
cc_unaltered_sem = np.std(cc_unaltered, axis=1)/np.sqrt(subject_num)
cc_broadband_ave = np.average(cc_broadband, axis=1)
cc_broadband_sem = np.std(cc_broadband, axis=1)/np.sqrt(subject_num)

y = [cc_unaltered_ave[0],cc_broadband_ave[0],cc_unaltered_ave[1],cc_broadband_ave[1],cc_unaltered_ave[2],cc_broadband_ave[2]]
yerr = [cc_unaltered_sem[0],cc_broadband_sem[0],cc_unaltered_sem[1],cc_broadband_sem[1],cc_unaltered_sem[2],cc_broadband_sem[2]]

barWidth = 0.25
br1 = np.arange(2)
br2 = [x + barWidth for x in br1]
br3 = [x + barWidth for x in br2]
fig = plt.figure(dpi=dpi)
fig.set_size_inches(6, 3)
plt.bar(br1, [cc_unaltered_ave[0],cc_broadband_ave[0]], color='C0', width=barWidth, label='HWR')
plt.bar(br2, [cc_unaltered_ave[1],cc_broadband_ave[1]], color='C2', width=barWidth, label='GP')
plt.bar(br3, [cc_unaltered_ave[2],cc_broadband_ave[2]], color='C4', width=barWidth, label='ANM')
plt.errorbar([br1[0],br1[1],br2[0],br2[1],br3[0],br3[1]], y=y, yerr=yerr,linestyle="", capsize=3.5, color="k",zorder=4,linewidth=1)
plt.xlabel('Stimulus')
plt.ylabel("Prediction Accuracy \n(Pearson's r)")
#plt.ylim(0,0.04)
plt.xticks([r + barWidth for r in range(2)], ['Unaltered', 'Peaky'])
for si in range(subject_num):
    plt.plot([br1[0],br2[0], br3[0]],[cc_unaltered[0][si],cc_unaltered[1][si], cc_unaltered[2][si]], ".-", linewidth=0.5, c='gray')
    plt.plot([br1[1],br2[1], br3[1]],[cc_broadband[0][si],cc_broadband[1][si], cc_broadband[2][si]], ".-", linewidth=0.5, c='gray')
plt.grid(alpha=0.5)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
lg = plt.legend(fontsize=10, bbox_to_anchor=(1.03, 1.0), loc='upper left')
plt.savefig(figure_path + 'cc_comparison_kernel_200ms.svg', dpi=dpi, format='svg', bbox_extra_artists=(lg,), bbox_inches='tight')

# 15 ms HP at 40 Hz
regressors = ['rect', 'pulse', 'ANM']
predicted_eeg_path = opath + 'predicted_eeg/'
cc_unaltered = np.zeros((len(regressors), subject_num))
cc_broadband = np.zeros((len(regressors), subject_num))
for regressor in regressors:
    ri = regressors.index(regressor)
    data = read_hdf5(predicted_eeg_path + regressor + '_coherence_slice_200ms_kernel_15ms.hdf5')
    cc_unaltered[ri] = data['corr_unaltered_hp'][:,0]
    cc_broadband[ri] = data['corr_broadband_hp'][:,0]

cc_unaltered_ave = np.average(cc_unaltered, axis=1)
cc_unaltered_sem = np.std(cc_unaltered, axis=1)/np.sqrt(subject_num)
cc_broadband_ave = np.average(cc_broadband, axis=1)
cc_broadband_sem = np.std(cc_broadband, axis=1)/np.sqrt(subject_num)

y = [cc_unaltered_ave[0],cc_broadband_ave[0],cc_unaltered_ave[1],cc_broadband_ave[1],cc_unaltered_ave[2],cc_broadband_ave[2]]
yerr = [cc_unaltered_sem[0],cc_broadband_sem[0],cc_unaltered_sem[1],cc_broadband_sem[1],cc_unaltered_sem[2],cc_broadband_sem[2]]

barWidth = 0.25
br1 = np.arange(2)
br2 = [x + barWidth for x in br1]
br3 = [x + barWidth for x in br2]
fig = plt.figure(dpi=dpi)
fig.set_size_inches(6, 3)
plt.bar(br1, [cc_unaltered_ave[0],cc_broadband_ave[0]], color='C0', width=barWidth, label='HWR')
plt.bar(br2, [cc_unaltered_ave[1],cc_broadband_ave[1]], color='C2', width=barWidth, label='GP')
plt.bar(br3, [cc_unaltered_ave[2],cc_broadband_ave[2]], color='C4', width=barWidth, label='ANM')
plt.errorbar([br1[0],br1[1],br2[0],br2[1],br3[0],br3[1]], y=y, yerr=yerr,linestyle="", capsize=3.5, color="k",zorder=4,linewidth=1)
plt.xlabel('Stimulus')
plt.ylabel("Prediction Accuracy \n(Pearson's r)")
#plt.ylim(0,0.04)
plt.xticks([r + barWidth for r in range(2)], ['Unaltered', 'Peaky'])
for si in range(subject_num):
    plt.plot([br1[0],br2[0], br3[0]],[cc_unaltered[0][si],cc_unaltered[1][si], cc_unaltered[2][si]], ".-", linewidth=0.5, c='gray')
    plt.plot([br1[1],br2[1], br3[1]],[cc_broadband[0][si],cc_broadband[1][si], cc_broadband[2][si]], ".-", linewidth=0.5, c='gray')
plt.grid(alpha=0.5)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
lg = plt.legend(fontsize=10, bbox_to_anchor=(1.03, 1.0), loc='upper left')
plt.savefig(figure_path + 'cc_comparison_kernel_15ms_hp40.svg', dpi=dpi, format='svg', bbox_extra_artists=(lg,), bbox_inches='tight')

# %% COHERENCE
#Loading noise floor
predicted_eeg_path = opath + 'predicted_eeg/'
noise_floor_data = read_hdf5(predicted_eeg_path + '/coherence_noise_floor.hdf5')
coh_noise_floor_01 = noise_floor_data['coh_noise_floor_01']
coh_01_lb =noise_floor_data['coh_01_lb']
coh_01_ub =noise_floor_data['coh_01_ub']
coh_noise_floor_02 = noise_floor_data['coh_noise_floor_02']
coh_02_lb =noise_floor_data['coh_02_lb']
coh_02_ub =noise_floor_data['coh_02_ub']
coh_noise_floor_1 = noise_floor_data['coh_noise_floor_1']
coh_1_lb =noise_floor_data['coh_1_lb']
coh_1_ub =noise_floor_data['coh_1_ub']

# Coherence params dur 1
dur_slice = 0.2
n_slices = int(len_stim / dur_slice)
len_slice = int(dur_slice * eeg_fs)
n_bands = int((eeg_fs / 2) * dur_slice + 1)
# Lists
regressors = ['rect', 'pulse', 'ANM']
# Plot 12 types averaged

coh_abs_unaltered_ave = np.zeros((len(regressors), n_bands))
coh_abs_broadband_ave = np.zeros((len(regressors), n_bands))
coh_unaltered_band_1 = np.zeros((len(regressors),subject_num,4))
coh_broadband_band_1 = np.zeros((len(regressors),subject_num,4))
coh_unaltered_band_2 = np.zeros((len(regressors),subject_num,5))
coh_broadband_band_2 = np.zeros((len(regressors),subject_num,5))
coh_unaltered_band_3 = np.zeros((len(regressors),subject_num,9))
coh_broadband_band_3 = np.zeros((len(regressors),subject_num,9))
for reg in regressors:
    ri = regressors.index(reg)
    data = read_hdf5(predicted_eeg_path + reg + '_coherence_slice_200ms_kernel_200ms.hdf5')
    coh_unaltered = data['coh_unaltered']
    coh_unaltered_band_1[ri,:] = abs(coh_unaltered[:,1:5])
    coh_unaltered_band_2[ri,:] = abs(coh_unaltered[:,8:13])
    coh_unaltered_band_3[ri,:] = abs(coh_unaltered[:,16:25])
    coh_abs_unaltered_ave[ri,:] = np.average(abs(coh_unaltered), axis=0)
    coh_broadband = data['coh_broadband']
    coh_broadband_band_1[ri,:] = abs(coh_broadband[:,1:5])
    coh_broadband_band_2[ri,:] = abs(coh_broadband[:,8:13])
    coh_broadband_band_3[ri,:] = abs(coh_broadband[:,16:25])
    coh_abs_broadband_ave[ri,:] = np.average(abs(coh_broadband), axis=0)
    freq = data['freq']

fig = plt.figure(dpi=dpi)
fig.set_size_inches(4, 3)
plt.plot(freq[1:], coh_abs_unaltered_ave[0, 1:], c='C0', linewidth=1)
plt.plot(freq[1:], coh_abs_unaltered_ave[1, 1:], c='C2', linewidth=1)
plt.plot(freq[1:], coh_abs_unaltered_ave[2, 1:], c='C4', linewidth=1)
plt.hlines(0.007882034989989706, 10, 400, color="grey", linewidth=1, linestyles="-.")
plt.xlim([10, 151])
plt.ylim([0.005, 0.055])
plt.fill_betweenx(np.arange(0.005, 0.056, 0.001),freq[1],freq[4], color='grey',ec=None, alpha=0.3, linewidth=0,zorder=2)
plt.fill_betweenx(np.arange(0.005, 0.056, 0.001),freq[8],freq[12], color='grey',ec=None, alpha=0.3, linewidth=0,zorder=2)
plt.fill_betweenx(np.arange(0.005, 0.056, 0.001),freq[16],freq[24], color='grey',ec=None, alpha=0.3, linewidth=0,zorder=2)
plt.legend(['HWR', 'GP', 'ANM'],
           fontsize=10)
#plt.text(8, 0.016,  'Noise Floor (95 percentile)', fontsize=10, c='C7')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Coherence (absolute)')
plt.title("Unaltered")
plt.grid(alpha=0.5)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(figure_path + 'coherence_lines_unaltered.svg', dpi=dpi, format='svg')


fig = plt.figure(dpi=dpi)
fig.set_size_inches(4, 3)
plt.plot(freq[1:], coh_abs_broadband_ave[0, 1:], c='C0', linewidth=1)
plt.plot(freq[1:], coh_abs_broadband_ave[1, 1:], c='C2', linewidth=1)
plt.plot(freq[1:], coh_abs_broadband_ave[2, 1:], c='C4', linewidth=1)
plt.hlines(0.007882034989989706, 10, 400, color="grey", linewidth=1, linestyles="-.")
plt.xlim([10, 151])
plt.ylim([0.005, 0.055])
plt.fill_betweenx(np.arange(0.005, 0.056, 0.001),freq[1],freq[4], color='grey',ec=None, alpha=0.3, linewidth=0,zorder=2)
plt.fill_betweenx(np.arange(0.005, 0.056, 0.001),freq[8],freq[12], color='grey',ec=None, alpha=0.3, linewidth=0,zorder=2)
plt.fill_betweenx(np.arange(0.005, 0.056, 0.001),freq[16],freq[24], color='grey',ec=None, alpha=0.3, linewidth=0,zorder=2)
plt.legend(['HWR', 'GP', 'ANM'],
           fontsize=10)
#plt.text(8, 0.016,  'Noise Floor (95 percentile)', fontsize=10, c='C7')
plt.xlabel('Frequency (Hz)')
plt.ylabel('Coherence (absolute)')
plt.title("Peaky")
plt.grid(alpha=0.5)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
plt.tight_layout()
plt.savefig(figure_path + 'coherence_lines_peaky.svg', dpi=dpi, format='svg')


# %% Making coherence dataframe
coherence_df = pd.DataFrame({'subject':list(np.arange(22)),
                             "regressor":list(np.repeat("HWR", 22)),
                             "stim":list(np.repeat("unaltered", 22)),
                             "bands":list(np.repeat("band1", 22)),
                             "coherence":list(coh_unaltered_band_1[0].mean(axis=-1))})
coherence_df = pd.concat([coherence_df, pd.DataFrame({'subject':list(np.arange(22)),
                                                      "regressor":list(np.repeat("GPT", 22)),
                                                      "stim":list(np.repeat("unaltered", 22)),
                                                      "bands":list(np.repeat("band1", 22)),
                                                      "coherence":list(coh_unaltered_band_1[1].mean(axis=-1))})])
coherence_df = pd.concat([coherence_df, pd.DataFrame({'subject':list(np.arange(22)),
                                                      "regressor":list(np.repeat("ANM", 22)),
                                                      "stim":list(np.repeat("unaltered", 22)),
                                                      "bands":list(np.repeat("band1", 22)),
                                                      "coherence":list(coh_unaltered_band_1[2].mean(axis=-1))})])
coherence_df = pd.concat([coherence_df, pd.DataFrame({'subject':list(np.arange(22)),
                                                      "regressor":list(np.repeat("HWR", 22)),
                                                      "stim":list(np.repeat("broadband", 22)),
                                                      "bands":list(np.repeat("band1", 22)),
                                                      "coherence":list(coh_broadband_band_1[0].mean(axis=-1))})])
coherence_df = pd.concat([coherence_df, pd.DataFrame({'subject':list(np.arange(22)),
                                                      "regressor":list(np.repeat("GPT", 22)),
                                                      "stim":list(np.repeat("broadband", 22)),
                                                      "bands":list(np.repeat("band1", 22)),
                                                      "coherence":list(coh_broadband_band_1[1].mean(axis=-1))})])
coherence_df = pd.concat([coherence_df, pd.DataFrame({'subject':list(np.arange(22)),
                                                      "regressor":list(np.repeat("ANM", 22)),
                                                      "stim":list(np.repeat("broadband", 22)),
                                                      "bands":list(np.repeat("band1", 22)),
                                                      "coherence":list(coh_broadband_band_1[2].mean(axis=-1))})])
coherence_df = pd.concat([coherence_df, pd.DataFrame({'subject':list(np.arange(22)),
                                                      "regressor":list(np.repeat("HWR", 22)),
                                                      "stim":list(np.repeat("unaltered", 22)),
                                                      "bands":list(np.repeat("band2", 22)),
                                                      "coherence":list(coh_unaltered_band_2[0].mean(axis=-1))})])
coherence_df = pd.concat([coherence_df, pd.DataFrame({'subject':list(np.arange(22)),
                                                      "regressor":list(np.repeat("GPT", 22)),
                                                      "stim":list(np.repeat("unaltered", 22)),
                                                      "bands":list(np.repeat("band2", 22)),
                                                      "coherence":list(coh_unaltered_band_2[1].mean(axis=-1))})])
coherence_df = pd.concat([coherence_df, pd.DataFrame({'subject':list(np.arange(22)),
                                                      "regressor":list(np.repeat("ANM", 22)),
                                                      "stim":list(np.repeat("unaltered", 22)),
                                                      "bands":list(np.repeat("band2", 22)),
                                                      "coherence":list(coh_unaltered_band_2[2].mean(axis=-1))})])
coherence_df = pd.concat([coherence_df, pd.DataFrame({'subject':list(np.arange(22)),
                                                      "regressor":list(np.repeat("HWR", 22)),
                                                      "stim":list(np.repeat("broadband", 22)),
                                                      "bands":list(np.repeat("band2", 22)),
                                                      "coherence":list(coh_broadband_band_2[0].mean(axis=-1))})])
coherence_df = pd.concat([coherence_df, pd.DataFrame({'subject':list(np.arange(22)),
                                                      "regressor":list(np.repeat("GPT", 22)),
                                                      "stim":list(np.repeat("broadband", 22)),
                                                      "bands":list(np.repeat("band2", 22)),
                                                      "coherence":list(coh_broadband_band_2[1].mean(axis=-1))})])
coherence_df = pd.concat([coherence_df, pd.DataFrame({'subject':list(np.arange(22)),
                                                      "regressor":list(np.repeat("ANM", 22)),
                                                      "stim":list(np.repeat("broadband", 22)),
                                                      "bands":list(np.repeat("band2", 22)),
                                                      "coherence":list(coh_broadband_band_2[2].mean(axis=-1))})])
coherence_df = pd.concat([coherence_df, pd.DataFrame({'subject':list(np.arange(22)),
                                                      "regressor":list(np.repeat("HWR", 22)),
                                                      "stim":list(np.repeat("unaltered", 22)),
                                                      "bands":list(np.repeat("band3", 22)),
                                                      "coherence":list(coh_unaltered_band_3[0].mean(axis=-1))})])
coherence_df = pd.concat([coherence_df, pd.DataFrame({'subject':list(np.arange(22)),
                                                      "regressor":list(np.repeat("GPT", 22)),
                                                      "stim":list(np.repeat("unaltered", 22)),
                                                      "bands":list(np.repeat("band3", 22)),
                                                      "coherence":list(coh_unaltered_band_3[1].mean(axis=-1))})])
coherence_df = pd.concat([coherence_df, pd.DataFrame({'subject':list(np.arange(22)),
                                                      "regressor":list(np.repeat("ANM", 22)),
                                                      "stim":list(np.repeat("unaltered", 22)),
                                                      "bands":list(np.repeat("band3", 22)),
                                                      "coherence":list(coh_unaltered_band_3[2].mean(axis=-1))})])
coherence_df = pd.concat([coherence_df, pd.DataFrame({'subject':list(np.arange(22)),
                                                      "regressor":list(np.repeat("HWR", 22)),
                                                      "stim":list(np.repeat("broadband", 22)),
                                                      "bands":list(np.repeat("band3", 22)),
                                                      "coherence":list(coh_broadband_band_3[0].mean(axis=-1))})])
coherence_df = pd.concat([coherence_df, pd.DataFrame({'subject':list(np.arange(22)),
                                                      "regressor":list(np.repeat("GPT", 22)),
                                                      "stim":list(np.repeat("broadband", 22)),
                                                      "bands":list(np.repeat("band3", 22)),
                                                      "coherence":list(coh_broadband_band_3[1].mean(axis=-1))})])
coherence_df = pd.concat([coherence_df, pd.DataFrame({'subject':list(np.arange(22)),
                                                      "regressor":list(np.repeat("ANM", 22)),
                                                      "stim":list(np.repeat("broadband", 22)),
                                                      "bands":list(np.repeat("band3", 22)),
                                                      "coherence":list(coh_broadband_band_3[2].mean(axis=-1))})])
coherence_df.to_csv(opath+"coherence_df.csv")
# %% Statistics
coherence_df = pd.read_csv(opath+"coherence_df.csv")
# Coherence for each frequency bands
# 0-20 - band_1
# RM ANOVA
anova_unaltered_band1 = pingouin.rm_anova(data=coherence_df[(coherence_df["stim"]=="unaltered") & (coherence_df["bands"]=="band1")], dv="coherence", subject="subject", within=["regressor"], detailed=True)
anova_unaltered_band1.round(3)
anova_broadband_band1 = pingouin.rm_anova(data=coherence_df[(coherence_df["stim"]=="broadband") & (coherence_df["bands"]=="band1")], dv="coherence", subject="subject", within=["regressor"], detailed=True)
anova_broadband_band1.round(3)
# Pairwise T-TEST
pairwise_unaltered = pingouin.pairwise_ttests(data=coherence_df[(coherence_df["stim"]=="unaltered") & (coherence_df["bands"]=="band1")], dv="coherence", subject="subject", within=["regressor"], parametric=True, alpha=0.05, padjust='holm', nan_policy='pairwise')
pairwise_unaltered.round(3)
pairwise_broadband = pingouin.pairwise_ttests(data=coherence_df[(coherence_df["stim"]=="broadband") & (coherence_df["bands"]=="band1")], dv="coherence", subject="subject", within=["regressor"], parametric=True, alpha=0.05, padjust='holm', nan_policy='pairwise')
pairwise_broadband.round(3)
# 40-60 - band_2
# RM ANOVA
anova_unaltered_band2 = pingouin.rm_anova(data=coherence_df[(coherence_df["stim"]=="unaltered") & (coherence_df["bands"]=="band2")], dv="coherence", subject="subject", within=["regressor"], detailed=True)
anova_unaltered_band2.round(3)
anova_broadband_band2 = pingouin.rm_anova(data=coherence_df[(coherence_df["stim"]=="broadband") & (coherence_df["bands"]=="band2")], dv="coherence", subject="subject", within=["regressor"], detailed=True)
anova_broadband_band2.round(3)
# Pairwise T-TEST
pairwise_unaltered = pingouin.pairwise_ttests(data=coherence_df[(coherence_df["stim"]=="unaltered") & (coherence_df["bands"]=="band2")], dv="coherence", subject="subject", within=["regressor"], parametric=True, alpha=0.05, padjust='holm', nan_policy='pairwise')
pairwise_unaltered.round(3)
pairwise_broadband = pingouin.pairwise_ttests(data=coherence_df[(coherence_df["stim"]=="broadband") & (coherence_df["bands"]=="band2")], dv="coherence", subject="subject", within=["regressor"], parametric=True, alpha=0.05, padjust='holm', nan_policy='pairwise')
pairwise_broadband.round(3)
# 80-120 - band_3
# RM ANOVA
anova_unaltered_band3 = pingouin.rm_anova(data=coherence_df[(coherence_df["stim"]=="unaltered") & (coherence_df["bands"]=="band3")], dv="coherence", subject="subject", within=["regressor"], detailed=True)
anova_unaltered_band3.round(3)
anova_broadband_band3 = pingouin.rm_anova(data=coherence_df[(coherence_df["stim"]=="broadband") & (coherence_df["bands"]=="band3")], dv="coherence", subject="subject", within=["regressor"], detailed=True)
anova_broadband_band3.round(3)
band3_df = coherence_df[coherence_df["bands"]=="band3"]

# Pairwise T-TEST
pairwise_unaltered = pingouin.pairwise_ttests(data=coherence_df[(coherence_df["stim"]=="unaltered") & (coherence_df["bands"]=="band3")], dv="coherence", subject="subject", within=["regressor"], parametric=True, alpha=0.05, padjust='holm', nan_policy='pairwise')
pairwise_unaltered.round(3)
pairwise_broadband = pingouin.pairwise_ttests(data=coherence_df[(coherence_df["stim"]=="broadband") & (coherence_df["bands"]=="band3")], dv="coherence", subject="subject", within=["regressor"], parametric=True, alpha=0.05, padjust='holm', nan_policy='pairwise')
pairwise_broadband.round(3)
pairwise_coherence = pingouin.pairwise_tests(data=band3_df, dv="coherence", subject="subject", within=["regressor","stim"], parametric=True, alpha=0.05, padjust='holm', nan_policy='pairwise')

# %% Plot Bar plot of coherence
unaltered_band1 = coherence_df[(coherence_df["stim"]=="unaltered") & (coherence_df["bands"]=="band1")]
unaltered_band2 = coherence_df[(coherence_df["stim"]=="unaltered") & (coherence_df["bands"]=="band2")]
unaltered_band3 = coherence_df[(coherence_df["stim"]=="unaltered") & (coherence_df["bands"]=="band3")]

y = [unaltered_band1[unaltered_band1['regressor']=='HWR']['coherence'].mean(),
     unaltered_band2[unaltered_band2['regressor']=='HWR']['coherence'].mean(),
     unaltered_band3[unaltered_band3['regressor']=='HWR']['coherence'].mean(),
     unaltered_band1[unaltered_band1['regressor']=='GPT']['coherence'].mean(),
     unaltered_band2[unaltered_band2['regressor']=='GPT']['coherence'].mean(),
     unaltered_band3[unaltered_band3['regressor']=='GPT']['coherence'].mean(),
     unaltered_band1[unaltered_band1['regressor']=='ANM']['coherence'].mean(),
     unaltered_band2[unaltered_band2['regressor']=='ANM']['coherence'].mean(),
     unaltered_band3[unaltered_band3['regressor']=='ANM']['coherence'].mean(),]
yerr = [unaltered_band1[unaltered_band1['regressor']=='HWR']['coherence'].sem(),
        unaltered_band2[unaltered_band2['regressor']=='HWR']['coherence'].sem(),
        unaltered_band3[unaltered_band3['regressor']=='HWR']['coherence'].sem(),
        unaltered_band1[unaltered_band1['regressor']=='GPT']['coherence'].sem(),
        unaltered_band2[unaltered_band2['regressor']=='GPT']['coherence'].sem(),
        unaltered_band3[unaltered_band3['regressor']=='GPT']['coherence'].sem(),
        unaltered_band1[unaltered_band1['regressor']=='ANM']['coherence'].sem(),
        unaltered_band2[unaltered_band2['regressor']=='ANM']['coherence'].sem(),
        unaltered_band3[unaltered_band3['regressor']=='ANM']['coherence'].sem(),]

barWidth = 0.25
br1 = np.arange(3)
br2 = [x + barWidth for x in br1]
br3 = [x + barWidth for x in br2]
fig = plt.figure(dpi=dpi)
fig.set_size_inches(4, 3)
plt.bar(br1, y[0:3], color='C0', width=barWidth, label='HWR')
plt.bar(br2, y[3:6], color='C2', width=barWidth, label='GP')
plt.bar(br3, y[6:9], color='C4', width=barWidth, label='ANM')
plt.errorbar([br1[0],br1[1],br1[2], br2[0],br2[1],br2[2],br3[0],br3[1],br3[2]], y=y, yerr=yerr,linestyle="", capsize=3.5, color="k",zorder=4,linewidth=1)
plt.xlabel('Frequency band (Hz)')
plt.ylabel("Coherence (absolute)")
plt.ylim(0,0.082)
plt.xticks([r + barWidth for r in range(3)], ['(0, 20]', '[40, 60]', '[80, 120]'])
for si in range(subject_num):
    plt.plot([br1[0],br2[0], br3[0]],[unaltered_band1[(unaltered_band1['regressor']=='HWR') & (unaltered_band1['subject']==si)]['coherence'],
                                      unaltered_band1[(unaltered_band1['regressor']=='GPT') & (unaltered_band1['subject']==si)]['coherence'],
                                      unaltered_band1[(unaltered_band1['regressor']=='ANM') & (unaltered_band1['subject']==si)]['coherence']],
             ".-", linewidth=0.5, c='gray')
    plt.plot([br1[1],br2[1], br3[1]],[unaltered_band2[(unaltered_band2['regressor']=='HWR') & (unaltered_band2['subject']==si)]['coherence'],
                                      unaltered_band2[(unaltered_band2['regressor']=='GPT') & (unaltered_band2['subject']==si)]['coherence'],
                                      unaltered_band2[(unaltered_band2['regressor']=='ANM') & (unaltered_band2['subject']==si)]['coherence']],
             ".-", linewidth=0.5, c='gray')
    plt.plot([br1[2],br2[2], br3[2]],[unaltered_band3[(unaltered_band3['regressor']=='HWR') & (unaltered_band3['subject']==si)]['coherence'],
                                      unaltered_band3[(unaltered_band3['regressor']=='GPT') & (unaltered_band3['subject']==si)]['coherence'],
                                      unaltered_band3[(unaltered_band3['regressor']=='ANM') & (unaltered_band3['subject']==si)]['coherence']],
             ".-", linewidth=0.5, c='gray')
plt.grid(alpha=0.5)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
lg = plt.legend(fontsize=10, bbox_to_anchor=(1.03, 1.0), loc='upper left')
plt.savefig(figure_path + 'coherence_freq_band_unaltered.svg', dpi=dpi, format='svg', bbox_extra_artists=(lg,), bbox_inches='tight')


broadband_band1 = coherence_df[(coherence_df["stim"]=="broadband") & (coherence_df["bands"]=="band1")]
broadband_band2 = coherence_df[(coherence_df["stim"]=="broadband") & (coherence_df["bands"]=="band2")]
broadband_band3 = coherence_df[(coherence_df["stim"]=="broadband") & (coherence_df["bands"]=="band3")]

y = [broadband_band1[broadband_band1['regressor']=='HWR']['coherence'].mean(),
     broadband_band2[broadband_band2['regressor']=='HWR']['coherence'].mean(),
     broadband_band3[broadband_band3['regressor']=='HWR']['coherence'].mean(),
     broadband_band1[broadband_band1['regressor']=='GPT']['coherence'].mean(),
     broadband_band2[broadband_band2['regressor']=='GPT']['coherence'].mean(),
     broadband_band3[broadband_band3['regressor']=='GPT']['coherence'].mean(),
     broadband_band1[broadband_band1['regressor']=='ANM']['coherence'].mean(),
     broadband_band2[broadband_band2['regressor']=='ANM']['coherence'].mean(),
     broadband_band3[broadband_band3['regressor']=='ANM']['coherence'].mean(),]
yerr = [broadband_band1[broadband_band1['regressor']=='HWR']['coherence'].sem(),
        broadband_band2[broadband_band2['regressor']=='HWR']['coherence'].sem(),
        broadband_band3[broadband_band3['regressor']=='HWR']['coherence'].sem(),
        broadband_band1[broadband_band1['regressor']=='GPT']['coherence'].sem(),
        broadband_band2[broadband_band2['regressor']=='GPT']['coherence'].sem(),
        broadband_band3[broadband_band3['regressor']=='GPT']['coherence'].sem(),
        broadband_band1[broadband_band1['regressor']=='ANM']['coherence'].sem(),
        broadband_band2[broadband_band2['regressor']=='ANM']['coherence'].sem(),
        broadband_band3[broadband_band3['regressor']=='ANM']['coherence'].sem(),]

barWidth = 0.25
br1 = np.arange(3)
br2 = [x + barWidth for x in br1]
br3 = [x + barWidth for x in br2]
fig = plt.figure(dpi=dpi)
fig.set_size_inches(4, 3)
plt.bar(br1, y[0:3], color='C0', width=barWidth, label='HWR')
plt.bar(br2, y[3:6], color='C2', width=barWidth, label='GP')
plt.bar(br3, y[6:9], color='C4', width=barWidth, label='ANM')
plt.errorbar([br1[0],br1[1],br1[2], br2[0],br2[1],br2[2],br3[0],br3[1],br3[2]], y=y, yerr=yerr,linestyle="", capsize=3.5, color="k",zorder=4,linewidth=1)
plt.xlabel('Frequency band (Hz)')
plt.ylabel("Coherence (absolute)")
plt.ylim(0,0.082)
plt.xticks([r + barWidth for r in range(3)], ['(0, 20]', '[40, 60]', '[80, 120]'])
for si in range(subject_num):
    plt.plot([br1[0],br2[0], br3[0]],[broadband_band1[(broadband_band1['regressor']=='HWR') & (broadband_band1['subject']==si)]['coherence'],
                                      broadband_band1[(broadband_band1['regressor']=='GPT') & (broadband_band1['subject']==si)]['coherence'],
                                      broadband_band1[(broadband_band1['regressor']=='ANM') & (broadband_band1['subject']==si)]['coherence']],
             ".-", linewidth=0.5, c='gray')
    plt.plot([br1[1],br2[1], br3[1]],[broadband_band2[(broadband_band2['regressor']=='HWR') & (broadband_band2['subject']==si)]['coherence'],
                                      broadband_band2[(broadband_band2['regressor']=='GPT') & (broadband_band2['subject']==si)]['coherence'],
                                      broadband_band2[(broadband_band2['regressor']=='ANM') & (broadband_band2['subject']==si)]['coherence']],
             ".-", linewidth=0.5, c='gray')
    plt.plot([br1[2],br2[2], br3[2]],[broadband_band3[(broadband_band3['regressor']=='HWR') & (broadband_band3['subject']==si)]['coherence'],
                                      broadband_band3[(broadband_band3['regressor']=='GPT') & (broadband_band3['subject']==si)]['coherence'],
                                      broadband_band3[(broadband_band3['regressor']=='ANM') & (broadband_band3['subject']==si)]['coherence']],
             ".-", linewidth=0.5, c='gray')
plt.grid(alpha=0.5)
plt.gca().spines['top'].set_visible(False)
plt.gca().spines['right'].set_visible(False)
lg = plt.legend(fontsize=10, bbox_to_anchor=(1.03, 1.0), loc='upper left')
plt.savefig(figure_path + 'coherence_freq_band_peaky.svg', dpi=dpi, format='svg', bbox_extra_artists=(lg,), bbox_inches='tight')

# %% Plot waveform
data = read_hdf5(opath + 'present_files/Exp1_stimuli.hdf5')
unaltered = data['audio'][0][0]
peaky = data['audio'][1][0]
stim_fs = 44100

plt.figure()
plt.plot(unaltered[int(4.45*stim_fs):int(5.25*stim_fs)],c='C0')
plt.show()
#plt.gca().spines.set_visible(False)
plt.savefig(figure_path + 'waveform_unaltered.png', dpi=dpi, format='png', transparent=True)

plt.figure()
plt.plot(peaky[int(10*stim_fs):int(10.8*stim_fs)],c='C0')
#plt.gca().spines.set_visible(False)
plt.savefig(figure_path + 'waveform_peaky.png', dpi=dpi, format='png', transparent=True)

wav_unaltered, fs = read_wav(opath + "audio_sample/MaleNarrator_unaltered.wav")
wav_peaky, fs = read_wav(opath + "audio_sample/MaleNarrator_broadband-peaky.wav")

plt.figure()
plt.plot(wav_unaltered[0][int(4.45*stim_fs):int(5.25*stim_fs)])
plt.savefig(figure_path + 'waveform_unaltered.png', dpi=dpi, format='png', transparent=True)

plt.figure()
plt.plot(wav_peaky[0][int(4.45*stim_fs):int(5.25*stim_fs)])
plt.savefig(figure_path + 'waveform_peaky.png', dpi=dpi, format='png', transparent=True)

ANM_data = read_hdf5(opath+"audio_sample/MaleNarrator_unaltered_ANM.hdf5")
ANM = ANM_data["waves_pos_resmp"]

plt.figure()
plt.plot(ANM[int(4.45*eeg_fs):int(5.25*eeg_fs)])
plt.savefig(figure_path + 'ANM_regressor_unaltered.png', dpi=dpi, format='png', transparent=True)