#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 18:49:13 2021

@author: tom
"""
# TODO: Correct M1, M3, M5 (probably just scale by 43/401)
# TODO: Test and verify
import numpy as np
import cochlea
from mne.filter import resample
from joblib import Parallel, delayed
import ic_cn2018 as nuclei
import re

import scipy.signal as signal
from numpy.fft import irfft, rfft, fft, ifft
import matplotlib.pyplot as plt

from expyfun.io import read_wav, read_tab
from expyfun.io import write_hdf5, read_hdf5

#%%

def findstring(ref, check):
    r = re.compile("(?:" + "|".join(check) + ")*$")
    if r.match(ref) is not None:
        return True
    return False


def get_rates(stim_up, cf):
    fs_up = int(100e3)
    return(np.array(cochlea.run_zilany2014_rate(stim_up,
                                                fs_up,
                                                anf_types='hsr',
                                                cf=cf,
                                                species='human',
                                                cohc=1,
                                                cihc=1))[:, 0])


def anm(stim, fs_in, stim_pres_db, parallel=True, n_jobs=-1,
        stim_gen_rms=0.01, cf_low=125, cf_high=16e3, shift_cfs=False,
        shift_vals=None):
    """
     shift_cfs: boolean
        shift each CF indpendently so maximum values align at zero
     shift_vals: array-like
        the values (in seconds) by which to shift each cf if shift_cfs == True
    """
    # Resample your stimuli to a higher fs for the model
    fs_up = int(100e3)
    stim_up = resample(stim, fs_up, fs_in, npad='auto', n_jobs=n_jobs)

    # Put stim in correct units
    # scalar to put it in units of pascals (double-checked and good)
    sine_rms_at_0db = 20e-6
    db_conv = ((sine_rms_at_0db / stim_gen_rms) * 10 ** (stim_pres_db / 20.))
    stim_up = db_conv * stim_up

    # run the model
    dOct = 1. / 6
    cfs = 2 ** np.arange(np.log2(cf_low), np.log2(cf_high + 1), dOct)
    anf_rates_up = np.zeros([len(cfs), len(stim)*fs_up//fs_in])

    if parallel:
        anf_rates_up = Parallel(n_jobs=n_jobs)([delayed(get_rates)(stim_up, cf)
                                               for cf in cfs])
    else:
        for cfi, cf in enumerate(cfs):
            anf_rates_up[cfi] = get_rates(stim_up, cf)

    # Downsample to match input fs
    anf_rates = resample(anf_rates_up, fs_in, fs_up, npad='auto',
                         n_jobs=n_jobs)

    # Optionally, shift each cf independently
    final_shift = int(fs_in*0.001)  # shift w1 by 1ms if not shifting each cf
    if shift_cfs:
        final_shift = 0  # don't shift everything after aligning channels at 0
        if shift_vals is None:
            # default shift_cfs values (based on 75 dB click)
            shift_vals = np.array([0.0046875, 0.0045625, 0.00447917,
                                   0.00435417, 0.00422917, 0.00416667,
                                   0.00402083, 0.0039375, 0.0038125, 0.0036875,
                                   0.003625, 0.00354167, 0.00341667,
                                   0.00327083, 0.00316667, 0.0030625,
                                   0.00302083, 0.00291667, 0.0028125,
                                   0.0026875, 0.00258333, 0.00247917,
                                   0.00239583, 0.0023125, 0.00220833,
                                   0.00210417, 0.00204167, 0.002, 0.001875,
                                   0.00185417, 0.00175, 0.00170833, 0.001625,
                                   0.0015625, 0.0015, 0.00147917, 0.0014375,
                                   0.00135417, 0.0014375, 0.00129167,
                                   0.00129167, 0.00125, 0.00122917])

        # Allow fewer CFs while still using defaults
        if len(cfs) != len(shift_vals):
            ref_cfs = 2 ** np.arange(np.log2(125), np.log2(16e3 + 1), 1/6)
            picks = [cf in np.round(cfs, 3) for cf in np.round(ref_cfs, 3)]
            shift_vals = shift_vals[picks]

        # Ensure the number of shift values matches the number of cfs
        msg = 'Number of CFs does not match number of known shift values'
        assert(len(shift_vals) == len(cfs)), msg
        lags = np.round(shift_vals * fs_in).astype(int)

        # Shift each channel
        for cfi in range(len(cfs)):
            anf_rates[cfi] = np.roll(anf_rates[cfi], -lags[cfi])
            anf_rates[cfi, -lags[cfi]:] = anf_rates[cfi, -(lags[cfi]+1)]

    # Shift, scale, and sum
    M1 = nuclei.M1
    anm = M1*anf_rates.sum(0)
    anm = np.roll(anm, final_shift)
    anm[:final_shift] = anm[final_shift+1]
    return(anm)


def model_abr(stim, fs_in, stim_pres_db, parallel=True, n_jobs=-1,
              stim_gen_rms=0.01, cf_low=125, cf_high=16e3, return_flag='abr'):
    """
    return_flag: str
     Indicates which waves of the abr to return. Defaults to 'abr' which
     returns a single abr waveform containing waves I, III, and V. Can also be
     '1', '3', or '5' to get individual waves. Combining these option will
     return a dict with the desired waveforms. e.g. '13abr' will return a
     dict with keys 'w1', 'w3', and 'abr'
    """

    return_flag = str(return_flag)
    known_flags = ['1', '3', '5', 'abr', 'rates']
    msg = ('return_flag must be a combination of the following: ' +
           str(known_flags))
    assert(findstring(return_flag, known_flags)), msg

    # Resample your stimuli to a higher fs for the model
    fs_up = int(100e3)
    stim_up = resample(stim, fs_up, fs_in, npad='auto', n_jobs=n_jobs)

    # Put stim in correct units
    # scalar to put it in units of pascals (double-checked and good)
    sine_rms_at_0db = 20e-6
    db_conv = ((sine_rms_at_0db / stim_gen_rms) * 10 ** (stim_pres_db / 20.))
    stim_up = db_conv * stim_up

    # run the model
    dOct = 1. / 6
    cfs = 2 ** np.arange(np.log2(cf_low), np.log2(cf_high + 1), dOct)
    anf_rates_up = np.zeros([len(cfs), len(stim)*fs_up//fs_in])

    if parallel:
        anf_rates_up = Parallel(n_jobs=n_jobs)([delayed(get_rates)(stim_up, cf)
                                               for cf in cfs])
    else:
        for cfi, cf in enumerate(cfs):
            anf_rates_up[cfi] = get_rates(stim_up, cf)

    # Downsample to match input fs
    anf_rates = resample(anf_rates_up, fs_in, fs_up, npad='auto',
                         n_jobs=n_jobs)

    # sum and filter to get AN and IC response, only use hsf to save time
    w3, w1 = nuclei.cochlearNuclei(anf_rates.T, anf_rates.T, anf_rates.T,
                                   1, 0, 0, fs_in)
    # filter to get IC response
    w5 = nuclei.inferiorColliculus(w3, fs_in)

    # shift, scale, and sum responses
    w1_shift = int(fs_in*0.001)
    w3_shift = int(fs_in*0.00225)
    w5_shift = int(fs_in*0.0035)
    w1 = np.roll(np.sum(w1, axis=1)*nuclei.M1, w1_shift)
    w3 = np.roll(np.sum(w3, axis=1)*nuclei.M3, w3_shift)
    w5 = np.roll(np.sum(w5, axis=1)*nuclei.M3, w5_shift)

    # clean up the roll
    w1[:w1_shift] = w1[w1_shift+1]
    w3[:w3_shift] = w3[w3_shift+1]
    w5[:w5_shift] = w5[w5_shift+1]

    # Handle output
    if return_flag == 'abr':
        return w1+w3+w5

    waves = {}
    if 'abr' in return_flag:
        waves['abr'] = w1+w3+w5
    if '1' in return_flag:
        waves['w1'] = w1
    if '3' in return_flag:
        waves['w3'] = w3
    if '5' in return_flag:
        waves['w5'] = w5
    if 'rates' in return_flag:
        waves['rates'] = anf_rates

    return waves


def get_ihc_voltage(stim_up, cf):
    fs_up = int(100e3)
    return(np.array(cochlea.zilany2014._zilany2014.run_ihc(stim_up,
                                                           cf,
                                                           fs_up,
                                                           species='human',
                                                           cohc=1,
                                                           cihc=1)))

def ihc(stim, fs_in, stim_pres_db, parallel=True, n_jobs=-1,
        stim_gen_rms=0.01, cf_low=125, cf_high=16e3):

    # Resample your stimuli to a higher fs for the model
    fs_up = int(100e3)
    stim_up = resample(stim, fs_up, fs_in, npad='auto', n_jobs=n_jobs)

    # Put stim in correct units
    # scalar to put it in units of pascals (double-checked and good)
    sine_rms_at_0db = 20e-6
    db_conv = ((sine_rms_at_0db / stim_gen_rms) * 10 ** (stim_pres_db / 20.))
    stim_up = db_conv * stim_up

    # run the model
    dOct = 1. / 6
    cfs = 2 ** np.arange(np.log2(cf_low), np.log2(cf_high + 1), dOct)
    ihc_out_up = np.zeros([len(cfs), len(stim)*fs_up//fs_in])

    if parallel:
        ihc_out_up = Parallel(n_jobs=n_jobs)([delayed(get_ihc_voltage)(stim_up, cf) for cf in cfs])
    else:
        for cfi, cf in enumerate(cfs):
            ihc_out_up[cfi] = get_ihc_voltage(stim_up, cf)

    # Downsample to match input fs
    ihc_out = resample(ihc_out_up, fs_in, fs_up, npad='auto', n_jobs=n_jobs)
    ihc_out_sum = ihc_out.sum(0)
    
    return ihc_out_sum
    
    

# %%
stim_fs = 44100
stim_pres_db = 65
t_mus = 64
eeg_fs = 10000
n_epoch = 40

file_path = "/media/tong/Elements/AMPLab/peaky_vs_ANM/present_files/"
data = read_hdf5(file_path + 'Exp1_stimuli.hdf5')


# %% 
len_eeg = int(t_mus*eeg_fs)
# unaltered x_in
x_in_unaltered_pos = np.zeros((n_epoch, len_eeg))
x_in_unaltered_neg = np.zeros((n_epoch, len_eeg))

for ei in range(n_epoch):
    print('unaltered {}'.format(ei))
    temp = data['audio'][0][ei, :]
    waves_pos = anm(temp, stim_fs, stim_pres_db)
    waves_pos_resmp = resample(waves_pos, down=stim_fs/eeg_fs)
    
    waves_neg = anm(-temp, stim_fs, stim_pres_db)
    waves_neg_resmp = resample(waves_neg, down=stim_fs/eeg_fs)
    
    x_in_unaltered_pos[ei, :] = waves_pos_resmp
    x_in_unaltered_neg[ei, :] = waves_neg_resmp
write_hdf5(file_path + '/ANM/unaltered_x_in.hdf5',
           dict(x_in_unaltered_pos=x_in_unaltered_pos,
                x_in_unaltered_neg=x_in_unaltered_neg,
                fs=eeg_fs), overwrite=True)
#%%
# braodband x_in
x_in_broadband_pos = np.zeros((n_epoch, len_eeg))
x_in_broadband_neg = np.zeros((n_epoch, len_eeg))
for ei in range(n_epoch):
    print('peaky {}'.format(ei))
    temp = data['audio'][1][ei, :]
    waves_pos = anm(temp, stim_fs, stim_pres_db)
    
    
    waves_neg = anm(-temp, stim_fs, stim_pres_db)
    waves_neg_resmp = resample(waves_neg, down=stim_fs/eeg_fs)
    
    x_in_broadband_pos[ei, :] = waves_pos_resmp
    x_in_broadband_neg[ei, :] = waves_neg_resmp
    
write_hdf5(file_path + '/ANM/broadband_x_in.hdf5',
           dict(x_in_broadband_pos=x_in_broadband_pos,
                x_in_broadband_neg=x_in_broadband_neg,
                fs=eeg_fs), overwrite=True)


# %% for audio sample
opath = "/media/tshan@urmc-sh.rochester.edu/Elements/AMPLab/peaky_vs_ANM/exp1/"
wav_unaltered, fs = read_wav(opath + "audio_sample/MaleNarrator_unaltered.wav")
waves_pos = anm(wav_unaltered[0], 44100, 65)
waves_pos_resmp = resample(waves_pos, down=44100/10000)
write_hdf5(opath+"audio_sample/MaleNarrator_unaltered_ANM.hdf5",
           dict(waves_pos=waves_pos, waves_pos_resmp=waves_pos_resmp), overwrite=True)