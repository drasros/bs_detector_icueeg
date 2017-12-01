# This short script demonstrates burst-suppression on an ICU EEG channel
# using a very simple method based on thresholding local variance. 
# method is described in :
# 'Real-time segmentation of burst suppression patterns in critical care EEG monitoring', 
# by Westover et al. 

# RE reimplement it here and briefly demonstrate it on
# on a record from the TUH EEG Corpus

from __future__ import print_function

import pyedflib
import numpy as np
import matplotlib.pyplot as plt

###################################
# PARAMS
rec_to_segment = '/Users/arnaudsors/Downloads/v1.1.2_train_full/edf/train/abnormal/015/00002869/s03_2013_11_05/00002869_s03_a00.edf'
plot_interval = [5*60 + 4, 5*60 + 4 + 6]
theta = 75.

rec_to_segment = '/Users/arnaudsors/Downloads/v1.1.2_train_full/edf/train/abnormal/016/00003029/s02_2012_10_18/00003029_s02_a00.edf'
plot_interval = [11*60+42, 11*60+42 + 20]
theta = 150.

# plot interval in seconds: example [10., 20.5]
# if you do not want to plot, use None 
# if you want to plot everything, use [0., -1.]
#plot_interval = [0, -1.]

###################################

def read_edfrecord(filename, channel):
    f = pyedflib.EdfReader(filename)
    print("Startdatetime: ", f.getStartdatetime())
    signal_labels = f.getSignalLabels()
    print(signal_labels)
    assert channel in signal_labels
    idx_chan = [i for i, x in enumerate(signal_labels) if x==channel][0]
    sigbuf = np.zeros((f.getNSamples()[idx_chan]))
    sigbuf[:] = f.readSignal(idx_chan)
    samplingrate = len(sigbuf) / f.file_duration
    print("sampling rate: ", samplingrate)
    return sigbuf, samplingrate

def recursive_variance_estimation(sig, beta=0.9633, theta=50., min_suppr_len=50):
    # beta: forgetting factor. 0.9534 in article with Fs=200Hz
    # tau = -1/(Fs * (ln beta))
    # therefore beta = exp(-1/(tau*Fs))
    # and we have Fs=250 for the TUH EEG data
    # So let's use beta=0.9633
    # theta: threshold for declaring 'suppressed EEG'
    # Recommended approach: use base beta but tune theta for each patient

    # algo can't be vectorized because it is recursive...
    sig = list(sig)
    res = []
    res_var = []
    res_smoothed = []

    # start with mu=0 and sigsquare=0 (this could be improved)
    mu = 0
    sigsquare = 0

    for i in range(len(sig)):
        mu = beta * mu + (1. - beta) * sig[i]
        sigsquare = beta * sigsquare + (1. - beta) * (sig[i] - mu)**2
        res_var += [sigsquare]
        res += [int(not sigsquare < theta)]

    # added compared to original method:
    # smoothing: if a 'suppression' period is shorter than min_suppr
    # samples, delete it
    is_suppressed_keep = False
    for i in range(len(res)):
        if res[i] == 0: # suppressed
            if is_suppressed_keep:
                # we already know that this suppression is long enough and
                # we can keep it
                res_smoothed += [0]
            else:
                # check that enough suppressed samples come after
                if max(res[i:i+min_suppr_len])==0: 
                    res_smoothed += [0]
                    is_suppressed_keep = True
                else:
                    # 'artifact' suppression too short. Consider it not suppressed
                    res_smoothed += [1] 
        else: # not suppressed
            # reset counter
            is_suppressed_keep = False
            res_smoothed += [1]

    return res, res_smoothed, res_var


# read and plot
sigbuf, samplingrate = read_edfrecord(rec_to_segment, channel='EEG O2-REF')
if plot_interval is not None:
    begin = int(plot_interval[0] * samplingrate)
    if plot_interval[-1] == -1.:
        end = len(sigbuf)
    else:
        end = int(plot_interval[1] * samplingrate)
    sig_to_plot = sigbuf[begin:end]
print(len(sigbuf))
print(len(sig_to_plot))
#plt.plot(sig_to_plot)

# segment
_, sig_segmented, sig_recvar = recursive_variance_estimation(sigbuf, theta=theta)
sig_segmented_to_plot = sig_segmented[begin:end]
sig_recvar_to_plot = sig_recvar[begin:end]

print(sig_segmented)

def normalize(sig):
    #amplitude
    sig = np.array(sig)
    sig -= np.mean(sig)
    #sig /= np.linalg.norm(sig)
    signorm = 5 * np.sqrt(np.mean(np.square(sig))) #mean not sum here.. (otherwise depends on length)
    sig /= signorm
    return sig

# plot again with original signal and segmentation
# NORMALIZE BEFORE PLOTTING
sig_to_plot = normalize(sig_to_plot)
plt.figure(figsize=(20, 2))
plt.plot(np.arange(len(sig_to_plot))/float(samplingrate), sig_to_plot, label='EEG signal')
plt.plot(np.arange(len(sig_to_plot))/float(samplingrate), sig_segmented_to_plot, label='segmentation')#,
#         range(len(sig_to_plot)), sig_recvar_to_plot)
plt.xlabel('time(s)', fontsize=16)
plt.ylabel('normalized\namplitude', fontsize=16)
plt.legend(fontsize=12, loc='lower right')
plt.ylim([-1.1, 1.1])
plt.tight_layout()
plt.savefig('segmentation.eps')




