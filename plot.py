# -*- coding: utf-8 -*-
"""
Created on Wed May 22 14:19:34 2019

@author: sdran
"""
import numpy  as np
import matplotlib.pyplot as plt
import pickle
# from matplotlib import rc
# rc('text', usetex=True)

plt.rcParams.update({'font.size': 16})

snr_plot = [10,15]
nplot = len(snr_plot)
plt.figure(figsize=(10,5))
plt.rcParams.update({'font.size': 12})

for iplot, snr in enumerate(snr_plot):

    keras_fn = ('saved_expts/adam_snr%d.pkl' % snr)
    with open(keras_fn,'rb') as fp:
        mse_ts_keras,mse_tr,ntr_keras,snr_keras = pickle.load(fp)
        
    vamp_fn = ('saved_expts/ml_mat_vamp_snr%d.pkl'% snr)
    with open(vamp_fn,'rb') as fp:
        mse_ts_vamp,ntr_vamp,nin,snr_vamp,se_test = pickle.load(fp)
    
    vamp_se_fn = ('saved_expts/ml_mat_vamp_se_snr%d.pkl' % snr)
    with open(vamp_se_fn,'rb') as fp:
        mse_ts_se,ntr_se,nin,snr_vamp,se_test = pickle.load(fp)
        
    
    mse_avg_se = np.median(mse_ts_se[-1,:,:],axis=0)    
    mse_avg_vamp = np.median(mse_ts_vamp[-1,:,:],axis=0)
    mse_avg_keras = np.median(mse_ts_keras[-1,:,:],axis=0)
    
    plt.subplot(1,nplot,iplot+1)
    plt.plot(ntr_keras, mse_avg_keras, 'o-', fillstyle='none', lw=2, ms=10)
    plt.plot(ntr_vamp, mse_avg_vamp, 's-', fillstyle='none', lw=2, ms=10)
    plt.plot(ntr_se, mse_avg_se, '-', lw=2)
    plt.grid()
    plt.title('SNR=%d dB' % int(snr))
    plt.ylim((1,2.5))
    plt.xlabel('Num training samples')
    plt.ylabel('Normalized test MSE')
    plt.legend(['ADAM-MAP', 'ML-Mat-VAMP', 'ML-Mat-VAMP (SE)'])
    
plt.tight_layout()    
fig_name = 'mse_vs_ntr.png'
plt.savefig(fig_name)