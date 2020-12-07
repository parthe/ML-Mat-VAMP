# -*- coding: utf-8 -*-
"""
keras_map:  Keras MAP estimation of the two layer neural network
"""
import numpy as np
import matplotlib.pyplot as plt
import pickle
import argparse
import twolayer
import json
from types import SimpleNamespace

"""
Parse arguments from command line
"""
parser = argparse.ArgumentParser(description='MAP estimation with keras')
parser.add_argument('--ntr',action='store',nargs='+',\
    default=[ 200,  400,  600,  800, 1000, 1500, 2000, 2500, 3000],type=int,\
    help='num training samples to test')
parser.add_argument('--nts',action='store',default=1000,type=int,\
    help='number of test samples')
parser.add_argument('--nin',action='store',default=100,type=int,\
    help='number of inputs')
parser.add_argument('--nepochs',action='store',default=100,type=int,\
    help='number of epochs for each model')
parser.add_argument('--snr',action='store',default=10.0,type=float,\
    help='SNR in dB')
parser.add_argument('--ntest',action='store',default=10,type=int,\
    help='number of tests per trial')
parser.add_argument('--fn_suffix',action='store',\
    default=0,type=int,\
    help='filename suffix')
parser.add_argument('--act',action='store',\
    default='sigmoid',help='activation (sigmoid,relu or linear)')
parser.add_argument('--wcorr_init',action='store',default=0.25,type=float,\
    help='Initial correlation with W1 matrix')
parser.add_argument('--w2train', dest='w2train', action='store_true',\
    help="Learns W2")

args = parser.parse_args()
ntr_test  = args.ntr
nts  = args.nts
nin  = args.nin
nepochs = args.nepochs
snr = args.snr
ntest = args.ntest
act = args.act
fn_suffix = args.fn_suffix
wcorr_init = args.wcorr_init
w2train = args.w2train
#%%
# Check if activation is valid
if act not in ['sigmoid', 'relu', 'linear']:
    raise ValueError('Unknown activation')

ntr_test = np.array([200,400,600,800,1000,1500,2000,2500,3000])
nparam = len(ntr_test)

# Initialize arrays
mse_ts = np.zeros((nepochs,ntest,nparam))
mse_tr = np.zeros((nepochs,ntest,nparam))

# Main simulation loop
for iparam, ntr in enumerate(ntr_test):
    for it in range(ntest):

        # Generate a random model
        mod = twolayer.TwoLayerMod(nin=nin,snr=snr,act_type=act)
        mod.gen_rand_param()
        
        # Get initial condition
        W1init, w1err = mod.gen_W1init(wcorr_init)
        if w2train:
            W2init = mod.gen_W2init(wcorr_init)
        else:
            W2init = mod.W2
        
        # Generate data and split into training and test
        nsamp = ntr + nts
        X, Y = mod.gen_rand_dat(nsamp) 
        Z0 = mod.Z0
        Y0 = mod.Y0
        Xtr = X[:ntr]
        Ytr = Y[:ntr]
        Xts = X[ntr:]
        Yts = Y[ntr:]
        
        # Fit keras model
        mod_est = twolayer.TwoLayerKerasMod(mod0=mod,act_type=act,\
                    W1init=W1init,W2init=W2init,w2train=w2train)
        mod_est.fit(Xtr,Ytr,Xts,Yts,nepochs=nepochs)
        
        mse_tsi = mod_est.hist.history['val_mse']/mod.yvar
        mse_tri = mod_est.hist.history['mse']/mod.yvar
        
        # Save results
        mse_ts[:,it,iparam] = mse_tsi
        mse_tr[:,it,iparam] = mse_tri
        print('ntr=%d it=%d mse: %12.4e %12.4e' %\
              (ntr,it,mse_tri[-1],mse_tsi[-1]))

if 1:
    fn = 'adam_snr%d_%s.pkl' % (snr, fn_suffix)
    with open(fn,'wb') as fp:
        pickle.dump([mse_ts,mse_tr,ntr_test,snr], fp)