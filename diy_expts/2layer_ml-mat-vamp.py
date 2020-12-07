# -*- coding: utf-8 -*-
"""
mad_vamp_test.py:  Main mad_vamp test
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from matrix_msg import MsgHandler
from madvamp import MadVamp
from madvamp_se import MadVampSE
import argparse
import pickle

import twolayer
import neural_est

"""
Parse arguments from command line
"""
parser = argparse.ArgumentParser(description='MAP estimation with keras')
parser.add_argument('--ntr',action='store',nargs='+',\
    default=[200,  400, 600,  800, 1000, 1500, 2000, 2500, 3000, 4000],type=int,\
    help='num training samples to test')
parser.add_argument('--nts',action='store',default=1000,type=int,\
    help='number of test samples')
parser.add_argument('--nin',action='store',default=100,type=int,\
    help='number of inputs')
parser.add_argument('--nit',action='store',default=20,type=int,\
    help='number of SE iterations')
parser.add_argument('--snr',action='store',default=10.0,type=float,\
    help='SNR in dB')
parser.add_argument('--wcorr_init',action='store',default=0.25,type=float,\
    help='Initial correlation with W1 matrix')
parser.add_argument('--damp',action='store',default=0.0,type=float,\
    help='Damping factor (0=no damping)')
parser.add_argument('--ntest',action='store',default=10,type=int,\
    help='number of tests per trial')
parser.add_argument('--fn_suffix',action='store',\
    default=0,type=int, help='filename suffix')
parser.add_argument('--act',action='store',\
    default='sigmoid',help='activation (sigmoid,relu or linear)')
parser.add_argument('--se_test', dest='se_test', action='store_true',\
    help="Performs state evolution instead of actual simulation")
parser.add_argument('--w2train', dest='w2train', action='store_true',\
    help="Learns W2")  
parser.set_defaults(w2train=False)   
parser.set_defaults(se_test=False)   

args = parser.parse_args()



ntr_test  = args.ntr
nts  = args.nts
nin  = args.nin
nit = args.nit
snr = args.snr
ntest = args.ntest
damp = args.damp
fn_suffix = args.fn_suffix
act_type = args.act
wcorr_init = args.wcorr_init
se_test = args.se_test
w2train = args.w2train
nparam = len(ntr_test)
vamp_verbose = False

#%%

class ValMse(object):
    def __init__(self,Xts,Yts,est0,est1,yvar,act_type='sigmoid',\
                 verbose=False):
        """
        Class to compute the validation MSE
        """
        self.Xts = Xts
        self.Yts = Yts
        self.est0 = est0
        self.est1 = est1
        self.yvar = yvar
        self.it = 0
        self.verbose = verbose
        self.act_type = act_type
        
    def act(self,Z):
        if self.act_type == 'sigmoid':
            U = 1/(1+np.exp(-Z))
        elif self.act_type == 'relu':
            U = np.maximum(0,Z)
        elif self.act_type == 'linear':
            U = Z
        else:
            raise ValueError('Unknown activation type')
        return U
        
        
    def metric(self):
        """
        Compute the test error
        """
        Zhat = self.Xts.dot(self.est0.What)
        Uhat = self.act(Zhat)
        Yhat = Uhat.dot(self.est1.What)
        mse = np.mean((Yhat-self.Yts)**2)/self.yvar
        
        if self.verbose:
            print('it=%d mse=%12.4e' % (self.it, mse))
        self.it += 1
        return mse
        
#%%        
# Initialize variables
mse_ts = np.zeros((nit,ntest,nparam))
    
# Main simulation loop
for iparam, ntr in enumerate(ntr_test):
    for it in range(ntest):

        # Generate a random model
        mod = twolayer.TwoLayerMod(nin=nin,snr=snr)
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
        Z0tr = Z0[:ntr]
        Z0ts = Z0[ntr:]
        Xtr = X[:ntr]
        Ytr = Y[:ntr]
        Xts = X[ntr:]
        Yts = Y[ntr:]
    
        # Create estimators
        d = mod.d
        wvar1 = np.mean(mod.W1**2)
        wvar2 = np.mean(mod.W2**2)
        
        # Input estimator and message handler
        lam_min = 1e-3
        lam_max = 1e3
        est0 = neural_est.NeuralInEst(X=Xtr,d=d,wvar=wvar1,\
                                      Winit=W1init, werr=w1err)
        msg0 = MsgHandler(lam_min=lam_min,lam_max=lam_max,damp=damp)
        
        # Output estimator and message handler        
        est1 = neural_est.NeuralOutEst(Y=Ytr,shape=Z0tr.shape,yvar=mod.yvar,\
                        act_type=act_type,wtrain=w2train, Winit=W2init)
        msg1 = MsgHandler(lam_min=lam_min,lam_max=lam_max,damp=damp)

        # Create a test error callback
        val_mse = ValMse(Xts,Yts,est0,est1,mod.yvar,act_type=act_type,\
                         verbose=vamp_verbose)

        # Create VAMP-SE solver
        if se_test:
            solver = MadVampSE(est0,est1,msg0,msg1,Z0=Z0tr,\
                           nit=nit,metric_cb = val_mse.metric,\
                           hist_list=['metric'])
        else:
            solver = MadVampSE(est0,est1,msg0,msg1,Z0=Z0tr,\
                           nit=nit,metric_cb = val_mse.metric,\
                           hist_list=['metric'])
            
        # Run the solver
        try:
            solver.solve()
        
            # Get metric
            mse_tsi = np.array(solver.hist_dict['metric'])            
        except:
            # Fill in metric with a large value
            print('solver failed')
            mse_tsi = 10*np.ones(nit)
        mse_ts[:,it,iparam] = mse_tsi
                        
        # Print result
        print('ntr=%d it=%d mse=%12.4e' % (ntr,it,mse_tsi[-1]))
        
if 1:
    if se_test:
        se_str = '_se'
    else:
        se_str = ''
    fn = ('ml_mat_vamp%s_snr%d_%s.pkl' % (se_str,int(snr),fn_suffix))
    with open(fn,'wb') as fp:
        pickle.dump([mse_ts,ntr_test,nin,snr,se_test], fp)

