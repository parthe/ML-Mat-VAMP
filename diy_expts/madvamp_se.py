# -*- coding: utf-8 -*-
"""
mad_vamp_test.py:  Main mad_vamp test
"""
import numpy as np
import matplotlib.pyplot as plt
import solver


class MadVampSE(solver.Solver):
    def __init__(self,est0,est1,msg0,msg1,Z0,nit=1,nv=1,\
                 rvar_init_corr=0.25, bias_se=False,\
                 metric_cb=None,hist_list=[]):
        """
        Mad-VAMP State Evolution
        
        bias_se:  Allow for biased SE
        nit:  Number of iterations
        """
        # Save estimators and message handlers
        self.est0 = est0
        self.est1 = est1
        self.msg0 = msg0
        self.msg1 = msg1
        self.Z0 = Z0
        self.rvar_init_scale = 4
        self.metric_cb = metric_cb
        
        # Check shapes
        if self.Z0.shape != self.est0.shape:
            raise ValueError('Z0.shape and est0.shape do not match')
        if self.Z0.shape != self.est1.shape:
            raise ValueError('Z0.shape and est1.shape do not match')
        
        # Get dimensions
        self.n, self.d = self.Z0.shape
        
        # Other parameters
        self.nit = nit        
        solver.Solver.__init__(self, hist_list)
        
    def lin_fit(self,Rt):
        """
        Finds a linear model:  Rt = Z0*A + N(0,Rerr)
        Also, generates a new R with the same statistics    
        """
        # This appears to sometimes not converge
        A = np.linalg.lstsq(self.Z0, Rt, rcond=1e-8)[0]
        D = self.Z0.dot(A)-Rt
        n = D.shape[0]
        Rerr = (1/n)*D.T.dot(D)
        return A, Rerr
        
    def solve(self):
        
        # Initial estimate
        Zhat0,Zvar0 = self.est0.est_init()
        R1t = Zhat0
        self.Rvar1 = Zvar0
        
        if 0:
            self.A0 = np.eye(self.d)
            rvar_init = self.rvar_init_scale*np.mean(self.Z0**2)
            self.Rvar0 = rvar_init*np.eye(self.d)
            self.Rerr0 = rvar_init*np.eye(self.d)
            
        # Initialize
        self.metric = []
        mu = np.zeros(self.d)
                    
        
        for it in range(self.nit):
            
            # Find statistics on R1
            self.A1, self.Rerr1  = self.lin_fit(R1t)
            
            # Generate random data R1
            self.R1 = self.Z0.dot(self.A1) +\
                np.random.multivariate_normal(mu,self.Rerr1,size=self.n)
                
            # Run estimator 0 on random data
            self.Zhat1, self.Zvar1 = self.est1.est(self.R1, self.Rvar1)            
            
            # Message to estimator 0
            R0t, self.Rvar0 = self.msg0.send_msg(self.Zhat1,self.R1,\
                                 self.Zvar1,self.Rvar1)
            
            # Find statistics on R0
            self.A0, self.Rerr0  = self.lin_fit(R0t)
            
            # Generate random data R0
            self.R0 = self.Z0.dot(self.A0) +\
                np.random.multivariate_normal(mu,self.Rerr0,size=self.n)
                
            # Run estimator 0 on random data
            self.Zhat0, self.Zvar0 = self.est0.est(self.R0, self.Rvar0)
            
            # Compute metric, if needed
            if not (self.metric_cb is None):
                self.metric = self.metric_cb()
                        
            # Message to estimator 1
            R1t, self.Rvar1 = self.msg1.send_msg(self.Zhat0,self.R0,\
                                 self.Zvar0,self.Rvar0)
            
                          
            self.save_hist()
