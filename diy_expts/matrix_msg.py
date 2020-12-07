# -*- coding: utf-8 -*-
"""
matrix_msg.py:  Matrix-valued message handler
"""

import numpy as np

class MsgHandler(object):
    def __init__(self,lam_min=0,lam_max=1e6,damp=0.5):
        """
        Sends messages from one node to another
        
        lam_min, lam_max:  Min and maximum eigenvalues
            of output message covariance.  This is used to 
            stabilize the algorithm
        """
        self.lam_min = lam_min
        self.lam_max = lam_max
        self.it = 0
        self.damp = damp
        self.Rinv0 = None
        
    def send_msg(self,Z,R,Zvar,Rvar):
        """
        Computes the MAD-VAMP message:
            
        Rvar1 = inv(inv(Zvar)-inv(Rvar)) 
        R1 = Rvar1*(Z1*Lam1 - R*Gam))
        """
        Gam = np.linalg.inv(Rvar)
        Lam = np.linalg.inv(Zvar)
        Rinv1 = Lam-Gam
        if self.Rinv0 is None:
            Rinv = Rinv1
        else:
            Rinv = (1-self.damp)*Rinv1 + self.damp*self.Rinv0             
        lamr, Vr = np.linalg.eigh(Rinv)
        lamr = np.maximum(self.lam_min,lamr)    
        lamr = np.minimum(self.lam_max,lamr)    
        self.Rinv0 = (Vr*lamr[None,:]).dot(Vr.T)
        Rvar1 = (Vr*(1/lamr[None,:])).dot(Vr.T)
        R1 = (Z.dot(Lam)-R.dot(Gam)).dot(Rvar1)
        return R1, Rvar1