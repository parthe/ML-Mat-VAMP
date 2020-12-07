"""
neural_out.py:  Estimators for the two-layer neural network
"""


import numpy as np
import matplotlib.pyplot as plt

class NeuralInEst:
    def __init__(self, X, d=4, wvar=1.0,\
                 Winit=None,werr=1):
        """
        Proximal operator corresponding to the penalty:
            
            f(Z) := 0.5*||W||^2/wvar,  where Z=X.dot(W)
                    
        wvar:  Variance of W
        Winit:   True W, used for debugging
        wcorr_init:  Initial correlation between true W and
               estimate
        """
        
        # Set dimensions
        self.X = X
        nsamp = self.X.shape[0]
        self.shape = (nsamp,d)
        self.Winit = Winit
        self.werr = werr
        
        
        # Pre-compute an SVD of X
        Vx,sx,Uxtr = np.linalg.svd(X,full_matrices=False)
        self.Vx = Vx
        self.sx = sx
        self.Uxtr = Uxtr
        
        # Save regualarization variance
        self.wvar = wvar

        
    def est_init(self):
        """
        Initial estimate
        """
        Zhat = self.X.dot(self.Winit)
        d = Zhat.shape[1]
        Zvar = self.werr*np.mean(self.X**2)*np.eye(d)
        return Zhat, Zvar
        
        
    def est(self,R,Rvar):
        """
        Proximal operator.  Solves:
            
            J(Z) := 0.5*||Z-R||^2_Rvar + f(Z),
        
        Since we have Z=X.dot(W), this can be solved as:
            
            Zhat = X.dot(What)
            What = argmin ||X.dot(W)-R||^2_{Rvar} + ||W||^2/wvar
            
        This is solved via taking an eigenvalue decomposition:
            
            Rvar = Vr*diag(lamr)*Vr.T
            
        Also, we have the SVD:  X=Vx.dot(diag(sx)).dot(Uxtr)
        
        Define:
            Wv = Uxtr.dot(W).dot(Vr)
            Zv = Vx.T.dot(Z).dot(Vr) 
               = Vx.T.dot(X.dot(W)).dot(Vr) = diag(sx)*Wv
            Rv = Vx.T.dot(R).dot(Vr)
        
        Hence, we have
           Wvhat:= argmin J(Wv)
           J(Wv) := ||diag(sx)*Wv-Rv||^2_{lamr} + ||Wv||^2_F
                                   
        Each row is the minimum of
        
          ||sx[i]*Wv[i,:] + Rv[i,:]||^2_{lamr} + ||Wv[i,:]||^2/wvar
                
        The minimum is
        
           Wv[i,:] = wvar*sx[i]/(sx[i]**2+lamr)*D[i,:]
           Zv[i,:] = (wvar*sx[i]**2)/(sx[i]**2+lamr)*D[i,:]
        """
                
        # Take eigenvalue decomposition of Rvar
        # Rvar = Vr*diag(lamr)*Vr.T
        lamr, Vr = np.linalg.eigh(Rvar)
        nsamp, ns = self.Vx.shape
        
        # Transform R to basis of Vx and Vr
        Rv = self.Vx.T.dot(R).dot(Vr)
        
        # Compute z gain
        swvar = (np.abs(self.sx)**2)*self.wvar
        #gain = swvar[:,None]/(swvar[:,None] + lamr[None,:])
        gain = self.sx[:,None]*self.wvar/(swvar[:,None] + lamr[None,:])
        Wv = gain*Rv
        Zv = self.sx[:,None]*Wv
        
                    
        # Transform back
        self.What = self.Uxtr.T.dot(Wv).dot(Vr.T)
        Zhat  = self.Vx.dot(Zv).dot(Vr.T)
        
        # Compute variance
        lamzhat = np.mean(gain*lamr[None,:], axis=0)*ns/nsamp
        Zvar = (Vr*lamzhat[None,:]).dot(Vr.T)
        return Zhat, Zvar

class NeuralOutEst:
    def __init__(self, Y, shape, yvar=1.0, wtrain=True, Winit=None,
                 act_type='sigmoid'):
        """
        Proximal operator corresponding to the penalty:
        
             f(Z) := min_W (1/2/yvar)*||Y-act(Z).dot(W)||^2,
             
        where act(Z) is an activation.  This penalty arises
        in the output layer of a two-layer neural network.
        
        Currently, this program only supports ReLU.
        Also, the estimator only supports scalar targets Y.
        
        wtrain:  Enables training of W
        Winit:  Initial estimate for W
        """
        self.Y = Y
        self.yvar = yvar        
        self.Zinit = None
        self.shape = shape
        self.var_axes = (0,)
        self.wtrain = wtrain
        self.Winit = Winit
        self.What = np.copy(Winit)

        # Gradient descent parameters        
        self.nitmax = 1000
        self.lr = 0.001
        self.lr_min = 1e-5
        self.gnormtol = 1e-3
        self.wtrain_per = 10
        
        # Check if shapes match
        if shape[0] != self.Y.shape[0]:
            err_msg = 'shape ' + str(shape) + 'not consistent with'\
                      'shape of Y ' + str(Y.shape)
            raise ValueError(err_msg)
        
        # Check if there is a single output
        if (len(Y.shape) > 1):
            if Y.shape[1] > 1:
                raise ValueError('Y must be shape (n,) or (n,1)')
            
        # Check for a valid activation type
        self.act_type = act_type
        if self.act_type not in ['sigmoid', 'relu', 'linear']:
            raise ValueError('Unknown activation type')
        
    def act(self,Z):
        """
        Activation with gradient
        """
        if self.act_type == 'sigmoid':
            U = 1/(1+np.exp(-Z))
            Ugrad = U*(1-U)
        elif self.act_type == 'relu':
            U = np.maximum(0,Z)
            Ugrad = (U > 0)
        elif self.act_type == 'linear':
            U = Z
            Ugrad = np.ones(Z.shape)
        else:
            raise ValueError('Unknown activation type')
        return U, Ugrad
    
    def estW(self,Z):
        """
        Updates estimate of W
        """
        U, Ugrad = self.act(Z)
        self.What = np.linalg.lstsq(U,self.Y,rcond=None)[0]
        
    
    def Jeval(self,Z):
        """
        Objective and gradient of the proximal optimization:
            
            J(Z) := 0.5*||Z-R||^2_Rvar + f(Z),
            
        where f(Z) is describe above and \|X\|_{Rvar} is the weighted
        two norm:
        
            ||X||^2_{Rvar} = \sum_i Z[i,:].dot(inv(Rvar)).dot(X[i,:])
        """
        # Function and gradient of the proximal term
        D0 = Z-self.R
        DR = D0.dot(self.Rinv)
        J0 = 0.5*np.sum(DR*D0)
        Jgrad0 = DR
        
        # Function and gradient of the penalty term
        U, Ugrad = self.act(Z)
        D1 = U.dot(self.What)-self.Y
        J1 = 0.5*np.sum(D1**2)/self.yvar
        Jgrad1 = D1.dot(self.What.T)*Ugrad/self.yvar
        
        # Sum 
        J = J0 + J1
        Jgrad = Jgrad0 + Jgrad1
        
        return J, Jgrad
    
    def computeZvar(self,Zhat,Rvar):
        """
        Computes the row-wise average second derivative.
        We assume that nout=1, so What is shape(d)
        Define, U = act(Z) and Ugrad = dU/dZ, 
        
           UW[i,j] = Ugrad[i,j]*What[j],  j=1,..,d
           
        The Hessian is 
        
           Zvar = E[ inv(UW[i,:]*UW[i,:]'/yvar + inv(Rvar)) ]
           
        This is computed via the matrix inversion Lemma,
        
            Zvar = E[ Rvar - RUW[i,:]*RUW[i,:]/(yvar + RUW[i,:]*RU[i,:])]
        """
        # Output estimate
        U, Ugrad = self.act(Zhat)
        UW = Ugrad*self.What[None,:,0]
        RUW = UW.dot(Rvar)
        
        
        scale = np.sum(RUW*UW,axis=1)+self.yvar
        Zvar = Rvar[None,:,:] - RUW[:,:,None]*RUW[:,None,:]/scale[:,None,None]
        Zvar = np.mean(Zvar,axis=0)
        return Zvar                
                
    
    def Jopt(self, Zinit=None):
        """
        Gradient descent optimization on Z
        """      
        
        # Intialize the estimate for Z
        if self.Zinit is None:
            self.Zinit = np.copy(self.R)
        Z = self.Zinit
        
        # Initialize estimate for W
        if self.Winit is None:
            self.estW(Z)
        else:
            self.What = np.copy(self.Winit)
        
        # Perform gradient descent
        self.Jhist = []
        self.gnormhist = []
        self.lrhist = []
        J, Jgrad = self.Jeval(Z)
        done = False
        it = 0        
        while not done:
            
            # Compute fn and gradient test point
            Z1 = Z - self.lr*Jgrad
            J1, Jgrad1 = self.Jeval(Z1)
            
            # See if decrease passes the Armijo rule
            dJest = np.sum(Jgrad*(Z1-Z))
            if (J1-J < 0.5*dJest):
                self.lr = self.lr*2
                J = J1
                Jgrad = Jgrad1
                Z = Z1
            else:
                self.lr = np.maximum(self.lr*0.5,self.lr_min)
            
            # Add history for debugging
            gnorm = np.sqrt(np.mean(Jgrad**2))             
            self.Jhist.append(gnorm)
            self.gnormhist.append(gnorm)
            self.lrhist.append(self.lr)
            
            # Test stopping condition
            it += 1
            if (gnorm < self.gnormtol) or (it >= self.nitmax):
                done = True
                
            if (np.mod(it, self.wtrain_per) == 0) and self.wtrain:
                self.estW(Z)
                
        # For warm start in next iteration save the initial condition
        self.Zinit = np.copy(Z)
        
        
        return Z
                        
        
    def est(self, R, Rvar):
        """
        Computes the proximal operator:
            
            Zhat = argmin J(Z,R,Rvar)
            Zvar = J''(Zhat,R,Rvar)
        """
        
        # Check dimensions
        if R.shape != self.shape:
            raise ValueError('R shape %s is not correct' % str(R.shape))
        d = self.shape[1]
        if Rvar.shape != (d,d):
            raise ValueError('Rvar shape %s is not correct' % str(Rvar.shape))
            
        # Set the variables
        self.Rinv = np.linalg.inv(Rvar)        
        self.R = R
        
        # Run the optimization
        Zhat = self.Jopt()
        
        # Compute the second derivative
        Zvar = self.computeZvar(Zhat,Rvar)
        
        return Zhat, Zvar
        
    
def rand_cov(d,rho=0.9,scale=0.1):
    """
    Generates random covariance matrix
    """
    A = np.random.normal(0,1,(d,d))
    U,s,Vtr =np.linalg.svd(A)
    lam = rho**np.arange(d)
    lam = scale*lam/np.mean(lam)
    Rvar = (U*lam[None,:]).dot(U.T)
    return Rvar    

def neural_out_test(nsamp=1000,d=4,yvar=0.01,rvar=0.04):
    """
    Unit test for the output estimator
    """
    
    # Set parameters
    nout = 1
    Rvar = rand_cov(d=d,rho=0.5,scale=rvar)

    # Creat random test data
    Z0 = np.random.normal(0,1,(nsamp,d))
    W0 = np.random.normal(0,np.sqrt(1/nout),(d,nout))
    U0 = np.maximum(0,Z0)
    Y0 = U0.dot(W0)
    V = np.random.normal(0,np.sqrt(yvar),(nsamp,nout))
    Y = Y0+V
    V = np.random.multivariate_normal(np.zeros(d),Rvar,(nsamp,))
    R = Z0 + V

    # Create estimator
    est = NeuralOutEst(Y,shape=(nsamp,d), yvar=yvar, act_type='relu')
    Zhat, Zvar =  est.est(R,Rvar)
    
    # Measure the empirical error
    D = Z0-Zhat
    Zvar0 = 1/nsamp*(D.T.dot(D))
    
    zerr = np.sum((Zvar0-Zvar)**2)/np.sum(Zvar0**2)
    print('Relative error = %12.4e' % zerr)
    
def neural_in_test(nsamp=1000,nin=400,d=4,rvar=0.1,wvar=None):
    """
    Unit test for the neural input estimator
    """
    if wvar is None:
        wvar = 1/nin

    # Generate random data
    Rvar = rand_cov(d=d,rho=0.5,scale=rvar)    
    X = np.random.normal(0,1,(nsamp,nin))    
    W0 = np.random.normal(0,np.sqrt(wvar),(nin,d))
    Z0 = X.dot(W0)
    V = np.random.multivariate_normal(np.zeros(d),Rvar,(nsamp,))
    R = Z0 + V
    
    # Create and run estimator
    input_est = NeuralInEst(X,wvar=wvar)
    Zhat, Zvar = input_est.est(R,Rvar)
    
    # Measure true error variance
    Zvar0 = 1/nsamp*(Zhat-Z0).T.dot(Zhat-Z0)
    zerr = np.sum((Zvar0-Zvar)**2)/np.sum(Zvar0**2)
    print('Relative error = %12.4e' % zerr)
    
    


