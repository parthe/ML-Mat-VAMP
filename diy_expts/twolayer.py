"""
twolayer.py:  Two layer neural network model
"""
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
import tensorflow.keras.backend as K
from sklearn.model_selection import train_test_split
from tensorflow.keras import regularizers

class TwoLayerMod:
    def __init__(self,nin=10,d=4,nout=1,bias=False,snr=10,\
                 act_type='sigmoid'):
        """
        Two layer neural network model class
        """
        self.nin = nin
        self.nout = nout
        self.d = d
        self.bias = bias
        self.snr = snr
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
    
        
    def gen_rand_param(self):
        """
        Generates random parameters
        """
        self.W1 = np.random.normal(0,1,(self.nin,self.d))
        self.W2 = np.random.normal(0,1,(self.d,self.nout))
        
    def gen_W1init(self,wcorr=0.25):
        """
        Generates a random initial condition with a target
        correlation to the true W1
        """
        W1init = np.sqrt(wcorr)*self.W1\
            + np.random.normal(0,np.sqrt(1-wcorr), self.W1.shape)
        werr = (1-np.sqrt(wcorr))**2
        return W1init, werr
    
    def gen_W2init(self,wcorr=0.25):
        """
        Generates a random initial condition with a target
        correlation to the true W2
        """
        W2init = np.sqrt(wcorr)*self.W2\
            + np.random.normal(0,np.sqrt(1-wcorr), self.W2.shape)
        return W2init

    
    def predict(self,X):
        """
        Computes the output given input    
        """
        Z = X.dot(self.W1)
        U = self.act(Z)
        Y = U.dot(self.W2)
        return Z,Y
        
    def gen_rand_dat(self,nsamp):
        """
        Generates random data
        """
        if self.bias:   
            if self.nin < 2:
                raise ValueError('nin >= 2 for bias term')
            xvar = 1/(self.nin-1)
            X = np.random.normal(0,np.sqrt(xvar),(nsamp,self.nin-1))
            X = np.hstack((np.ones((nsamp,1)), X))
        else:
            X = np.random.normal(0,np.sqrt(1/self.nin),(nsamp,self.nin))
            
        Z0,Y0 = self.predict(X)
        
        self.Z0 = Z0
        self.Y0 = Y0
        
        # Add noise
        self.yvar = np.mean(Y0**2)*(10**(-0.1*self.snr))
        Y = self.Y0 + np.random.normal(0,np.sqrt(self.yvar), (nsamp,self.nout))        
        
        return X,Y 
    
class TwoLayerKerasMod(object):
    def __init__(self,mod0,map_est=True,w1_init_corr=0.2,lr=0.01,\
                 batch_size=32,act_type='sigmoid',W1init=None,W2init=None,\
                 w2train=False):
        """
        Keras two-layer model
        
        mod0:  True two layer model 
        map_est:  Add regularization on W1 for MAP estimation
        w1_init_corr:  Initial correlation with W1
        lr:  Learning rate
        """
    
        # Save params
        self.lr = lr
        self.batch_size = batch_size
        self.act_type = act_type
        self.W1init = W1init
        self.W2init = W2init
        self.w2train = w2train
        
        # Compute regularization level
        w1var = np.mean(np.abs(mod0.W1)**2)
        if map_est:
            reg1 = mod0.yvar/w1var/mod0.nin/mod0.d
        else:
            reg1 = 0.0
    
            
        # Create ther keras model
        K.clear_session()
        self.mod = Sequential()
        self.mod.add(Dense(input_shape=(mod0.nin,),units=mod0.d,\
                           activation=self.act_type,name='Input',\
                           use_bias=False,\
                           kernel_regularizer=regularizers.l2(reg1)))
        self.mod.add(Dense(units=mod0.nout,\
                           activation='linear',name='Output',
                           use_bias=False))  
        
        
        # Set the keras model weights to the weights in a base model, mod0        
        self.mod.set_weights((self.W1init,self.W2init))
        
        # Set output layer to fixed value
        if not self.w2train:
            layer = self.mod.get_layer('Output')
            layer.trainable = False                    

        
    def fit(self,Xtr,Ytr,Xts,Yts,nepochs=20,verbose=False):
        """
        Fit the model parameters using keras.
        """
        # Compile the model
        opt = Adam(self.lr)
        self.mod.compile(optimizer=opt,loss='mse',metrics=['mse'])
        
        # Fit on the training data
        self.hist = self.mod.fit(Xtr,Ytr, epochs=nepochs,verbose=verbose,\
                                 batch_size=self.batch_size,\
                                 validation_data=(Xts,Yts))        
        
def twolayer_keras_test(nsamp=10000,nin=100,nepochs=40):        
    """
    Unit test for the two layer model
    
    We create a two layer model and see if the W1 parameters
    can be fit from keras
    """
    
    # Generate a random model
    mod = TwoLayerMod(nin=nin)
    mod.gen_rand_param()
    
    # Generate random data
    X, Y = mod.gen_rand_dat(nsamp) 
    
    # Splut into training and test
    Xtr,Xts,Ytr,Yts = train_test_split(X,Y,test_size=0.3)
    
    # Fit keras model
    mod.create_keras_mod()  
    mod.keras_fit(Xtr,Ytr,Xts,Yts,nepochs=nepochs,oracle_out=True)
    layer = mod.mod.get_layer('Input')
    W1hat = layer.get_weights()[0]
    
    # Measure parameter error up to a transformation
    T = np.linalg.lstsq(W1hat,mod.W1,rcond=None)[0]
    D = mod.W1 - W1hat.dot(T) 
    w1err = np.mean(D**2)/np.mean(mod.W1**2)
    print(w1err)
    


