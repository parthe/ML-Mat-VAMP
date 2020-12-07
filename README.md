# ML-Mat-VAMP (Multi-layer - Matrix - VAMP)

Code to reproduce results from **Matrix Inference and Estimation in Multi-Layer Models**, *NeurIPS 2020.*

```
@article{pandit2020matrix,
  title={Matrix Inference and Estimation in Multi-Layer Models},
  author={Pandit, Parthe and Sahraee Ardakan, Mojtaba and Rangan, Sundeep and Schniter, Philip and Fletcher, Alyson K},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  year={2020}
}
```

# Dependencies
tensorflow >= 2.3.1  
scikit-learn  
matplotlib

# Plot results using stored data from our simulations

The following command will generate 2 subplots (Fig. 2 from the paper) and save them in mse_vs_ntr.png

```
python plot.py
```

# Run your own experiments to plot results
To run multiple experiments:
```
python diy_expts/2layer_ALGO.py --act relu --snr 10.0 --fn_suffix k
```
for k = 0,1,..K-1.  
The above command creates the file adam_snr10_k.pkl  
By default k=0.  
ALGO can be 'adam' or 'ml-mat-vamp'

# Run experiments for Adam. This uses Keras
The following creates files adam_snr10_0.pkl and adam_snr15_0.pkl
```
python diy_expts/2layer_adam.py --act relu --snr 10.0
python diy_expts/2layer_adam.py --act relu --snr 15.0
```

# Run experiments for ML-Mat-VAMP
```
python diy_expts/2layer_ml-mat-vamp.py --act relu --snr 10.0
python diy_expts/2layer_ml-mat-vamp.py --act relu --snr 15.0
```

# Run experiments for State Evolution of ML-Mat-VAMP
```
python diy_expts/2layer_ml-mat-vamp.py --se_test --act relu --snr 10.0
python diy_expts/2layer_ml-mat-vamp.py --se_test --act relu --snr 15.0
```
