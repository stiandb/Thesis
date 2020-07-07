import numpy as np 
import pandas as pd 
from matplotlib.pylab import *
from QDNN import *
from AutoEncoder import *
import matplotlib.pylab as plt

np.random.seed(7)
X = np.random.randn(8)
autoencoder = AutoEncoder(U_1=y_rotation_ansatz,U_2=AmplitudeEncoder(inverse=True),n_qubits=3,n_weights=3,seed_simulator=42)
autoencoder.fit(X,print_loss=True)
np.save('autoencoder_1_samp_loss.npy',autoencoder.loss_train)


loss = np.load('autoencoder_1_samp_loss.npy')
print('Inner product:',np.sqrt(-np.min(loss)))
plt.plot(np.sqrt(-loss))
plt.xlabel('Function evaluations')
plt.ylabel('|Inner Product|')
plt.title(r'Comparison of $R_y(\theta)$ ansatz and amplitude encoding of eight random variables')
plt.show()



