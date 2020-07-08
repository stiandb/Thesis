import numpy as np 
import pandas as pd 
from matplotlib.pylab import *
from QDNN import *
from SubsetAutoencoder import *
from layers import *
from loss import *
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from settings import ibmq_london_noise_model as noise_model, ibmq_london_basis_gates as basis_gates, ibmq_london_coupling_map as coupling_map
import matplotlib.pylab as plt

np.random.seed(7)
X = np.random.randn(2,4)

layer = GeneralLinear(n_qubits=1,n_outputs=2*3,n_weights_a=3,n_weights_ent=1,U_enc=AmplitudeEncoder(),U_a=EulerRotationAnsatz(linear_entangler),U_ent=EntanglementRotation(bias=True),shots=1000,seed_simulator=42)
layers = [layer]
autoencoder = SubsetAutoencoder(k=2,layers=layers,U_subenc=EulerRotationAnsatz(linear_entangler),k_encoders=False)

autoencoder.fit(X,print_loss=True,method='COBYLA')
loss_sample = autoencoder.loss_list

plt.plot(np.sqrt(loss_sample[0:-1:5,0]),label='Sample 1')
plt.plot(np.sqrt(loss_sample[0:-1:5,1]),label='Sample 2')
plt.title('Subset Autoencoder with k=2 on two samples with four predictors')
plt.legend()
plt.xticks(range(0,loss_sample[0:-1:5,0].shape[0],5))
plt.xlabel('Function evaluations')
plt.ylabel('|Inner Product|')
plt.show()