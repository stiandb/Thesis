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
from hamiltonian import *
from settings import ibmq_london_noise_model as noise_model, ibmq_london_basis_gates as basis_gates, ibmq_london_coupling_map as coupling_map
import matplotlib.pylab as plt

"""np.random.seed(7)
X = np.random.randn(1,16)
layer = GeneralLinear(n_qubits=2,n_outputs=3*5,n_weights_a=3*2*2,n_weights_ent=1,U_enc=AmplitudeEncoder(),U_a=EulerRotationAnsatz(linear_entangler),U_ent=EntanglementRotation(bias=True),shots=1000,seed_simulator=42)
layers = [layer]
autoencoder = SubsetAutoencoder(k=4,layers=layers,U_subenc=EulerRotationAnsatz(linear_entangler))
autoencoder.fit(X,print_loss=True)
np.save('autoencoder_loss.npy',autoencoder.loss_train)"""



X = np.random.randn(2,16)
layer = GeneralLinear(n_qubits=2,n_outputs=3*6,n_weights_a=3*2*2,n_weights_ent=1,U_enc=AmplitudeEncoder(),U_a=EulerRotationAnsatz(linear_entangler),U_ent=EntanglementRotation(bias=True),shots=1000,seed_simulator=42)
layers = [layer]
autoencoder = SubsetAutoencoder(k=4,layers=layers,U_subenc=EulerRotationAnsatz(linear_entangler),k_encoders=True)
autoencoder.fit(X,print_loss=True)
np.save('autoencoder_loss_multi.npy',autoencoder.loss_train)