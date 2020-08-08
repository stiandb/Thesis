import numpy as np 
import pandas as pd 
from matplotlib.pylab import *
from QDNN import *
from layers import *
from loss import *
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from settings import ibmq_london_noise_model as noise_model, ibmq_london_basis_gates as basis_gates, ibmq_london_coupling_map as coupling_map

np.random.seed(22)
seed_simulator = 13

shots = 10000

def f(x):
	return(3*sin(x) - 0.5*x)

n=2

x = np.array([np.pi/2,0.5])
X_train = np.zeros((n,1))
X_train[:,0] = x
y_train = np.zeros(n)
y_train[0] = 1
y_train[1] = 0.3

l1 = X_train.shape[1]
l2 = 6
l3 = 6


layer1 = GeneralLinear(n_qubits=1,n_outputs=l2,n_weights_ent=2,n_weights_a=1,U_enc=YRotationAnsatz(identity_circuit),U_a=YRotationAnsatz(identity_circuit),U_ent=YRotation(bias=True),seed_simulator=seed_simulator,shots=shots,backend=qk.Aer.get_backend('qasm_simulator'))
layer2 = GeneralLinear(n_qubits=3,n_outputs=1,n_weights_ent=4,n_weights_a=6,U_enc=YRotationAnsatz(linear_entangler),U_a=YRotationAnsatz(linear_entangler,inverse=True),U_ent=YRotation(bias=True),seed_simulator=seed_simulator,shots=shots,backend=qk.Aer.get_backend('qasm_simulator'))

layers = [layer1,layer2]
loss_fn = MSE()

model = QDNN(layers,loss_fn)

model.fit_numerical(X_train,y_train,epochs=100,dw=np.pi/2,lr=1)
w = model.w_opt

