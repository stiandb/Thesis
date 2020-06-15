import numpy as np 
import pandas as pd 
from matplotlib.pylab import *
from QDNN import *
from layers import *
from loss import *
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from hamiltonian import *
np.random.seed(7)
"""
y_train = np.zeros(2)
H, Eref = hamiltonian(1,2,1,1)
eigvals, eigvecs = np.linalg.eigh(H)
print(eigvals)
loss_fn = rayleigh_quotient(H)
x = np.ones(8).reshape(1,8)
l1 = AnsatzLinear(x.shape[1],8,3,ansatz=y_rotation_ansatz)
l2 = AnsatzLinear(8,2,3,ansatz=y_rotation_ansatz)
layers = [l1,l2]
model = QDNN(layers,loss_fn)

np.save('model_params.npy',model.fit(X=x,y=y_train,method='Powell'))"""


y_train = np.zeros(2)
H, Eref = hamiltonian(2,3,1,1)
eigvals, eigvecs = np.linalg.eigh(H)
print(eigvals)
loss_fn = rayleigh_quotient(H)
x = np.ones(8).reshape(1,8)
l1 = RotationLinear(x.shape[1],8,3,rotation=y_rotation,n_parallel=8)
l2 = RotationLinear(8,3,3,rotation=y_rotation,n_parallel=3)
layers = [l1,l2]
model = QDNN(layers,loss_fn)

np.save('model_params.npy',model.fit(X=x,y=y_train,method='Powell'))

"""
y_train = np.zeros(2)
H, Eref = hamiltonian(2,3,1,1)
eigvals, eigvecs = np.linalg.eigh(H)
print(eigvals)
dt = 1
loss_fn = eigenvector_ode(H,dt)

x = np.ones(10*8).reshape(1,10,8)
l1 = RotationRNN(3,3,2,rotation=y_rotation)

layers = [l1]
model = QDNN(layers,loss_fn)

np.save('model_params.npy',model.fit(X=x,y=y_train,method='Powell'))
"""