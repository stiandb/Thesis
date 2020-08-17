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


n=10

x = np.linspace(0,2*np.pi,n)
X_train = np.zeros((n,1))
X_train[:,0] = x/(4*np.pi)
y_train = f(x)
min_y = np.min(y_train)
y_train -= min_y
max_y = np.max(y_train)
y_train /= max_y
l1 = X_train.shape[1]
l2 = 6
l3 = 6


layer1 = GeneralLinear(n_qubits=1,n_outputs=l2,n_weights_ent=2,n_weights_a=1,U_enc=YRotationAnsatz(identity_circuit),U_a=YRotationAnsatz(identity_circuit),U_ent=YRotation(bias=True),seed_simulator=seed_simulator,shots=shots,backend=qk.Aer.get_backend('qasm_simulator'))
layer2 = GeneralLinear(n_qubits=3,n_outputs=l3,n_weights_ent=4,n_weights_a=6,U_enc=YRotationAnsatz(linear_entangler),U_a=YRotationAnsatz(linear_entangler,inverse=True),U_ent=YRotation(bias=True),seed_simulator=seed_simulator,shots=shots,backend=qk.Aer.get_backend('qasm_simulator'))
layer3 = GeneralLinear(n_qubits=3,n_outputs=1,n_weights_ent=4,n_weights_a=6,U_enc=YRotationAnsatz(linear_entangler),U_a=YRotationAnsatz(linear_entangler,inverse=True),U_ent=YRotation(bias=True),seed_simulator=seed_simulator,backend=qk.Aer.get_backend('qasm_simulator'),shots=shots)
layers = [layer1,layer2,layer3]
loss_fn = MSE()

model = QDNN(layers,loss_fn)

model.fit(X=X_train,y=y_train,print_loss=True,)
w = model.w_opt

n=20

x = np.linspace(0,2*np.pi,n)
X_train = np.zeros((n,1))
X_train[:,0] = x/(4*np.pi)
y_train = f(x)
min_y = np.min(y_train)
y_train -= min_y
max_y = np.max(y_train)
y_train /= max_y

model.set_weights(w)
y_pred = model.forward(X_train)


plt.plot(x,y_pred,'+r',label='Predicted')
plt.plot(x,y_train,'g',label='Actual')
plt.title(r'Learning $f(x) = 3 \sin{x} - \frac{1}{2}x$ with Neural Network')
plt.xlabel('x')
plt.ylabel('f(x) (Normalized between 0 and 1)')
plt.legend()
plt.show()


n=10

x = np.linspace(0,2*np.pi,n)
X_train = np.zeros((n,1))
X_train[:,0] = x/(4*np.pi)
y_train = f(x)
min_y = np.min(y_train)
y_train -= min_y
max_y = np.max(y_train)
y_train /= max_y
l1 = X_train.shape[1]
l2 = 6
l3 = 6


layer1 = GeneralLinear(n_qubits=1,n_outputs=l2,n_weights_ent=1,n_weights_a=1,U_enc=YRotationAnsatz(identity_circuit),U_a=YRotationAnsatz(identity_circuit),U_ent=EntanglementRotation(bias=True),seed_simulator=seed_simulator,shots=shots,backend=qk.Aer.get_backend('qasm_simulator'))
layer2 = GeneralLinear(n_qubits=3,n_outputs=l3,n_weights_ent=1,n_weights_a=6,U_enc=YRotationAnsatz(linear_entangler),U_a=YRotationAnsatz(linear_entangler,inverse=True),U_ent=EntanglementRotation(bias=True),seed_simulator=seed_simulator,shots=shots,backend=qk.Aer.get_backend('qasm_simulator'))
layer3 = GeneralLinear(n_qubits=3,n_outputs=1,n_weights_ent=1,n_weights_a=6,U_enc=YRotationAnsatz(linear_entangler),U_a=YRotationAnsatz(linear_entangler,inverse=True),U_ent=EntanglementRotation(bias=True),seed_simulator=seed_simulator,backend=qk.Aer.get_backend('qasm_simulator'),shots=shots)
layers = [layer1,layer2,layer3]
loss_fn = MSE()

model = QDNN(layers,loss_fn)

model.fit(X=X_train,y=y_train,print_loss=True,)
w = model.w_opt

n=20

x = np.linspace(0,2*np.pi,n)
X_train = np.zeros((n,1))
X_train[:,0] = x/(4*np.pi)
y_train = f(x)
min_y = np.min(y_train)
y_train -= min_y
max_y = np.max(y_train)
y_train /= max_y

model.set_weights(w)
y_pred = model.forward(X_train)


plt.plot(x,y_pred,'+r',label='Predicted')
plt.plot(x,y_train,'g',label='Actual')
plt.title(r'Learning $f(x) = 3 \sin{x} - \frac{1}{2}x$ with Neural Network. Inner product entangler')
plt.xlabel('x')
plt.ylabel('f(x) (Normalized between 0 and 1)')
plt.legend()
plt.show()




