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
seed_simulator = 47

iris = load_iris()
X = iris['data']
y = iris['target']
X, X_test, y, y_test = train_test_split(X,y,test_size=0.92,stratify=y)
X_train,X_val, y_train, y_val = train_test_split(X,y,test_size=0.5,stratify=y)
print('train dataset shape', X_train.shape)
print('val dataset shape', X_val.shape)
print('test dataset shape', X_test.shape)

l1 = X_train.shape[1]
l2 = 4

y_rotation = YRotation(bias=True)
layer1 = GeneralLinear(n_qubits=2,n_outputs=l2,n_weights_ent=2,n_weights_a=0,U_enc=AmplitudeEncoder(),U_a=identity_ansatz,U_ent=YRotation(bias=False),seed_simulator=seed_simulator,backend=qk.Aer.get_backend('qasm_simulator'),noise_model=noise_model,basis_gates=basis_gates,coupling_map=coupling_map,transpile=True,seed_transpiler=42,optimization_level=3,error_mitigator=ErrorMitigation(),shots=100)
layer2 = GeneralLinear(n_qubits=l2,n_outputs=3,n_weights_ent=l2,n_weights_a=0,U_enc=YRotationAnsatz(linear_entangler),U_a=identity_ansatz,U_ent=YRotation(bias=False),seed_simulator=seed_simulator,backend=qk.Aer.get_backend('qasm_simulator'),noise_model=noise_model,basis_gates=basis_gates,coupling_map=coupling_map,transpile=True,seed_transpiler=42,optimization_level=3,error_mitigator=ErrorMitigation(),shots=100)

layers = [layer1,layer2]
loss_fn = cross_entropy()

model = QDNN(layers,loss_fn,classification=True)

model.fit(X=X_train,y=y_train,print_loss=True)
y_pred = model.forward(X_test)
y_pred = np.argmax(y_pred,axis=1)

print('Confusion Matrix:')
print(confusion_matrix(y_test,y_pred))

