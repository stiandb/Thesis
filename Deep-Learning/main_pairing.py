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
from settings import ibmq_london_noise_model as noise_model, ibmq_london_basis_gates as basis_gates, ibmq_london_coupling_map as coupling_map
import matplotlib.pylab as plt
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

y_rotation = YRotation(bias=True)
y = np.zeros(2)
n_fermi=4
n_spin_orbitals=8
delta = 1
g = 5
H, Eref = hamiltonian(int(n_fermi/2),int(n_spin_orbitals/2),delta,g)
eigvals, eigvecs = np.linalg.eigh(H)
print(eigvals)
loss_fn = rayleigh_quotient(H)
x = np.ones(4).reshape(1,4)
out1=2
l1 = RotationLinear(x.shape[1],out1,3,rotation=y_rotation,n_parallel=1,shots=1000,seed_simulator=42,backend=qk.Aer.get_backend('qasm_simulator'),noise_model=noise_model,basis_gates=basis_gates,coupling_map=coupling_map,transpile=True,seed_transpiler=42,optimization_level=3,error_mitigator=ErrorMitigation())
l2 = IntermediateAnsatzRotationLinear(n_qubits=out1,n_outputs=eigvecs[0].shape[0],n_weights_a=0,n_weights_r=out1+1,ansatz_i=y_rotation_ansatz,ansatz_a=identity_ansatz,rotation=y_rotation,n_parallel = 1,shots=1000,seed_simulator=42,backend=qk.Aer.get_backend('qasm_simulator'),noise_model=noise_model,basis_gates=basis_gates,classical_bits=None,coupling_map=coupling_map,transpile=True,seed_transpiler=42,optimization_level=3,error_mitigator=ErrorMitigation())
layers = [l1,l2]
model = QDNN(layers,loss_fn)

model.fit(x,y,seed=42)

loss_noise = model.loss_train

l1 = RotationLinear(x.shape[1],out1,3,rotation=y_rotation,n_parallel=1,shots=1000,seed_simulator=42,backend=qk.Aer.get_backend('qasm_simulator'))
l2 = IntermediateAnsatzRotationLinear(n_qubits=out1,n_outputs=eigvecs[0].shape[0],n_weights_a=0,n_weights_r=out1+1,ansatz_i=y_rotation_ansatz,ansatz_a=identity_ansatz,rotation=y_rotation)
layers = [l1,l2]
model = QDNN(layers,loss_fn)

model.fit(x,y,seed=42)

loss_ideal = model.loss_train

E_ci_ideal = np.ones(len(loss_ideal))*eigvals[0]
E_ci_noise = np.ones(len(loss_noise))*eigvals[0]

plt.plot(list(range(len(loss_ideal))),loss_ideal,label='Ideal Simulation: Loss')
plt.plot(list(range(len(loss_ideal))),E_ci_ideal,label='FCI')
plt.legend()
plt.xlabel('Function Evaluations')
plt.ylabel('Rayleigh Quotient')
plt.title(r'Ideal Quantum Neural Network minimization of Rayleigh Quotient for {} particles, {} spin orbitals, $\delta = ${}, $g = ${}'.format(n_fermi,n_spin_orbitals,delta,g))
plt.show()

plt.plot(list(range(len(loss_noise))),loss_noise,label='Noisy Simulation: Loss')
plt.plot(list(range(len(loss_noise))),E_ci_noise,label='FCI')
plt.legend()
plt.xlabel('Function Evaluations')
plt.ylabel('Rayleigh Quotient')
plt.title(r'Noisy Quantum Neural Network minimization of Rayleigh Quotient for {} particles, {} spin orbitals, $\delta = ${}, $g = ${}'.format(n_fermi,n_spin_orbitals,delta,g))
plt.show()


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
