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

y_rotation = YRotation(bias=True)
y = np.zeros(2)
n_fermi=2
n_spin_orbitals=4
delta = 1
n = 20
x = np.ones(4).reshape(1,4)
out1=2
E_fci = np.zeros(n)
E_noisy = np.zeros(n)
E_ideal = np.zeros(n)




for i,g in enumerate(np.linspace(1,5,n)):
	H, Eref = hamiltonian(int(n_fermi/2),int(n_spin_orbitals/2),delta,g)
	eigvals, eigvecs = np.linalg.eigh(H)
	E_fci[i] = eigvals[0]
	loss_fn = rayleigh_quotient(H)
	

	l1 = RotationLinear(x.shape[1],out1,3,rotation=y_rotation,n_parallel=1,shots=1000,seed_simulator=42,backend=qk.Aer.get_backend('qasm_simulator'),noise_model=noise_model,basis_gates=basis_gates,coupling_map=coupling_map,transpile=True,seed_transpiler=42,optimization_level=3,error_mitigator=ErrorMitigation())
	l2 = IntermediateAnsatzRotationLinear(n_qubits=out1,n_outputs=eigvecs[0].shape[0],n_weights_a=0,n_weights_r=out1+1,ansatz_i=y_rotation_ansatz,ansatz_a=identity_ansatz,rotation=y_rotation,n_parallel = 1,shots=1000,seed_simulator=42,backend=qk.Aer.get_backend('qasm_simulator'),noise_model=noise_model,basis_gates=basis_gates,classical_bits=None,coupling_map=coupling_map,transpile=True,seed_transpiler=42,optimization_level=3,error_mitigator=ErrorMitigation())
	layers = [l1,l2]
	model = QDNN(layers,loss_fn)

	model.fit(x,y,seed=42)
	E_noisy[i] = np.min(model.loss_train)


	l1 = RotationLinear(x.shape[1],out1,3,rotation=y_rotation,n_parallel=1,shots=1000,seed_simulator=42,backend=qk.Aer.get_backend('qasm_simulator'))
	l2 = IntermediateAnsatzRotationLinear(n_qubits=out1,n_outputs=eigvecs[0].shape[0],n_weights_a=0,n_weights_r=out1+1,ansatz_i=y_rotation_ansatz,ansatz_a=identity_ansatz,rotation=y_rotation)
	layers = [l1,l2]
	model = QDNN(layers,loss_fn)

	model.fit(x,y,seed=42)
	E_ideal[i] = np.min(model.loss_train)
	print(i+1)

np.save('pairing{}_{}_E_fci.npy'.format(n_fermi,n_spin_orbitals),E_fci)
np.save('pairing{}_{}_E_noisy.npy'.format(n_fermi,n_spin_orbitals),E_noisy)
np.save('pairing{}_{}_E_ideal.npy'.format(n_fermi,n_spin_orbitals),E_ideal)




