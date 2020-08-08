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
import matplotlib.pylab as plt
np.random.seed(7)

y_rotation = YRotation(bias=True)
y = np.zeros(2)
n_fermi=4#2
n_spin_orbitals=8#4
delta = 1
n = 5
x = np.ones(4).reshape(1,4)
out1=2
E_fci = np.zeros(n)
E_noisy = np.zeros(n)
E_ideal = np.zeros(n)
g_array = np.linspace(1,5,n)



for i,g in enumerate(g_array):
	H, Eref = PairingFCIMatrix()(int(n_fermi/2),int(n_spin_orbitals/2),delta,g)
	eigvals, eigvecs = np.linalg.eigh(H)
	E_fci[i] = eigvals[0]
	loss_fn = rayleigh_quotient(H)
	
	l1 = GeneralLinear(n_qubits=2,n_outputs=out1,n_weights_a = 0,n_weights_ent=3,U_enc=AmplitudeEncoder(),U_a = identity_ansatz,U_ent=YRotation(bias=True),shots=1000,seed_simulator=42,backend=qk.Aer.get_backend('qasm_simulator'),noise_model=noise_model,basis_gates=basis_gates,coupling_map=coupling_map,transpile=True,seed_transpiler=42,optimization_level=3,error_mitigator=ErrorMitigation())
	l2 = GeneralLinear(n_qubits=out1,n_outputs=eigvecs[0].shape[0],n_weights_a = 0, n_weights_ent = out1 + 1, U_enc = YRotationAnsatz(linear_entangler),U_a =identity_ansatz,U_ent =YRotation(bias=True),shots=1000,seed_simulator=42,backend=qk.Aer.get_backend('qasm_simulator'),noise_model=noise_model,basis_gates=basis_gates,coupling_map=coupling_map,transpile=True,seed_transpiler=42,optimization_level=3,error_mitigator=ErrorMitigation())
	
	layers = [l1,l2]
	model = QDNN(layers,loss_fn)

	model.fit(x,y,seed=42)
	E_noisy[i] = np.min(model.loss_train)

	l1 = GeneralLinear(n_qubits=2,n_outputs=out1,n_weights_a = 0,n_weights_ent=3,U_enc=AmplitudeEncoder(),U_a = identity_ansatz,U_ent=YRotation(bias=True),shots=1000,seed_simulator=42,backend=qk.Aer.get_backend('qasm_simulator'))
	l2 = GeneralLinear(n_qubits=out1,n_outputs=eigvecs[0].shape[0],n_weights_a = 0, n_weights_ent = out1 + 1, U_enc = YRotationAnsatz(linear_entangler),U_a =identity_ansatz,U_ent =YRotation(bias=True),shots=1000,seed_simulator=42,backend=qk.Aer.get_backend('qasm_simulator'))
	
	layers = [l1,l2]
	model = QDNN(layers,loss_fn)

	model.fit(x,y,seed=42)
	E_ideal[i] = np.min(model.loss_train)
	print(i+1)


np.save('pairing{}_{}_E_fci.npy'.format(n_fermi,n_spin_orbitals),E_fci)
np.save('pairing{}_{}_E_noisy.npy'.format(n_fermi,n_spin_orbitals),E_noisy)
np.save('pairing{}_{}_E_ideal.npy'.format(n_fermi,n_spin_orbitals),E_ideal)


E_fci = np.load('pairing{}_{}_E_fci.npy'.format(n_fermi,n_spin_orbitals))
E_noisy = np.load('pairing{}_{}_E_noisy.npy'.format(n_fermi,n_spin_orbitals))
E_ideal = np.load('pairing{}_{}_E_ideal.npy'.format(n_fermi,n_spin_orbitals))

fig1 = plt.figure(1)

frame1=fig1.add_axes((.1,.3,.8,.6))
plt.plot(g_array,E_fci,label='FCI')
plt.plot(g_array,E_ideal,'r*',label='Ideal',alpha=0.8)
plt.plot(g_array,E_noisy,'g+',label='Noisy')
plt.title(r'Rayleigh quotient minimization. {} particles, {} spin orbitals, $\delta = {}$.'.format(n_fermi,n_spin_orbitals,delta))
plt.ylabel('Energy [u.l]')
plt.legend()


#Residual plot
difference_ideal = np.abs(E_fci- E_ideal)
difference_noisy = np.abs(E_fci - E_noisy)
frame2=fig1.add_axes((.1,.1,.8,.2))       
plt.plot(g_array,difference_ideal,'r*',alpha=0.8)
plt.plot(g_array,difference_noisy,'g+')
plt.ylabel(r'|$E_{FCI} - E_{NN}$|')
plt.xlabel('g')
plt.grid()
plt.show()














