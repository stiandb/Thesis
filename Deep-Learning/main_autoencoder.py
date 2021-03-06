import numpy as np 
import pandas as pd 
from matplotlib.pylab import *
from QDNN import *
from AutoEncoder import *
import matplotlib.pylab as plt


shots=1000
n = 4
np.random.seed(47)
inner1 = np.zeros(n)
inner2 = np.zeros(n)
inner3 = np.zeros(n)
i = 0
for n_qubits,n_predictors in zip([1,2,3,4],[2,4,8,16]):
	X = np.random.randn(n_predictors)
	autoencoder = AutoEncoder(U_1=EulerRotationAnsatz(linear_entangler),U_2=AmplitudeEncoder(inverse=True),n_qubits=n_qubits,n_weights=3*n_qubits,seed_simulator=42,shots=shots)
	autoencoder.fit(X,print_loss=True)
	inner1[i] = np.sqrt(-min(autoencoder.loss_train))
	autoencoder = AutoEncoder(U_1=EulerRotationAnsatz(linear_entangler),U_2=AmplitudeEncoder(inverse=True),n_qubits=n_qubits,n_weights=3*2*n_qubits,seed_simulator=42,shots=shots)
	autoencoder.fit(X,print_loss=True)
	inner2[i] = np.sqrt(-min(autoencoder.loss_train))
	autoencoder = AutoEncoder(U_1=EulerRotationAnsatz(linear_entangler),U_2=AmplitudeEncoder(inverse=True),n_qubits=n_qubits,n_weights=3*3*n_qubits,seed_simulator=42,shots=shots)
	autoencoder.fit(X,print_loss=True)
	inner3[i] = np.sqrt(-min(autoencoder.loss_train))
	i+= 1

np.save('inner1e.npy',inner1)
np.save('inner2e.npy',inner2)
np.save('inner3e.npy',inner3)
inner1 = np.load('inner1e.npy')
inner2 = np.load('inner2e.npy')
inner3 = np.load('inner3e.npy')

plt.plot([2,4,8,16],inner1,'--',label=r'Euler rotation ansatz. d=1')
plt.plot([2,4,8,16],inner2,'--',label=r'Euler rotation ansatz. d=2')
plt.plot([2,4,8,16],inner3,'--',label=r'Euler rotation ansatz. d=3')
plt.title(r'Learning Amplitude Encoding with Ansatz')
plt.xlabel('Number of predictors')
plt.ylabel('|Inner product|')
plt.legend()
plt.show()


