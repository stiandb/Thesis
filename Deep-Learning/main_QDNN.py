import numpy as np 
import pandas as pd 
from matplotlib.pylab import *
from QDNN import *
from layers import *
from loss import *
from settings import ibmq_london_noise_model as noise_model, ibmq_london_basis_gates as basis_gates, ibmq_london_coupling_map as coupling_map
np.random.seed(42)


def f(x):
	return(np.exp(2 - x + 3*x**2))

n = 10

x = np.linspace(0,1,n)


y = f(x) #Create target vector

x -= np.min(x)
x /= np.max(x)

X = np.zeros((y.shape[0],1))
X[:,0] = x

y -= np.min(y)
y /= np.max(y) #Normalize target vector

plt.plot(x,y)
plt.show()

in1 = 1
out1 = 3

loss_fn = MSE()
y_rotation=YRotation(bias=True)


l1 = RotationLinear(in1,3*out1,in1+1,rotation=y_rotation,n_parallel=1,shots=1000,seed_simulator=42,backend=qk.Aer.get_backend('qasm_simulator'))
l2_1 = IntermediateAnsatzRotationLinear(n_qubits=out1,n_outputs=1,n_weights_a=out1,n_weights_r=out1+1,ansatz_i=y_rotation_ansatz,ansatz_a=identity_ansatz,rotation=y_rotation,n_parallel = 1,shots=1000,seed_simulator=42,backend=qk.Aer.get_backend('qasm_simulator'))
l2_2 = IntermediateAnsatzRotationLinear(n_qubits=out1,n_outputs=1,n_weights_a=out1,n_weights_r=out1+1,ansatz_i=y_rotation_ansatz,ansatz_a=identity_ansatz,rotation=y_rotation,n_parallel = 1,shots=1000,seed_simulator=42,backend=qk.Aer.get_backend('qasm_simulator'))
l2_3 = IntermediateAnsatzRotationLinear(n_qubits=out1,n_outputs=1,n_weights_a=out1,n_weights_r=out1+1,ansatz_i=y_rotation_ansatz,ansatz_a=identity_ansatz,rotation=y_rotation,n_parallel = 1,shots=1000,seed_simulator=42,backend=qk.Aer.get_backend('qasm_simulator'))
l3 = IntermediateAnsatzRotationLinear(n_qubits=3,n_outputs=1,n_weights_a=3,n_weights_r=4,ansatz_i=y_rotation_ansatz,ansatz_a=identity_ansatz,rotation=y_rotation,n_parallel = 1,shots=1000,seed_simulator=42,backend=qk.Aer.get_backend('qasm_simulator'))

layers = [l1,[l2_1,l2_2,l2_3],l3]
model = QDNN(layers,loss_fn)

model.fit(X,y,seed=42)

np.save('optimal_weights.npy',model.w_opt)
np.save('loss_train_1d.npy',model.loss_train)






