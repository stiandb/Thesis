import numpy as np 
import pandas as pd 
from matplotlib.pylab import *
from QDNN import *
from layers import *
from loss import *
from settings import ibmq_london_noise_model as noise_model, ibmq_london_basis_gates as basis_gates, ibmq_london_coupling_map as coupling_map
np.random.seed(42)


def f(x1):
	return(3-3*x1**2 + x1)

n = 5

x = np.linspace(-1,1,n)




y = f(x) #Create target vector


x -= np.min(x)
x /= np.max(x)*np.pi*2



X = np.zeros((y.shape[0],1))
X[:,0] = x


y -= np.min(y)
y /= np.max(y) #Normalize target vector



loss_fn = MSE()
y_rotation=YRotation(bias=True)


l1 = GeneralLinear(n_qubits=1,n_outputs=4,n_weights_a=3,n_weights_ent=1,U_enc=AmplitudeEncoder(),U_a=EulerRotationAnsatz(linear_entangler),U_ent=EntanglementRotation(bias=True),shots=200,seed_simulator=42)
l2 = GeneralLinear(n_qubits=2,n_outputs=1,n_weights_a=3*2*3,n_weights_ent=1,U_enc=AmplitudeEncoder(),U_a=EulerRotationAnsatz(linear_entangler),U_ent=EntanglementRotation(bias=True),shots=200,seed_simulator=42)

layers = [l1,l2]
model = QDNN(layers,loss_fn)

model.fit(X,y,method='Powell',seed=42,print_loss=True)

np.save('optimal_weights.npy',model.w_opt)
np.save('loss_train_1d.npy',model.loss_train)

w = np.load('optimal_weights.npy')
model.set_weights(w)

y_pred = model.forward(X)







