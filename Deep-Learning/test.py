
from layers import AnsatzRotationLinear, AnsatzRotationRNN
from dl_utils import YRotation
from utils import y_rotation_ansatz
import numpy as np

np.random.seed(45)
seed_simulator = 46

x = np.random.randn(10,8,4) #Generate random data
h_0 = np.random.randn(2) #Generate initial hidden vector

y_rotation = YRotation(bias=True)

recurrent_layer = AnsatzRotationRNN(n_hidden=2,n_wxa=2,n_wxr=3,n_wha=1,n_whr=2,rotation=y_rotation,ansatz=y_rotation_ansatz,n_parallel_x=1,n_parallel_h=1,seed_simulator=seed_simulator)
output_layer = AnsatzRotationLinear(n_inputs=2,n_outputs=3,n_weights_a=1,n_weights_r=2,ansatz=y_rotation_ansatz,rotation=y_rotation,n_parallel = 1,seed_simulator=seed_simulator+1)

output = recurrent_layer(x,h_0)[:,-1,:] #Chose the final hidden vector from recurrent layer
output = output_layer(output) 			#Feed this into dense output layer

print('Output:')
print(output)
