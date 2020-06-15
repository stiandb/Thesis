
from layers import Linear, RotationRNN
from dl_utils import YRotation
import numpy as np

np.random.seed(42)
seed_simulator = 42

x = np.random.randn(2,8,4) #Generate random data
h_0 = np.ones(2) #Generate initial hidden vector

y_rotation = YRotation(bias=True)
recurrent_layer = RotationRNN(n_hidden=2,n_wx=3,n_wh=2,n_parallel_x=2,n_parallel_h=2,rotation=y_rotation,seed_simulator=seed_simulator)
output_layer = Linear(n_inputs=2,n_outputs=3,bias=True,seed_simulator=seed_simulator+1)

output = recurrent_layer(x,h_0)[:,-1,:] #Chose the final hidden vector from recurrent layer
output = output_layer(output) 			#Feed this into dense output layer

print('Output:')
print(output)
