from layers import GeneralLinear, GeneralRecurrent
from utils import AmplitudeEncoder
from dl_utils import EntanglementRotation
import numpy as np
np.random.seed(42)
seed_simulator = 42

np.random.seed(42)
seed_simulator = 42

x = np.random.randn(2,8,4) #Generate random data
h_0 = np.ones(3) #Generate initial hidden vector

recurrent_layer = GeneralRecurrent(n_qubits=3,n_hidden=3,n_weights_ent=0,n_weights_a=7,bias=True,U_enc=AmplitudeEncoder(),U_a=AmplitudeEncoder(inverse=True),U_ent=EntanglementRotation(zero_condition=True),seed_simulator=seed_simulator)
output_layer = GeneralLinear(n_qubits=2,n_outputs=4,n_weights_ent=0,n_weights_a=4,bias=True,U_enc=AmplitudeEncoder(),U_a=AmplitudeEncoder(inverse=True),U_ent=EntanglementRotation(zero_condition=True),seed_simulator=seed_simulator+1)

output = recurrent_layer(x,h_0)[:,-1,:] #Chose the final hidden vector from recurrent layer
output = output_layer(output) 			#Feed this into dense output layer

print('Output:')
print(output)