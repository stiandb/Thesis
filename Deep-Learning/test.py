import sys
sys.path.append('../')
from layers import Linear
import numpy as np
np.random.seed(42)
seed_simulator = 42

x = np.random.randn(2,8)

hidden_layer = Linear(n_inputs=8,n_outputs=8,bias=True,seed_simulator=seed_simulator)
output_layer = Linear(n_inputs=8,n_outputs=3,bias=True,seed_simulator=seed_simulator+1)

output = hidden_layer(x)
output = output_layer(output)

print('Output:',output)