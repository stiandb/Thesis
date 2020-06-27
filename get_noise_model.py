from qiskit import QuantumRegister, ClassicalRegister
from qiskit import QuantumCircuit, execute, Aer, IBMQ
from qiskit.providers.aer import noise
import pickle

# Choose a real device to simulate
provider = IBMQ.load_account()
device = provider.get_backend('ibmq_16_melbourne')
properties = device.properties()

# Generate an Aer noise model for device
noise_model = noise.device.basic_device_noise_model(properties)
basis_gates = noise_model.basis_gates

pickle.dump( noise_model, open( "noise_model.p", "wb" ) )
pickle.dump( basis_gates, open( "basis_gates.p", "wb" ) )