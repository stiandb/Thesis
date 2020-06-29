from qiskit import QuantumRegister, ClassicalRegister
from qiskit import QuantumCircuit, execute, Aer, IBMQ
from qiskit.providers.aer import noise
import pickle

provider = IBMQ.load_account()

backend_name='ibmq_london'
backend = provider.get_backend(backend_name)
coupling_map = backend.configuration().coupling_map


device = provider.get_backend(backend_name)
properties = device.properties()


noise_model = noise.device.basic_device_noise_model(properties)
basis_gates = noise_model.basis_gates

pickle.dump( noise_model, open( "noise_model_{}.p".format(backend_name), "wb" ) )
pickle.dump( basis_gates, open( "basis_gates_{}.p".format(backend_name), "wb" ) )
pickle.dump( coupling_map, open( "coupling_map_{}.p".format(backend_name), "wb" ) )