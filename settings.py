import pickle

with open("noise_model.p", "rb") as fp:   #Get properties from IBMQ 16 qubit Melbourne
	melbourne_noise_model = pickle.load(fp)

with open("basis_gates.p", "rb") as fp:   #Get properties from IBMQ 16 qubit Melbourne
	melbourne_basis_gates = pickle.load(fp)