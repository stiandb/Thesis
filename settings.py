import pickle
import os
path = os.path.dirname(os.path.realpath(__file__))

ibmq_16_melbourne_noise_model = pickle.load(open( path+"/noise_model_ibmq_16_melbourne.p", "rb" ))
ibmq_16_melbourne_basis_gates = pickle.load(open( path+"/basis_gates_ibmq_16_melbourne.p", "rb" ))
ibmq_16_melbourne_coupling_map = pickle.load(open( path+"/coupling_map_ibmq_16_melbourne.p", "rb" ))
ibmqx2_noise_model = pickle.load(open( path+"/noise_model_ibmqx2.p", "rb" ))
ibmqx2_basis_gates = pickle.load(open( path+"/basis_gates_ibmqx2.p", "rb" ))
ibmqx2_coupling_map = pickle.load(open( path+"/coupling_map_ibmqx2.p", "rb" ))
ibmq_london_noise_model = pickle.load(open( path+"/noise_model_ibmq_london.p", "rb" ))
ibmq_london_basis_gates = pickle.load(open( path+"/basis_gates_ibmq_london.p", "rb" ))
ibmq_london_coupling_map = pickle.load(open( path+"/coupling_map_ibmq_london.p", "rb" ))