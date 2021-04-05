from quimb import *
from quimb.tensor import *
import quimb as qu
import numpy as np
import quimb.tensor as qtn
import quf
import matplotlib.pyplot as plt
from numpy import linalg as LA
import autoray

L=3
circ = qtn.Circuit( L, MPS_computational_state(["0"]*L))
params = [2.]*15
params_0 = [0.,0,0,   0,-pi/2,0,   0,pi/2,0,   0,0,0, -pi/2,pi/2,-pi/2  ]
circ.apply_gate( "SU4", *params, 0, 1, parametrize=True,  gate_round=1, contract=False )
circ.apply_gate( "SU4", *params, 1, 2, parametrize=True,  gate_round=2, contract=False )
#print (circ.to_dense())
print (type(circ.gates), circ.gates, circ.psi, "\n")

L=3
circ1 = qtn.Circuit( L, MPS_computational_state(["0"]*L))
circ1.apply_gate( "SU4", *params, 0, 1, parametrize=True,  gate_round=1, contract=False )
circ1.apply_gate( "SU4", *params, 1, 2, parametrize=True,  gate_round=2, contract=False )
circ1.apply_gate( "SU4", *params_0, 0, 1, parametrize=True,  gate_round=3, contract=False )
print (type(circ1.gates), circ1.gates, circ1.psi, circ1.psi["ROUND_3"].data)

print ( fidelity(circ1.to_dense(),circ.to_dense()) )



