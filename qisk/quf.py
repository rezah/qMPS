from quimb import *
from quimb.tensor import *
import quimb as qu
import numpy as np
import matplotlib.pyplot as plt
from qiskit import *
from qiskit.circuit import Parameter
from qiskit.visualization import plot_state_city, plot_bloch_multivector
from qiskit.visualization import plot_state_paulivec, plot_state_hinton
from qiskit.visualization import plot_state_qsphere




def make_circuit(circ, param_1, where ):
 
 theta, zeta, chi, gamma, phi=param_1
 q0, q1=where
 #print (where, q0, q1, theta, gamma, phi)


 circ.rz(-2.*phi,q0)

 circ.rz(chi+zeta,q1)


 circ.h(q0)
 circ.cz(q0, q1)
 circ.h(q0)

 circ.rz(-2.*gamma,q0)
 circ.ry(-theta,q1)


 circ.h(q1)
 circ.cz(q0, q1)
 circ.h(q1)

 circ.ry(theta,q1)

 circ.h(q0)
 circ.cz(q0, q1)
 circ.h(q0)

 circ.rz(-zeta,q0)
 circ.rz(-chi,q1)


 return circ

def gen_1Q_circ(circ, theta, phi, lam,q_0):


 circ.rz(lam,q_0)
 circ.rx(pi/2,q_0)
 circ.rz(theta,q_0)
 circ.rx(-pi/2,q_0)
 circ.rz(phi,q_0)





# circ.u3(theta, phi, lam,q_0)

 return circ


def make_circuit_gen(circ, param_1, where ):
 
 theta1, phi1, lamda1, theta2, phi2, lamda2, theta3, phi3, lamda3, theta4, phi4, lamda4, t1, t2, t3=param_1
 q0, q1=where
 circ=gen_1Q_circ(circ, theta1, phi1, lamda1, q0)
 circ=gen_1Q_circ(circ, theta2, phi2, lamda2, q1)


 circ.h(q0)
 circ.cz(q0, q1)
 circ.h(q0)

 circ.rz(t1,q0)
 circ.ry(t2,q1)

 circ.h(q1)
 circ.cz(q0, q1)
 circ.h(q1)

 circ.ry(t3,q1)

 circ.h(q0)
 circ.cz(q0, q1)
 circ.h(q0)

 circ=gen_1Q_circ(circ, theta3, phi3, lamda3, q0)
 circ=gen_1Q_circ(circ, theta4, phi4, lamda4, q1)


 return circ



def mpo_particle(L):

 We = np.zeros([1, 1, 2, 2])
 Wo = np.zeros([1, 1, 2, 2])

 Z = qu.pauli('Z')
 X = qu.pauli('X')
 Y = qu.pauli('Y')
 I = qu.pauli('I')


 S_up=(X+1.0j*Y)*(0.5)
 S_down=(X-1.0j*Y)*(0.5)

#  S_up=S_up.astype('float64')
#  S_down=S_down.astype('float64')
#  Z=Z.astype('float64')


 MPO_I=MPO_identity(2*L, phys_dim=2)
 MPO_result=MPO_identity(2*L, phys_dim=2)
 MPO_result=MPO_result*0.0


 max_bond_val=200
 cutoff_val=1.0e-12

 MPO_result=MPO_identity(2*L, phys_dim=2)
 MPO_result=MPO_result*0.0

 for i in range(L):
  Wl = np.zeros([ 1, 2, 2],dtype='complex128')
  W = np.zeros([1, 1, 2, 2],dtype='complex128')
  Wr = np.zeros([ 1, 2, 2],dtype='complex128')
  Wl[ 0,:,:]=S_up@S_down
  W[ 0,0,:,:]=S_up@S_down
  Wr[ 0,:,:]=S_up@S_down

  W_list=[Wl]+[W]*(2*L-2)+[Wr]

  MPO_I=MPO_identity(2*L, phys_dim=2 )
  MPO_I[2*i].modify(data=W_list[2*i])
  MPO_result=MPO_result+MPO_I
  MPO_result.compress( max_bond=max_bond_val, cutoff=cutoff_val )

  MPO_I=MPO_identity(2*L, phys_dim=2 )
  MPO_I[2*i+1].modify(data=W_list[2*i+1])
  MPO_result=MPO_result+MPO_I
  MPO_result.compress( max_bond=max_bond_val, cutoff=cutoff_val )


 return MPO_result



def mpo_spin(L):

 We = np.zeros([1, 1, 2, 2],dtype='complex128')
 Wo = np.zeros([1, 1, 2, 2],dtype='complex128')

 Z = qu.pauli('Z')
 X = qu.pauli('X')
 Y = qu.pauli('Y')
 I = qu.pauli('I')


 S_up=(X+1.0j*Y)*(0.5)
 S_down=(X-1.0j*Y)*(0.5)

#  S_up=S_up.astype('float64')
#  S_down=S_down.astype('float64')
#  Z=Z.astype('float64')


 MPO_I=MPO_identity(2*L, phys_dim=2)
 MPO_result=MPO_identity(2*L, phys_dim=2)
 MPO_result=MPO_result*0.0


 max_bond_val=200
 cutoff_val=1.0e-12

 MPO_result=MPO_identity(2*L, phys_dim=2)
 MPO_result=MPO_result*0.0

 for i in range(L):
  Wl = np.zeros([ 1, 2, 2],dtype='complex128')
  W = np.zeros([1, 1, 2, 2],dtype='complex128')
  Wr = np.zeros([ 1, 2, 2],dtype='complex128')
  Wl[ 0,:,:]=S_up@S_down
  W[ 0,0,:,:]=S_up@S_down
  Wr[ 0,:,:]=S_up@S_down

  W_list=[Wl]+[W]*(2*L-2)+[Wr]

  MPO_I=MPO_identity(2*L, phys_dim=2 )
  MPO_I[2*i].modify(data=W_list[2*i])
  MPO_result=MPO_result+MPO_I
  MPO_result.compress( max_bond=max_bond_val, cutoff=cutoff_val )



 MPO_result1=MPO_identity(2*L, phys_dim=2)
 MPO_result1=MPO_result*0.0

 for i in range(L):
  Wl = np.zeros([ 1, 2, 2],dtype='complex128')
  W = np.zeros([1, 1, 2, 2],dtype='complex128')
  Wr = np.zeros([ 1, 2, 2],dtype='complex128')
  Wl[ 0,:,:]=S_up@S_down
  W[ 0,0,:,:]=S_up@S_down
  Wr[ 0,:,:]=S_up@S_down

  W_list=[Wl]+[W]*(2*L-2)+[Wr]


  MPO_I=MPO_identity(2*L, phys_dim=2 )
  MPO_I[2*i+1].modify(data=W_list[2*i+1])
  MPO_result1=MPO_result1+MPO_I
  MPO_result1.compress( max_bond=max_bond_val, cutoff=cutoff_val )

 return MPO_result, MPO_result1






def mpo_Fermi_Hubburd(L, U, t, mu):

 #print ( "L,", L, "U,", U, "t,", t,  "mu,", mu)
 We = np.zeros([1, 1, 2, 2])
 Wo = np.zeros([1, 1, 2, 2])

 Z = qu.pauli('Z')
 X = qu.pauli('X')
 Y = qu.pauli('Y')
 I = qu.pauli('I')


 S_up=(X+1.0j*Y)*(0.5)
 S_down=(X-1.0j*Y)*(0.5)

#  S_up=S_up.astype('float64')
#  S_down=S_down.astype('float64')
#  Z=Z.astype('float64')


 MPO_I=MPO_identity(2*L, phys_dim=2)
 MPO_result=MPO_identity(2*L, phys_dim=2)
 MPO_result=MPO_result*0.0
 MPO_f=MPO_result*0.0


 max_bond_val=200
 cutoff_val=1.0e-12
 if abs(U) > 1.0e-9:
  for i in range(L): 
   Wl = np.zeros([ 1, 2, 2], dtype='complex128')
   W = np.zeros([1, 1, 2, 2],dtype='complex128')
   Wr = np.zeros([ 1, 2, 2],dtype='complex128')

   Wl[ 0,:,:]=S_up@S_down
   W[ 0,0,:,:]=S_up@S_down
   Wr[ 0,:,:]=S_up@S_down
   W_list=[Wl]+[W]*(2*L-2)+[Wr]

   MPO_I=MPO_identity(2*L, phys_dim=2 )
   MPO_I[2*i].modify(data=W_list[2*i])
   MPO_I[2*i+1].modify(data=W_list[2*i+1])
   MPO_result=MPO_result+MPO_I
   MPO_result.compress( max_bond=max_bond_val, cutoff=cutoff_val )


  MPO_f=MPO_result*U
  MPO_f.compress( max_bond=max_bond_val, cutoff=cutoff_val )

 MPO_result=MPO_identity(2*L, phys_dim=2)
 MPO_result=MPO_result*0.0

 if abs(mu) > 1.0e-9:
  for i in range(L):
   Wl = np.zeros([ 1, 2, 2],dtype='complex128')
   W = np.zeros([1, 1, 2, 2],dtype='complex128')
   Wr = np.zeros([ 1, 2, 2],dtype='complex128')
   Wl[ 0,:,:]=S_up@S_down
   W[ 0,0,:,:]=S_up@S_down
   Wr[ 0,:,:]=S_up@S_down

   W_list=[Wl]+[W]*(2*L-2)+[Wr]

   MPO_I=MPO_identity(2*L, phys_dim=2 )
   MPO_I[2*i].modify(data=W_list[2*i])
   MPO_result=MPO_result+MPO_I
   MPO_result.compress( max_bond=max_bond_val, cutoff=cutoff_val )

   MPO_I=MPO_identity(2*L, phys_dim=2 )
   MPO_I[2*i+1].modify(data=W_list[2*i+1])
   MPO_result=MPO_result+MPO_I
   MPO_result.compress( max_bond=max_bond_val, cutoff=cutoff_val )

  MPO_f=MPO_f+MPO_result*(-mu)
  MPO_f.compress( max_bond=max_bond_val, cutoff=cutoff_val )

 MPO_result=MPO_identity(2*L, phys_dim=2)
 MPO_result=MPO_result*0.0

 if abs(t) > 1.0e-9:
  for i in range(L-1):
    Wl = np.zeros([ 1, 2, 2],dtype='complex128')
    W = np.zeros([1, 1, 2, 2],dtype='complex128')
    Wr = np.zeros([ 1, 2, 2],dtype='complex128')

    Wl[ 0,:,:]=S_up
    W[ 0,0,:,:]=S_up
    Wr[ 0,:,:]=S_up
    W_1=[Wl]+[W]*(2*L-2)+[Wr]

    Wl = np.zeros([ 1, 2, 2],dtype='complex128')
    W = np.zeros([1, 1, 2, 2],dtype='complex128')
    Wr = np.zeros([ 1, 2, 2],dtype='complex128')

    Wl[ 0,:,:]=S_down
    W[ 0,0,:,:]=S_down
    Wr[ 0,:,:]=S_down
    W_2=[Wl]+[W]*(2*L-2)+[Wr]


    MPO_I=MPO_identity(2*L, phys_dim=2 )
    MPO_I[2*i].modify(data=W_1[2*i])
    MPO_I[2*i+2].modify(data=W_2[2*i+2])
    MPO_result=MPO_result+MPO_I
    MPO_result.compress( max_bond=max_bond_val, cutoff=cutoff_val )

    MPO_I=MPO_identity(2*L, phys_dim=2 )
    MPO_I[2*i].modify(data=W_2[2*i])
    MPO_I[2*i+2].modify(data=W_1[2*i+2])
    MPO_result=MPO_result+MPO_I
    MPO_result.compress( max_bond=max_bond_val, cutoff=cutoff_val )


    MPO_I=MPO_identity(2*L, phys_dim=2 )
    MPO_I[2*i+1].modify(data=W_1[2*i+1])
    MPO_I[2*i+3].modify(data=W_2[2*i+3])
    MPO_result=MPO_result+MPO_I
    MPO_result.compress( max_bond=max_bond_val, cutoff=cutoff_val )


    MPO_I=MPO_identity(2*L, phys_dim=2 )
    MPO_I[2*i+1].modify(data=W_2[2*i+1])
    MPO_I[2*i+3].modify(data=W_1[2*i+3])
    MPO_result=MPO_result+MPO_I
    MPO_result.compress( max_bond=max_bond_val, cutoff=cutoff_val )


 MPO_f=MPO_f+MPO_result*(t)
 MPO_f.compress( max_bond=max_bond_val, cutoff=cutoff_val )

 return MPO_f
