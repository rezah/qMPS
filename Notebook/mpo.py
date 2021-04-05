from quimb import *
from quimb.tensor import *
import quimb as qu
import numpy as np
import quimb.tensor as qtn
import quf
import matplotlib.pyplot as plt
from numpy import linalg as LA

#L is even
L=10
U= 5.0
t=1.0
mu=0.0


print ( "L,", L, "U,", U, "t,", t,  "mu,", mu)
We = np.zeros([1, 1, 2, 2], dtype='float64')
Wo = np.zeros([1, 1, 2, 2], dtype='float64')

Z = qu.pauli('Z')
X = qu.pauli('X')
Y = qu.pauli('Y')
I = qu.pauli('I')


S_up=(X+1.0j*Y)*(0.5)
S_down=(X-1.0j*Y)*(0.5)

S_up=S_up.astype('float64')
S_down=S_down.astype('float64')
Z=Z.astype('float64')


MPO_I=MPO_identity(2*L, phys_dim=2)
MPO_result=MPO_identity(2*L, phys_dim=2)
MPO_result=MPO_result*0.0
MPO_f=MPO_result*0.0


max_bond_val=200
cutoff_val=1.0e-12
if abs(U) > 1.0e-9:
 for i in range(L): 
  Wl = np.zeros([ 1, 2, 2], dtype='float64')
  W = np.zeros([1, 1, 2, 2], dtype='float64')
  Wr = np.zeros([ 1, 2, 2], dtype='float64')

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
  Wl = np.zeros([ 1, 2, 2], dtype='float64')
  W = np.zeros([1, 1, 2, 2], dtype='float64')
  Wr = np.zeros([ 1, 2, 2], dtype='float64')
  Wl[ 0,:,:]=S_up@S_down
  W[ 0,0,:,:]=S_up@S_down
  Wr[ 0,:,:]=S_up@S_down

  W_list=[Wl]+[W]*(2*L-2)+[Wr]

  MPO_I=MPO_identity(2*L, phys_dim=2 )
  MPO_I[2*i].modify(data=W_list[2*i])
  MPO_I[2*i+1].modify(data=W_list[2*i+1])
  MPO_result=MPO_result+MPO_I
  MPO_result.compress( max_bond=max_bond_val, cutoff=cutoff_val )

 MPO_f=MPO_f+MPO_result*(-mu)
 MPO_f.compress( max_bond=max_bond_val, cutoff=cutoff_val )

MPO_result=MPO_identity(2*L, phys_dim=2)
MPO_result=MPO_result*0.0

if abs(t) > 1.0e-9:
 for i in range(L-1):
   Wl = np.zeros([ 1, 2, 2], dtype='float64')
   W = np.zeros([1, 1, 2, 2], dtype='float64')
   Wr = np.zeros([ 1, 2, 2], dtype='float64')

   Wl[ 0,:,:]=S_up
   W[ 0,0,:,:]=S_up
   Wr[ 0,:,:]=S_up
   W_1=[Wl]+[W]*(2*L-2)+[Wr]

   Wl = np.zeros([ 1, 2, 2], dtype='float64')
   W = np.zeros([1, 1, 2, 2], dtype='float64')
   Wr = np.zeros([ 1, 2, 2], dtype='float64')

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

print("Start_DMRG")
#psi_init=load_from_disk("Store/psi_init")
dmrg = DMRG2(MPO_f, bond_dims=[20,  60, 80, 100, 200, 300], cutoffs=1.e-12)
dmrg.solve(tol=1e-12, verbosity=0 )
psi=dmrg.state
print("DMRG_Fermi_Hubburd=", dmrg.energy, psi.show())

#save_to_disk(psi, "Store/psi_init")


# 
# MPO_XY=MPO_ham_XY(L=L, j=1.0, bz=0.0, S=0.5, cyclic=False)
# dmrg = DMRG2(MPO_XY, bond_dims=[20,100], cutoffs=1.e-12)
# dmrg.solve(tol=1e-12, verbosity=0 )
# psi_xy=dmrg.state
# print("DMRG_XY=", dmrg.energy*4.0, psi_xy.show())



MPO_I=MPO_identity(2*L, phys_dim=2, cyclic=False )
MPO_UP=MPO_identity(2*L, phys_dim=2, cyclic=False )
MPO_UP=MPO_UP*0.0



max_bond_val=120
cutoff_val=1.0e-12

for i in range(L):
 Wl = np.zeros([ 1, 2, 2], dtype='float64')
 W = np.zeros([1, 1, 2, 2], dtype='float64')
 Wr = np.zeros([ 1, 2, 2], dtype='float64')

 Wl[ 0,:,:]=S_up@S_down
 W[ 0,0,:,:]=S_up@S_down
 Wr[ 0,:,:]=S_up@S_down

 W_l=[Wl]+[W]*(2*L-2)+[Wr]

 MPO_I=MPO_identity(2*L, phys_dim=2 )
 MPO_I[2*i].modify(data=W_l[2*i])
 MPO_UP=MPO_UP+MPO_I
 MPO_UP.compress( max_bond=max_bond_val, cutoff=cutoff_val )




MPO_I=MPO_identity(2*L, phys_dim=2, cyclic=False )
MPO_down=MPO_identity(2*L, phys_dim=2, cyclic=False )
MPO_down=MPO_down*0.0



max_bond_val=20
cutoff_val=1.0e-16

for i in range(L):

 Wl = np.zeros([ 1, 2, 2], dtype='float64')
 W = np.zeros([1, 1, 2, 2], dtype='float64')
 Wr = np.zeros([ 1, 2, 2], dtype='float64')

 Wl[ 0,:,:]=S_up@S_down
 W[ 0,0,:,:]=S_up@S_down
 Wr[ 0,:,:]=S_up@S_down

 W_l=[Wl]+[W]*(2*L-2)+[Wr]

 MPO_I=MPO_identity(2*L, phys_dim=2 )
 MPO_I[2*i+1].modify(data=W_l[2*i+1])
 MPO_down=MPO_down+MPO_I
 MPO_down.compress( max_bond=max_bond_val, cutoff=cutoff_val )




p_0=MPO_down.apply(psi, compress=False)
print (   "down",   (psi | p_0)^all   )
N_d=(psi | p_0)^all

p_0=MPO_UP.apply(psi, compress=False)
print (   "up",   (psi | p_0)^all   )
N_u=(psi | p_0)^all


print (N_d + N_u, N_d - N_u)


