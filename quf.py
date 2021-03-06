from quimb import *
from quimb.tensor import *
import quimb as qu
import numpy as np
import quimb.tensor as qtn
import itertools
from operator import add
import operator
import matplotlib.pyplot as plt
import math
import cmath
from numpy.linalg import inv
from cmath import exp, pi, sin, cos, acos, log, polar
import cotengra as ctg
import copy
import autoray
from progress.bar import Bar
import tqdm
import warnings
from collections import Counter

val_intense=0.0

#'FSIM': apply_fsim,   2
#'FSIMT': apply_fsimt,  1 
#'FSIMG': apply_fsimg,   5
#'SU4': apply_su4,   15






#i_start: all-the-way left position, where the first unitary acts
def range_unitary(psi, i_start, n_apply, list_u3, depth, n_Qbit,data_type,seed_val, Qubit_ara):
    gate_round=None
    if n_Qbit==0: depth=1
    if n_Qbit==1: depth=1

    c_val=0
    for r in range(depth):

     if r%2==0:
      for i in range(i_start, i_start+n_Qbit, 2):
         #print("U_e", i, i + 1, n_apply)
         if seed_val==0:
            G=np.eye(4, dtype=data_type).reshape((2, 2, 2, 2))
            Grand=qu.randn((2, 2, 2, 2),dtype=data_type, dist="uniform", seed=i+seed_val)
            G=G+Grand*val_intense
         else:
            G = qu.randn((2, 2, 2, 2),dtype=data_type, dist="uniform", seed=i+seed_val)
         
         
         psi.gate_(G, (i, i + 1), tags={'U',f'G{n_apply}', f'lay{Qubit_ara}',f'P{Qubit_ara}L{i}D{r}'})
         list_u3.append(f'G{n_apply}')
         n_apply+=1
         c_val+=1

     elif r%2!=0:
      for i in range(i_start, i_start+n_Qbit-1, 2):
         #print("U_o", i+1, i + 2, n_apply)

         if seed_val==0:
            G=np.eye(4, dtype=data_type).reshape((2, 2, 2, 2))
            Grand=qu.randn((2, 2, 2, 2),dtype=data_type, dist="uniform", seed=i+seed_val)
            G=G+Grand*val_intense
         else:
            G = qu.randn((2, 2, 2, 2),dtype=data_type, dist="uniform", seed=i+seed_val)

         psi.gate_(G, (i+1, i + 2), tags={'U',f'G{n_apply}',f'lay{Qubit_ara}',f'P{Qubit_ara}L{i}D{r}'})
         list_u3.append(f'G{n_apply}')
         n_apply+=1
         c_val+=1

    return n_apply, list_u3

def range_unitary_reverse(psi, i_start, n_apply, list_u3, depth, n_Qbit,data_type,seed_val,Qubit_ara):
    gate_round=None
    if abs(n_Qbit)==0: depth=1
    if abs(n_Qbit)==1: depth=1

    c_val=0
    for r in range(depth):

     if r%2==0:
      for i in range(i_start, i_start+n_Qbit, -2):
         if seed_val==0:
            G=np.eye(4, dtype=data_type).reshape((2, 2, 2, 2))
            Grand=qu.randn((2, 2, 2, 2),dtype=data_type, dist="uniform", seed=i+seed_val)
            G=G+Grand*val_intense
         else:
            G = qu.randn((2, 2, 2, 2),dtype=data_type, dist="uniform", seed=i+seed_val)

         psi.gate_(G, (i, i - 1), tags={'U',f'G{n_apply}',f'lay{Qubit_ara}',f'P{Qubit_ara}L{i}D{r}'})
         list_u3.append(f'G{n_apply}')
         n_apply+=1
         c_val+=1


     elif r%2!=0:
      for i in range(i_start, i_start+n_Qbit+1, -2):
         if seed_val==0:
            G=np.eye(4, dtype=data_type).reshape((2, 2, 2, 2))
            Grand=qu.randn((2, 2, 2, 2),dtype=data_type, dist="uniform", seed=i+seed_val)
            G=G+Grand*val_intense
         else:
            G = qu.randn((2, 2, 2, 2),dtype=data_type, dist="uniform", seed=i+seed_val)

         psi.gate_(G, (i-1, i - 2), tags={'U',f'G{n_apply}',f'lay{Qubit_ara}',f'P{Qubit_ara}L{i}D{r}'})
         list_u3.append(f'G{n_apply}')
         n_apply+=1
         c_val+=1

    return n_apply, list_u3




def range_unitary_pollmann_reverse(psi, i_start, n_apply, list_u3, depth, n_Qbit,data_type,seed_val,Qubit_ara):
    if n_Qbit==0: depth=1
    if n_Qbit==1: depth=1
    c_val=0
    for r in range(depth):
      for i in range(i_start, i_start+n_Qbit, -1):
         if seed_val==0:
            G=np.eye(4, dtype=data_type).reshape((2, 2, 2, 2))
            Grand=qu.randn((2, 2, 2, 2),dtype=data_type, dist="uniform", seed=i+seed_val)
            G=G+Grand*val_intense
         else:
            G = qu.randn((2, 2, 2, 2),dtype=data_type, dist="uniform", seed=i+seed_val)

         psi.gate_(G, (i, i - 1), tags={'U',f'G{n_apply}',f'lay{Qubit_ara}',f'P{Qubit_ara}L{i}D{r}'})
         list_u3.append(f'G{n_apply}')
         n_apply+=1
         c_val+=1

    return n_apply, list_u3






def range_unitary_pollmann(psi, i_start, n_apply, list_u3, depth, n_Qbit,data_type,seed_val,Qubit_ara):
    gate_round=None
    if n_Qbit==0: depth=1
    if n_Qbit==1: depth=1
    c_val=0
    for r in range(depth):
      for i in range(i_start, i_start+n_Qbit, 1):
         #print("U_e", i, i + 1, n_apply)
         if seed_val==0:
            G=np.eye(4, dtype=data_type).reshape((2, 2, 2, 2))
            Grand=qu.randn((2, 2, 2, 2),dtype=data_type, dist="uniform", seed=i+seed_val)
            G=G+Grand*val_intense
         else:
            G = qu.randn((2, 2, 2, 2),dtype=data_type, dist="uniform", seed=i+seed_val)

         psi.gate_(G, (i, i + 1), tags={'U',f'G{n_apply}',f'lay{Qubit_ara}',f'P{Qubit_ara}L{i}D{r}'})
         list_u3.append(f'G{n_apply}')
         n_apply+=1
         c_val+=1

    return n_apply, list_u3



def MERA_internal( psi, i_start, L, in_depth, n_apply, list_u3,n_Qbit, Qubit_ara, data_type='float64', qmera_type="brickwall"):

   depth_total=int(math.log2(L))
   
   #print ("i_start, n_Qbit, in_depth, L, depth_total",i_start, n_Qbit, in_depth, L, depth_total)
   
   seed_val=10
   

   for j in range(depth_total):
    depth_mera=int(j*1)
    even_depth=int(2*depth_mera)
    odd_depth=int(2*depth_mera+1)
    if j==0:  
     in_depth_tmp=1
    else:
     in_depth_tmp=in_depth
    

    if 2**(j+1)>L: 
     print("2^depth>L", 2**(j+1), L)
     break
    
    if  n_Qbit > 2**(j):
        n_Qbit_temp = 2**(j)
    else  :
        n_Qbit_temp = n_Qbit


    for i in range( 0, L, 2**(j+1)):

     list_Qubit1=[(i-i_nq)%L  for i_nq in reversed(range(n_Qbit_temp))]
     list_Qubit2=[ (i-i_nq+2**(j))%L    for i_nq in reversed(range(n_Qbit_temp))  ]
     list_Qubit=list_Qubit1+list_Qubit2
     list_Qubit=[ i+ i_start  for i in list_Qubit]

     #print("1_Qubit", list_Qubit, "site=",i, "layer=", j,"max_bond=", 2**(j+1))
     if qmera_type=="brickwall":
      n_apply, list_u3=general_unitary_qmera_list(psi, list_Qubit, n_apply, list_u3, in_depth_tmp, L,data_type,seed_val, depth_mera,even_depth, Qubit_ara, i)
     if qmera_type=="pollmann":
      n_apply, list_u3=general_unitary_qmera_list_pollmann(psi, list_Qubit, n_apply, list_u3, in_depth_tmp, L,data_type,seed_val, depth_mera,even_depth)

    for i in range( 0, L, 2**(j+1)):

     if 2**(j+1)<L:
      list_Qubit1=[ (i-i_nq+2**(j))%L  for i_nq in reversed(range(n_Qbit_temp))]
      list_Qubit2=[ (i-i_nq+2**(j+1))%L  for i_nq in reversed(range(n_Qbit_temp))]
      list_Qubit=list_Qubit1+list_Qubit2
      list_Qubit=[ i + i_start  for i in list_Qubit]

      #print("2_Qubit", list_Qubit, "site=", i, "layer=",j)

      if qmera_type=="brickwall":
       n_apply, list_u3=general_unitary_qmera_list(psi, list_Qubit, n_apply, list_u3, in_depth_tmp, L,data_type,seed_val, depth_mera,odd_depth, Qubit_ara,i)
      if qmera_type=="pollmann":
       n_apply, list_u3=general_unitary_qmera_list_pollmann(psi, list_Qubit, n_apply, list_u3, in_depth_tmp, L,data_type,seed_val, depth_mera,odd_depth)


   return n_apply, list_u3








def qmps_f( L=16, in_depth=2, n_Qbit=3, data_type='float64', qmps_structure="brickwall", canon="left",  n_q_mera=2, seed_init=10, internal_mera="brickwall"):

   seed_val=seed_init
   list_u3=[]
   n_apply=0
   psi = qtn.MPS_computational_state('0' * L)
   for t in psi:
     t.modify(left_inds=())

   for t in  range(L):
     psi[t].modify(tags=[f"I{t}", "MPS"])




   if canon=="left":
#    for i in range(0,n_Qbit,1):
#     #print ("Qbit_0", i)
#     Qubit_ara=i
#     if qmps_structure=="brickwall":
#      n_apply, list_u3=range_unitary(psi, 0, n_apply, list_u3, in_depth, i,data_type,seed_val, Qubit_ara)
#     elif qmps_structure=="pollmann":
#      n_apply, list_u3=range_unitary_pollmann(psi, 0, n_apply, list_u3, in_depth, i,data_type,seed_val, Qubit_ara)
#     elif qmps_structure=="mera":
#      n_apply, list_u3=range_unitary(psi, 0, n_apply, list_u3, in_depth, i,data_type,seed_val, Qubit_ara)




    for i in range(0,L-n_Qbit,1):
     #print ("quibit", i+n_Qbit, n_Qbit)
     Qubit_ara=i+n_Qbit
     if qmps_structure=="brickwall":
      n_apply, list_u3=range_unitary(psi, i, n_apply, list_u3, in_depth, n_Qbit,data_type,seed_val, Qubit_ara)
     elif qmps_structure=="pollmann":
      n_apply, list_u3=range_unitary_pollmann(psi, i, n_apply, list_u3, in_depth, n_Qbit,data_type,seed_val, Qubit_ara)
     elif qmps_structure=="mera":
         n_apply, list_u3=MERA_internal( psi, i, n_Qbit, in_depth, n_apply,list_u3, n_q_mera, Qubit_ara, qmera_type=internal_mera)



    if qmps_structure=="mera":
        n_apply, list_u3=MERA_internal( psi, L-n_Qbit, n_Qbit, in_depth, n_apply,list_u3, n_q_mera, Qubit_ara, qmera_type=internal_mera)


   if canon=="mixed":

#range(start, stop, step)
 #first step:
    for i in range(0,n_Qbit,1):
     #print ("Qbit0", i)
     Qubit_ara=i
     if qmps_structure=="brickwall":
      n_apply, list_u3=range_unitary(psi, 0, n_apply, list_u3, in_depth, i,data_type,seed_val, Qubit_ara)
     elif qmps_structure=="pollmann":
      n_apply, list_u3=range_unitary_pollmann(psi, 0, n_apply, list_u3, in_depth, i,data_type,seed_val, Qubit_ara)
     elif qmps_structure=="mera":
      n_apply, list_u3=range_unitary(psi, 0, n_apply, list_u3, in_depth, i,data_type,seed_val, Qubit_ara)


    #print (list_u3)
    for i in range(0,L//2-n_Qbit,1):
     #print ("quibit", i+n_Qbit)
     Qubit_ara=i+n_Qbit
     if qmps_structure=="brickwall":
      n_apply, list_u3=range_unitary(psi, i, n_apply, list_u3, in_depth, n_Qbit,data_type,seed_val, Qubit_ara)
     elif qmps_structure=="pollmann":
      n_apply, list_u3=range_unitary_pollmann(psi, i, n_apply, list_u3, in_depth, n_Qbit,data_type,seed_val, Qubit_ara)
     if qmps_structure=="mera":
         #print ("i0", i)
         n_apply, list_u3=MERA_internal( psi, i, n_Qbit, in_depth, n_apply,list_u3, n_q_mera, Qubit_ara, qmera_type=internal_mera)


    if qmps_structure=="mera":
        #print ("i0", L//2-n_Qbit)
        n_apply, list_u3=MERA_internal( psi, L//2-n_Qbit, n_Qbit, in_depth, n_apply,list_u3, n_q_mera, Qubit_ara,qmera_type=internal_mera)


#range(start, stop, step)
 #first step:
    for i in range(0,n_Qbit,1):
     #print ("n_Qbit", L-1-i)
     Qubit_ara=L-1-i
     if qmps_structure=="brickwall":
      n_apply, list_u3=range_unitary_reverse(psi, L-1, n_apply, list_u3, in_depth, -i,data_type,seed_val, Qubit_ara)
     elif qmps_structure=="pollmann":
      n_apply, list_u3=range_unitary_pollmann_reverse(psi, L-1, n_apply, list_u3, in_depth, -i,data_type,seed_val, Qubit_ara)
     if qmps_structure=="mera":
      n_apply, list_u3=range_unitary_reverse(psi, L-1, n_apply, list_u3, in_depth, -i,data_type,seed_val, Qubit_ara,qmera_type=internal_mera)

    #print (list_u3)
    for i in range( L-1,L//2+n_Qbit-1,-1):
     #print ("quibit", i-n_Qbit)
     Qubit_ara=i-n_Qbit
     if qmps_structure=="brickwall":
      n_apply, list_u3=range_unitary_reverse(psi, i, n_apply, list_u3, in_depth, -n_Qbit,data_type,seed_val, Qubit_ara)
     elif qmps_structure=="pollmann":
      n_apply, list_u3=range_unitary_pollmann_reverse(psi, i, n_apply, list_u3, in_depth, -n_Qbit,data_type,seed_val, Qubit_ara)
     if qmps_structure=="mera":
         #print ("im", i)
         n_apply, list_u3=MERA_internal_reverse( psi, i, n_Qbit, in_depth, n_apply,list_u3, n_q_mera, Qubit_ara,qmera_type=internal_mera)



    if qmps_structure=="mera":
        #print ("im", L//2+n_Qbit-1)
        n_apply, list_u3=MERA_internal_reverse( psi, L//2+n_Qbit-1, n_Qbit, in_depth, n_apply,list_u3, n_q_mera, Qubit_ara,qmera_type=internal_mera)


    #print ("quibit", L//2-n_Qbit)
    Qubit_ara=L//2-n_Qbit
    if qmps_structure=="brickwall":
     n_apply, list_u3=range_unitary(psi, L//2-n_Qbit, n_apply, list_u3, in_depth, 2*n_Qbit,data_type, seed_val, Qubit_ara)
    elif qmps_structure=="pollmann":
     n_apply, list_u3=range_unitary_pollmann(psi, L//2-n_Qbit, n_apply, list_u3, in_depth, 2*n_Qbit, data_type, seed_val, Qubit_ara)
    if qmps_structure=="mera":
         print ("if", L//2-n_Qbit)
         n_apply, list_u3=MERA_internal( psi, L//2-n_Qbit, 2*n_Qbit, in_depth, n_apply,list_u3, n_q_mera, Qubit_ara,qmera_type=internal_mera)


   psi=convert_wave_function_to_real(psi, L, list_u3)
   return psi, list_u3





def convert_wave_function_to_real(qmps, L_L, tags):


 mps_product=qmps.select('MPS', which='any')

 T=qmps.select('U', which='any')
 list_inds=[]
 list_k=[]
 list_b=[]
 for i in range(L_L):
    list_inds.append(   list(mps_product[i].inds)[-1]   )
    list_k.append(  f"k{i}"   )
    list_b.append(  f"b{i}"   )

 list_tag=[]
 for i in list_k:
   for j in range(len(tags)):
     t=list(qmps[tags[j]].inds)
     for ii in range(len(t)):
         if t[ii]==i:
           A=qmps[tags[j]].tags
           list_A=list(A)
           #print (list_A)
           list_A = [x for x in list_A if not x.startswith('G')]
           list_A = [x for x in list_A if not x.startswith('lay')]
           list_A = [x for x in list_A if not x.startswith('U')]
           list_A = [x for x in list_A if not x.startswith('P')]
           list_A=list_A+["ket0"]
           list_tag.append(list_A)






 ops = { list_inds[i] : list_b[i] for i in range(L_L)}
 T.reindex_(ops)
 ops = { list_k[i] : list_inds[i] for i in range(L_L)}
 T.reindex_(ops)
 ops = { list_b[i] : list_k[i] for i in range(L_L)}
 T.reindex_(ops)
 for i in range(L_L):
     mps_product[[f"I{i}", "MPS"]].modify(tags=list_tag[i])

 qmps=mps_product | T

 qmps.view_as_(
     qtn.tensor_1d.TensorNetwork1DVector,
     L=L_L,
     site_tag_id='I{}',
     site_ind_id='k{}'
 )

 return qmps




def convert_wave_function_to_real_2D(psi, L_x, L_y, tags):

 mps_product=psi.select('PEPS', which='any')
 #tags=list_u3

 T=psi.select('U', which='any')
 list_inds=[]
 list_k=[]
 list_b=[]

 for t in mps_product:
     list_inds.append(   list(t.inds)[-1]   )

 for i in range(L_x):
  for j in range(L_y):
     list_k.append(  f"k{i},{j}"   )
     list_b.append(  f"b{i},{j}"   )



 list_tag=[]
 for i in list_k:
  for j in range(len(tags)):
   t=list(psi[tags[j]].inds)
   for ii in range(len(t)):
     if t[ii]==i:
         A=psi[tags[j]].tags
         list_A=list(A)
         for jj in range(len(tags)):
                try:
                  list_A.remove(f"G{jj}")
                except ValueError:
                     pass  # do nothing!

                
                try:
                  list_A.remove(f"lay{jj}")
                except ValueError:
                     pass  # do nothing!


                try:
                  list_A.remove("U")
                except ValueError:
                     pass  # do nothing!


                
                
         list_A=list_A+["ket0"]
         list_tag.append(list_A)

            
            
 ops = { list_inds[i] : list_b[i] for i in range(len(list_inds))}
 T.reindex_(ops)
 ops = { list_k[i] : list_inds[i] for i in range(len(list_k))}
 T.reindex_(ops)
 ops = { list_b[i] : list_k[i] for i in range(len(list_k))}
 T.reindex_(ops)


 for i in range(L_x):
  for j in range(L_y):
     #print (i,j, list_k[i*L_y+j], list_tag[i*L_y+j])
     mps_product[[f"I{i},{j}", f'ROW{i}', f'COL{j}', "PEPS"]].modify(tags=list_tag[i*L_y+j])

    
    
    
 psi1=mps_product | T



 psi1.view_as_(
      qtn.tensor_2d.TensorNetwork2DVector,
     Lx=L_x,
     Ly=L_y,
     site_ind_id='k{},{}',
     site_tag_id='I{},{}',
     row_tag_id='ROW{}',
     col_tag_id='COL{}')



 return psi1













def qmera_f( L=16, in_depth=4, n_Qbit=1, depth_total=3 ,data_type='float64', qmera_type="brickwall", seed_init=10):

   seed_val=seed_init
   list_u3=[]
   n_apply=0
   psi = qtn.MPS_computational_state('0' * L)
   for t in psi:
     t.modify(left_inds=())

   for t in  range(L):
     psi[t].modify(tags=[f"I{t}", "MPS"])


   for j in range(depth_total):
    depth_mera=int(j*1)
    even_depth=int(2*depth_mera)
    odd_depth=int(2*depth_mera+1)
    if j==0:  
     in_depth_tmp=1
    else:
     in_depth_tmp=in_depth
    

    if 2**(j+1)>L: 
     print("2^depth>L", 2**(j+1), L)
     break
    
    if  n_Qbit > 2**(j):
        n_Qbit_temp = 2**(j)
    else  :
        n_Qbit_temp = n_Qbit


    for i in range( 0, L, 2**(j+1)):

     list_Qubit1=[(i-i_nq)%L  for i_nq in reversed(range(n_Qbit_temp))]
     list_Qubit2=[ (i-i_nq+2**(j))%L    for i_nq in reversed(range(n_Qbit_temp))  ]
     list_Qubit=list_Qubit1+list_Qubit2
     #print("1_Qubit", list_Qubit, "site=", i, "layer=", j,"max_bond=", 2**(j+1))
     if qmera_type=="brickwall":
      n_apply, list_u3=general_unitary_qmera_list(psi, list_Qubit, n_apply, list_u3, in_depth_tmp, L,data_type,seed_val, depth_mera, even_depth, i, i)
     if qmera_type=="pollmann":
      n_apply, list_u3=general_unitary_qmera_list_pollmann(psi, list_Qubit, n_apply, list_u3, in_depth_tmp, L,data_type,seed_val, depth_mera,even_depth)

    for i in range( 0, L, 2**(j+1)):
     if 2**(j+1)<L:
      list_Qubit1=[ (i-i_nq+2**(j))%L  for i_nq in reversed(range(n_Qbit_temp))]
      list_Qubit2=[ (i-i_nq+2**(j+1))%L  for i_nq in reversed(range(n_Qbit_temp))]
      list_Qubit=list_Qubit1+list_Qubit2
      #print("2_Qubit", list_Qubit, "site=", i, "layer=",j)

      if qmera_type=="brickwall":
       n_apply, list_u3=general_unitary_qmera_list(psi, list_Qubit, n_apply, list_u3, in_depth_tmp, L,data_type,seed_val, depth_mera, odd_depth, i, i)
      if qmera_type=="pollmann":
       n_apply, list_u3=general_unitary_qmera_list_pollmann(psi, list_Qubit, n_apply, list_u3, in_depth_tmp, L,data_type,seed_val, depth_mera, odd_depth)


   psi=convert_wave_function_to_real(psi, L, list_u3)
   return psi, list_u3





def  general_unitary_qmera_list_pollmann(psi, list_Qubit, n_apply, list_u3, depth, L,data_type,seed_val, depth_mera,even_depth):
    gate_round=None

    #print ("depth", depth)
    for r in range(depth):
      for i in range(0,len(list_Qubit)-1, 1):
         print ("r", r, "qubits", list_Qubit[i], list_Qubit[i + 1])
         if seed_val==0:
            G=np.eye(4, dtype=data_type).reshape((2, 2, 2, 2))
            Grand=qu.randn((2, 2, 2, 2),dtype=data_type, dist="uniform", seed=i+seed_val)
            G=G+Grand*val_intense
         else:
            G = qu.randn((2, 2, 2, 2),dtype=data_type, dist="uniform", seed=i+seed_val)

         psi.gate_(G, (list_Qubit[i], list_Qubit[i + 1]), tags={'U',f'G{n_apply}', f'lay{depth_mera}', f'lay{depth_mera}', f'uni{even_depth}',f'P{depth_mera}L{even_depth}D{r}R{list_Qubit[i]}'})
         list_u3.append(f'G{n_apply}')
         n_apply+=1

    return n_apply, list_u3




def general_unitary_qmera_list(psi, list_Qubit, n_apply, list_u3, depth, L,data_type,seed_val, depth_mera, even_depth, Qubit_ara, site):
    gate_round=None

    #print ("depth", depth)
    for r in range(depth):
     if r%2==0:

      for i in range(0,len(list_Qubit), 2):
         #print("G_e", list_Qubit[i], list_Qubit[i + 1], n_apply)
         if seed_val==0:
            G=np.eye(4, dtype=data_type).reshape((2, 2, 2, 2))
            Grand=qu.randn((2, 2, 2, 2),dtype=data_type, dist="uniform", seed=i+seed_val)
            G=G+Grand*val_intense
         else:
          G = qu.randn((2, 2, 2, 2),dtype=data_type, dist="uniform", seed=i+seed_val)

         psi.gate_(G, (list_Qubit[i], list_Qubit[i + 1]), tags={'U',f'G{n_apply}',f'layy{even_depth}', f'lay{Qubit_ara}',f'uni{even_depth}',f'P{depth_mera}L{even_depth}D{r}R{list_Qubit[i]}M{Qubit_ara}S{site}'})
         list_u3.append(f'G{n_apply}')
         n_apply+=1


     elif r%2!=0:
      for i in range(0,len(list_Qubit)-1, 2):
         if i+2 < len(list_Qubit)-1: 

          #print("G_o", list_Qubit[(i+1)], list_Qubit[i+2], n_apply)
          if seed_val==0:
             G=np.eye(4, dtype=data_type).reshape((2, 2, 2, 2))
             Grand=qu.randn((2, 2, 2, 2),dtype=data_type, dist="uniform", seed=i+seed_val)
             G=G+Grand*val_intense
          else:
            G = qu.randn((2, 2, 2, 2),dtype=data_type, dist="uniform", seed=i+seed_val)
          psi.gate_(G, (list_Qubit[(i+1)], list_Qubit[i+2]), tags={'U',f'G{n_apply}',f'lay{Qubit_ara}', f'uni{even_depth}',f'layy{even_depth}',f'P{depth_mera}L{even_depth}D{r}R{list_Qubit[i]}M{Qubit_ara}S{site}'})
          list_u3.append(f'G{n_apply}')

          n_apply+=1


    return n_apply, list_u3







def MERA_internal_reverse( psi, i_start, L, in_depth, n_apply, list_u3,n_Qbit,Qubit_ara, data_type='float64', qmera_type="brickwall"):


   depth_total=int(math.log2(L))
   
   print ("i_start, n_Qbit, in_depth, L, depth_total",i_start, n_Qbit, in_depth, L, depth_total)
   
   seed_val=10
   

   for j in range(depth_total):
    depth_mera=int(Qubit_ara)
    if j==0:  
     in_depth_tmp=1
    else:
     in_depth_tmp=in_depth
    

    if 2**(j+1)>L: 
     print("2^depth>L", 2**(j+1), L)
     break
    
    if  n_Qbit > 2**(j):
        n_Qbit_temp = 2**(j)
    else  :
        n_Qbit_temp = n_Qbit


    for i in range( 0, L, 2**(j+1)):

     list_Qubit1=[(i-i_nq)%L  for i_nq in reversed(range(n_Qbit_temp))]
     list_Qubit2=[ (i-i_nq+2**(j))%L    for i_nq in reversed(range(n_Qbit_temp))  ]
     list_Qubit=list_Qubit1+list_Qubit2
     list_Qubit=[ -i+ i_start  for i in list_Qubit]

     #print("1_Qubit", list_Qubit, "site=",i, "layer=", j,"max_bond=", 2**(j+1))
     if qmera_type=="brickwall":
      n_apply, list_u3=general_unitary_qmera_list(psi, list_Qubit, n_apply, list_u3, in_depth_tmp, L,data_type,seed_val, depth_mera)
     if qmera_type=="pollmann":
      n_apply, list_u3=general_unitary_qmera_list_pollmann(psi, list_Qubit, n_apply, list_u3, in_depth_tmp, L,data_type,seed_val, depth_mera)

    for i in range( 0, L, 2**(j+1)):

     if 2**(j+1)<L:
      list_Qubit1=[ (i-i_nq+2**(j))%L  for i_nq in reversed(range(n_Qbit_temp))]
      list_Qubit2=[ (i-i_nq+2**(j+1))%L  for i_nq in reversed(range(n_Qbit_temp))]
      list_Qubit=list_Qubit1+list_Qubit2
      list_Qubit=[ -i + i_start  for i in list_Qubit]

      #print("2_Qubit", list_Qubit, "site=", i, "layer=",j)

      if qmera_type=="brickwall":
       n_apply, list_u3=general_unitary_qmera_list(psi, list_Qubit, n_apply, list_u3, in_depth_tmp, L,data_type,seed_val, depth_mera)
      if qmera_type=="pollmann":
       n_apply, list_u3=general_unitary_qmera_list_pollmann(psi, list_Qubit, n_apply, list_u3, in_depth_tmp, L,data_type,seed_val, depth_mera)


   return n_apply, list_u3





def   brickwall_circuit( L=16, in_depth=4, n_Qbit=4, depth_total=3, data_type='float64', qmps_structure="brickwall" , n_q_mera=2,seed_init=0, internal_mera="brickwall"):

   seed_val=seed_init
   list_u3=[]
   n_apply=0
   psi = qtn.MPS_computational_state('0' * L)
   for t in psi:
     t.modify(left_inds=())

   for t in  range(L):
     psi[t].modify(tags=[f"I{t}", "MPS"])

 


   for r in range(depth_total):
    Qubit_ara=r
    if r%2==0:
     for i in range(0, L-n_Qbit+1, n_Qbit):

       #print ("r", r, "i", i)
       if qmps_structure=="brickwall":
        n_apply, list_u3=range_unitary(psi, i, n_apply, list_u3, in_depth, n_Qbit-1, data_type, seed_val, Qubit_ara)
       if qmps_structure=="pollmann":
        n_apply, list_u3=range_unitary_pollmann(psi, i, n_apply, list_u3, in_depth, n_Qbit-1, data_type, seed_val, Qubit_ara)
       if qmps_structure=="mera":
         n_apply, list_u3=MERA_internal( psi, i, n_Qbit, in_depth, n_apply,list_u3, n_q_mera, Qubit_ara, qmera_type=internal_mera)


    if r%2!=0:
     for i in range(n_Qbit//2, L-n_Qbit, n_Qbit):
       #print ("r", r, "i", i)
       if qmps_structure=="brickwall":
        n_apply, list_u3=range_unitary(psi, i, n_apply, list_u3, in_depth, n_Qbit-1, data_type, seed_val, Qubit_ara)
       if qmps_structure=="pollmann":
        n_apply, list_u3=range_unitary_pollmann(psi, i, n_apply, list_u3, in_depth, n_Qbit-1, data_type, seed_val, Qubit_ara)
       if qmps_structure=="mera":
         n_apply, list_u3=MERA_internal( psi, i, n_Qbit, in_depth, n_apply,list_u3,n_q_mera, Qubit_ara, qmera_type=internal_mera)



   #print (psi, L, type(list_u3), list_u3)
   psi=convert_wave_function_to_real(psi, L, list_u3)

   return psi, list_u3

def pollmann_circuit( L=16, in_depth=4, n_Qbit=4, depth_total=3, data_type='float64', qmps_structure="pollmann" , n_q_mera=2, seed_init=60):

   seed_val=seed_init
   list_u3=[]
   n_apply=0
   psi = qtn.MPS_computational_state('0' * L)
   for t in psi:
     t.modify(left_inds=())

   for t in  range(L):
     psi[t].modify(tags=[f"I{t}", "MPS"])

   for r in range(depth_total):
    Qubit_ara=r
    for i in range(0, L-n_Qbit+1, n_Qbit//2):
       #print("i", i)
       if qmps_structure=="pollmann":
        n_apply, list_u3=range_unitary_pollmann(psi, i, n_apply, list_u3, in_depth, n_Qbit-1, data_type, seed_val, Qubit_ara)
       if qmps_structure=="brickwall":
        n_apply, list_u3=range_unitary(psi, i, n_apply, list_u3, in_depth, n_Qbit-1, data_type, seed_val, Qubit_ara)
       if qmps_structure=="mera":
         n_apply, list_u3=MERA_internal( psi, i, n_Qbit, in_depth, n_apply,list_u3, n_q_mera, Qubit_ara)

   #print (psi, L, type(list_u3), list_u3)
   psi=convert_wave_function_to_real(psi, L, list_u3)

   return psi, list_u3


def Smart_gate(qmps):
  dic_mps=load_from_disk("Store/gateInfo")
  for i in dic_mps:
      try:
            qmps[i].params=dic_mps[i]
      except (KeyError, AttributeError):
                  pass  # do nothing!
  return qmps



def range_unitary_gate( circ, i_start, depth, n_Qbit, seed_val, Qubit_ara, **kwargs):

    gt=kwargs["gate_type"]
    lp=kwargs["len_parma"]
    gate_round=None
    if n_Qbit==0: depth=1

    for r in range(depth):
     if r%2==0:

      for i in range(i_start, i_start+n_Qbit, 2):
         #print("F-even", i_start, i_start+n_Qbit, i, (i + 1))
         if seed_val==0:
              params = [0.]*lp
         else:
              params = qu.randn(lp, dist='uniform', seed=i+seed_val+seed_val+Qubit_ara)
         
         circ.apply_gate( gt, *params, i, i + 1, parametrize=True, gate_round=f'P{Qubit_ara}L{i}D{r}', contract=False )

     elif r%2!=0:
      for i in range( i_start, i_start+n_Qbit-1, 2):
         if seed_val==0:
             params = [0.]*lp
         else:
             params = qu.randn(lp, dist='uniform', seed=i+seed_val+seed_val+Qubit_ara)

         circ.apply_gate( gt, *params, i+1, i + 2, parametrize=True,  gate_round=f'P{Qubit_ara}L{i}D{r}', contract=False )



def qmps_gate_f( L=16, in_depth=2, n_Qbit=3, seed_val=10, **kwargs):
   list_basis=[]
   for i in range(L):
    if i%2==0:
     list_basis.append("0")
    else:    
     list_basis.append("1")
    
#   list_basis=["1", "1"]
   print (list_basis)

   circ = qtn.Circuit( L, MPS_computational_state(list_basis))
   for i in range(0,L-n_Qbit,1):
     Qubit_ara=i+n_Qbit
     range_unitary_gate(circ, i, in_depth, n_Qbit, seed_val, Qubit_ara, **kwargs )
   return circ


def energy_gate(qmps, MPO):
   psi_h=qmps.H 
   qmps.align_(MPO, psi_h)
   E_complex=(( psi_h & MPO & qmps).contract(all, optimize='auto-hq'))
   return  autoray.do('real',  E_complex)


def auto_diff_gate(qmps,MPO, GATE, optimizer_c='L-BFGS-B'):

 tnopt_qmps= qtn.TNOptimizer(
    qmps,                      
    loss_fn=energy_gate,                    
    loss_constants={ "MPO": MPO},
    constant_tags=[ 'PSI0'],    
    tags=[GATE],
    autodiff_backend="tensorflow",   # use 'autograd' for non-compiled optimization
    optimizer='L-BFGS-B',     # the optimization algorithm
)
 return tnopt_qmps


def state_gate(qmps, pdmrg):
   return  1-abs(( pdmrg.H & qmps).contract(all, optimize='auto-hq'))


def auto_diff_stateGATE( qmps, pdmrg, GATE, optimizer_c='L-BFGS-B'):

 pdmrg=pdmrg.astype('complex128')

 tnopt_qmps= qtn.TNOptimizer(
    qmps,                      
    loss_fn=state_gate,                    
    loss_constants={ "pdmrg": pdmrg},
    constant_tags=[ 'PSI0'],    
    tags=[GATE],
    autodiff_backend="tensorflow",   # use 'autograd' for non-compiled optimization
    optimizer='L-BFGS-B',     # the optimization algorithm
)

 return tnopt_qmps



def Gate_qmps_finite( ):

 L_L=8
 U=6.0
 t=1.0
 mu=U//2
 opt="auto-hq"
 Qbit=3
 Depth=8
 D=8
 relative_error=[]
 relative_error_Q=[]
 
 
 
 
 GATE="FSIMG"
 PARAM=5
 circ=qmps_gate_f( L=L_L, in_depth=Depth, n_Qbit=Qbit, seed_val=10, gate_type=GATE, len_parma=PARAM)
 qmps=circ.psi
 print ( "Info", "L", L_L, "Qbit", Qbit, "Depth", Depth, "D", D, "U", U, "t", t, "mu", mu)
 print (  "N_gates", len(circ.gates)*3  )


 circ.psi.draw(color=['PSI0'] + [f'ROUND_lay{i}' for i in range(L_L)], iterations=600, figsize=(80, 80),  return_fig=True,node_size=700 , edge_scale=6, initial_layout='spectral', edge_alpha=0.633)
 plt.savefig('Gate.pdf')
 plt.clf()


 #MPO_origin=mpo_Fermi_Hubburd(L_L//2, U, t, mu)
 MPO_origin=MPO_ham_heis(L=L_L, j=(1.0,1.0,1.0), bz=0.0, S=0.5, cyclic=False)
 MPO_origin=MPO_origin.astype('complex128')
 #print (MPO_origin[0].data)

 

 print ( "param_init", qmps["GATE_0"].params)

 #qmps=Smart_gate(qmps)
 psi_h=qmps.H 
 qmps.align_(MPO_origin, psi_h)
 print ("E_init", ( psi_h & MPO_origin & qmps).contract(all, optimize='auto-hq').real, ( qmps.H & qmps).contract(all, optimize='auto-hq'))
 #Hubburd_correlation( qmps, L_L, opt) 

#############################################
 dmrg = DMRG2(MPO_origin, bond_dims=[10, 20, 60, 80, 100, 160, 220, 250], cutoffs=1.e-12) 
 dmrg.solve(tol=1.e-12, verbosity=0 )
 E_exact=dmrg.energy
 p_DMRG=dmrg.state
 N_particle, N_up, N_down=Hubburd_correlation( p_DMRG, L_L, opt) 
 print ("DMRG-part", N_particle, N_up, N_down)
 print( "E_exact", E_exact)  #p_DMRG.show()

 dmrg = DMRG2(MPO_origin, bond_dims=[10, 20, 60, 80, 100], cutoffs=1.e-12) 
 dmrg.solve(tol=1.e-12, verbosity=0 )
 p_DMRG=dmrg.state
 print( "E_DMRG", dmrg.energy)  #p_DMRG.show()



 tnopt_qmps=auto_diff_gate(qmps, MPO_origin, GATE, optimizer_c='L-BFGS-B')
 #tnopt_qmps=auto_diff_stateGATE(qmps, p_DMRG, optimizer_c='L-BFGS-B')


 tnopt_qmps.optimizer = 'L-BFGS-B' 
 qmps = tnopt_qmps.optimize( n=400, ftol= 2.220e-10, maxfun= 10e+9, gtol= 1e-12, eps= 1.49016e-08, maxls=400, iprint = 5, disp=True )

#SLSQP, CG
#  tnopt_qmps.optimizer = 'adam' 
#  qmps = tnopt_qmps.optimize(n=50, gtol= 1e-12,eps= 1e-08, ftol= 2.220e-12, maxfun= 10e+10)
 N_particle, N_up, N_down=Hubburd_correlation( qmps, L_L, opt) 
 print ("QMPS-part", N_particle, N_up, N_down)


#, ftol= 2.220e-10, gtol= 1e-10
#'eps': 1e-08: the absolute step size used for numerical approximation of the jacobian via forward differences.



 print("DMRG")
 dmrg = DMRG2(MPO_origin, bond_dims=[2, D], cutoffs=1e-10) 
 dmrg.solve(tol=1e-8, verbosity=0 )
 E_DMRG=dmrg.energy
 print( "DMRG", D,  E_DMRG)





 psi_h=qmps.H 
 qmps.align_(MPO_origin, psi_h)
 E_final=( psi_h & MPO_origin & qmps).contract(all, optimize='auto-hq').real
 print ("E_final", E_final,  abs((E_final-E_exact)/E_exact) )


 y=tnopt_qmps.losses[:]
 relative_error.append( abs((E_final-E_exact)/E_exact))
 relative_error_Q.append( abs((E_DMRG-E_exact)/E_exact))

 print ("error_qmps" ,relative_error, "\n", "error_dmrg", relative_error_Q)


 file = open("Data/Gate.txt", "w")
 for index in range(len(y)):
    file.write( str(index) + "  "+ str(y[index])+ "  "+ str(abs((y[index]-E_exact)/E_exact)) + "\n")
 file.close()





 circ.update_params_from(qmps)
 #psi_denc=circ.to_dense()
 #print ("psi_dense,", circ.to_dense(), "\n")
 
 #print ("E_dense,", "\n", psi_denc.H @ ham_heis(L_L) @ psi_denc, "\n", "norm",  psi_denc.H @ psi_denc)

 
 #save_to_disk(qmps, "Store/circ")
 print ( "param_final", qmps["GATE_0"].params)
 #print ("\n", "param_final", np.equal(qmps["GATE_0"].data.reshape(4,4), fsimg(*qmps["GATE_0"].params)),"\n", qmps["GATE_0"].params,"\n",qmps["GATE_0"].data,"\n",  fsimg(*qmps["GATE_0"].params), "\n")




#  print (type(circ.gates), circ.gates)

 save_to_disk(circ.gates, "Store/gates")
 list_gates=load_from_disk("Store/gates")


 list_qubits=[]
 list_params=[]
 for  i in range(len(list_gates)):

  t,theta, Zeta, chi, gamma, phi, qubit1, qubit2 =list_gates[i]
  #t,theta1, phi1, lamda1,theta2, phi2, lamda2,theta3, phi3, lamda3,theta4, phi4, lamda4,t1, t2, t3,qubit1, qubit2=list_gates[i]

  list_params.append((theta, Zeta, chi, gamma, phi))
#  list_params.append((theta1, phi1, lamda1,theta2, phi2, lamda2,theta3, phi3, lamda3,theta4, phi4, lamda4,t1, t2, t3))
  list_qubits.append((qubit1, qubit2))


 save_to_disk(list_params, "Store/list_params")
 save_to_disk(list_qubits, "Store/list_qubits")



 tag_list=list(qmps.tags)
 tag_final=[]
 for i_index in tag_list: 
     if i_index.startswith('R'): tag_final.append(i_index)

 dic_mps= {      i : qmps[i].params   for   i   in   tag_final }
 save_to_disk(dic_mps, "Store/gateInfo")






def range_unitary_gate_inf( circ, where_gates, depth, seed_val, Qubit_ara,block_size,list_sharedtags,**kwargs):

    gt=kwargs["gate_type"]
    lp=kwargs["len_parma"]
    gate_round=None
    lis_in_tags_block=[]
    for r in range(depth):
     try:
      if r%2==0:

       for i in range(0,len(where_gates), 2):
          if seed_val==0:
               params = [0.]*lp
               params =[0.,0,0,   0,-pi/2,0,   0,pi/2,0,   0,0,0, -pi/2,pi/2,-pi/2  ]
          else:
               params = qu.randn(lp, dist='uniform', seed=i+seed_val)

          #print ("where_e", where_gates[i], where_gates[i + 1], f'P{Qubit_ara%block_size}L{where_gates[i]}D{r}')
          circ.apply_gate( gt, *params, where_gates[i], where_gates[i + 1], parametrize=True, gate_round=f'P{Qubit_ara%block_size}L{where_gates[i]}D{r}', contract=False )
          list_sharedtags.add (f'ROUND_P{Qubit_ara%block_size}L{where_gates[i]}D{r}')
          lis_in_tags_block.append (f'ROUND_P{Qubit_ara%block_size}L{where_gates[i]}D{r}')
      elif r%2!=0:
       for i in range(0,len(where_gates), 2):
          if seed_val==0:
               params = [0.]*lp
               params =[0.,0,0,   0,-pi/2,0,   0,pi/2,0,   0,0,0, -pi/2,pi/2,-pi/2  ]
          else:
               params = qu.randn(lp, dist='uniform', seed=i+seed_val)

          circ.apply_gate( gt, *params, where_gates[i+1], where_gates[i + 2], parametrize=True,  gate_round=f'P{Qubit_ara%block_size}L{where_gates[i+1]}D{r}', contract=False )
          #print ("where_o", where_gates[i+1], where_gates[i + 2], f'P{Qubit_ara%block_size}L{where_gates[i+1]}D{r}')
          list_sharedtags.add (f'ROUND_P{Qubit_ara%block_size}L{where_gates[i+1]}D{r}')
          lis_in_tags_block.append (f'ROUND_P{Qubit_ara%block_size}L{where_gates[i+1]}D{r}')

     except:    
         pass

    return list_sharedtags, lis_in_tags_block






def range_unitary_inf( psi, where_gates, depth, seed_val, Qubit_ara,block_size,list_sharedtags, list_u3, n_apply, data_type="float64", val_intense=0.):

    gate_round=None

    for r in range(depth):
     try:
       if r%2==0:
        for i in range(0,len(where_gates), 2):
          if seed_val==0:
             G=np.eye(4, dtype=data_type).reshape((2, 2, 2, 2))
             Grand=qu.randn((2, 2, 2, 2),dtype=data_type, dist="uniform", seed=i+seed_val)
             G=G+Grand*val_intense
          else:
             G = qu.randn((2, 2, 2, 2),dtype=data_type, dist="uniform", seed=i+seed_val)


          #print ("where_e", where_gates[i], where_gates[i + 1], f'P{Qubit_ara%block_size}L{where_gates[i]}D{r}')
          psi.gate_(G, (where_gates[i], where_gates[i + 1]), tags={f'P{Qubit_ara%block_size}L{where_gates[i]}D{r}', f'G{n_apply}'})
          list_sharedtags.add (f'P{Qubit_ara%block_size}L{where_gates[i]}D{r}')
          list_u3.append(f'G{n_apply}')
          n_apply+=1

       elif r%2!=0:
        for i in range(0,len(where_gates), 2):
          if seed_val==0:
             G=np.eye(4, dtype=data_type).reshape((2, 2, 2, 2))
             Grand=qu.randn((2, 2, 2, 2),dtype=data_type, dist="uniform", seed=i+seed_val)
             G=G+Grand*val_intense
          else:
             G = qu.randn((2, 2, 2, 2),dtype=data_type, dist="uniform", seed=i+seed_val)

          #print ("where_o", where_gates[i+1], where_gates[i + 2], f'P{Qubit_ara%block_size}L{where_gates[i+1]}D{r}')
          psi.gate_(G, (where_gates[i+1], where_gates[i + 2]), tags={f'P{Qubit_ara%block_size}L{where_gates[i+1]}D{r}', f'G{n_apply}'})
          list_sharedtags.add (f'P{Qubit_ara%block_size}L{where_gates[i+1]}D{r}')
          list_u3.append(f'G{n_apply}')
          n_apply+=1

     except:    
         pass

    return list_sharedtags, list_u3, n_apply



def range_unitary_inf_dens( psi, where_gates, depth, seed_val, Qubit_ara,block_size,list_sharedtags, list_u3, n_apply,n_Qbit ,data_type="float64", val_intense=0.):

  if seed_val==0:
     G=np.eye(2**(2*n_Qbit+2), dtype=data_type).reshape( (2,)*(2*n_Qbit+2))
     Grand=qu.randn((2,)*(2*n_Qbit+2),dtype=data_type, dist="uniform", seed=seed_val)
     G=G+Grand*val_intense
  else:
     G = qu.randn((2,)*(2*n_Qbit+2),dtype=data_type, dist="uniform", seed=seed_val)


  #print (where_gates, (*where_gates), G.shape, f'P{Qubit_ara%block_size}L0D0')
  psi.gate_(G, where_gates, tags={f'P{Qubit_ara%block_size}', f'U{n_apply}'})
  list_sharedtags.add (f'P{Qubit_ara%block_size}')
  list_u3.append(f'U{n_apply}')
  n_apply+=1

  return list_sharedtags, list_u3, n_apply



def  qmps_inf_f(L=16, block_size=4, in_depth=2, n_Qbit=3, seed_val=10):
 list_basis=[]
 list_u3=[]
 n_apply=0
 for i in range(L+n_Qbit):
  if i%2==0:
   list_basis.append("0")
  else:    
   list_basis.append("0")
  
#   list_basis=["1", "1"]

# list_basis=["0"]*n_Qbit+list_basis
 print ("init_basis", list_basis)

 psi = qtn.MPS_computational_state(list_basis)
 for t in psi:
   t.modify(left_inds=())

 for t in  range(L+n_Qbit):
   psi[t].modify(tags=[f"I{t}", "MPS"])



 where_gates=[ i   for i in range (n_Qbit)]
 list_sharedtags=qu.oset()
 for j in range( L//block_size):
  for i in range( 0, block_size, 1):
  
    Qubit_ara=j*block_size+i
    where_gates_update=where_gates+[j*block_size+i+n_Qbit]
    #print (where_gates_update, "block", j, "wher_in_block", i, "position", j*block_size+i)
    list_sharedtags, list_u3, n_apply=range_unitary_inf(psi, where_gates_update, in_depth, seed_val,Qubit_ara, block_size, list_sharedtags,list_u3, n_apply)

 list_sharedtags=list(list_sharedtags)
 #psi=convert_wave_function_to_real(psi, L+n_Qbit, list_u3)
 return psi, list_sharedtags, list_u3






def  mps_inf_f(L=16, block_size=4, in_depth=2, n_Qbit=3, seed_val=10):
 list_basis=[]
 list_u3=[]
 n_apply=0
 for i in range(L+n_Qbit):
  if i%2==0:
   list_basis.append("0")
  else:    
   list_basis.append("0")
  
#   list_basis=["1", "1"]

# list_basis=["0"]*n_Qbit+list_basis
 print ("init_basis", list_basis)

 psi = qtn.MPS_computational_state(list_basis)
 for t in psi:
   t.modify(left_inds=())

 for t in  range(L+n_Qbit):
   psi[t].modify(tags=[f"I{t}", "MPS"])



 where_gates=[ i   for i in range (n_Qbit)]
 list_sharedtags=qu.oset()
 for j in range( L//block_size):
  for i in range( 0, block_size, 1):
  
    Qubit_ara=j*block_size+i
    where_gates_update=where_gates+[j*block_size+i+n_Qbit]
    #print (where_gates_update, "block", j, "wher_in_block", i, "position", j*block_size+i)
    list_sharedtags, list_u3, n_apply=range_unitary_inf_dens(psi, where_gates_update, in_depth, seed_val,Qubit_ara, block_size, list_sharedtags,list_u3, n_apply, n_Qbit)

 list_sharedtags=list(list_sharedtags)
 #psi=convert_wave_function_to_real(psi, L+n_Qbit, list_u3)
 return psi, list_sharedtags, list_u3




def energy_infgate(qmps, MPO, L, l_mpo, Qbit):
   psi_h=qmps.H
   psi_h=psi_h.reindex({ f"k{L+Qbit-i-1}":f"b{L+Qbit-i-1}"  for i in range(l_mpo) } )
   E_complex=(( psi_h & MPO & qmps).contract(all, optimize='auto-hq'))
   return  autoray.do('real',  E_complex)


def auto_diff_infgate(qmps, MPO, GATE, sharedtags,  L,
 l_mpo, Qbit, Qbitoptimizer_c='L-BFGS-B'):

 MPO=MPO.astype('complex128')
 tnopt_qmps= qtn.TNOptimizer(
    qmps,                      
    loss_fn=energy_infgate,                    
    loss_constants={ "MPO": MPO },
    constant_tags=[],
    loss_kwargs={"L":L, "l_mpo":l_mpo, "Qbit":Qbit},    
    tags=sharedtags,
    shared_tags=sharedtags,
    autodiff_backend="tensorflow",  
    optimizer='L-BFGS-B',     
)
 return tnopt_qmps



def energy_infqmps(qmps, MPO, L, l_mpo, Qbit):
   psi_h=qmps.H
   psi_h=psi_h.reindex({ f"k{L+Qbit-i-1}":f"b{L+Qbit-i-1}"  for i in range(l_mpo) } )
   E_complex=(( psi_h & MPO & qmps).contract(all, optimize='auto-hq'))
   return  E_complex

def auto_diff_infmps(qmps, MPO, GATE, sharedtags,  L,
 l_mpo, Qbit, Qbitoptimizer_c='L-BFGS-B'):
 tnopt_qmps= qtn.TNOptimizer(
    qmps,                      
    loss_fn=energy_infqmps,
    norm_fn=norm_f_energy,                    
    loss_constants={ "MPO": MPO },
    constant_tags=[],
    loss_kwargs={"L":L, "l_mpo":l_mpo, "Qbit":Qbit},    
    tags=sharedtags,
    shared_tags=sharedtags,
    autodiff_backend="torch",  
    optimizer='L-BFGS-B',     
)
 return tnopt_qmps


def state_gate(qmps, pdmrg):
   return  1-abs(( pdmrg.H & qmps).contract(all, optimize='auto-hq'))

def auto_diff_stateGATE_inf( qmps, pdmrg, sharedtags, optimizer_c='L-BFGS-B'):

 pdmrg=pdmrg.astype('complex128')

 tnopt_qmps= qtn.TNOptimizer(
    qmps,                      
    loss_fn=state_gate,                    
    loss_constants={ "pdmrg": pdmrg},
    tags=sharedtags,
    shared_tags=sharedtags,
    autodiff_backend="tensorflow",   # use 'autograd' for non-compiled optimization
    optimizer='L-BFGS-B',     # the optimization algorithm
)

 return tnopt_qmps


def qmps_gate_f_inf( L=16, block_size=4, in_depth=2, n_Qbit=3, seed_val=10,b_s_g=2, **kwargs):

   list_basis=[]
   for i in range(L):
    if i%b_s_g==0:
       list_basis.append("0")
    if i%b_s_g==1:
       list_basis.append("1")
    if i%b_s_g==2:
       list_basis.append("1")
    if i%b_s_g==3:
       list_basis.append("0")



#   list_basis=[]
#   for i in range(L):
#    if i%b_s_g==0:
#       list_basis.append("1")
#    if i%b_s_g==1:
#       list_basis.append("0")
#    if i%b_s_g==2:
#       list_basis.append("1")
#    if i%b_s_g==3:
#       list_basis.append("1")
#    if i%b_s_g==4:
#       list_basis.append("1")
#    if i%b_s_g==5:
#       list_basis.append("0")
#     if i%b_s_g==6:
#        list_basis.append("1")
#     if i%b_s_g==7:
#        list_basis.append("1")





   list_basis=["0"]*n_Qbit+list_basis
   print ("init_basis", list_basis)
   
   circ = qtn.Circuit( L+n_Qbit, MPS_computational_state(list_basis))
   where_gates=[ i   for i in range (n_Qbit)]
   list_sharedtags=qu.oset()
   list_tag_block=[]

   for j in range( L):
#   for j in range( L//block_size):
    #for i in range( 0, block_size, 1):
#      Qubit_ara=j*block_size+i
#      where_gates_update=where_gates+[j*block_size+i+n_Qbit]
      Qubit_ara=j
      where_gates_update=where_gates+[j+n_Qbit]

      #print (where_gates_update, "block", j, "wher_in_block", i, "position", j*block_size+i)
      list_sharedtags, list_tempo=range_unitary_gate_inf(circ, where_gates_update, in_depth, seed_val,Qubit_ara, block_size, list_sharedtags, **kwargs )
      list_tag_block.append(list_tempo)


   #print (list_sharedtags)
   return circ, list(list_sharedtags), list_tag_block



def Smart_infgate(qmps):
  dic_mps=load_from_disk("Store/infgateInfo")
  for i in dic_mps:
      #print (i, dic_mps[i])
      t=qmps[i]
      t = t if isinstance(t, tuple) else [t]
      for j in range(len(t)):
       try:  
             #print ("Reza",j,i)
             if len(t)==1:
              qmps[i].params=dic_mps[i]
             else: 
              qmps[i][j].params=dic_mps[i]

       except (KeyError, AttributeError):
                   pass  # do nothing!
  return qmps




def Smart_guess_infmps(qmps):

  qmpsGuess=load_from_disk("Store/mpsguess")
  tag_list=list(qmpsGuess.tags)
  tag_final=[]

  for i_index in tag_list: 
      if i_index.startswith('P'): tag_final.append(i_index)

  #print (tag_final)
  for i in tag_final:

      t=qmps[i]
      t = t if isinstance(t, tuple) else [t]
      #print (i, t)
      for j in range(len(t)):
       try:
             if len(t)==1:
                 qmps[i].modify(data=qmpsGuess[i].data)
             else: 
                 qmps[i][j].modify(data=qmpsGuess[i][0].data)

       except (KeyError, AttributeError):
                   pass  # do nothing!


  qmps.unitize_(method='mgs')
  return qmps





def imps( ):

 U=1.0
 t=1.0
 mu=.588


 Qbit=2
 D=16
 L_L=8                        
 b_s=2                          #----ABCABCABC----
 l_mpo=2 * ( b_s//2 + 1  )
 GATE="FSIMG"
 PARAM=5


 opt="auto-hq"
 relative_error=[]
 relative_error_Q=[]


 qmps, list_sharedtags, list_tags=mps_inf_f(L=L_L, block_size=b_s, in_depth=1, n_Qbit=Qbit, seed_val=60)
 qmps.unitize_(method='mgs')


 #print (list_sharedtags, "\n", list_tags)
 
 print ( "Info", "L", L_L, "Qbit", Qbit,"b_s", b_s, "D", D, "U", U, "t", t, "mu", mu, "GATE", GATE)


 qmps.draw(color=[i  for i in list_sharedtags ], iterations=600, figsize=(40, 40),return_fig=True,show_inds=False, show_tags=False, show_scalars=False,node_size=1000,edge_scale=6,initial_layout='spectral', edge_alpha=0.633)
 plt.savefig('iMPS.pdf')
 plt.clf()


 L_dmrg=96
 #E_exact=DMRG_test( L_dmrg, U, t, mu)
 E_exact=DMRG_test_1( L_dmrg, U, t, mu)

 #E_exact=  -0.414754
 #E_exact=-4.0/pi
 #E_exact=-0.443147188
 print("E_exact=", E_exact)

 #qmps=Smart_guess_infmps(qmps)

 #MPO_origin=mpo_Fermi_Hubburd_inf( l_mpo//2, U, t, mu)
 MPO_origin=mpo_Fermi_Hubburd_inf_1( l_mpo//2, U, t, mu)

 MPO_origin.reindex_({ f"k{i}":f"k{L_L+Qbit-i-1}"  for i in range(l_mpo) } )
 MPO_origin.reindex_({ f"b{i}":f"b{L_L+Qbit-i-1}"  for i in range(l_mpo) } )



 psi_h=qmps.H 
 psi_h.reindex_({ f"k{L_L+Qbit-i-1}":f"b{L_L+Qbit-i-1}"  for i in range(l_mpo) } )
 print  ("E_init", ( psi_h & MPO_origin & qmps).contract(all, optimize=opt).real, ( qmps.H & qmps).contract(all, optimize=opt))
 N_par, Spin_up, Spin_down=part_inf_Hubburd(qmps, l_mpo,L_L, Qbit)
 
#############################################

#energy
 tnopt_qmps=auto_diff_infmps( qmps, MPO_origin, GATE, list_sharedtags, L_L, l_mpo, Qbit)


 tnopt_qmps.optimizer = 'L-BFGS-B' 
 qmps = tnopt_qmps.optimize( n=40, ftol= 2.220e-10, maxfun= 10e+9, gtol= 1e-12, eps= 1.49016e-08, maxls=400, iprint = 0, disp=False )

 save_to_disk(qmps, "Store/mps")
 save_to_disk(qmps, "Store/mpsguess")
 
###########################################################################



 energy_f_local(qmps,l_mpo, L_L, Qbit,MPO_origin,mu)
 #local_particle_info(qmps, L_L, Qbit)
 #Hubburd_correlation_inf(qmps, L_L, Qbit, b_s, opt,shift_AB=b_s)




 y=tnopt_qmps.losses[:]
 file = open("Data/infqmps.txt", "w")
 for index in range(len(y)):
    E_val=(y[index]+N_par*mu)  / ( (l_mpo/2) - 1.)
    file.write( str(index) + "  "+ str(y[index])+ "  "+ str(abs((E_val-E_exact)/E_exact)) + "\n")
 file.close()







def  Gate_qmps_infinit( ):

 U=4.0
 t=1.0
 mu=U/2.
 #mu=0

 Qbit=2
 Depth=4
 D=8
 L_L=120                       
 b_s=4                          #----ABCABCABC----
 l_mpo=2 * ( b_s//2 + 1  )
 GATE="FSIMG"
 PARAM=5

 GATE="SU4"
 PARAM=15


 opt="auto-hq"
 relative_error=[]
 relative_error_Q=[]

 circ, list_sharedtags, list_tag_block=qmps_gate_f_inf( L=L_L, block_size=b_s, in_depth=Depth, n_Qbit=Qbit, seed_val=10, b_s_g=b_s, gate_type=GATE, len_parma=PARAM)
 qmps=circ.psi



 #print (list_sharedtags, "\n", list_tag_block)
 
 print ( "Info", "L", L_L, "Qbit", Qbit,"b_s", b_s, "Depth", Depth, "D", D, "U", U, "t", t, "mu", mu, "GATE", GATE)
 print (  "N_gates", len(circ.gates)*3  )


 qmps.draw(color=[i  for i in list_sharedtags ], iterations=600, figsize=(40, 80),return_fig=True,node_size=1200,edge_scale=6,initial_layout='spectral', edge_alpha=0.633)
 plt.savefig('Gate.pdf')
 plt.clf()




 #qmps=Smart_infgate(qmps)

 #MPO_origin=mpo_Fermi_Hubburd_inf( l_mpo//2, U, t, mu)
 MPO_origin=mpo_Fermi_Hubburd_inf_1( l_mpo//2, U, t, mu)

 MPO_origin.reindex_({ f"k{i}":f"k{L_L+Qbit-i-1}"  for i in range(l_mpo) } )
 MPO_origin.reindex_({ f"b{i}":f"b{L_L+Qbit-i-1}"  for i in range(l_mpo) } )



# psi_h=qmps.H 
# psi_h.reindex_({ f"k{L_L+Qbit-i-1}":f"b{L_L+Qbit-i-1}"  for i in range(l_mpo) } )
# print  ("E_init", ( psi_h & MPO_origin & qmps).contract(all, optimize=opt).real, ( qmps.H & qmps).contract(all, optimize=opt))
# N_par, Spin_up, Spin_down=part_inf_Hubburd(qmps, l_mpo,L_L, Qbit)
 
#############################################
 #mps=load_from_disk("Store/mps")
#state
 #tnopt_qmps=auto_diff_stateGATE_inf(qmps, mps, list_sharedtags, optimizer_c='L-BFGS-B')

#energy
 tnopt_qmps=auto_diff_infgate( qmps, MPO_origin, GATE, list_sharedtags, L_L, l_mpo, Qbit)

 tnopt_qmps.optimizer = 'L-BFGS-B' 
 qmps = tnopt_qmps.optimize( n=100, ftol= 2.220e-10, maxfun= 10e+9, gtol= 1e-12, eps= 1.49016e-08, maxls=400, iprint = 0, disp=False )
 
###########################################################################
 save_info_QMPS(qmps,circ, GATE ,list_tag_block)

 energy_f_local(qmps,l_mpo, L_L, Qbit,MPO_origin,mu)
 #local_particle_info(qmps, L_L, Qbit)
 #Hubburd_correlation_inf(qmps, L_L, Qbit, b_s, opt, shift_AB=1)


 #save_info_QMPS_gate(qmps,circ, GATE ,list_tag_block)
 #Build_up_U_burnin(qmps,Qbit,PARAM,GATE,L_L,depth_local=4,ancilla_qubit=2)


# qmps_u, n_ancilla=rebuil_qmps_U(Qbit,GATE,L_L,b_s,b_s_g=b_s)
#  
# MPO_origin=mpo_Fermi_Hubburd_inf_1( l_mpo//2, U, t, mu)
# MPO_origin.reindex_({ f"k{i}":f"k{L_L+Qbit+n_ancilla-i-1}"  for i in range(l_mpo) } )
# MPO_origin.reindex_({ f"b{i}":f"b{L_L+Qbit+n_ancilla-i-1}"  for i in range(l_mpo) } )
# 
# 
# energy_f_local(qmps_u,l_mpo, L_L, Qbit+n_ancilla,MPO_origin,mu)
# local_particle_info(qmps_u, L_L, Qbit+n_ancilla)
# Hubburd_correlation_inf(qmps_u, L_L, Qbit+n_ancilla, b_s, opt,shift_AB=1)


 E_exact=-0.571380
 y=tnopt_qmps.losses[:]
 file = open("Data/infqmps.txt", "w")
 for index in range(len(y)):
    E_val=(y[index]+mu*2.0)  / ( (l_mpo/2) - 1.)
    file.write( str(index) + "  "+ str(y[index])+ "  "+ str(E_val)+ "  "+ str(abs((E_val-E_exact)/E_exact)) + "\n")
 file.close()






def    Build_up_U_burnin(qmps,n_Qbit,lp,gt,L_L,depth_local=2, ancilla_qubit=2):


 print ("info", depth_local, ancilla_qubit)
 GATE=gt
 psi_h=qmps.H 
 psi_h.reindex_( {  f"k{i}":f"b{i}"  for  i  in  range(n_Qbit)  } )
 rho=psi_h & qmps
 rho=rho.contract(all, optimize='auto-hq')
 save_to_disk(rho, "Store/rho")
 rho=load_from_disk("Store/rho")
 rho.modify(tags={"rho"})

 
 
 MPO_part=mpo_particle(n_Qbit//2)
 print ("rho_particle", (rho&MPO_part)^all )
 e,v=eigh(rho.to_dense( [f"b{i}"  for  i  in  range(n_Qbit)], [f"k{i}"  for  i  in  range(n_Qbit)] ))
 print ("e_rho", e, sum(e))
 #print ("e_rho", sum(e))

 list_basis=[]
 list_sharedtags=[]
 list_basis=["0"]+["0"]+["0"]+["1"]+["0"]+["1"]
 #list_basis=["0"]+["0"]+["1"]+["0"]

 #list_basis=["0"]*ancilla_qubit
 print ("init_basis_build_up", list_basis)


 circ = qtn.Circuit( n_Qbit+ancilla_qubit, MPS_computational_state(list_basis) )
 seed_val=20
 for r in range(depth_local):
   if r%2==0:
    for i in range(0,n_Qbit+ancilla_qubit, 2):
       params = qu.randn(lp, dist='uniform', seed=i+r+seed_val)
       #print  ("even", i, i+1)
       circ.apply_gate( gt, *params, i, i + 1, parametrize=True, gate_round=f'L{i}D{r}', contract=False )
       list_sharedtags.append(f'ROUND_L{i}D{r}')

   elif r%2!=0:
    for i in range(1,n_Qbit+ancilla_qubit-1, 2):
       params = qu.randn(lp, dist='uniform', seed=i+r+seed_val)
       #print  ("odd", i, i+1)
       circ.apply_gate( gt, *params, i, i + 1, parametrize=True,  gate_round=f'L{i}D{r}', contract=False )
       list_sharedtags.append(f'ROUND_L{i}D{r}')


 #print ( circ.psi, list_sharedtags )

 v_opt=circ.psi
 psi_h=v_opt.H 
 psi_h.reindex_( {  f"k{i}":f"b{i}"  for  i  in  range(n_Qbit)  } )
 rho_1=psi_h & v_opt
 rho_1=rho_1.contract(all, optimize='auto-hq')
 #print ("circ_particle", (rho_1&MPO_part)^all )

 e,v=eigh(rho_1.to_dense( [f"b{i}"  for  i  in  range(n_Qbit)], [f"k{i}"  for  i  in  range(n_Qbit)] ))
 #print ("e_circuit", e, sum(e))


 def loss_in(v_opt, rho, n_Qbit,ancilla_qubit):
    v_opt_h=v_opt.H
    v_opt_h.reindex_({ f"k{i}":f"b{i}"  for i in range(n_Qbit) } )
    sigma=(v_opt_h & v_opt).contract(all, optimize='auto-hq')
    sigma_1=sigma*1.0

    sigma_1.reindex_({ f"k{i}":f"A{i}"  for i in range(n_Qbit) } )
    sigma_1.reindex_({ f"b{i}":f"k{i}"  for i in range(n_Qbit) } )
    sigma_1.reindex_({ f"A{i}":f"b{i}"  for i in range(n_Qbit) } )

    A=(sigma & sigma_1).contract(all, optimize='auto-hq')
    rho_1=rho*1.0

    rho_1.reindex_({ f"k{i}":f"A{i}"  for i in range(n_Qbit) } )
    rho_1.reindex_({ f"b{i}":f"k{i}"  for i in range(n_Qbit) } )
    rho_1.reindex_({ f"A{i}":f"b{i}"  for i in range(n_Qbit) } )


    B=(rho & rho_1).contract(all, optimize='auto-hq')

    sigma.reindex_({ f"k{i}":f"A{i}"  for i in range(n_Qbit) } )
    sigma.reindex_({ f"b{i}":f"k{i}"  for i in range(n_Qbit) } )
    sigma.reindex_({ f"A{i}":f"b{i}"  for i in range(n_Qbit) } )

    C=2.0*( rho & sigma).contract(all, optimize='auto-hq')
    #print (abs(A), abs(B), abs(C))
    return ( abs(A)+abs(B)-abs(C) )


 #print (v_opt['GATE_2'].params)
 print ("loss:=", loss_in(v_opt, rho, n_Qbit,ancilla_qubit) )
 def optimizer_local(v_opt,rho,n_Qbit,ancilla_qubit): 
  rho=rho.astype('complex128')
  tnopt = qtn.TNOptimizer(
      v_opt,                        # the tensor network we want to optimize
      loss_fn=loss_in,                     # the function we want to minimize
      loss_constants={'rho': rho},  # supply U to the loss function as a constant TN
      tags=list_sharedtags,              # only optimize U3 tensors
      loss_kwargs={"n_Qbit":n_Qbit, "ancilla_qubit":ancilla_qubit},    
      autodiff_backend='tensorflow',   # use 'autograd' for non-compiled optimization
      optimizer='L-BFGS-B',     # the optimization algorithm
#      optimizer='CG',     # the optimization algorithm

  )
  return tnopt

 v_opt = optimizer_local(v_opt,rho,n_Qbit,ancilla_qubit).optimize(n=600,ftol= 2.220e-10, maxfun= 10e+9, gtol= 1e-12, eps= 1.49016e-08, maxls=600, iprint = 0, disp=False)
# v_opt = optimizer_local(v_opt,rho,n_Qbit).optimize(n=100,ftol= 2.220e-10, maxfun= 10e+9, gtol= 1e-12, eps= 1.49016e-08, maxls=400, iprint = 0, disp=False)

 circ.update_params_from(v_opt)
 v_opt=circ.psi
 #print (v_opt['GATE_2'].params)
 print ( "loss:=", loss_in(v_opt, rho, n_Qbit,ancilla_qubit) )


 v_opt=circ.psi
 psi_h=v_opt.H 
 psi_h.reindex_( {  f"k{i}":f"b{i}"  for  i  in  range(n_Qbit)  } )
 rho_1=psi_h & v_opt
 rho_1=rho_1.contract(all, optimize='auto-hq')
 print ("circ_particle", (rho_1&MPO_part)^all )

 e,v=eigh(rho_1.to_dense( [f"b{i}"  for  i  in  range(n_Qbit)], [f"k{i}"  for  i  in  range(n_Qbit)] ))
 print ("e_circuit",e, sum(e))



 save_to_disk(circ.gates, "Store/U_gates")
 list_gates=load_from_disk("Store/U_gates")

 list_qubits_U=[]
 list_params_U=[]
 for  i in range(len(list_gates)):
  if  GATE=="SU4":
   t,theta1, phi1, lamda1,theta2, phi2, lamda2,theta3, phi3, lamda3,theta4, phi4, lamda4,t1, t2, t3,qubit1, qubit2=list_gates[i]
   list_params_U.append((theta1, phi1, lamda1,theta2, phi2, lamda2,theta3, phi3, lamda3,theta4, phi4, lamda4,t1, t2, t3))
   list_qubits_U.append((qubit1, qubit2))
  elif  GATE=="FSIMG":
   t , theta, Zeta, chi, gamma, phi, qubit1, qubit2 =list_gates[i]
   list_params_U.append((theta, Zeta, chi, gamma, phi))
   list_qubits_U.append((qubit1, qubit2))


 save_to_disk(list_params_U, f"Store/list_params_U{GATE}")
 save_to_disk(list_qubits_U, f"Store/list_qubits_U{GATE}")

 save_to_disk(list_basis, f"Store/list_basis_rho")
 save_to_disk(len(list_basis)-n_Qbit, f"Store/Qbit_rho")



def rebuil_qmps_U(n_Qbit,gt,L_L,block_size,b_s_g=2):
 GATE=gt

 list_basis_Q=load_from_disk( f"Store/list_basis_rho")
 n_ancilla=load_from_disk( f"Store/Qbit_rho")
 list_params_U=load_from_disk(f"Store/list_params_U{GATE}")
 list_qubits_U=load_from_disk(f"Store/list_qubits_U{GATE}")
 list_params=load_from_disk(f"Store/list_params{GATE}")
 list_qubits=load_from_disk(f"Store/list_qubits{GATE}")




 list_basis=[]
 for i in range(L_L):
  if i%b_s_g==0:
     list_basis.append("0")
  if i%b_s_g==1:
     list_basis.append("1")
  if i%b_s_g==2:
     list_basis.append("1")
  if i%b_s_g==3:
     list_basis.append("0")
 list_basis=list_basis_Q+list_basis

 
 print ("init_basis_new", list_basis)
 save_to_disk(list_basis, f"Store/list_basis_cirq")

 circ_new = qtn.Circuit( n_Qbit+n_ancilla+L_L, MPS_computational_state(list_basis))


 for i in range(len(list_params_U)):
  where=list_qubits_U[i]
  t0, t1=where
  #print ("Start",where)
  circ_new.apply_gate( gt, *list_params_U[i], t0, t1, parametrize=True,  gate_round=f'{i}', contract=False )


 for i in range(len(list_params)):
  where=list_qubits[i]
  t0, t1=where
  if t0 >= n_Qbit:
    t0=t0+n_ancilla
  if t1 >= n_Qbit:
    t1=t1+n_ancilla


  #print ("F", where, t0, t1)
  circ_new.apply_gate( gt, *list_params[i], t0, t1, parametrize=True,  gate_round=f'{i}', contract=False )


 print (  "N_gates", len(circ_new.gates)*3  )

 return  circ_new.psi, n_ancilla




def  save_info_QMPS_gate(qmps,circ, GATE ,list_tag_block):


 circ.update_params_from(qmps)

 save_to_disk(circ.gates, "Store/infgates")
 list_gates=load_from_disk("Store/infgates")

 list_qubits=[]
 list_params=[]
 for  i in range(len(list_gates)):
  if  GATE=="SU4":
   t,theta1, phi1, lamda1,theta2, phi2, lamda2,theta3, phi3, lamda3,theta4, phi4, lamda4,t1, t2, t3,qubit1, qubit2=list_gates[i]
   list_params.append((theta1, phi1, lamda1,theta2, phi2, lamda2,theta3, phi3, lamda3,theta4, phi4, lamda4,t1, t2, t3))
   list_qubits.append((qubit1, qubit2))
  elif  GATE=="FSIMG":
   t , theta, Zeta, chi, gamma, phi, qubit1, qubit2 =list_gates[i]
   list_params.append((theta, Zeta, chi, gamma, phi))
   list_qubits.append((qubit1, qubit2))


 save_to_disk(list_params, f"Store/list_params{GATE}")
 save_to_disk(list_qubits, f"Store/list_qubits{GATE}")
 save_to_disk(list_tag_block, "Store/list_tag_block")









def  save_info_QMPS(qmps,circ, GATE ,list_tag_block):

 tag_list=list(qmps.tags)
 tag_final=[]
 for i_index in tag_list: 
     if i_index.startswith('R'): tag_final.append(i_index)


 dic_mps={}
 for   i   in   tag_final:
   t = qmps[i]
   t = t if isinstance(t, tuple) else [t]
   dic_mps[i] = t[0].params
 save_to_disk(dic_mps, "Store/infgateInfo")






def   energy_f_local(qmps,l_mpo, L_L, Qbit,MPO_origin,mu):

 opt="auto-hq"
 psi_h=qmps.H 
 psi_h.reindex_({ f"k{L_L+Qbit-i-1}":f"b{L_L+Qbit-i-1}"  for i in range(l_mpo) } )
 E_f=( psi_h & MPO_origin & qmps).contract(all, optimize=opt).real
 N_par, Spin_up, Spin_down=part_inf_Hubburd(qmps, l_mpo, L_L, Qbit)
 print  ("E_f", E_f, (E_f+N_par*mu)  / ( (l_mpo/2) - 1.) )
  




def  local_particle_info(qmps, L_L, Qbit):

 Local_order=[]
 for i in range( L_L+Qbit):

  MPO_I=MPO_identity(L=L_L+Qbit, phys_dim=2)
  W = np.zeros([ 1, 1, 2, 2])
  Z = qu.pauli('Z')
  X = qu.pauli('X')
  Y = qu.pauli('Y')
  I = qu.pauli('I')
  S_up=(X+1.0j*Y)*(0.5)
  S_down=(X-1.0j*Y)*(0.5)
  W[ 0, 0,:,:]=S_up@S_down
  if i==L_L+Qbit-1:
   W = np.zeros([ 1, 2, 2])
   W[ 0,:,:]=S_up@S_down
  if i==0:
   W = np.zeros([ 1, 2, 2])
   W[ 0,:,:]=S_up@S_down
  
  MPO_I[i].modify(data=W)
  psi_h=qmps.H
  qmps.align_(MPO_I, psi_h)
  E_final=( psi_h & MPO_I & qmps).contract(all, optimize='auto-hq').real
  print ("i", i, "X", E_final )
  Local_order.append(E_final)

  file = open("Data/infTI.txt", "w")
  for index in range(0, len(Local_order), 2):
     file.write( str(index) + "  "+ str(Local_order[index])+ "  "+ str(abs((Local_order[index]-Local_order[-1])/Local_order[-1])) + "\n")
  file.close()



def  part_inf_Hubburd_1(qmps, l_mpo,L_L, Qbit):


 psi_h=qmps.H
 mpo_spin_up, mpo_spin_down=mpo_spin_inf(l_mpo//2)
 mpo_part=mpo_particle_inf(l_mpo//2)

 mpo_spin_up.reindex_({ f"k{i}":f"k{L_L+Qbit-i-1}"  for i in range(l_mpo) } )
 mpo_spin_up.reindex_({ f"b{i}":f"b{L_L+Qbit-i-1}"  for i in range(l_mpo) } )
 mpo_spin_down.reindex_({ f"k{i}":f"k{L_L+Qbit-i-1}"  for i in range(l_mpo) } )
 mpo_spin_down.reindex_({ f"b{i}":f"b{L_L+Qbit-i-1}"  for i in range(l_mpo) } )
 mpo_part.reindex_({ f"k{i}":f"k{L_L+Qbit-i-1}"  for i in range(l_mpo) } )
 mpo_part.reindex_({ f"b{i}":f"b{L_L+Qbit-i-1}"  for i in range(l_mpo) } )

 psi_h.reindex_({ f"k{L_L+Qbit-i-1}":f"b{L_L+Qbit-i-1}"  for i in range(l_mpo) } )
 Spin_up=( psi_h & mpo_spin_up & qmps).contract(all, optimize='auto-hq').real
 print ("Spin_up", Spin_up )
 Spin_down=( psi_h & mpo_spin_down & qmps).contract(all, optimize='auto-hq').real
 print ("Spin_down", Spin_down )
 N_par=( psi_h & mpo_part & qmps).contract(all, optimize='auto-hq').real
 print ("Part", N_par, "dopping", N_par/((l_mpo//2)-1) )

 return N_par, Spin_up, Spin_down





def  part_inf_Hubburd(qmps, l_mpo,L_L, Qbit):


 psi_h=qmps.H
 mpo_spin_up, mpo_spin_down=mpo_spin_inf(l_mpo//2)
 mpo_part=mpo_particle_inf(l_mpo//2)

 mpo_spin_up.reindex_({ f"k{i}":f"k{L_L+Qbit-i-1}"  for i in range(l_mpo) } )
 mpo_spin_up.reindex_({ f"b{i}":f"b{L_L+Qbit-i-1}"  for i in range(l_mpo) } )
 mpo_spin_down.reindex_({ f"k{i}":f"k{L_L+Qbit-i-1}"  for i in range(l_mpo) } )
 mpo_spin_down.reindex_({ f"b{i}":f"b{L_L+Qbit-i-1}"  for i in range(l_mpo) } )
 mpo_part.reindex_({ f"k{i}":f"k{L_L+Qbit-i-1}"  for i in range(l_mpo) } )
 mpo_part.reindex_({ f"b{i}":f"b{L_L+Qbit-i-1}"  for i in range(l_mpo) } )

 psi_h.reindex_({ f"k{L_L+Qbit-i-1}":f"b{L_L+Qbit-i-1}"  for i in range(l_mpo) } )
 Spin_up=( psi_h & mpo_spin_up & qmps).contract(all, optimize='auto-hq').real
 print ("Spin_up", Spin_up )
 Spin_down=( psi_h & mpo_spin_down & qmps).contract(all, optimize='auto-hq').real
 print ("Spin_down", Spin_down )
 N_par=( psi_h & mpo_part & qmps).contract(all, optimize='auto-hq').real
 print ("Part", N_par, "dopping", ((l_mpo//2)-1)/N_par )

 return N_par, Spin_up, Spin_down








def MPO_ham_ITF():

 MPO_I=MPO_identity(L=3, phys_dim=2)
 MPO_II=MPO_identity(L=3, phys_dim=2)
 MPO_III=MPO_identity(L=3, phys_dim=2)
 MPO_Z=MPO_identity(L=3, phys_dim=2)
 MPO_Z1=MPO_identity(L=3, phys_dim=2)

 Wr = np.zeros([ 1, 2, 2])
 Wl = np.zeros([ 1, 2, 2])
 W = np.zeros([ 1, 1, 2, 2])
 Wl[ 0,:,:]=pauli('X')/2.
 W[ 0,:,:]=pauli('X')
 Wr[ 0,:,:]=pauli('X')/2.
 MPO_I[0].modify(data=Wl)
 MPO_II[1].modify(data=W)
 MPO_III[2].modify(data=Wr)

 Wr = np.zeros([ 1, 2, 2])
 Wl = np.zeros([ 1, 2, 2])
 W = np.zeros([ 1, 1, 2, 2])
 Wl[ 0,:,:]=pauli('Z')
 W[ 0,:,:]=pauli('Z')
 Wr[ 0,:,:]=pauli('Z')

 MPO_Z[0].modify(data=Wl)
 MPO_Z[1].modify(data=W)
 MPO_Z1[1].modify(data=W)
 MPO_Z1[2].modify(data=Wr)
 MPO_origin=MPO_Z+MPO_Z1+MPO_I+MPO_II+MPO_III
 return MPO_origin









def get_parameters(U_unitary):
 x_r, rotation_val=polar(U_unitary[0][0])   
 #print("XR", x_r*exp(+1j *rotation_val), U_unitary[0][0])
 U_new=U_unitary*(1./exp(+1j * rotation_val))
 #print(U_new, "U_new")
 teta=(2*acos(U_new[0][0])).real
 Lambda=log(  -U_new[0][1] / sin(teta/2) ).imag       
 phi=log(  (U_new[1][0]) / sin(teta/2) ).imag
 return    (teta, phi, Lambda, exp(+1j *rotation_val)) 


def get_unitary_form(teta, phi, Lambda):
 return     numpy.array([[ cos( teta/2 ),-exp(+1j * Lambda )*sin(teta/2 ) ],[exp(+1j * phi )*sin(teta/2 ), exp(+1j * (Lambda+phi) )*cos( teta/2 )]] )




 #print (  qmps["GATE_1"].params,  qmps["GATE_5"].params)
 #print (  type(qmps["GATE_1"].params), *qmps["GATE_1"].params )

#  for i in list_sharedtags:
#    #print (i) 
#    t=qmps[i]
#    target=t[0]
#    for j in t:
#       #print ( np.array_equal(j.data, target.data) )
       
#  print ( np.array_equal(qmps["G4"].data, qmps["G12"].data) )
#  print ( np.array_equal(qmps["G6"].data, qmps["G14"].data) )





def DMRG_style_gate(p_DMRG, list_tag, V_opt1, MPO_origin, L, N_iter=20):


 V_opt=V_opt1*1.0
 list_update=[]
 list_iter=[]
 iter_val=0
 Energy_list=[]

 E_1=0.001
 for j in range(N_iter):
  overlap_TN=p_DMRG.H & V_opt
  Dis_val=1-abs(  overlap_TN.contract(all,optimize='auto-hq',backend="numpy")
)
  list_update.append(Dis_val)
  #print ("DMRG", Dis_val, len(list_tag))

  inds_old=[f'k{i}' for i in range(L)]
  inds_new=[f'b{i}' for i in range(L)]
  V_opt_h=V_opt.reindex(dict(zip(inds_old, inds_new)))
  
  #E_dmrg=(V_opt_h.H & MPO_origin & V_opt).contract(all,optimize='auto-hq',backend="numpy") 
  E_dmrg=1.
  #print ( "Energy_DMRG", E_dmrg)
  Energy_list.append(E_dmrg)



  list_iter.append(iter_val)
  iter_val+=1

  for i_ter in  range( len(list_tag) ):

      overlap_TN=p_DMRG.H & V_opt
      t, = overlap_TN.select_tensors(list_tag[i_ter], which='any')
      tn_env = overlap_TN.select(list_tag[i_ter], which='!any')
      t_env = tn_env.contract(all,optimize='auto-hq', get=None, backend="numpy", output_inds=t.inds)
      t_unpa=t.unparametrize()
      E_1=(t & t_env)^all
      #print("E_1", 1-abs(E_1))
     
     
     
      t1,t2=t.inds
      t_m=t_env.transpose(t2,t1)
      t_env.modify(data=t_m.data)

    
      (u, s, v)=t_env.split(left_inds=[t1], method='svd',get='tensors', absorb=None, cutoff=-1e-10, right_inds=[t2]) 
      U_update=(v.H @ u.H) 
      U_update=U_update.transpose(t2,t1)
    
      #print(s.data)
    
    
      teta, phi, Lambda, overal_phase = get_parameters(U_update.data)
      V_opt[list_tag[i_ter]].params=teta, phi, Lambda
      #V_opt[list_tag[i_ter]].modify(data=U_update.data)
      #print ( quf.get_unitary_form(teta, phi, Lambda) * overal_phase, "\n,\n", U_update.data, overal_phase, abs(overal_phase))
  
 return list_update, list_iter, V_opt, Energy_list
 
 
 
 
 
 
def DMRG_style(p_DMRG, list_tag, V_opt1, MPO_origin,MPO_ten, L, E_exact,N_iter=20, N_iter_inner=10, opt_method='auto-hq'):

 y_list=[]
 V_opt=V_opt1*1.0
 list_update=[]
 list_iter=[]
 iter_val=0
 Energy_list=[]
 E_0=10
 E_11=100.
 E_1=100.

 pbar = tqdm.tqdm(total=N_iter, disable=not True)

 for j in range(N_iter):

  overlap_TN=p_DMRG.H & V_opt
  Dis_val=1-abs(overlap_TN.contract( all, optimize=opt_method)
)
  list_update.append(Dis_val)
  #print ("DMRG", j, Dis_val, len(list_tag))
 #backend='SCIPY'
  V_opt_h=V_opt.H 
  V_opt.align_(MPO_origin, V_opt_h)
  #path=(V_opt_h & MPO_origin & V_opt).contract(all,optimize=opt_method, backend='auto',get='path-info')
  E_0=E_11*1.0
  E_dmrg=(V_opt_h | MPO_origin | V_opt).contract(all,optimize=opt_method) 
  E_11=E_dmrg*1.0
  #print (path)
  msg = f"{Dis_val} [energy: {E_dmrg}] "
  pbar.update()
  pbar.set_description(msg)




  Energy_list.append(E_dmrg.real)
  y_list.append((E_exact-E_dmrg.real)/E_exact)

  file = open("Data/dmrgtem.txt", "w")
  for index in range(len(Energy_list)):
     file.write(str(index) + "  "+ str(Energy_list[index])+ "  "+ str(y_list[index]) + "\n")


  if (abs(E_0-E_11)/abs(E_11))<1.0e-10:
   break

  #print ( "Energy_DMRG", j, E_dmrg.real, "fidel", Dis_val, "error", (E_exact-E_dmrg.real)/E_exact, "step-error", abs(E_0-E_11)/abs(E_11))


  list_iter.append(iter_val)
  iter_val+=1

  for i_ter in  range( len(list_tag) ):
   for ii_iter in range(N_iter_inner):
#        V_opt_h=V_opt.H 
#        V_opt.align_(MPO_ten, V_opt_h)
#        for i in range(len(list_tag)):
#          V_opt_h[list_tag[i]].modify(tags="GGGGGG")


       #overlap_TN=V_opt_h | MPO_ten | V_opt
       overlap_TN=p_DMRG.H & V_opt
       t, = overlap_TN.select_tensors(list_tag[i_ter], which='any')
       tn_env = overlap_TN.select(list_tag[i_ter], which='!any')
       t_env = tn_env.contract(all,optimize=opt_method, get=None, output_inds=t.inds)
       #t_unpa=t.unparametrize()
       E_1=(t & t_env)^all
       #print(i_ter, ii_iter, "E_1", E_1)
     
     
     
       t1,t2,t3,t4=t.inds
       t_m=t_env.transpose(t3,t4,t1,t2)
       t_env.modify(data=t_m.data)

    
       (u, s, v)=t_env.split(left_inds=[t1,t2], method='svd',get='tensors', absorb=None, cutoff=-1e-10, right_inds=[t3,t4]) 
       U_update=(v.H @ u.H) *(-1.0) 
       U_update=U_update.transpose(t3,t4,t1,t2)

       #teta, phi, Lambda, overal_phase = get_parameters(U_update.data)
       #V_opt[list_tag[i_ter]].params=teta, phi, Lambda
       V_opt[list_tag[i_ter]].modify(data=U_update.data)

  
 return list_update, list_iter, V_opt, Energy_list
 
 
 
 
 
 
 
 
 
 
 
 
 
def auto_diff_function_gate(psi_opt, p_DMRG, optimizer_c='L-BFGS-B'):

 tnopt_BFGS = qtn.TNOptimizer(
    psi_opt,                        # the tensor network we want to optimize
    loss,                     # the function we want to minimize
    loss_constants={'p_DMRG': p_DMRG},  # supply U to the loss function as a constant TN
    constant_tags=['cz', 'PSI0'],    # within V we also want to keep all the CZ gates constant
    tags=['U3'],
    autodiff_backend='jax',   # use 'autograd' for non-compiled optimization
    optimizer=optimizer_c,     # the optimization algorithm
)

 return tnopt_BFGS




def line_search_armijo( Z_decent, Gamma, E1, Norm_Z, list_tag, V_opt, p_DMRG, MPO,technique):

 V_opt1=V_opt*1.0
 u_exp=[None]*len(list_tag)

 Break_loop=1
 count=0
 while Break_loop == 1:
  count+=1

  for i in range(len(list_tag)):
    if technique=="cg":
     ex_t=expm( -2*Gamma*Z_decent[i].data.reshape(4,4), herm=False )
     u_t=ex_t @ (V_opt[list_tag[i]].data.reshape(4,4))
    if technique=="sd":
     A=V_opt[list_tag[i]].data.reshape(4,4)+2*Gamma*Z_decent[i].data.reshape(4,4)
     u, s, vh=svd(A)
     u_t=u@vh
    u_exp[i]=u_t.reshape(2,2,2,2)
    V_opt1[list_tag[i]].modify(data=u_exp[i])
  

  #print (V_opt1[list_tag[2]].params, V_opt[list_tag[2]].params)
#   T_u=p_DMRG.H & V_opt1
#   E2=1.0-abs(T_u.contract(all,optimize='auto-hq', get=None, backend="numpy"))

  psi_h=V_opt1.H 
  V_opt1.align_(MPO, psi_h)
  E2=( psi_h & MPO & V_opt1).contract(all, optimize='auto-hq')


  if E1-E2 >= +Norm_Z*Gamma:
   Gamma*=2.000
  else:
   Break_loop=0
 Break_loop=1


 #print("2", Gamma)
 while Break_loop == 1:
  count+=1

  for i in range(len(list_tag)):
    if technique=="cg":
     ex_t=expm( -1.*Gamma*Z_decent[i].data.reshape(4,4), herm=False )
     u_t=ex_t @ V_opt[list_tag[i]].data.reshape(4,4)

    if technique=="sd":
     A=V_opt[list_tag[i]].data.reshape(4,4)+Gamma*Z_decent[i].data.reshape(4,4)
     u, s, vh=svd(A)
     u_t=u@vh

    u_exp[i]=u_t.reshape(2,2,2,2)*1.0
    V_opt1[list_tag[i]].modify(data=u_exp[i])


  #T_u=p_DMRG.H & V_opt1
  #E2=1.0-abs(T_u.contract(all,optimize='auto-hq', get=None, backend="numpy"))

  psi_h=V_opt1.H 
  V_opt1.align_(MPO, psi_h)
  E2=( psi_h & MPO & V_opt1).contract(all, optimize='auto-hq')



  #print("E2", E2)
  if Gamma < 1.0e-11:
   Break_loop=0
  
  if E1-E2 < (0.5)*Norm_Z*Gamma:
   Gamma*=0.50
  else:
   Break_loop=0


 #print(Gamma , count)
 return Gamma , count




def grad_list(p_DMRG, list_tag, V_opt, MPO, technique):


   list_grad=[None]*len(list_tag)

#    overlap_TN=p_DMRG.H & V_opt
#    val_const=overlap_TN.contract( all, optimize='auto-hq', get=None, backend="numpy")

   psi_h=V_opt.H 
   V_opt.align_(MPO, psi_h)
   
   for i in range(len(list_tag)):
    psi_h[list_tag[i]].modify(tags="GGGGGG")


   overlap_TN=( psi_h & MPO & V_opt)


   for i in range(len(list_tag)):

      t, = overlap_TN.select_tensors(list_tag[i], which='any')
      tn_env = overlap_TN.select(list_tag[i], which='!any')
      t_env = tn_env.contract( all, optimize='auto-hq', get=None, output_inds=t.inds)
      #t_env.modify(data=np.array(np.matrix(t_env.data.reshape(4, 4)).T).reshape(4,4))
      
      A=np.array(np.matrix(t_env.data.reshape(4, 4)).conjugate()).reshape(2,2,2,2)
      t_env.modify(data=A)
      #list_grad[i]=t_env*(cmath.sqrt(val_const/val_const.conjugate()))*(-1.0)
      list_grad[i]=t_env*2.0

   return  list_grad


def grad_direction(p_DMRG, list_tag, V_opt, list_grad,technique):

###init
  grad_direct=[None]*len(list_tag)
  for i in range(len(list_tag)):
    grad_direct[i] = qtn.Tensor(qu.rand(16).reshape(4, 4), inds=('k0', 'b0'))  



  for i in range(len(list_tag)):

    env_t=list_grad[i].data*1.0
    u_t,=V_opt.select_tensors(list_tag[i], which='any')


    if technique=="cg":
     result_0 = np.matrix(env_t.reshape(4, 4)) @ np.matrix(u_t.data.reshape(4, 4)).getH()
     result_1 = np.matrix(u_t.data.reshape(4, 4)) @ np.matrix(env_t.reshape(4, 4)*1.).getH()

    if technique=="sd":
     result_0 = np.matrix(u_t.data.reshape(4, 4)) @ np.matrix(env_t.reshape(4, 4)).getH() @ np.matrix(u_t.data.reshape(4, 4))
     result_1 = np.matrix(env_t.reshape(4, 4))


    result=(result_0-result_1)
    result=np.array( result )


    
    grad_direct[i].modify(data=result.reshape(4,4))
    grad_direct[i].modify(tags=list_grad[i].tags)
     
  return grad_direct




def CG_style( p_DMRG, list_tag, V_opt, MPO, iter_total=20, accuracy_t = +1.0e-8, show_level = 0, opt_method="global", technique="cg"):
 

  print("technique=", technique)

  E_final_list=[]
  count_list=[]
  Energy_val_0=0
  Energy_val_1=20
  relative_erro=0

  if opt_method=="global":

   for qq_iter in range(iter_total):
      #overlap_TN=p_DMRG.H & V_opt

    list_grad=grad_list( p_DMRG, list_tag, V_opt, MPO, technique)
    grad_direct=grad_direction(p_DMRG, list_tag, V_opt, list_grad, technique)


    val_norm=0
    for i in range(len(list_tag)):
      val_norm=(np.trace( np.matrix(grad_direct[i].data.reshape(4,4)*1.0) @  np.matrix(grad_direct[i].data.reshape(4,4)*1.0).getH()  ) *(0.5)).real



    if show_level==1:
     Energy_val_0=Energy_val_1*1.0
     #overlap_TN=p_DMRG.H & V_opt
#    Energy_val_1=1.-abs(overlap_TN.contract(all,optimize='auto-hq', get=None, backend="numpy"))

     psi_h=V_opt.H 
     V_opt.align_(MPO, psi_h)
     overlap_TN=psi_h & MPO & V_opt
     var_energy=overlap_TN.contract(all, optimize='auto-hq')


     Energy_val_1=var_energy.real


     Energy_val_1=Energy_val_1
     E_final_list.append(Energy_val_1)
     count_list.append(qq_iter)
     print( qq_iter, Energy_val_1,  abs((Energy_val_1-Energy_val_0)/Energy_val_1), val_norm ) 
     relative_erro= abs((Energy_val_1-Energy_val_0)/Energy_val_1)
     if abs((Energy_val_1-Energy_val_0)/Energy_val_1)<accuracy_t: 
       print ( "Optimized_break", abs((Energy_val_1-Energy_val_0)/Energy_val_1) )
       break
    V_opt_copy=V_opt*1.0
    Gamma=1.00
    Gamma , count=line_search_armijo(grad_direct, Gamma, Energy_val_1, val_norm, list_tag, V_opt_copy, p_DMRG, MPO, technique)
    #print ("Gamma", Gamma)
    for i in range(len(list_tag)):
      if technique=="cg":
       ex_t=expm( -1*Gamma*grad_direct[i].data.reshape(4,4), herm=False )
       u_t=ex_t @ (V_opt[list_tag[i]].data.reshape(4,4))
      if technique=="sd":
       A=V_opt[list_tag[i]].data.reshape(4,4)+Gamma*grad_direct[i].data.reshape(4,4)
    
       u, s, vh=svd(A)
       u_t=u@vh


      V_opt[list_tag[i]].modify(data=u_t.reshape(2,2,2,2))




  return  E_final_list,  count_list,  V_opt


def norm_f(psi):
    # method='qr' is the default but the gradient seems very unstable
    # 'mgs' is a manual modified gram-schmidt orthog routine
    return psi.unitize(method='mgs')

def loss_f(psi, p_DMRG):
    return  1-abs((psi & p_DMRG.H).contract(all, optimize='auto-hq'))

def auto_diff_function(psi, p_DMRG,optimizer_c='L-BFGS-B'):
 tnopt_BFGS = qtn.TNOptimizer(
    psi,                      
    loss_fn=loss_f,                    
    norm_fn=norm_f,
    loss_constants={'p_DMRG': p_DMRG},
    constant_tags=[ 'PSI0'],    
    tags=[],
    autodiff_backend="torch",   # use 'autograd' for non-compiled optimization
    optimizer=optimizer_c,     # the optimization algorithm
)

 return tnopt_BFGS


def norm_f_energy(qmps):
    # method='qr' is the default but the gradient seems very unstable
    # 'mgs' is a manual modified gram-schmidt orthog routine
    return qmps.unitize(method='mgs', allow_no_left_inds=True)


def energy_f(qmps, p_DMRG, MPO):
   psi_h=qmps.H 
   qmps.align_(MPO, psi_h)
   return ( psi_h & MPO & qmps).contract(all, optimize='auto-hq')

def state_qmps(qmps, p_DMRG):
   return  1-abs(( p_DMRG.H & qmps).contract(all, optimize='auto-hq'))


def auto_diff_stateQMPS(qmps, p_DMRG, optimizer_c='L-BFGS-B'):

 tnopt_qmps= qtn.TNOptimizer(
    qmps,                      
    loss_fn=state_qmps,                    
    norm_fn=norm_f_energy,
    loss_constants={'p_DMRG': p_DMRG},
    constant_tags=[ 'ket0'],    
    tags=["U"],
    autodiff_backend="torch",   # use 'autograd' for non-compiled optimization
    optimizer='L-BFGS-B',     # the optimization algorithm
)
 return tnopt_qmps




def auto_diff_energy(qmps, p_DMRG,MPO, optimizer_c='L-BFGS-B'):

 tnopt_qmps= qtn.TNOptimizer(
    qmps,                      
    loss_fn=energy_f,                    
    norm_fn=norm_f_energy,
    loss_constants={'p_DMRG': p_DMRG, "MPO": MPO},
    constant_tags=[ 'ket0'],    
    tags=["U"],
    autodiff_backend="torch",   # use 'autograd' for non-compiled optimization
    optimizer='L-BFGS-B',     # the optimization algorithm
)
 return tnopt_qmps



def Smart_guess(qmps, tag, L, Qbit , val_iden=0):

  qmpsGuess=load_from_disk("Store/qmpsGuess")
  tag_list=list(qmpsGuess.tags)
  tag_final=[]

  for i_index in tag_list: 
      if i_index.startswith('P'): tag_final.append(i_index)


  #print (tag_final)
  for i in tag_final:
      try:
            qmps[i].modify(data=qmpsGuess[i].data)
      except (KeyError, AttributeError):
                  pass  # do nothing!

  for i in tag:
   A=qmps[i].data
   G=np.eye(4, dtype="float64").reshape((2, 2, 2, 2))
   Grand=qu.randn((2, 2, 2, 2),dtype="float64", dist="uniform")
   A=A+Grand*val_iden
   qmps[i].modify(data=A)

  qmps.unitize_(method='mgs')
  return qmps



def auto_diff_qmps( ):

 relative_error=[]
 relative_error_p=[]
 relative_error_Q=[]

 U=3.0
 t=1.0
 mu=U/10

 opt="auto-hq"
 J_l=[]
 L_L=12
 Qbit=4
 Depth=2
 D=46
 depth_total_f=1

 print ( "Info_QMPS", "L", L_L, "Qbit", Qbit, "Depth", Depth, "D", D, "depth_total_f", depth_total_f, "U", U, "t", t, "mu", mu)

 qmps, tag=qmps_f( L=L_L, in_depth=Depth, n_Qbit=Qbit, data_type='float64', qmps_structure='brickwall', canon="left",  seed_init=10, internal_mera="brickwall", n_q_mera=2)
 qmps.unitize_(method='mgs')



 #print ("Defined qmps", qmps)

# qmps, tag=brickwall_circuit( L=L_L, in_depth=Depth, n_Qbit=Qbit, depth_total=depth_total_f, qmps_structure="brickwall", seed_init=0, internal_mera="brickwall",n_q_mera=2 )
# qmps.unitize_(method='mgs')

# qmps, tag=pollmann_circuit( L=L_L, in_depth=Depth, n_Qbit=Qbit, depth_total=depth_total_f, qmps_structure="pollmann", n_q_mera=10,seed_init=10 )
# qmps.unitize_(method='mgs')

 #qmps, tag=qmera_f(L=L_L,in_depth=Depth,n_Qbit=Qbit,depth_total=int(math.log2(L_L)),data_type='float64',qmera_type='brickwall',seed_init=90) 
 #qmps.unitize_(method='mgs')


 #print (qmps.ind_map)

 count_val = 0
 for key in qmps.ind_map:
  for key1 in qmps.ind_map:
   if key1 != key and len(qmps.ind_map[key])==2:
    if qmps.ind_map[key1] == qmps.ind_map[key]:
       count_val+=1
       #print (key, key1, qmps.ind_map[key1], qmps.ind_map[key]) 


 #print (qmps.inner_inds())

 print (count_val, count_val//2, "links", qmps.num_indices-(2*L_L))
 print ( "number_gates", len(tag), qmps.num_tensors-L_L,  len(tag)*6 - (qmps.num_indices-(2*L_L)) - (count_val//2)  )







 qmps.draw( color=[f"lay{i}" for i in range(100)], iterations=2500, figsize=(120, 120),  return_fig=True,node_size=3000 , edge_scale=6, initial_layout='spectral', edge_alpha=0.633)
 plt.savefig('qmps.pdf')
 plt.clf()

 #MPO_origin=mpo_Fermi_Hubburd(L_L//2, U, t, mu)
 MPO_origin=MPO_ham_heis(L=L_L, j=(1.0,1.0,1.0), bz=0.0, S=0.5, cyclic=False)
 #qmps=Smart_guess(qmps, tag, L_L, Qbit, val_iden=0.001)
 #qmps=Smart_guessmera(qmps, tag, L_L, Qbit, val_iden=0.00)

 psi_h=qmps.H 
 qmps.align_(MPO_origin, psi_h)
 print ("E_init", ( psi_h & MPO_origin & qmps).contract(all, optimize='auto-hq'))
 #save_to_disk(qmps, "Store/qmpsGuess")




#  info=( psi_h & MPO_origin & qmps).contract(all,optimize=opt, get='path-info') 
#  #opt.plot_trials()
# 
# 
#  tree = ctg.ContractionTree.from_info(info)
#  tree.plot_tent(node_scale= 1. / 1, edge_scale=1 / 1, return_fig=True, figsize=(7, 7))
#  #opt.plot_trials()
#  plt.savefig('tree.pdf',bbox_inches='tight', dpi=200)
#  #fig.savefig('tree.pdf', dpi=fig.dpi)
#  #plt.clf()
# 
#  tree.plot_ring(node_scale= 1. / 1, edge_scale=1 / 1, return_fig=True, figsize=(7, 7))
#  plt.savefig('treeRing.pdf',bbox_inches='tight', dpi=200)
#  #plt.clf()
# 
#  #print (info, opt )
# 
#  print (info, opt )





 coupling=-0.10
 for iter in range(1):
  coupling=coupling+0.1
  MPO_origin=mpo_Fermi_Hubburd(L_L//2, U, t, mu)
  #MPO_origin=MPO_ham_heis(L=L_L, j=(1.0,1.0,1.0), bz=0.0, S=0.5, cyclic=False)
  #MPO_origin=mpo_longrange_Heisenberg(L_L)
  #DMRG_test( L_L, U, t, mu)
  #print ("MPO", MPO_origin.show())
  dmrg = DMRG2(MPO_origin, bond_dims=[10, 20, 60, 80, 100, 150, 200,300], cutoffs=1.e-14) 
  dmrg.solve(tol=1.e-14, verbosity=0 )
  E_exact=dmrg.energy
  p_DMRG=dmrg.state
  #N_particle, N_up, N_down=Hubburd_correlation( p_DMRG, L_L, opt) 
  #print ("DMRG-part", N_particle, N_up, N_down)
  print( "E_dmrg", E_exact, p_DMRG.show())
  #Hubburd_correlation( p_DMRG, L_L, opt) 
  #correlation( p_DMRG, L_L, opt) 






  print("qmps")
  #qmps=load_from_disk("Store/qmps")
  tnopt_qmps=auto_diff_energy(qmps, p_DMRG, MPO_origin, optimizer_c='L-BFGS-B')
  #tnopt_qmps=auto_diff_stateQMPS(qmps, p_DMRG, optimizer_c='L-BFGS-B')



  tnopt_qmps.optimizer = 'L-BFGS-B' 
  qmps = tnopt_qmps.optimize(n=1 ,ftol= 2.220e-10, maxfun= 10e+9, gtol= 1e-12, eps= 1.49016e-08, maxls=400, iprint = 0, disp=False)


  #qmps = tnopt_qmps.optimize_basinhopping(n=100, nhop=10, temperature=0.5 ,ftol= 2.220e-10, maxfun= 10e+9, gtol= 1e-12, eps= 1.49016e-08, maxls=400, iprint = 1, disp=False)


#  tnopt_qmps.optimizer = 'TNC' 
#  qmps = tnopt_qmps.optimize( n=1000, stepmx=200, eta=0.25, maxCGit=200, accuracy=1e-12, maxfun=int(10e+8), gtol= 1e-10, disp=True)


#   print ("nevals", tnopt_qmps.nevals)
#   tnopt_qmps.optimizer = 'adam'
#   qmps = tnopt_qmps.optimize(n=0, ftol= 2.220e-10, gtol= 1e-10)
  save_to_disk(qmps, "Store/qmps")
  save_to_disk(qmps, "Store/qmpsGuess")
  #save_to_disk(qmps, "Store/qmeraGuess")

  psi_h=qmps.H 
  qmps.align_(MPO_origin, psi_h)
  E_final=( psi_h & MPO_origin & qmps).contract(all, optimize='auto-hq').real
  print ("E_final", E_final,  abs((E_final-E_exact)/E_exact) )


  #Hubburd_correlation( qmps, L_L, opt) 
  #correlation( qmps, L_L, opt) 

  #N_particle, N_up, N_down=Hubburd_correlation( qmps, L_L, opt) 
  #print ("QMPS-part", N_particle, N_up, N_down)





#  for D in range(2,44,2):
#   #print("DMRG")
#   dmrg = DMRG2(MPO_origin, bond_dims=[2,D//2, D], cutoffs=1e-10) 
#   dmrg.solve(tol=1e-8, verbosity=0 )auto_diff_qmps
#   E_DMRG=dmrg.energy
#   if D<=4:
#    print(  D, (  2*(2-1) + 4*(4-1)   ) + (2*D*D+2*D*D-1) +  (L_L-6)* (  ((3*D-1)*(D))/2   ) , abs(E_DMRG-E_exact)/abs(E_exact))
#   elif D<=8:
#    print(  D, (  2*(2-1) + 4*(4-1)+8*(8-1)   ) + (2*D*D+2*D*D-1) +  (L_L-8)* (  ((3*D-1)*(D))/2   ) , abs(E_DMRG-E_exact)/abs(E_exact))
#   elif D<=16:
#    print(  D, (  2*(2-1) + 4*(4-1)+8*(8-1)+16*(16-1)   ) + (2*D*D+2*D*D-1) +  (L_L-10)* (  ((3*D-1)*(D))/2   ) , abs(E_DMRG-E_exact)/abs(E_exact))
#   elif D<=32:
#    print(  D, (  2*(2-1) + 4*(4-1)+8*(8-1)+16*(16-1)+32*(32-1)   ) + (2*D*D+2*D*D-1) +  (L_L-12)* (  ((3*D-1)*(D))/2   ) , abs(E_DMRG-E_exact)/abs(E_exact))


  dmrg = DMRG2(MPO_origin, bond_dims=[2, D//2, D], cutoffs=1e-12) 
  dmrg.solve(tol=1e-9, verbosity=0 )
  E_DMRG=dmrg.energy
  p_DMRG=dmrg.state
  print ( p_DMRG.show())
  #correlation( p_DMRG, L_L, opt) 
  Hubburd_correlation( p_DMRG, L_L, opt) 



  y=tnopt_qmps.losses[:]
  relative_error.append( abs((y[-1]-E_exact)/E_exact))
#   y=tnopt_qmps_p.losses[:]
#   relative_error_p.append( abs((y[-1]-E_exact)/E_exact))
  #y=tnopt_mps_w.losses[:]
#  relative_error_Q.append( abs((y[-1]-E_exact)/E_exact))
  relative_error_Q.append( abs((E_DMRG-E_exact)/E_exact))

  J_l.append(coupling)
  print ("error_qmps" ,relative_error, "\n", "error_dmrg", relative_error_Q)



 
 file = open("Data/QmpsPoints.txt", "w")
 for index in range(len(y)):
    file.write( str(index) + "  "+ str(y[index])+ "  "+ str(abs((y[index]-E_exact)/E_exact)) + "\n")
 file.close()





 y_list=[  abs((y[i]-E_exact)/E_exact)  for i in range(len(y)) ]
 x_list=[ i for i in range(len(y)) ]
 plt.loglog(x_list, y_list, '4', color = '#e3360b', label='q=3, lay=6-random')
# plt.yscale('log')
 plt.title('qmps')
 plt.ylabel(r'$\delta$ E')
 plt.ylabel(r'$h$')
 #plt.axhline(1.149e-07, color='black', label='D=8')
 plt.legend(loc='upper left')

 plt.grid(True)
 plt.savefig('qmps-plot.pdf')
 plt.clf()




def DMRG_qmps( ):

 relative_error_DMRG=[]
 relative_error_p=[]
 relative_error=[]
 J_l=[]
 L_L=16
 Qbit=3
 Depth=4
 D=16
 depth_total_f=1

 J=1.0
 h=1.0
 U=3.0
 t=1.0
 mu=U/2
 print ( "Info_DMRG", "L", L_L, "Qbit", Qbit, "Depth", Depth, "D", D, "depth_total_f", depth_total_f, "U", U, "t", t, "mu", mu)


 qmps, tag=qmps_f( L=L_L, in_depth=Depth, n_Qbit=Qbit, data_type='float64', qmps_structure='brickwall', canon="left",  seed_init=40)
 qmps.unitize_(method='mgs')

#  qmps, tag=brickwall_circuit( L=L_L, in_depth=Depth, n_Qbit=Qbit, depth_total=depth_total_f, qmps_structure="brickwall", seed_init=10, internal_mera="brickwall",n_q_mera=2 )
#  qmps.unitize_(method='mgs')
# 
# 
#  qmps, tag=pollmann_circuit( L=L_L, in_depth=Depth, n_Qbit=Qbit, depth_total=depth_total_f, qmps_structure="brickwall", n_q_mera=2,seed_init=10 )
#  qmps.unitize_(method='mgs')
# 
#  qmps, tag=qmera_f(L=L_L,in_depth=Depth,n_Qbit=Qbit,depth_total=int(math.log2(L_L)),data_type='float64',qmera_type='brickwall',seed_init=90) 
#  qmps.unitize_(method='mgs')


 qmps.draw( color=[f"lay{i}" for i in range(L_L)],  show_inds=False, show_tags=False, iterations=600, k=None, fix=None, figsize=(60, 60) , legend=True, return_fig=True, node_size=700 , edge_scale=6, highlight_inds=( ), initial_layout='spectral', edge_alpha=0.6333333333333333,)
 plt.savefig('qmps_dmrg.pdf')
 plt.clf()



 #print (qmps_brickwall)
 opt = ctg.ReusableHyperOptimizer(
     progbar=True,
     reconf_opts={},
     max_repeats=20,
     parallel=True,
     directory="cash/"
 )
 #opt='auto-hq'


 MPO_ten, MPO_origin, E_exact, p_DMRG, H_ham_list=make_Hamiltonian( L_L, J, h ,U, t, mu)



 #qmps=Smart_guessdmrg(qmps, tag, L_L, Qbit, val_iden=0.00)
 psi_h=qmps.H 
 qmps.align_(MPO_origin, psi_h)
 print ("E_init", ( psi_h & MPO_origin & qmps).contract(all, optimize='auto-hq'))
 #save_to_disk(qmps, "Store/qmpsGuess")




 coupling=-0.0
 for iter in range(1):
  coupling=coupling+0.05
  J=1.0
  h=coupling*1.0
  MPO_ten, MPO_origin, E_exact, p_DMRG, H_ham_list=make_Hamiltonian( L_L, J, h ,U, t, mu)

  print("L", L_L,"exact", E_exact)
  print("qmps")
  #qmps=load_from_disk("Store/qmpsdmrg")
  qmps_fidel, qmps_iter, qmps, E_qmps_brickwall=DMRG_style(p_DMRG, tag, qmps, MPO_origin, MPO_ten, L_L, E_exact, N_iter=1000, N_iter_inner=1, opt_method=opt)
  save_to_disk(qmps, "Store/qmpsdmrg")
  save_to_disk(qmps, "Store/qmpsGuessdmrg")



  print("DMRG")
  dmrg = DMRG2(MPO_origin, bond_dims=[D], cutoffs=1e-10) 
  dmrg.solve(tol=1e-8, verbosity=0 )
  E_DMRG=dmrg.energy
  print( "DMRG", "D", D, "Energy", E_DMRG)



  relative_error.append( abs((E_qmps_brickwall[-1]-E_exact)/E_exact))
  relative_error_DMRG.append( abs((E_DMRG-E_exact)/E_exact))
  J_l.append(coupling)
  print (J_l, relative_error, "\n",relative_error_p, "\n", relative_error_DMRG)



 file = open("Data/QmpsPointdmrg.txt", "w")
 for index in range(len(E_qmps_brickwall)):
    file.write(str(index) + "  "+ str(E_qmps_brickwall[index])+ "  "+ str(abs((E_qmps_brickwall[index]-E_exact)/E_exact)) + "\n")
 file.close()


 file = open("Data/fideldmrg.txt", "w")
 for index in range(len(E_qmps_brickwall)):
    file.write(str(index) + "  "+ str(qmps_fidel[index])+ "  " + "\n")
 file.close()



 y_list=[  abs((E_qmps_brickwall[i]-E_exact)/E_exact)  for i in range(len(E_qmps_brickwall)) ]
 x_list=[ i for i in range(len(E_qmps_brickwall)) ]


 plt.loglog(x_list,y_list, 'o', color = '#c30be3', label=r'qmps-brikwall, $n_q=2, N_{lay}=4$')

 plt.title('qmps')
 plt.ylabel(r'$\delta$ E')
 plt.xlabel(r'$h$')
 #plt.axhline(1.149e-07, color='black', label='D=8')
 plt.legend(loc='upper left')
 plt.grid(True)
 plt.savefig('DMRG-plot.pdf')
 plt.clf()










def auto_diff_qmps_time( ):

 relative_error=[]
 relative_error_p=[]
 relative_error_Q=[]

 U=5.0
 t=1.0
 mu=0  #U/2.

 opt="auto-hq"
 J_l=[]
 L_L=32
 Qbit=2
 Depth=1
 D=64
 depth_total_f=5

 print ( "Info", "L", L_L, "Qbit", Qbit, "Depth", Depth, "D", D, "depth_total_f", depth_total_f, "U", U, "t", t, "mu", mu)


# qmps, tag=qmps_f( L=L_L, in_depth=Depth, n_Qbit=Qbit, data_type='float64', qmps_structure='brickwall', canon="left",  seed_init=10, internal_mera="brickwall", n_q_mera=4)
# qmps.unitize_(method='mgs')

 #print ("Defined qmps", qmps)

 qmps, tag=brickwall_circuit( L=L_L, in_depth=Depth, n_Qbit=Qbit, depth_total=depth_total_f, qmps_structure="brickwall", seed_init=0, internal_mera="brickwall",n_q_mera=2 )
 qmps.unitize_(method='mgs')

# qmps, tag=pollmann_circuit( L=L_L, in_depth=Depth, n_Qbit=Qbit, depth_total=depth_total_f, qmps_structure="pollmann", n_q_mera=2,seed_init=0 )
# qmps.unitize_(method='mgs')

 #qmps, tag=qmera_f(L=L_L,in_depth=Depth,n_Qbit=Qbit,depth_total=int(math.log2(L_L)),data_type='float64',qmera_type='brickwall',seed_init=90) 
 #qmps.unitize_(method='mgs')

 print ( "number_gates", len(tag) )

 qmps.draw( color=[f"lay{i}" for i in range(L_L)], iterations=600, figsize=(80, 80),  return_fig=True,node_size=700 , edge_scale=6, initial_layout='spectral', edge_alpha=0.633)
 plt.savefig('qmps.pdf')
 plt.clf()

 #qmps=Smart_guess(qmps, tag, L_L, Qbit, val_iden=0.00)
 p_0=MPS_computational_state('1' * L_L)
 p_0=MPS_product_state( [[0.450+2.j, 0.250-4.j]] * L_L, cyclic=False)
 p_0=MPS_rand_state(L=L_L, bond_dim=1)
 #p_0.left_canonize()
 p_0=p_0.astype('complex128')
 norm_val=(p_0.H & p_0).contract(all, optimize=opt)
 p_0=p_0*(norm_val**(-0.5))
 print ( "norm", (p_0.H & p_0).contract(all, optimize=opt) )
 
 
 MPO_origin=MPO_ham_heis(L=L_L, j=(1.0,1.0,1.0), bz=0.0, S=0.5, cyclic=False)
 MPO_origin=MPO_origin.astype('complex128')
 
 grid_time=0.2
 for iter in range(10):

  print( "time", (iter+1)*grid_time, "grid_time", grid_time)

  MPO_exp, MPO_exp_H=exp_ham_Tylor( L_L, MPO_origin, complex(0.,-1.)*grid_time, order_expan=12, cutoff_val=1e-14, max_bond_val=120)


  print ( p_0.show())
  p_0=MPO_exp.apply(p_0, compress=True, max_bond=120, cutoff=+10e-12)
  print ( p_0.show())






  print("qmps")
  #qmps=load_from_disk("Store/qmps")
#   tnopt_qmps=auto_diff_stateQMPS(qmps, p_0, optimizer_c='L-BFGS-B')
#   tnopt_qmps.optimizer = 'L-BFGS-B' 
#   qmps = tnopt_qmps.optimize(n=10 ,ftol= 2.220e-10, maxfun= 10e+9, gtol= 1e-12, eps= 1.49016e-08, maxls=400, iprint = 0, disp=False)
  #qmps = tnopt_qmps.optimize_basinhopping(n=100, nhop=10, temperature=0.5 ,ftol= 2.220e-10, maxfun= 10e+9, gtol= 1e-12, eps= 1.49016e-08, maxls=400, iprint = 1, disp=False)


#   tnopt_qmps.optimizer = 'adam'
#   qmps = tnopt_qmps.optimize(n=0, ftol= 2.220e-10, gtol= 1e-10)
  save_to_disk(qmps, "Store/qmps")
  save_to_disk(qmps, "Store/qmpsGuess")
  #save_to_disk(qmps, "Store/qmeraGuess")




#  
#  file = open("Data/QmpsPoints.txt", "w")
#  for index in range(len(y)):
#     file.write( str(index) + "  "+ str(y[index])+ "  "+ str(abs((y[index]-E_exact)/E_exact)) + "\n")
#  file.close()









def exp_ham_Tylor(L, MPO, delta, order_expan=2, bra_index='b',ket_index='k', cutoff_val=1e-10, max_bond_val=50):


 print ("delta", delta, "order_expan", order_expan)
 
 MPO_0=MPO*1.0
 for i in range(L):
  if i==0:
   t=list(MPO[i].inds)
   MPO_0[i].modify(inds=[ t[0], f'%s{i}' %ket_index, f'%s{i}' %bra_index])
  elif i==L-1:
   t=list(MPO[i].inds)
   MPO_0[i].modify(inds=[ t[0], f'%s{i}' %ket_index, f'%s{i}' %bra_index])
  else:
   t=list(MPO[i].inds)
   MPO_0[i].modify(inds=[ t[0], t[1], f'%s{i}' %ket_index, f'%s{i}' %bra_index])


 MPO_result=MPO_0*0.0
 for i in range(order_expan):
  if i==0:
   MPO_result=MPO_identity(L, phys_dim=2, cyclic=False )
  elif i==1:
   MPO_result=MPO_result+MPO_0*((delta**(i))/math.factorial(i))   
   MPO_result.compress( max_bond=max_bond_val, cutoff=cutoff_val )
  else:
   MPO_i=MPO_0*1.0
   MPO_j=MPO_0*1.0
   for j in range(i-1):
    MPO_i=MPO_i.apply(MPO_j)

    MPO_i.compress(  max_bond=max_bond_val, cutoff=cutoff_val )
   MPO_result=MPO_result+MPO_i*((delta**(i))/math.factorial(i))
   MPO_result.compress(  max_bond=max_bond_val, cutoff=cutoff_val )


 norm=(MPO_result @ MPO_result.H)**(0.5)
 print("norm", norm)
 MPO_result.show()
 #MPO_result=MPO_result*(1./norm)
 #MPO_result.show()

 
 MPO_0_H=MPO_result*1.0
 for i in range(L):
  if i==0:
   t=MPO_result[i]*1.0
   ind_list=list(MPO_result[i].inds)
   t=t.transpose(ind_list[0], f'%s{i}' %bra_index, f'%s{i}' %ket_index)
   MPO_0_H[i].modify(data=t.H.data)
  elif i==L-1:
   t=MPO_result[i]*1.0
   ind_list=list(MPO_result[i].inds)
   t=t.transpose(ind_list[0],  f'%s{i}' %bra_index, f'%s{i}' %ket_index)
   MPO_0_H[i].modify(data=t.H.data)
  else:
   t=MPO_result[i]*1.0
   ind_list=list(MPO_result[i].inds)
   t=t.transpose(ind_list[0], ind_list[1], f'%s{i}' %bra_index, f'%s{i}' %ket_index)
   MPO_0_H[i].modify(data=t.H.data)

 return   MPO_result, MPO_0_H










def CG_results():


 MPO_origin=MPO_ham_ising(L=32, j=1.0, bx=-0.5, S=0.5, cyclic=False)
 #MPO_origin=MPO_ham_heis(L=12, j=(1.0,1.0,0.0), bz=0.0, S=0.5, cyclic=False)
 dmrg = DMRG2(MPO_origin, bond_dims=[10, 20, 60, 80, 100, 200, 200, 300], cutoffs=1e-10) 
 dmrg.solve(tol=1e-8, verbosity=0 )
 E_exact=dmrg.energy
 p_DMRG=dmrg.state
 print( E_exact)
 #save_to_disk(p_DMRG, "Data/p_DMRG")
 #p_DMRG=load_from_disk("Data/p_DMRG")
 seed_val=40
 qmps, tag=qmps_f( seed_val, L=32, in_depth=2, n_Qbit=2, data_type='float64', qmps_structure='pollmann')
 qmps.unitize_(method='mgs')
 #save_to_disk(qmps, "Data/qmps")
 #qmps=load_from_disk("Data/qmps")


 #qmps=load_from_disk("Data/qmpsp")

 brickwall_fidel, brickwall_iter, V_brickwall=CG_style(p_DMRG, tag, qmps,MPO_origin, iter_total=70, show_level = 1, technique="sd", accuracy_t = +1.0e-15)



 plt.plot(qtree_iter,qtree_fidel, 'h', color = '#e30b36', label='qtree')



 #plt.plot( list_iter_cg, list_energy_cg, marker ='^', markersize=50,label='direct-cg', color = '#340B8C' )
 #plt.axhline(E_exact, color='black')
 plt.title('Global Autodiff Optimize Convergence')
 plt.ylabel('1-F')
 plt.legend(loc='upper right')
 #, fontsize=60
 #plt.xticks(size = 60)
 #plt.yticks(size = 60)
 plt.ylim(0, 1)
 #plt.xlim(0, 200, fontsize=60)
 #plt.xlabel('Iteration',fontsize=60)
 plt.grid(True)
 plt.savefig('Fidel_CG.pdf')
 plt.clf()





def local_expectation_mera(mera, list_sites, list_inter, i, optimize='auto-hq'):
 where=list_sites[i]
 tags = [mera.site_tag(coo) for coo in where]
 mera_ij = mera.select(tags, which='any')
 mera_ij_G=mera_ij.gate(list_inter[i], where)
 mera_ij_ex = (mera_ij_G & mera_ij.H)
 return mera_ij_ex.contract(all, optimize=optimize) 




def energy_f_qmera(psi, list_sites, list_inter, **kwargs):
    """Compute the total energy as a sum of all terms.
    """
    return sum(
        local_expectation_mera(psi, list_sites, list_inter, iter, **kwargs)
        for iter in range(len(list_sites))
    )



def auto_diff_energy_qmera(psi, list_sites,list_inter, opt, optimizer_c='L-BFGS-B'):
 tnopt = qtn.TNOptimizer(
      psi,                          # the initial TN
      loss_fn=energy_f_qmera,                         # the loss function
      norm_fn=norm_f,                         # this is the function that 'prepares'/constrains the tn
      constant_tags=['ket0'],                 
      tags=["U"],
      loss_constants={'list_sites': list_sites, 'list_inter': list_inter },  # additional tensor/tn kwargs
      loss_kwargs={'optimize': opt},
     autodiff_backend="torch",   # use 'autograd' for non-compiled optimization
     optimizer=optimizer_c,     # the optimization algorithm
  )
 return tnopt
 
 
def Smart_guessmera(qmps, tag, L, Qbit , val_iden=0):

  qmpsGuess=load_from_disk("Store/qmeraGuess")
  tag_list=list(qmpsGuess.tags)
  tag_final=[]
  for i_index in tag_list: 
      if i_index.startswith('P'): tag_final.append(i_index)



  for i in tag_final:
      try:
            qmps[i].modify(data=qmpsGuess[i].data)
      except (KeyError, AttributeError):
                  pass  # do nothing!


  for i in tag:
   A=qmps[i].data
   G=np.eye(4, dtype="float64").reshape((2, 2, 2, 2))
   Grand=qu.randn((2, 2, 2, 2),dtype="float64", dist="uniform")
   A=A+Grand*val_iden
   qmps[i].modify(data=A)

  qmps.unitize_(method='mgs')
  return qmps

def U_counting(n,m):
 return ((n**2)*(m**2) )-(  ((m**2)*((m**2)+1.))/2.)
def V_counting(n,m):
 return ((n**2)*m)-(  (m*(m+1.))/2.)


def  H_terms_Fermi(L, U, t, mu):
 ZZ = pauli('Z', dtype="float64") & pauli('Z',dtype="float64")
 YY = pauli('Y') & pauli('Y')
 XX = pauli('X', dtype="float64") & pauli('X',dtype="float64")
 H=ZZ+XX+YY
 H=H.astype("float64") 
 
 Z = qu.pauli('Z')
 X = qu.pauli('X')
 Y = qu.pauli('Y')
 I = qu.pauli('I')

 S_up=(X+1.0j*Y)*(0.5)
 S_down=(X-1.0j*Y)*(0.5)

 S_up=S_up.astype('float64')
 S_down=S_down.astype('float64')
 Z=Z.astype('float64')
 I=I.astype('float64')

 list_sites=[]
 list_inter=[]

 #print (S_up@S_down &  S_up@S_down, (S_down & S_up) + (S_up & S_down))

 for i in range(L): 
  list_sites.append( (2*i, 2*i+1)  )
  A=((S_up@S_down)  &  (S_up@S_down)) * U 
  B= ( (S_up@S_down)  &  I ) * (-mu) 
  C= (  I  &  (S_up@S_down) ) * (-mu) 
  list_inter.append( A+B+C)



 for i in range(L-1): 
  list_sites.append( (2*i, 2*i+2)  )
  list_inter.append( ( (S_down & S_up) + (S_up & S_down) ) * t)
  list_sites.append( (2*i+1, 2*i+3)  )
  list_inter.append( ( (S_down & S_up) + (S_up & S_down) ) * t)

 return list_sites, list_inter



def  H_terms_Hisenberg(L):
 ZZ = pauli('Z', dtype="float64") & pauli('Z',dtype="float64")
 YY = pauli('Y') & pauli('Y')
 XX = pauli('X', dtype="float64") & pauli('X',dtype="float64")
 H=(ZZ+XX+YY)*(1./4.)
 H=H.astype("float64") 
 
 list_sites=[]
 list_inter=[]

 for i in range(L-1): 
  list_sites.append( (i, i+1)  )
  list_inter.append( H)

 return list_sites, list_inter


def  auto_diff_qmps_local( ):

 J_l=[]
 L_L=2**5
 Qbit=2
 Depth=2
 depth_total_f=1
 D=2
 U=3.
 t=1.
 mu=U/10.
 
 print ("info", "L", L_L, "Qubit", Qbit, "Depth", Depth, "Depth_total",depth_total_f )
 list_sites, list_inter=H_terms_Fermi(L_L//2, U, t, mu)
 list_sites, list_inter=H_terms_Hisenberg(L_L)

 opt = ctg.ReusableHyperOptimizer(
     progbar=True,
     reconf_opts={},
     max_repeats=32,
     parallel=True,
     directory="cash/"
 )



 #opt='auto-hq'
 
 
 relative_error=[]
 relative_error_p=[]
 relative_error_Q=[]

 

 qmera = qtn.MERA.rand(L_L, max_bond=D, dtype='float64')
 qmera.add_tag("U", which='all')
 qmera.unitize_()

 #bond_list=[ (2,2), (2,2), (2,2),  (2,2),  (2,1)  ]
 #bond_list=[ (2,3), (3,3), (3,3),  (3,3),  (3,1)  ]
 #bond_list=[ (2,4), (4,4), (4,4),  (4,4),  (4,1)  ]
 bond_list=[ (2,4), (5,5), (5,5),  (5,5),  (5,1)  ]
 bond_list=[ (2,4), (4,6), (6,6),  (6,6),  (6,1)  ]
 bond_list=[ (2,4), (4,7), (7,7),  (7,7),  (7,1)  ]
 #bond_list=[ (2,4), (4,8), (8,8),  (8,8),  (8,1)  ]
 #bond_list=[ (2,4), (4,9), (9,9),  (9,9),  (9,1)  ]
 #bond_list=[ (2,4), (4,10), (10,10),  (10,10),  (10,1)  ]

 count=0
 lay_count=1
 for i in bond_list:
     if lay_count<5:
      m,n=i
      count+=( L_L * (2**(-lay_count)) )*(U_counting(m,m)+V_counting(m,n)) - ( L_L * (2**(-lay_count)) ) * ( m*(m-1)+n*(n-1) ) 
      print ("U",U_counting(m,m), ( L_L * (2**(-lay_count)) ) * ( m*(m-1)+n*(n-1) ) *(0.5))
      print ("V",V_counting(m,n), L_L * (2**(-lay_count)))

     else :
      m,n=i
      count+=( L_L * (2**(-lay_count)) )*(V_counting(m,n))
      print ("Last_V",V_counting(m,n), L_L * (2**(-lay_count)))

     lay_count+=1
     
     
 print ("mera", D,qmera.num_indices, count)



 #qmera,tag=qmera_f(L=L_L,in_depth=Depth,n_Qbit=Qbit,depth_total=int(math.log2(L_L)),data_type='float64',qmera_type='brickwall', seed_init=10) 
 #qmera.unitize_(method='mgs')
 #print (  Qbit, Depth, len(tag)*15   )


# count_val = 0
# for key in qmera.ind_map:
#  for key1 in qmera.ind_map:
#   if key1 != key and len(qmera.ind_map[key])==2:
#    if qmera.ind_map[key1] == qmera.ind_map[key]:
#       count_val+=1
#       print (key, key1, qmera.ind_map[key1], qmera.ind_map[key]) 


# print (qmera.inner_inds())

# print (count_val, count_val//2, "links", qmera.num_indices-(2*L_L))
# print ( "number_gates", len(tag), qmera.num_tensors-L_L,  len(tag)*6 - (qmera.num_indices-(2*L_L)) - (count_val//2)  )






# qmera, tag=qmps_f( L=L_L, in_depth=Depth, n_Qbit=Qbit, data_type='float64', qmps_structure='brickwall', canon="left",  seed_init=0, internal_mera="brickwall", n_q_mera=2)
# qmera.unitize_(method='mgs')
# print (  Qbit,  Depth,  len(tag)*15   )



#  qmera, tag=brickwall_circuit( L=L_L, in_depth=Depth, n_Qbit=Qbit, depth_total=depth_total_f, qmps_structure="brickwall", seed_init=0, internal_mera="brickwall",n_q_mera=2 )
#  qmera.unitize_(method='mgs')
#  print (  depth_total_f, len(tag)*12   )
# 


 #qmera, tag=pollmann_circuit( L=L_L, in_depth=Depth, n_Qbit=Qbit, depth_total=depth_total_f, qmps_structure="pollmann", n_q_mera=2,seed_init=0 )
 #qmera.unitize_(method='mgs')
 #print (  depth_total_f, len(tag)*12   )


#color=[f'_LAYER{i}' for i in range(7)]
#color=[f"layy{i}" for i in range(100)]

 qmera.draw( color=[f'_LAYER{i}' for i in range(100)], iterations=600, figsize=(100, 100),  return_fig=True,node_size=8900 , edge_scale=6, initial_layout='spectral', edge_alpha=0.633)
 plt.savefig('fig.pdf')
 plt.clf()



 #print ( "number_gates", len(tag)   )

 for i in range(len(list_sites)):
  where=(i,(i+1)%L_L)
  where=list_sites[i]
  tags = [ qmera.site_tag(coo) for coo in where ]
  mera_ij = qmera.select(tags, which='any')
  #mera_ij.draw( color=[f'lay{i}' for i in range(int(math.log2(L_L)))], iterations=600, figsize=(100, 100),node_size=700 , edge_scale=6,  initial_layout='spectral', edge_alpha=0.63333)
  #plt.savefig(f'qmera-brickwall{i}.pdf')
  #plt.clf()
  mera_ij_G=mera_ij.gate(list_inter[i], list_sites[i])
  mera_ij_ex = (mera_ij_G & mera_ij.H)
  print ( "contract", i, mera_ij_ex.contraction_width( optimize=opt) )
  #print ( mera_ij_ex.contract(all, optimize=opt) )

#  info=(mera_ij_ex).contract(all,optimize=opt, get='path-info') 
#  #opt.plot_trials()
#  tree = ctg.ContractionTree.from_info(info)
#  tree.plot_tent(node_scale= 1. / 1, edge_scale=1 / 1, return_fig=True, figsize=(7, 7))
#  #opt.plot_trials()
#  plt.savefig('tree.pdf',bbox_inches='tight', dpi=300)
#  plt.clf()

#  tree.plot_ring(node_scale= 1. / 1, edge_scale=1 / 1, return_fig=True, figsize=(7, 7))
#  plt.savefig('treeRing.pdf',bbox_inches='tight', dpi=300)
#  plt.clf()

#  #print (info, opt )

#  opt.plot_trials(return_fig=True, figsize=(7, 7))
#  plt.savefig('plot.pdf',bbox_inches='tight', dpi=300)

#  print (info, opt )






 #qmera=Smart_guessmera(qmera, tag, L_L, Qbit, val_iden=0.00)
 #save_to_disk(qmera, "Store/qmeraGuess")
 #qmera=Smart_guess(qmera, tag, L_L, Qbit, val_iden=0.00)
 qmera=load_from_disk("Store/qmera")

 print ("energy_init", energy_f_qmera(qmera, list_sites, list_inter))


 coupling=-0.0
 for iter in range(1):
  #E_exact = qu.heisenberg_energy(L_L)
  #print ("energy_exact", L_L, E_exact)
  #MPO_origin=MPO_ham_heis(L=L_L, j=(1.0,1.0,1.0), bz=0.0, S=0.5, cyclic=False)
  MPO_origin=mpo_Fermi_Hubburd(L_L//2, U, t, mu)
  #DMRG_test( L_L, U, t, mu)

  dmrg = DMRG2(MPO_origin, bond_dims=[10, 20, 60, 80, 100, 150, 200, 300, 350, 400], cutoffs=1.e-12) 
  dmrg.solve(tol=1.e-12, verbosity=0 )
  E_exact=dmrg.energy
  print( "DMRG", E_exact, dmrg.state.show(), E_exact/L_L)

#   for D in range(2, 26,1):
#    dmrg = DMRG2(MPO_origin, bond_dims=[D], cutoffs=1.e-12) 
#    dmrg.solve(tol=1.e-12, verbosity=0 )
#    E_dmrg=dmrg.energy
#    print( D,  D*D*2*(L_L-2)+2*D*2, (dmrg.energy-E_exact)/E_exact)

  print("qmera")
  #qmera=load_from_disk("Store/qmera")
  tnopt_qmera=auto_diff_energy_qmera(qmera, list_sites,list_inter , opt, optimizer_c='L-BFGS-B')

  tnopt_qmera.optimizer = 'L-BFGS-B' 
  qmera = tnopt_qmera.optimize(n=50 ,ftol= 2.220e-10, maxfun= 10e+9, gtol= 1e-12, eps= 1.49016e-08, maxls=400, iprint = 0, disp=False)




#   tnopt_qmera.optimizer = 'adam' 
#   qmera = tnopt_qmera.optimize(n=100, ftol= 2.220e-13, gtol= 1e-08, maxfun= 40000)
  save_to_disk(qmera, "Store/qmera")
  save_to_disk(qmera, "Store/qmps")
  save_to_disk(qmera, "Store/qmeraGuess")
  save_to_disk(qmera, "Store/qmpsGuess")


  #correlation_mera( qmera, L_L, opt) 
  #Hubburd_correlation_mera( qmera, L_L, opt) 




  y=tnopt_qmera.losses[:]
  relative_error.append( abs((y[-1]-E_exact)/E_exact))

  J_l.append(coupling)
  print ("error_qmera", relative_error, "\n")


#Depth,n_Qbit=Qbit

 y=tnopt_qmera.losses[:]
 y_list=[  abs((y[i]-E_exact)/E_exact)  for i in range(len(y)) ]
 x_list=[ i for i in range(len(y)) ]

 file = open("Data/qmera.txt", "w")
 for index in range(len(y_list)):
    file.write(str(x_list[index]) + "  "+ str(y[index])+ "  "+ str(y_list[index]) + "\n")




def Smart_guessdmrg(qmps, tag, L_L, Qbit, val_iden=0.00):

  qmpsGuess=load_from_disk("Store/qmpsGuessdmrg")
  tag_list=list(qmpsGuess.tags)
  tag_final=[]

  for i_index in tag_list: 
      if i_index.startswith('P'): tag_final.append(i_index)


  for i in tag_final:
      try:
            qmps[i].modify(data=qmpsGuess[i].data)
      except (KeyError, AttributeError):
                  pass  # do nothing!


  for i in tag:
   A=qmps[i].data
   G=np.eye(4, dtype="float64").reshape((2, 2, 2, 2))
   Grand=qu.randn((2, 2, 2, 2),dtype="float64", dist="uniform")
   A=A+Grand*val_iden
   qmps[i].modify(data=A)

  qmps.unitize_(method='mgs')
  return qmps




















def DMRG_qmps_gate(psi_brickwall, psi_qmps, list_tag_brickwall, list_tag_qmps, p_DMRG, MPO_origin, L=16):


#DMRG_brickwall
 print("brickwall_DMRG")
 brickwall_fidel, brickwall_iter, V_brickwall, E_dmrg_brickwall=DMRG_style(p_DMRG, list_tag_brickwall, psi_brickwall, MPO_origin, L, N_iter=80)



#  print("mps_DMRG")
#  mps_fidel, mps_iter, V_mps, E_dmrg_mps=DMRG_style(p_DMRG, list_tag_mps, psi_mps, MPO_origin, L, N_iter=80)



 print("qmps_DMRG")
 qmps_fidel, qmps_iter, V_qmps, E_dmrg_qmps=DMRG_style(p_DMRG, list_tag_qmps, psi_qmps, MPO_origin, L, N_iter=80)

#  print("qmera_DMRG")
# 
#  qmera_fidel, qmera_iter, V_qmera, E_dmrg_qmera=DMRG_style(p_DMRG, list_tag_qmera, psi_qmera, MPO_origin, L, N_iter=80)
# 
#  print("mera_DMRG")
# 
#  mera_fidel, mera_iter, V_mera, E_dmrg_mera=DMRG_style(p_DMRG, list_tag_mera, psi_mera, MPO_origin, L, N_iter=80)
# 
# 
#  print("qtree_DMRG")
# 
#  qtree_fidel, qtree_iter, V_qtree, E_dmrg_qtree=DMRG_style(p_DMRG, list_tag_qtree, psi_qtree, MPO_origin, L, N_iter=80)
# 

 plt.plot(brickwall_iter,brickwall_fidel,'>',color='#340B8C',label='brickwall')
# plt.plot(mps_iter,mps_fidel,'s', color ='#ce5c00',label='mps')
# plt.plot(mera_iter,mera_fidel, '*', color ='#a40000',label='mera')
 plt.plot(qmps_iter,qmps_fidel, 'd', color = '#c30be3', label='qmps')
# plt.plot(qmera_iter,qmera_fidel, '1', color = '#0b8de3', label='qmera')
# plt.plot(qtree_iter,qtree_fidel, 'h', color = '#e30b36', label='qtree')



 #plt.plot( list_iter_cg, list_energy_cg, marker ='^', markersize=50,label='direct-cg', color = '#340B8C' )
 #plt.axhline(E_exact, color='black')
 plt.title('Global Autodiff Optimize Convergence')
 plt.ylabel('1-F')
 plt.legend(loc='upper right')
 #, fontsize=60
 #plt.xticks(size = 60)
 #plt.yticks(size = 60)
 plt.ylim(0, 1)
 #plt.xlim(0, 200, fontsize=60)
 #plt.xlabel('Iteration',fontsize=60)
 plt.grid(True)
 plt.savefig('Fidel_DMRG.pdf')
 plt.clf()







def DMRG_qmera_gate(psi_mera, psi_qmera, list_tag_qmera, list_tag_mera, p_DMRG, MPO_origin, L=16):


#DMRG_brickwall
#  print("brickwall_DMRG")
#  brickwall_fidel, brickwall_iter, V_brickwall, E_dmrg_brickwall=DMRG_style(p_DMRG, list_tag_brickwall, psi_brickwall, MPO_origin, L, N_iter=80)
# 
# 
# 
#  print("mps_DMRG")
#  mps_fidel, mps_iter, V_mps, E_dmrg_mps=DMRG_style(p_DMRG, list_tag_mps, psi_mps, MPO_origin, L, N_iter=80)
# 


#  print("qmps_DMRG")
#  qmps_fidel, qmps_iter, V_qmps, E_dmrg_qmps=DMRG_style(p_DMRG, list_tag_qmps, psi_qmps, MPO_origin, L, N_iter=80)

 print("qmera_DMRG")

 qmera_fidel, qmera_iter, V_qmera, E_dmrg_qmera=DMRG_style(p_DMRG, list_tag_qmera, psi_qmera, MPO_origin, L, N_iter=80)

 print("mera_DMRG")

 mera_fidel, mera_iter, V_mera, E_dmrg_mera=DMRG_style(p_DMRG, list_tag_mera, psi_mera, MPO_origin, L, N_iter=80)

# 
#  print("qtree_DMRG")
# 
#  qtree_fidel, qtree_iter, V_qtree, E_dmrg_qtree=DMRG_style(p_DMRG, list_tag_qtree, psi_qtree, MPO_origin, L, N_iter=80)


# plt.plot(brickwall_iter,brickwall_fidel,'>',color='#340B8C',label='brickwall')
 #plt.plot(mps_iter,mps_fidel,'s', color ='#ce5c00',label='mps')
 plt.plot(mera_iter,mera_fidel, '*', color ='#a40000',label='mera')
 #plt.plot(qmps_iter,qmps_fidel, 'd', color = '#c30be3', label='qmps')
 plt.plot(qmera_iter,qmera_fidel, '1', color = '#0b8de3', label='qmera')
 #plt.plot(qtree_iter,qtree_fidel, 'h', color = '#e30b36', label='qtree')



 #plt.plot( list_iter_cg, list_energy_cg, marker ='^', markersize=50,label='direct-cg', color = '#340B8C' )
 #plt.axhline(E_exact, color='black')
 plt.title('Global Autodiff Optimize Convergence')
 plt.ylabel('1-F')
 plt.legend(loc='upper right')
 #, fontsize=60
 #plt.xticks(size = 60)
 #plt.yticks(size = 60)
 plt.ylim(0, 1)
 #plt.xlim(0, 200, fontsize=60)
 #plt.xlabel('Iteration',fontsize=60)
 plt.grid(True)
 plt.savefig('Fidel_DMRG-qmera.pdf')
 plt.clf()

















def DMRG_results(psi_brickwall, psi_mps, psi_mera, psi_qmps, psi_qmera, psi_qtree,list_tag_qmps, list_tag_qmera, list_tag_mps, list_tag_mera, list_tag_brickwall, list_tag_qtree, p_DMRG, MPO_origin, L=16):


#DMRG_brickwall
 print("brickwall_DMRG")
 brickwall_fidel, brickwall_iter, V_brickwall, E_dmrg_brickwall=DMRG_style(p_DMRG, list_tag_brickwall, psi_brickwall, MPO_origin, L, N_iter=80)



 print("mps_DMRG")
 mps_fidel, mps_iter, V_mps, E_dmrg_mps=DMRG_style(p_DMRG, list_tag_mps, psi_mps, MPO_origin, L, N_iter=80)



 print("qmps_DMRG")
 qmps_fidel, qmps_iter, V_qmps, E_dmrg_qmps=DMRG_style(p_DMRG, list_tag_qmps, psi_qmps, MPO_origin, L, N_iter=80)

 print("qmera_DMRG")

 qmera_fidel, qmera_iter, V_qmera, E_dmrg_qmera=DMRG_style(p_DMRG, list_tag_qmera, psi_qmera, MPO_origin, L, N_iter=80)

 print("mera_DMRG")

 mera_fidel, mera_iter, V_mera, E_dmrg_mera=DMRG_style(p_DMRG, list_tag_mera, psi_mera, MPO_origin, L, N_iter=80)


 print("qtree_DMRG")

 qtree_fidel, qtree_iter, V_qtree, E_dmrg_qtree=DMRG_style(p_DMRG, list_tag_qtree, psi_qtree, MPO_origin, L, N_iter=80)


 plt.plot(brickwall_iter,brickwall_fidel,'>',color='#340B8C',label='brickwall')
 plt.plot(mps_iter,mps_fidel,'s', color ='#ce5c00',label='mps')
 plt.plot(mera_iter,mera_fidel, '*', color ='#a40000',label='mera')
 plt.plot(qmps_iter,qmps_fidel, 'd', color = '#c30be3', label='qmps')
 plt.plot(qmera_iter,qmera_fidel, '1', color = '#0b8de3', label='qmera')
 plt.plot(qtree_iter,qtree_fidel, 'h', color = '#e30b36', label='qtree')



 #plt.plot( list_iter_cg, list_energy_cg, marker ='^', markersize=50,label='direct-cg', color = '#340B8C' )
 #plt.axhline(E_exact, color='black')
 plt.title('Global Autodiff Optimize Convergence')
 plt.ylabel('1-F')
 plt.legend(loc='upper right')
 #, fontsize=60
 #plt.xticks(size = 60)
 #plt.yticks(size = 60)
 plt.ylim(0, 1)
 #plt.xlim(0, 200, fontsize=60)
 #plt.xlabel('Iteration',fontsize=60)
 plt.grid(True)
 plt.savefig('Fidel_DMRG.pdf')
 plt.clf()





def   make_Hamiltonian( L, J, h , U, t, mu):

 ZZ = pauli('Z', dtype="float64") & pauli('Z',dtype="float64")
 YY = pauli('Y') & pauli('Y')
 XX = pauli('X', dtype="float64") & pauli('X',dtype="float64")
 XI = pauli('X',dtype="float64") & pauli('I',dtype="float64")
 IX = pauli('I',dtype="float64") & pauli('X',dtype="float64")
 
 H_ham=0.25*( J*ZZ+h*(XI+XI) )
 H_ham=4.0*J*( ZZ+XX+YY )
 H_ham=H_ham.astype("float64")

 H_list=[]
 for i in range(L):
  if i==0:
#   H_list.append( 0.25*( -J*ZZ+h*(2.0*IX+IX) ) )
   H_list.append( 0.25*( ZZ+XX+YY ) )
  elif i==L-1:
#   H_list.append( 0.25*( -J*ZZ+h*(IX+2.0*IX) ) )
   H_list.append( 0.25*( ZZ+XX+YY ) )
  else: 
#   H_list.append( 0.25*( -J*ZZ+h*(XI+XI) ) )
   H_list.append( 0.25*( ZZ+XX+YY ) )


 H_list= [   H_list[i].astype("float64")   for i in range(L-1)  ] 
 #H_ham=H_ham.astype("float64")
 #print (H_ham)
 e=eigensystem(H_ham, True,k=-1, sort=True, return_vecs=False)
 Landa=max(e, key=abs)

 MPO_origin=MPO_ham_ising(L, j=J, bx=h, S=0.5, cyclic=False)
 MPO_origin=mpo_Fermi_Hubburd(L//2, U, t, mu)
 #MPO_origin=MPO_ham_heis(L, j=1.0, bz=0.0, S=0.5, cyclic=False)
 #MPO_origin=MPO_ham_XY(L, j=1.0, bz=0.0, S=0.5, cyclic=False)
 MPO_ten_iden=MPO_identity(L, phys_dim=2, cyclic=False )



 MPO_ten=MPO_origin.add_MPO(MPO_ten_iden*abs(Landa)*(L-1)*-1.0, inplace=False, compress=False)



 dmrg = DMRG2(MPO_origin, bond_dims=[10, 20, 60, 80, 100, 150], cutoffs=1e-11)
 #print (dmrg)
 #dmrg.opts(local_eig_backend='slepc')
 dmrg.opts['local_eig_backend'] = 'SCIPY'
 #dmrg.opts['local_eig_backend'] = 'SCIPY'
 
 #print(type(dmrg), dmrg.opts)
 dmrg.solve(tol=1e-12, verbosity=0 )
 #print("DMRG=", dmrg.energy)
 E_exact=dmrg.energy
 p_0=dmrg.state
 print("DMRG_state=", p_0.show())

#  T_u=qtn.TensorNetwork(MPO_origin)
#  list_ket=[  f'k{i}'  for i in range(L)]
#  list_bra=[  f'b{i}'  for i in range(L)]
#  list_f=list_ket+list_bra
#  ham_ED=T_u^all
#  final_list=list_ket+list_bra
#  ham_ED1=ham_ED.transpose(*final_list)
#  e=eigensystem(ham_ED1.data.reshape(2**L,2**L), True, k=-1, sort=True, return_vecs=False)
#  #print("ED=", e[0])
#  E_exact=e[0]*1.0



 #MPO_ten=MPO_ten.astype_(data_type)
 #MPO_origin=MPO_origin.astype_(data_type)
 
 return MPO_ten, MPO_origin, E_exact, p_0, H_list








def  mpo_Fermi_Hubburd(L, U, t, mu):

 #print ( "L,", L, "U,", U, "t,", t,  "mu,", mu)
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

 return MPO_f










def  mpo_Fermi_Hubburd_1(L, U, t, mu):

 #print ( "L,", L, "U,", U, "t,", t,  "mu,", mu)
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
    Wl = np.zeros([ 1, 2, 2], dtype='float64')
    W = np.zeros([1, 1, 2, 2], dtype='float64')
    Wr = np.zeros([ 1, 2, 2], dtype='float64')

    W_Z = np.zeros([1, 1, 2, 2], dtype='float64')
    W_Z[ 0,0,:,:]=Z


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
    MPO_I[2*i+1].modify(data=W_Z)
    MPO_I[2*i+2].modify(data=W_2[2*i+2])
    MPO_result=MPO_result+MPO_I
    MPO_result.compress( max_bond=max_bond_val, cutoff=cutoff_val )

    MPO_I=MPO_identity(2*L, phys_dim=2 )
    MPO_I[2*i].modify(data=W_2[2*i])
    MPO_I[2*i+1].modify(data=W_Z)
    MPO_I[2*i+2].modify(data=W_1[2*i+2])
    MPO_result=MPO_result+MPO_I
    MPO_result.compress( max_bond=max_bond_val, cutoff=cutoff_val )


    MPO_I=MPO_identity(2*L, phys_dim=2 )
    MPO_I[2*i+1].modify(data=W_1[2*i+1])
    MPO_I[2*i+2].modify(data=W_Z)
    MPO_I[2*i+3].modify(data=W_2[2*i+3])
    MPO_result=MPO_result+MPO_I
    MPO_result.compress( max_bond=max_bond_val, cutoff=cutoff_val )


    MPO_I=MPO_identity(2*L, phys_dim=2 )
    MPO_I[2*i+1].modify(data=W_2[2*i+1])
    MPO_I[2*i+2].modify(data=W_Z)
    MPO_I[2*i+3].modify(data=W_1[2*i+3])
    MPO_result=MPO_result+MPO_I
    MPO_result.compress( max_bond=max_bond_val, cutoff=cutoff_val )


 MPO_f=MPO_f+MPO_result*(t)*(1.0)
 MPO_f.compress( max_bond=max_bond_val, cutoff=cutoff_val )

 return MPO_f




def  mpo_Fermi_Hubburd_inf_1(L, U, t, mu):

 #print ( "L,", L, "U,", U, "t,", t,  "mu,", mu)
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
  for i in range(L-1): 
   Wl = np.zeros([ 1, 2, 2], dtype='float64')
   W = np.zeros([1, 1, 2, 2], dtype='float64')
   Wr = np.zeros([ 1, 2, 2], dtype='float64')
   
   Wl[ 0,:,:]=S_up@S_down
   W[ 0,0,:,:]=S_up@S_down
   Wr[ 0,:,:]=S_up@S_down


   W_list=[Wl]+[W]*(2*L-1)

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
  for i in range(L-1):
   Wl = np.zeros([ 1, 2, 2], dtype='float64')
   W = np.zeros([1, 1, 2, 2], dtype='float64')
   Wr = np.zeros([ 1, 2, 2], dtype='float64')
   
   Wl[ 0,:,:]=S_up@S_down
   W[ 0,0,:,:]=S_up@S_down
   Wr[ 0,:,:]=S_up@S_down


   W_list=[Wl]+[W]*(2*L-1)


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
    Wl = np.zeros([ 1, 2, 2], dtype='float64')
    W = np.zeros([1, 1, 2, 2], dtype='float64')
    W_Z = np.zeros([1, 1, 2, 2], dtype='float64')
    Wr = np.zeros([ 1, 2, 2], dtype='float64')

    W_Z[ 0,0,:,:]=Z

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
    MPO_I[2*i+1].modify(data=W_Z)
    MPO_I[2*i+2].modify(data=W_2[2*i+2])
    MPO_result=MPO_result+MPO_I
    MPO_result.compress( max_bond=max_bond_val, cutoff=cutoff_val )

    MPO_I=MPO_identity(2*L, phys_dim=2 )
    MPO_I[2*i].modify(data=W_2[2*i])
    MPO_I[2*i+1].modify(data=W_Z)
    MPO_I[2*i+2].modify(data=W_1[2*i+2])
    MPO_result=MPO_result+MPO_I
    MPO_result.compress( max_bond=max_bond_val, cutoff=cutoff_val )


    MPO_I=MPO_identity(2*L, phys_dim=2 )
    MPO_I[2*i+1].modify(data=W_1[2*i+1])
    MPO_I[2*i+2].modify(data=W_Z)
    MPO_I[2*i+3].modify(data=W_2[2*i+3])
    MPO_result=MPO_result+MPO_I
    MPO_result.compress( max_bond=max_bond_val, cutoff=cutoff_val )


    MPO_I=MPO_identity(2*L, phys_dim=2 )
    MPO_I[2*i+1].modify(data=W_2[2*i+1])
    MPO_I[2*i+2].modify(data=W_Z)
    MPO_I[2*i+3].modify(data=W_1[2*i+3])
    MPO_result=MPO_result+MPO_I
    MPO_result.compress( max_bond=max_bond_val, cutoff=cutoff_val )


 MPO_f=MPO_f+MPO_result*(t)
 MPO_f.compress( max_bond=max_bond_val, cutoff=cutoff_val )

 return MPO_f





def  mpo_Fermi_Hubburd_inf(L, U, t, mu):

 #print ( "L,", L, "U,", U, "t,", t,  "mu,", mu)
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
  for i in range(L-1): 
   Wl = np.zeros([ 1, 2, 2], dtype='float64')
   W = np.zeros([1, 1, 2, 2], dtype='float64')
   Wr = np.zeros([ 1, 2, 2], dtype='float64')
   
   Wl[ 0,:,:]=S_up@S_down
   W[ 0,0,:,:]=S_up@S_down
   Wr[ 0,:,:]=S_up@S_down


   W_list=[Wl]+[W]*(2*L-1)

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
  for i in range(L-1):
   Wl = np.zeros([ 1, 2, 2], dtype='float64')
   W = np.zeros([1, 1, 2, 2], dtype='float64')
   Wr = np.zeros([ 1, 2, 2], dtype='float64')
   Wl[ 0,:,:]=S_up@S_down
   W[ 0,0,:,:]=S_up@S_down
   Wr[ 0,:,:]=S_up@S_down

#   W_list=[Wl/2]+[W/2]+[W]*(2*L-4)+[W/2]+[Wr/2]
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

 return MPO_f





def mpo_particle_inf(L):

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


 max_bond_val=200
 cutoff_val=1.0e-12

 MPO_result=MPO_identity(2*L, phys_dim=2)
 MPO_result=MPO_result*0.0

 for i in range(L):
  Wl = np.zeros([ 1, 2, 2], dtype='float64')
  W = np.zeros([1, 1, 2, 2], dtype='float64')
  Wr = np.zeros([ 1, 2, 2], dtype='float64')
  Wl[ 0,:,:]=S_up@S_down
  W[ 0,0,:,:]=S_up@S_down
  Wr[ 0,:,:]=S_up@S_down

  W_list=[Wl/2]+[W/2]+[W]*(2*L-4)+[W/2]+[Wr/2]

  MPO_I=MPO_identity(2*L, phys_dim=2 )
  MPO_I[2*i].modify(data=W_list[2*i])
  MPO_result=MPO_result+MPO_I
  MPO_result.compress( max_bond=max_bond_val, cutoff=cutoff_val )

  MPO_I=MPO_identity(2*L, phys_dim=2 )
  MPO_I[2*i+1].modify(data=W_list[2*i+1])
  MPO_result=MPO_result+MPO_I
  MPO_result.compress( max_bond=max_bond_val, cutoff=cutoff_val )


 return MPO_result






def mpo_spin_inf(L):

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


 max_bond_val=200
 cutoff_val=1.0e-12

 MPO_result=MPO_identity(2*L, phys_dim=2)
 MPO_result=MPO_result*0.0

 for i in range(L):
  Wl = np.zeros([ 1, 2, 2], dtype='float64')
  W = np.zeros([1, 1, 2, 2], dtype='float64')
  Wr = np.zeros([ 1, 2, 2], dtype='float64')
  Wl[ 0,:,:]=S_up@S_down
  W[ 0,0,:,:]=S_up@S_down
  Wr[ 0,:,:]=S_up@S_down

  W_list=[Wl/2]+[W/2]+[W]*(2*L-4)+[W/2]+[Wr/2]

  MPO_I=MPO_identity(2*L, phys_dim=2 )
  MPO_I[2*i].modify(data=W_list[2*i])
  MPO_result=MPO_result+MPO_I
  MPO_result.compress( max_bond=max_bond_val, cutoff=cutoff_val )



 MPO_result1=MPO_identity(2*L, phys_dim=2)
 MPO_result1=MPO_result*0.0

 for i in range(L):
  Wl = np.zeros([ 1, 2, 2], dtype='float64')
  W = np.zeros([1, 1, 2, 2], dtype='float64')
  Wr = np.zeros([ 1, 2, 2], dtype='float64')
  Wl[ 0,:,:]=S_up@S_down
  W[ 0,0,:,:]=S_up@S_down
  Wr[ 0,:,:]=S_up@S_down

  W_list=[Wl/2]+[W/2]+[W]*(2*L-4)+[W/2]+[Wr/2]


  MPO_I=MPO_identity(2*L, phys_dim=2 )
  MPO_I[2*i+1].modify(data=W_list[2*i+1])
  MPO_result1=MPO_result1+MPO_I
  MPO_result1.compress( max_bond=max_bond_val, cutoff=cutoff_val )

 return MPO_result, MPO_result1



def mpo_particle(L):

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


 max_bond_val=200
 cutoff_val=1.0e-12

 MPO_result=MPO_identity(2*L, phys_dim=2)
 MPO_result=MPO_result*0.0

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
  MPO_result=MPO_result+MPO_I
  MPO_result.compress( max_bond=max_bond_val, cutoff=cutoff_val )

  MPO_I=MPO_identity(2*L, phys_dim=2 )
  MPO_I[2*i+1].modify(data=W_list[2*i+1])
  MPO_result=MPO_result+MPO_I
  MPO_result.compress( max_bond=max_bond_val, cutoff=cutoff_val )


 return MPO_result






def mpo_spin(L):

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


 max_bond_val=200
 cutoff_val=1.0e-12

 MPO_result=MPO_identity(2*L, phys_dim=2)
 MPO_result=MPO_result*0.0

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
  MPO_result=MPO_result+MPO_I
  MPO_result.compress( max_bond=max_bond_val, cutoff=cutoff_val )



 MPO_result1=MPO_identity(2*L, phys_dim=2)
 MPO_result1=MPO_result*0.0

 for i in range(L):
  Wl = np.zeros([ 1, 2, 2], dtype='float64')
  W = np.zeros([1, 1, 2, 2], dtype='float64')
  Wr = np.zeros([ 1, 2, 2], dtype='float64')
  Wl[ 0,:,:]=S_up@S_down
  W[ 0,0,:,:]=S_up@S_down
  Wr[ 0,:,:]=S_up@S_down

  W_list=[Wl]+[W]*(2*L-2)+[Wr]


  MPO_I=MPO_identity(2*L, phys_dim=2 )
  MPO_I[2*i+1].modify(data=W_list[2*i+1])
  MPO_result1=MPO_result1+MPO_I
  MPO_result1.compress( max_bond=max_bond_val, cutoff=cutoff_val )

 return MPO_result, MPO_result1




def mpo_longrange_Heisenberg(L):

 Z = qu.pauli('Z')
 X = qu.pauli('X')
 Y = qu.pauli('Y')
 I = qu.pauli('I')

 S_up=(1.0j*Y)
 S_up=S_up.astype('float64')

 MPO_I=MPO_identity(L, phys_dim=2)
 MPO_f=MPO_identity(L, phys_dim=2)


 max_bond_val=300
 cutoff_val=1.0e-12


 for i in range(L): 
  Wl = np.zeros([ 1, 2, 2], dtype='float64')
  W = np.zeros([1, 1, 2, 2], dtype='float64')
  Wr = np.zeros([ 1, 2, 2], dtype='float64')
  Wl[ 0,:,:]=Z
  W[ 0,0,:,:]=Z
  Wr[ 0,:,:]=Z
  W_list=[Wl]+[W]*(L-2)+[Wr]

  for j in range(L): 
   MPO_I=MPO_identity(L, phys_dim=2 )
   MPO_I[i].modify(data=W_list[i])
   if j != i:
     MPO_I[j].modify(data=W_list[j])
     MPO_f=MPO_f+MPO_I*(1./  (abs(i-j)**(2.))  )
     MPO_f.compress( max_bond=max_bond_val, cutoff=cutoff_val )



 for i in range(L): 
  Wl = np.zeros([ 1, 2, 2], dtype='float64')
  W = np.zeros([1, 1, 2, 2], dtype='float64')
  Wr = np.zeros([ 1, 2, 2], dtype='float64')
  Wl[ 0,:,:]=X
  W[ 0,0,:,:]=X
  Wr[ 0,:,:]=X


  W_list=[Wl]+[W]*(L-2)+[Wr]

  for j in range(L): 
   MPO_I=MPO_identity(L, phys_dim=2 )
   MPO_I[i].modify(data=W_list[i])
   if j != i:
     MPO_I[j].modify(data=W_list[j])
     MPO_f=MPO_f+MPO_I*(1./  (abs(i-j)**(2.))  )
     MPO_f.compress( max_bond=max_bond_val, cutoff=cutoff_val )



 for i in range(L): 
  Wl = np.zeros([ 1, 2, 2], dtype='float64')
  W = np.zeros([1, 1, 2, 2], dtype='float64')
  Wr = np.zeros([ 1, 2, 2], dtype='float64')
  Wl[ 0,:,:]=X
  W[ 0,0,:,:]=X
  Wr[ 0,:,:]=X


  W_list=[Wl]+[W]*(L-2)+[Wr]

  for j in range(L): 
   MPO_I=MPO_identity(L, phys_dim=2 )
   MPO_I[i].modify(data=W_list[i])
   if j != i:
     MPO_I[j].modify(data=W_list[j])
     MPO_f=MPO_f+MPO_I*(1./  (abs(i-j)**(2.))  )
     MPO_f.compress( max_bond=max_bond_val, cutoff=cutoff_val )


 for i in range(L): 
  Wl = np.zeros([ 1, 2, 2], dtype='float64')
  W = np.zeros([1, 1, 2, 2], dtype='float64')
  Wr = np.zeros([ 1, 2, 2], dtype='float64')
  Wl[ 0,:,:]=S_up
  W[ 0,0,:,:]=S_up
  Wr[ 0,:,:]=S_up
  #print (S_up)

  W_list=[Wl]+[W]*(L-2)+[Wr]

  for j in range(L): 
   MPO_I=MPO_identity(L, phys_dim=2 )
   MPO_I[i].modify(data=W_list[i])
   if j != i:
     MPO_I[j].modify(data=W_list[j])
     MPO_f=MPO_f+MPO_I*(1./  (abs(i-j)**(2.))  )
     MPO_f.compress( max_bond=max_bond_val, cutoff=cutoff_val )





 print(MPO_f.show())
 return MPO_f






#  overlap_TN=p_DMRG.H & qmps_pollmann
#  Dis_val=1-abs(overlap_TN.contract(all,optimize=opt)
# )
# 
#  print ("DMRG", Dis_val, qmps_pollmann.site_tags)
# 
#  V_opt_h=qmps_pollmann.H 
#  qmps_pollmann.align_(MPO_origin, V_opt_h)
#  info=(V_opt_h & MPO_origin & qmps_pollmann).contract(all,optimize=opt, get='path-info') 
#  E_dmrg=(V_opt_h & MPO_origin & qmps_pollmann).contract(all,optimize=opt) 

#  tree = ctg.ContractionTree.from_info(info)
#  tree.plot_tent(node_scale= 1 / 6, edge_scale=1 / 6)
#  #opt.plot_trials()
#  plt.savefig('tree.pdf')
#  plt.clf()
# 
#  tree.plot_ring(node_scale= 1 / 8, edge_scale=1 / 5)
#  plt.savefig('treeRing.pdf')
#  plt.clf()
# 
#  print (info, opt )
#  print ( "Energy_DMRG", E_dmrg.real, Dis_val, (E_exact-E_dmrg.real)/E_dmrg.real)
#  print (qmps_pollmann, list_qmps_pollmann)





def DMRG_test( L_int, U, t, mu):


 for L_L in range(L_int,L_int+1, 1):
  MPO_origin=mpo_Fermi_Hubburd(L_L//2, U, t, mu)
  dmrg = DMRG2(MPO_origin, bond_dims=[10, 20, 60, 80, 100, 150, 200], cutoffs=1.e-12) 
  dmrg.solve(tol=1.e-12, verbosity=0 )
  psi=dmrg.state
  E_exact=dmrg.energy
  print("L", L_L//2, "DMRG", E_exact, "E/L", 2.0*E_exact/L_L, dmrg.state.show())
  MPO_p=mpo_particle(L_L//2)
  psi_h=psi.H 
  psi.align_(MPO_p, psi_h)
  N_par=( psi_h & MPO_p & psi).contract(all, optimize='auto-hq') 
  print ("particles", N_par, "dopping", 2*N_par/L_L, "Energy/L", 2.0*(E_exact+mu*N_par)/L_L )
 return 2.0*(E_exact+mu*N_par)/L_L








def DMRG_test_1( L_int, U, t, mu):


 for L_L in range(L_int,L_int+1, 1):
  MPO_origin=mpo_Fermi_Hubburd_1(L_L//2, U, t, mu)
  dmrg = DMRG2(MPO_origin, bond_dims=[10, 20, 60, 80, 100, 150, 200], cutoffs=1.e-12) 
  dmrg.solve(tol=1.e-12, verbosity=0 )
  psi=dmrg.state
  E_exact=dmrg.energy
  print("L", L_L//2, "DMRG", E_exact, "E/L", 2.0*E_exact/L_L, dmrg.state.show())
  MPO_p=mpo_particle(L_L//2)
  psi_h=psi.H 
  psi.align_(MPO_p, psi_h)
  N_par=( psi_h & MPO_p & psi).contract(all, optimize='auto-hq') 
  print ("particles", N_par, "dopping", 2*N_par/L_L, "Energy/L", 2.0*(E_exact+mu*N_par)/L_L )
 return 2.0*(E_exact+mu*N_par)/L_L






def   correlation(qmera, L_L, opt):

  Z = qu.pauli('Z')
  X = qu.pauli('X')
  Y = qu.pauli('Y')
  I = qu.pauli('I')

  S_up=(X+1.0j*Y)*(0.5)
  S_down=(X-1.0j*Y)*(0.5)

  S_up=S_up.astype('float64')
  S_down=S_down.astype('float64')
  Z=Z.astype('float64')

  Wz = np.zeros([1, 1, 2, 2], dtype='float64')
  Wx = np.zeros([1, 1, 2, 2], dtype='float64')

  Wz = np.zeros([1, 1, 2, 2], dtype='float64')
  Wz[ 0, 0,:,:]=Z

  Wx = np.zeros([1, 1, 2, 2], dtype='float64')
  Wx[ 0, 0,:,:]=Z


  Wy = np.zeros([1, 1, 2, 2],dtype='float64')
  A=Y*1.0j
  A=A.astype('float64')
  Wy[ 0, 0,:,:]=A


#  MPO_p=mpo_particle(L_L//2)
#  qmera_h=qmera.H 
#  qmera.align_(MPO_p, qmera_h)
#  print ("particle_number",     (qmera_h  & MPO_p  &  qmera).contract(all, optimize=opt)   )

#  MPO_up, MPO_down=mpo_spin(L_L//2)
#  qmera_h=qmera.H 
#  qmera.align_(MPO_up, qmera_h)
#  print ("spin_u",     (qmera_h  & MPO_up  &  qmera).contract(all, optimize=opt)  )
#  qmera_h=qmera.H 
#  qmera.align_(MPO_down, qmera_h)
#  print ("spin_d",   (qmera_h  & MPO_down  &  qmera).contract(all, optimize=opt)   )



  list_z=[]
  list_r=[]
  list_i=[]
  list_j=[]
  i_init=(L_L//2)-8  #(3*L_L//8)
  for i in range(i_init+1, L_L-1, 1):
   MPO_f=MPO_identity(L_L, phys_dim=2)
   MPO_f[i_init].modify(data=Wz)
   qmera_h=qmera.H 
   qmera.align_(MPO_f, qmera_h)
   Z_1=(qmera_h  & MPO_f  &  qmera).contract(all, optimize=opt) 

   MPO_f=MPO_identity(L_L, phys_dim=2)
   MPO_f[i].modify(data=Wz)
   qmera_h=qmera.H 
   qmera.align_(MPO_f, qmera_h)
   Z_2=(qmera_h  & MPO_f  &  qmera).contract(all, optimize=opt) 


   MPO_f=MPO_identity(L_L, phys_dim=2)
   MPO_f[i_init].modify(data=Wx)
   qmera_h=qmera.H 
   qmera.align_(MPO_f, qmera_h)
   X_1=(qmera_h  & MPO_f  &  qmera).contract(all, optimize=opt) 

   MPO_f=MPO_identity(L_L, phys_dim=2)
   MPO_f[i].modify(data=Wx)
   qmera_h=qmera.H 
   qmera.align_(MPO_f, qmera_h)
   X_2=(qmera_h  & MPO_f  &  qmera).contract(all, optimize=opt) 



   MPO_f=MPO_identity(L_L, phys_dim=2)
   MPO_f[i_init].modify(data=Wy)
   qmera_h=qmera.H 
   qmera.align_(MPO_f, qmera_h)
   Y_1=(qmera_h  & MPO_f  &  qmera).contract(all, optimize=opt) 

   MPO_f=MPO_identity(L_L, phys_dim=2)
   MPO_f[i].modify(data=Wy)
   qmera_h=qmera.H 
   qmera.align_(MPO_f, qmera_h)
   Y_2=(qmera_h  & MPO_f  &  qmera).contract(all, optimize=opt) 




   MPO_f=MPO_identity(L_L, phys_dim=2)
   MPO_f[i].modify(data=Wz)
   MPO_f[i_init].modify(data=Wz)
   qmera_h=qmera.H 
   qmera.align_(MPO_f, qmera_h)
   C_1=(qmera_h  & MPO_f  &  qmera).contract(all, optimize=opt) 


   MPO_f=MPO_identity(L_L, phys_dim=2)
   MPO_f[i].modify(data=Wx)
   MPO_f[i_init].modify(data=Wx)
   qmera_h=qmera.H 
   qmera.align_(MPO_f, qmera_h)
   C_2=(qmera_h  & MPO_f  &  qmera).contract(all, optimize=opt) 


   MPO_f=MPO_identity(L_L, phys_dim=2)
   MPO_f[i].modify(data=Wy)
   MPO_f[i_init].modify(data=Wy)
   qmera_h=qmera.H 
   qmera.align_(MPO_f, qmera_h)
   C_3=(qmera_h  & MPO_f  &  qmera).contract(all, optimize=opt) 




   print ( i_init, i,abs(i_init-i), C_1, C_2, C_3,Z_1*Z_2,X_1*X_2,Y_1*Y_2,  Z_1*Z_2+X_1*X_2-Y_1*Y_2+C_1+C_2-C_3  )
   list_z.append(Z_1*Z_2+X_1*X_2-Y_1*Y_2+C_1+C_2-C_3 )
   list_r.append(abs(i_init-i))
   list_i.append(i_init)
   list_j.append(i)

  file = open("Data/corrZ.txt", "w")
  for index in range(len(list_z)):
     file.write( str(list_i[index])+ "  "+ str(list_j[index])+ "  "+str(list_r[index])+"  "+ str(list_z[index])+ "  " + "\n")
  file.close()



#  list_n=[]
#  list_r=[]
#  i_init=(L_L//2)   #(3*L_L//8)
#  for i in range(i_init+1, L_L-1, 1):
#   MPO_f=MPO_identity(L_L, phys_dim=2)
#   MPO_f[i_init].modify(data=Wn)
#   qmera_h=qmera.H 
#   qmera.align_(MPO_f, qmera_h)
#   A_1=(qmera_h  & MPO_f  &  qmera).contract(all, optimize=opt) 

#   MPO_f=MPO_identity(L_L, phys_dim=2)
#   MPO_f[i].modify(data=Wn)
#   qmera_h=qmera.H 
#   qmera.align_(MPO_f, qmera_h)
#   B_1=(qmera_h  & MPO_f  &  qmera).contract(all, optimize=opt) 


#   MPO_f=MPO_identity(L_L, phys_dim=2)
#   MPO_f[i].modify(data=Wn)
#   MPO_f[i_init].modify(data=Wn)
#   qmera_h=qmera.H 
#   qmera.align_(MPO_f, qmera_h)
#   C_1=(qmera_h  & MPO_f  &  qmera).contract(all, optimize=opt) 
#   list_n.append(C_1- A_1*B_1)
#   list_r.append(abs(i_init-i))
#   list_i.append(i_init)
#   list_j.append(i)
#   print ( i_init, i, A_1, B_1, C_1,  C_1- A_1*B_1  )


#  file = open("Data/corrN.txt", "w")
#  for index in range(len(list_n)):
#     file.write( str(list_i[index])+ "  "+ str(list_j[index])+ "  "+str(list_r[index]) + "  "+ str(list_n[index])+ "  " + "\n")
#  file.close()















def   correlation_mera(qmera, L_L, opt):

  Z = qu.pauli('Z')
  X = qu.pauli('X')
  Y = qu.pauli('Y')
  I = qu.pauli('I')

  S_up=(X+1.0j*Y)*(0.5)
  S_down=(X-1.0j*Y)*(0.5)

  S_up=S_up.astype('float64')
  S_down=S_down.astype('float64')
  Z=Z.astype('float64')

  Wz = np.zeros([1, 1, 2, 2], dtype='float64')
  Wx = np.zeros([1, 1, 2, 2], dtype='float64')

  Wz = np.zeros([1, 1, 2, 2], dtype='float64')
  Wz[ 0, 0,:,:]=Z

  Wx = np.zeros([1, 1, 2, 2], dtype='float64')
  Wx[ 0, 0,:,:]=Z


  Wy = np.zeros([1, 1, 2, 2],dtype='float64')
  A=Y*1.0j
  A=A.astype('float64')
  Wy[ 0, 0,:,:]=A


  Z=Z.astype('float64')
  X=X.astype('float64')



  list_z=[]
  list_r=[]
  list_i=[]
  list_j=[]
  i_init=(L_L//2)-8  #(3*L_L//8)
  for i in range(i_init+1, L_L-1, 1):
   A_1=0
   B_1=0



   where=(i,i_init)
   tags = [ qmera.site_tag(coo) for coo in where ]
   mera_ij = qmera.select(tags, which='any')
   mera_ij_G=mera_ij.gate(Z&Z, where)
   mera_ij_ex = (mera_ij_G & mera_ij.H)
   C_1=mera_ij_ex.contract(all, optimize=opt)
   

   where=(i,i_init)
   tags = [ qmera.site_tag(coo) for coo in where ]
   mera_ij = qmera.select(tags, which='any')
   mera_ij_G=mera_ij.gate(X&X, where)
   mera_ij_ex = (mera_ij_G & mera_ij.H)
   C_2=mera_ij_ex.contract(all, optimize=opt)
   
   where=(i,i_init)
   tags = [ qmera.site_tag(coo) for coo in where ]
   mera_ij = qmera.select(tags, which='any')
   Y_opt=Y&Y
   Y_opt=Y_opt.astype('float64')
   mera_ij_G=mera_ij.gate(Y_opt, where)
   mera_ij_ex = (mera_ij_G & mera_ij.H)
   C_3=mera_ij_ex.contract(all, optimize=opt)
   




   print ( i_init, i,abs(i_init-i), C_1, C_2, C_3,  C_1+C_2+C_3  )
   list_z.append(C_1+C_2+C_3)
   list_r.append(abs(i_init-i))
   list_i.append(i_init)
   list_j.append(i)

  file = open("Data/corrZ.txt", "w")
  for index in range(len(list_z)):
     file.write( str(list_i[index])+ "  "+ str(list_j[index])+ "  "+str(list_r[index])+"  "+ str(list_z[index])+ "  " + "\n")
  file.close()


def   Hubburd_correlation_mera( qmera, L_L, opt):

  Z = qu.pauli('Z')
  X = qu.pauli('X')
  Y = qu.pauli('Y')
  I = qu.pauli('I')

  S_up=(X+1.0j*Y)*(0.5)
  S_down=(X-1.0j*Y)*(0.5)

  S_up=S_up.astype('float64')
  S_down=S_down.astype('float64')
  Z=Z.astype('float64')

  Wz = np.zeros([1, 1, 2, 2], dtype='float64')
  Wn = np.zeros([1, 1, 2, 2], dtype='float64')

  Wz = np.zeros([1, 1, 2, 2], dtype='float64')
  Wz[ 0, 0,:,:]=Z

  Wn = np.zeros([1, 1, 2, 2], dtype='float64')
  Wn[ 0, 0,:,:]=S_up@S_down

  MPO_result=MPO_identity(L_L, phys_dim=2)
  MPO_result=MPO_result*0.0
  MPO_f=MPO_result*0.0
  max_bond_val=100
  cutoff_val=1.0e-10

  list_z=[]
  list_r=[]
  list_i=[]
  list_j=[]
  i_init=5
  for i in range(i_init+1, L_L//2-1, 1):
   A_1=0
   where=(2*i_init,)
   tags = [ qmera.site_tag(coo) for coo in where ]
   mera_ij = qmera.select(tags, which='any')
   mera_ij_G=mera_ij.gate(S_up@S_down, where)
   mera_ij_ex = (mera_ij_G & mera_ij.H)
   val_1=mera_ij_ex.contract(all, optimize=opt)

   A_1=A_1+val_1
   where=(2*i_init+1,)
   tags = [ qmera.site_tag(coo) for coo in where ]
   mera_ij = qmera.select(tags, which='any')
   mera_ij_G=mera_ij.gate(S_up@S_down, where)
   mera_ij_ex = (mera_ij_G & mera_ij.H)
   val_1=mera_ij_ex.contract(all, optimize=opt)
   A_1=A_1-val_1



   B_1=0
   where=(2*i,)
   tags = [ qmera.site_tag(coo) for coo in where ]
   mera_ij = qmera.select(tags, which='any')
   mera_ij_G=mera_ij.gate(S_up@S_down, where)
   mera_ij_ex = (mera_ij_G & mera_ij.H)
   val_1=mera_ij_ex.contract(all, optimize=opt)

   B_1=B_1+val_1
   where=(2*i+1,)
   tags = [ qmera.site_tag(coo) for coo in where ]
   mera_ij = qmera.select(tags, which='any')
   mera_ij_G=mera_ij.gate(S_up@S_down, where)
   mera_ij_ex = (mera_ij_G & mera_ij.H)
   val_1=mera_ij_ex.contract(all, optimize=opt)
   B_1=B_1-val_1



   C_1=0
   where=(2*i_init,2*i)
   tags = [ qmera.site_tag(coo) for coo in where ]
   mera_ij = qmera.select(tags, which='any')
   mera_ij_G=mera_ij.gate( (S_up@S_down) & (S_up@S_down), where)
   mera_ij_ex = (mera_ij_G & mera_ij.H)
   val_1=mera_ij_ex.contract(all, optimize=opt)
   C_1=C_1+val_1



   where=(2*i_init,2*i+1)
   tags = [ qmera.site_tag(coo) for coo in where ]
   mera_ij = qmera.select(tags, which='any')
   mera_ij_G=mera_ij.gate((S_up@S_down) & (S_up@S_down), where)
   mera_ij_ex = (mera_ij_G & mera_ij.H)
   val_1=mera_ij_ex.contract(all, optimize=opt)
   C_1=C_1-val_1


   where=(2*i_init+1,2*i)
   tags = [ qmera.site_tag(coo) for coo in where ]
   mera_ij = qmera.select(tags, which='any')
   mera_ij_G=mera_ij.gate((S_up@S_down) & (S_up@S_down), where)
   mera_ij_ex = (mera_ij_G & mera_ij.H)
   val_1=mera_ij_ex.contract(all, optimize=opt)
   C_1=C_1-val_1

   where=(2*i_init+1,2*i+1)
   tags = [ qmera.site_tag(coo) for coo in where ]
   mera_ij = qmera.select(tags, which='any')
   mera_ij_G=mera_ij.gate((S_up@S_down) & (S_up@S_down), where)
   mera_ij_ex = (mera_ij_G & mera_ij.H)
   val_1=mera_ij_ex.contract(all, optimize=opt)
   C_1=C_1+val_1



   print ( i_init, i,abs(i_init-i), A_1, B_1, C_1,  C_1- A_1*B_1  )
   list_z.append(C_1-A_1*B_1)
   list_r.append(abs(i_init-i))
   list_i.append(i_init)
   list_j.append(i)

  file = open("Data/corrZf.txt", "w")
  for index in range(len(list_z)):
     file.write( str(list_i[index])+ "  "+ str(list_j[index])+ "  "+str(list_r[index])+"  "+ str(list_z[index])+ "  " + "\n")
  file.close()

  plt.loglog( list_r, list_z, '>', color = '#0b8de3', label='spin, U=6')


  list_n=[]
  list_r=[]
  i_init=5
  for i in range(i_init+1, L_L//2-1, 1):
   A_1=0
   where=(2*i_init,)
   tags = [ qmera.site_tag(coo) for coo in where ]
   mera_ij = qmera.select(tags, which='any')
   mera_ij_G=mera_ij.gate(S_up@S_down, where)
   mera_ij_ex = (mera_ij_G & mera_ij.H)
   val_1=mera_ij_ex.contract(all, optimize=opt)

   A_1=A_1+val_1
   where=(2*i_init+1,)
   tags = [ qmera.site_tag(coo) for coo in where ]
   mera_ij = qmera.select(tags, which='any')
   mera_ij_G=mera_ij.gate(S_up@S_down, where)
   mera_ij_ex = (mera_ij_G & mera_ij.H)
   val_1=mera_ij_ex.contract(all, optimize=opt)
   A_1=A_1+val_1



   B_1=0
   where=(2*i,)
   tags = [ qmera.site_tag(coo) for coo in where ]
   mera_ij = qmera.select(tags, which='any')
   mera_ij_G=mera_ij.gate(S_up@S_down, where)
   mera_ij_ex = (mera_ij_G & mera_ij.H)
   val_1=mera_ij_ex.contract(all, optimize=opt)

   B_1=B_1+val_1
   where=(2*i+1,)
   tags = [ qmera.site_tag(coo) for coo in where ]
   mera_ij = qmera.select(tags, which='any')
   mera_ij_G=mera_ij.gate(S_up@S_down, where)
   mera_ij_ex = (mera_ij_G & mera_ij.H)
   val_1=mera_ij_ex.contract(all, optimize=opt)
   B_1=B_1+val_1



   C_1=0
   where=(2*i_init,2*i)
   tags = [ qmera.site_tag(coo) for coo in where ]
   mera_ij = qmera.select(tags, which='any')
   mera_ij_G=mera_ij.gate( (S_up@S_down) & (S_up@S_down), where)
   mera_ij_ex = (mera_ij_G & mera_ij.H)
   val_1=mera_ij_ex.contract(all, optimize=opt)
   C_1=C_1+val_1



   where=(2*i_init,2*i+1)
   tags = [ qmera.site_tag(coo) for coo in where ]
   mera_ij = qmera.select(tags, which='any')
   mera_ij_G=mera_ij.gate((S_up@S_down) & (S_up@S_down), where)
   mera_ij_ex = (mera_ij_G & mera_ij.H)
   val_1=mera_ij_ex.contract(all, optimize=opt)
   C_1=C_1+val_1


   where=(2*i_init+1,2*i)
   tags = [ qmera.site_tag(coo) for coo in where ]
   mera_ij = qmera.select(tags, which='any')
   mera_ij_G=mera_ij.gate((S_up@S_down) & (S_up@S_down), where)
   mera_ij_ex = (mera_ij_G & mera_ij.H)
   val_1=mera_ij_ex.contract(all, optimize=opt)
   C_1=C_1+val_1

   where=(2*i_init+1,2*i+1)
   tags = [ qmera.site_tag(coo) for coo in where ]
   mera_ij = qmera.select(tags, which='any')
   mera_ij_G=mera_ij.gate((S_up@S_down) & (S_up@S_down), where)
   mera_ij_ex = (mera_ij_G & mera_ij.H)
   val_1=mera_ij_ex.contract(all, optimize=opt)
   C_1=C_1+val_1


   print ( i_init, i,abs(i_init-i), A_1, B_1, C_1,  C_1- A_1*B_1  )
   list_n.append(C_1- A_1*B_1)
   list_r.append(abs(i_init-i))
   list_i.append(i_init)
   list_j.append(i)

  file = open("Data/corrNf.txt", "w")
  for index in range(len(list_n)):
     file.write( str(list_i[index])+ "  "+ str(list_j[index])+ "  "+str(list_r[index]) + "  "+ str(list_n[index])+ "  " + "\n")
  file.close()


  #return  N_particle,  N_up,  N_down



def   Hubburd_correlation(qmera, L_L, opt):

  Z = qu.pauli('Z')
  X = qu.pauli('X')
  Y = qu.pauli('Y')
  I = qu.pauli('I')

  S_up=(X+1.0j*Y)*(0.5)
  S_down=(X-1.0j*Y)*(0.5)

  S_up=S_up.astype('float64')
  S_down=S_down.astype('float64')
  Z=Z.astype('float64')

  Wz = np.zeros([1, 1, 2, 2], dtype='float64')
  Wn = np.zeros([1, 1, 2, 2], dtype='float64')

  Wz = np.zeros([1, 1, 2, 2], dtype='float64')
  Wz[ 0, 0,:,:]=Z

  Wn = np.zeros([1, 1, 2, 2], dtype='float64')
  Wn[ 0, 0,:,:]=S_up@S_down




#   MPO_p=mpo_particle(L_L//2)
#   qmera_h=qmera.H 
#   qmera.align_(MPO_p, qmera_h)
#   N_particle=(qmera_h  & MPO_p  &  qmera).contract(all, optimize=opt).real 
#   print ("particle_number",  N_particle     )
# 
# 
#   MPO_up, MPO_down=mpo_spin(L_L//2)
#   qmera_h=qmera.H 
#   qmera.align_(MPO_up, qmera_h)
#   N_up=(qmera_h  & MPO_up  &  qmera).contract(all, optimize=opt).real
#   print ("spin_u",   N_up    )
#   qmera_h=qmera.H 
#   qmera.align_(MPO_down, qmera_h)
#   N_down=(qmera_h  & MPO_down  &  qmera).contract(all, optimize=opt).real
#   print ("spin_d",   N_down   )


  MPO_result=MPO_identity(L_L, phys_dim=2)
  MPO_result=MPO_result*0.0
  MPO_f=MPO_result*0.0
  max_bond_val=100
  cutoff_val=1.0e-10


  list_z=[]
  list_r=[]
  list_i=[]
  list_j=[]
  i_init=5
  for i in range(i_init+1, L_L//2-1, 1):
   MPO_f=MPO_identity(L_L, phys_dim=2)
   MPO_f[2*i_init].modify(data=Wn)
   MPO_I=MPO_identity(L_L, phys_dim=2)
   MPO_I[2*i_init+1].modify(data=Wn)
   MPO_f=MPO_f+MPO_I*-1.
   MPO_f.compress( max_bond=max_bond_val, cutoff=cutoff_val )
   qmera_h=qmera.H 
   qmera.align_(MPO_f, qmera_h)
   A_1=(qmera_h  & MPO_f  &  qmera).contract(all, optimize=opt).real 

   MPO_ff=MPO_identity(L_L, phys_dim=2)
   MPO_ff[2*i].modify(data=Wn)
   MPO_I=MPO_identity(L_L, phys_dim=2)
   MPO_I[2*i+1].modify(data=Wn)
   MPO_ff=MPO_ff+MPO_I*-1.
   MPO_ff.compress( max_bond=max_bond_val, cutoff=cutoff_val )
   qmera_h=qmera.H 
   qmera.align_(MPO_ff, qmera_h)
   B_1=(qmera_h  & MPO_ff  &  qmera).contract(all, optimize=opt).real 

   
   MPO_ff=MPO_ff.apply(MPO_f)
   MPO_ff.compress( max_bond=max_bond_val, cutoff=cutoff_val )

   qmera_h=qmera.H 
   qmera.align_(MPO_ff, qmera_h)
   C_1=(qmera_h  & MPO_ff  &  qmera).contract(all, optimize=opt).real 

   #print (MPO_ff.show())

   print ( i_init, i,abs(i_init-i), A_1, B_1, C_1,  C_1- A_1*B_1  )
   list_z.append(C_1- A_1*B_1)
   list_r.append(abs(i_init-i))
   list_i.append(i_init)
   list_j.append(i)

  file = open("Data/corrZ.txt", "w")
  for index in range(len(list_z)):
     file.write( str(list_i[index])+ "  "+ str(list_j[index])+ "  "+str(list_r[index])+"  "+ str(list_z[index])+ "  " + "\n")
  file.close()

  plt.loglog( list_r, list_z, '>', color = '#0b8de3', label='spin, U=6')


  list_n=[]
  list_r=[]
  i_init=5
  for i in range(i_init+1, L_L//2-1, 1):
   MPO_f=MPO_identity(L_L, phys_dim=2)
   MPO_f[2*i_init].modify(data=Wn)
   MPO_I=MPO_identity(L_L, phys_dim=2)
   MPO_I[2*i_init+1].modify(data=Wn)
   MPO_f=MPO_f+MPO_I
   MPO_f.compress( max_bond=max_bond_val, cutoff=cutoff_val )
   qmera_h=qmera.H 
   qmera.align_(MPO_f, qmera_h)
   A_1=(qmera_h  & MPO_f  &  qmera).contract(all, optimize=opt).real 

   MPO_ff=MPO_identity(L_L, phys_dim=2)
   MPO_ff[2*i].modify(data=Wn)
   MPO_I=MPO_identity(L_L, phys_dim=2)
   MPO_I[2*i+1].modify(data=Wn)
   MPO_ff=MPO_ff+MPO_I
   MPO_ff.compress( max_bond=max_bond_val, cutoff=cutoff_val )
   qmera_h=qmera.H 
   qmera.align_(MPO_ff, qmera_h)
   B_1=(qmera_h  & MPO_ff  &  qmera).contract(all, optimize=opt).real 


   MPO_ff=MPO_ff.apply(MPO_f)
   MPO_ff.compress( max_bond=max_bond_val, cutoff=cutoff_val )


   qmera_h=qmera.H 
   qmera.align_(MPO_ff, qmera_h)
   C_1=(qmera_h  & MPO_ff  &  qmera).contract(all, optimize=opt).real 

   print ( i_init, i,abs(i_init-i), A_1, B_1, C_1,  C_1- A_1*B_1  )
   list_n.append(C_1- A_1*B_1)
   list_r.append(abs(i_init-i))
   list_i.append(i_init)
   list_j.append(i)

  file = open("Data/corrN.txt", "w")
  for index in range(len(list_n)):
     file.write( str(list_i[index])+ "  "+ str(list_j[index])+ "  "+str(list_r[index]) + "  "+ str(list_n[index])+ "  " + "\n")
  file.close()


  #return  N_particle,  N_up,  N_down










def   Hubburd_correlation_inf( qmps, L_L,Qbit, b_s, opt, shift_AB=1):

  Z = qu.pauli('Z')
  X = qu.pauli('X')
  Y = qu.pauli('Y')
  I = qu.pauli('I')

  S_up=(X+1.0j*Y)*(0.5)
  S_down=(X-1.0j*Y)*(0.5)

  S_up=S_up.astype('float64')
  S_down=S_down.astype('float64')
  Z=Z.astype('float64')

  Wz = np.zeros([1, 1, 2, 2], dtype='float64')
  Wn = np.zeros([1, 1, 2, 2], dtype='float64')
  Wnr = np.zeros([1, 2, 2], dtype='float64')
  Wnl = np.zeros([1, 2, 2], dtype='float64')

  Wz = np.zeros([1, 1, 2, 2], dtype='float64')
  Wz[ 0, 0,:,:]=Z

  Wn = np.zeros([1, 1, 2, 2], dtype='float64')
  Wn[ 0, 0,:,:]=S_up@S_down
  Wnr[ 0,:,:]=S_up@S_down
  Wnl[ 0,:,:]=S_up@S_down



  MPO_result=MPO_identity(L_L, phys_dim=2)
  MPO_result=MPO_result*0.0
  MPO_f=MPO_result*0.0
  max_bond_val=40
  cutoff_val=1.0e-7


  list_z=[]
  list_z_local=[]
  list_r=[]
  list_i=[]
  list_j=[]

  sign=[+1,-1]
  dic_label={ 1:"particle", -1:"spin" }
  for  sign_val   in    sign:
   print ( dic_label[sign_val] )
   list_z=[]
   list_z_local=[]
   list_r=[]
   list_i=[]
   list_j=[]
   long_val=4
   for Length_corr in range(1,long_val,1):
    corr_val=0
    iter_val=0
    #for p in range(0,2*b_s,2):
    for p in range(0,2*shift_AB,2):
     "p is acting like a shift to explor AB and BA in iMPS form ABAB "
     i_init=2*(Length_corr+1)
     W_list=[Wnl]+[Wn]*(i_init-2)+[Wnr]
     MPO_I=MPO_identity(i_init, phys_dim=2)
     MPO_I[0].modify(data=W_list[0])
     MPO_II=MPO_identity(i_init, phys_dim=2)
     MPO_II[1].modify(data=W_list[1])
     MPO_I=MPO_I+MPO_II*(sign_val)
     #print (MPO_I.show())
     i_init=2*(Length_corr+1)
     W_list=[Wnl]+[Wn]*(i_init-2)+[Wnr]
     MPO_f=MPO_identity(i_init, phys_dim=2)
     MPO_f[i_init-1].modify(data=W_list[i_init-1])
     MPO_ff=MPO_identity(i_init, phys_dim=2)
     MPO_ff[i_init-2].modify(data=W_list[i_init-2])
     MPO_f=MPO_f*(sign_val)+MPO_ff

     MPO_two=MPO_f.apply(MPO_I)
     MPO_two.compress( max_bond=max_bond_val, cutoff=cutoff_val )
     #print (MPO_two.show())


     MPO_I.reindex_({ f"k{i}":f"k{L_L+Qbit-i-p-1}"  for i in range( i_init) } )
     MPO_I.reindex_({ f"b{i}":f"b{L_L+Qbit-i-p-1}"  for i in range(i_init) } )
     psi_h=qmps.H 
     psi_h.reindex_({ f"k{L_L+Qbit-i-p-1}":f"b{L_L+Qbit-i-p-1}"  for i in range(i_init) } )
     A_1=(psi_h  & MPO_I  &  qmps).contract(all, optimize=opt).real

     MPO_f.reindex_({ f"k{i}":f"k{L_L+Qbit-i-p-1}"  for i in range( i_init) } )
     MPO_f.reindex_({ f"b{i}":f"b{L_L+Qbit-i-p-1}"  for i in range(i_init) } )
     psi_h=qmps.H 
     psi_h.reindex_({ f"k{L_L+Qbit-i-p-1}":f"b{L_L+Qbit-i-p-1}"  for i in range(i_init) } )
     B_1=(psi_h  & MPO_f  &  qmps).contract(all, optimize=opt).real 

     MPO_two.reindex_({ f"k{i}":f"k{L_L+Qbit-i-p-1}"  for i in range( i_init) } )
     MPO_two.reindex_({ f"b{i}":f"b{L_L+Qbit-i-p-1}"  for i in range(i_init) } )

     psi_h=qmps.H 
     psi_h.reindex_({ f"k{L_L+Qbit-i-p-1}":f"b{L_L+Qbit-i-p-1}"  for i in range(i_init) } )
     C_1=(psi_h  & MPO_two  &  qmps).contract(all, optimize=opt).real 

     print (dic_label[sign_val], p, "sites", L_L+Qbit-p-1, L_L+Qbit-p-i_init+1, A_1, B_1, C_1, C_1 - A_1*B_1)
     corr_val+=C_1- A_1*B_1
     iter_val+=1.
     list_i.append(L_L+Qbit-p-i_init+1)
     list_j.append(L_L+Qbit-p-1)
     list_z_local.append( C_1- A_1*B_1 )

    print ("final_corr", abs(Length_corr),  corr_val/iter_val)
    list_z.append(corr_val/iter_val)
    list_r.append(abs(Length_corr))



   file = open(f"Data/corr{dic_label[sign_val]}.txt", "w")
   for index in range(len(list_z)):
      file.write( str(list_r[index])+"  "+ str(list_z[index])+ "  " + "\n")
   file.close()




  Wn1 = np.zeros([1, 1, 2, 2], dtype='float64')
  Wnr1 = np.zeros([ 1, 2, 2], dtype='float64')
  Wnl1 = np.zeros([ 1, 2, 2], dtype='float64')
  Wn2 = np.zeros([1, 1, 2, 2], dtype='float64')
  Wnr2 = np.zeros([ 1, 2, 2], dtype='float64')
  Wnl2 = np.zeros([ 1, 2, 2], dtype='float64')

  Wn1[ 0, 0,:,:]=S_up
  Wnr1[ 0,:,:]=S_up
  Wnl1[ 0,:,:]=S_up
  Wn2 = np.zeros([1, 1, 2, 2], dtype='float64')
  Wn2[ 0, 0,:,:]=S_down
  Wnr2[ 0,:,:]=S_down
  Wnl2[ 0,:,:]=S_down

  sign=[+1,-1]
  dic_label={ 1:"X", -1:"Y" }
  for  sign_val   in    sign:
   print ( dic_label[sign_val] )
   list_z=[]
   list_z_local=[]
   list_r=[]
   list_i=[]
   list_j=[]
   long_val=4
   for Length_corr in range(1,long_val,1):
    corr_val=0
    iter_val=0
#    for p in range(0,2*b_s,2):
    for p in range(0,2*shift_AB,2):
     "p is acting like a shift to explor AB and BA in iMPS form ABAB "
     i_init=2*(Length_corr+1)
     W_list1=[Wnl1]+[Wn1]*(i_init-2)+[Wnr1]
     W_list2=[Wnl2]+[Wn2]*(i_init-2)+[Wnr2]

     MPO_I=MPO_identity(i_init, phys_dim=2)
     MPO_I[0].modify(data=W_list1[0])
     MPO_I[1].modify(data=W_list2[1])

     MPO_II=MPO_identity(i_init, phys_dim=2)
     MPO_II[0].modify(data=W_list2[0])
     MPO_II[1].modify(data=W_list1[1])
     MPO_I=(MPO_I+MPO_II*(sign_val))*(-1./1.)*sign_val



     i_init=2*(Length_corr+1)
     W_list1=[Wnl1]+[Wn1]*(i_init-2)+[Wnr1]
     W_list2=[Wnl2]+[Wn2]*(i_init-2)+[Wnr2]
     MPO_f=MPO_identity(i_init, phys_dim=2)
     MPO_f[i_init-1].modify(data=W_list1[i_init-1])
     MPO_f[i_init-2].modify(data=W_list2[i_init-2])

     MPO_ff=MPO_identity(i_init, phys_dim=2)
     MPO_ff[i_init-1].modify(data=W_list2[i_init-1])
     MPO_ff[i_init-2].modify(data=W_list1[i_init-2])

     MPO_f=(MPO_f*(sign_val)+MPO_ff)*(-1./1.)*sign_val

     MPO_two=MPO_f.apply(MPO_I)
     MPO_two.compress( max_bond=max_bond_val, cutoff=cutoff_val )

     MPO_I.reindex_({ f"k{i}":f"k{L_L+Qbit-i-p-1}"  for i in range( i_init) } )
     MPO_I.reindex_({ f"b{i}":f"b{L_L+Qbit-i-p-1}"  for i in range(i_init) } )
     psi_h=qmps.H 
     psi_h.reindex_({ f"k{L_L+Qbit-i-p-1}":f"b{L_L+Qbit-i-p-1}"  for i in range(i_init) } )
     A_1=(psi_h  & MPO_I  &  qmps).contract(all, optimize=opt).real

     MPO_f.reindex_({ f"k{i}":f"k{L_L+Qbit-i-p-1}"  for i in range( i_init) } )
     MPO_f.reindex_({ f"b{i}":f"b{L_L+Qbit-i-p-1}"  for i in range(i_init) } )
     psi_h=qmps.H 
     psi_h.reindex_({ f"k{L_L+Qbit-i-p-1}":f"b{L_L+Qbit-i-p-1}"  for i in range(i_init) } )
     B_1=(psi_h  & MPO_f  &  qmps).contract(all, optimize=opt).real 

     MPO_two.reindex_({ f"k{i}":f"k{L_L+Qbit-i-p-1}"  for i in range( i_init) } )
     MPO_two.reindex_({ f"b{i}":f"b{L_L+Qbit-i-p-1}"  for i in range(i_init) } )

     psi_h=qmps.H 
     psi_h.reindex_({ f"k{L_L+Qbit-i-p-1}":f"b{L_L+Qbit-i-p-1}"  for i in range(i_init) } )
     C_1=(psi_h  & MPO_two  &  qmps).contract(all, optimize=opt).real 

     print (dic_label[sign_val], p, "sites", L_L+Qbit-p-1, L_L+Qbit-p-i_init+1, A_1, B_1, C_1, C_1 - A_1*B_1)
     corr_val+=C_1- A_1*B_1
     iter_val+=1.
     list_i.append(L_L+Qbit-p-i_init+1)
     list_j.append(L_L+Qbit-p-1)
     list_z_local.append( C_1- A_1*B_1 )

    print ("final_corr", abs(Length_corr),  corr_val/iter_val)
    list_z.append(corr_val/iter_val)
    list_r.append(abs(Length_corr))

   file = open(f"Data/corr{dic_label[sign_val]}.txt", "w")
   for index in range(len(list_z)):
      file.write( str(list_r[index])+"  "+ str(list_z[index])+ "  " + "\n")
   file.close()

  return  

