#!/usr/bin/env python
# coding: utf-8
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jan 5 2021
@author: Yuxuan Zhang
based on the Iso-MPS codes
"""
#%% -- IMPORTS -- 
import sys
sys.path.append("..") # import one subdirectory up in files

# external packages
import numpy as np
import qiskit as qk
import networkx as nx
import tenpy

# custom things
from networks.isonetwork import IsoTensor, IsoNetwork, QKParamCircuit
import mps.mps as mps


#%%
class IsoMERA(IsoNetwork):
    """
    MPS defined by 
        - number of physical and bond qubits (sets up associated quantum registers accordingly)
        - l_uc - length of unit cell
        - L number of times to repeat unit cell
        - circuits for each site in the unit cell, and initial state of bond-qubits
    """
               
    def __init__(self,
                 preg, 
                 breg,
                 pcircs,
                 smax, #
                 **kwargs):
        """
        inputs:
            preg, list of lists of physical qubit registers on each site; 
                notice that in MERA setting we require len(preg) = 2^(smax-1)
            breg, list of lists of physical qubit registers on each site;
                notice that in MERA setting we require len(preg) = smax
                (for qiskit: register= quantum register)
            smax, # of layers; count from 0 to smax-1; total smax layers
            pcircs, list, of parameterized circuit objects:
                pcircs[0] - boundary circuit (acting only on bond-qubits)
                pcircs[1...l_uc] for each site in unit-cell
            param_names,list of sympy symbols, parameterized gate parameters (shared by all tensors)
            L, int (default=1), Length of System (number of times to repeat unit cell)
            bdry_circ, boundary vector circuit for prepping initial state of bond-qubits
            circuit_format, str, (default='cirq'), type of circuit editor/simulator used
        """
        # here, pcircs is a list of lists with length 1,2,4...2^(smax-1), respectively

#        self.n_params = len(param_names)
        
        # parse kwargs that don't depend on circuit_format
        if 'circuit_format' in kwargs.keys():
            self.circuit_format = kwargs['circuit_format']
        else: 
            self.circuit_format = 'qiskit'
        if 'L' in kwargs.keys():
            self.L = kwargs['L']
        else:
            self.L=1 
        if self.circuit_format == 'qiskit':
            # setup classical registers for measurement outcomes
            self.cregs = [[qk.ClassicalRegister(len(preg[z]))for z in range(2**(smax-1))]#label the thing on each layer
                         for x in range(self.L)]   
            self.nphys = 0
            self.nbond = 0
            for i in range(len(preg)):
                self.nphys += len(preg[i]) # number of physical qubits
            for i in range(len(breg)):
                self.nbond += len(breg[i])  # number of bond qubits
            if 'boundary_circuit' in kwargs.keys():
                bdry_circ = kwargs['boundary_circuit'] #this, as well, has to be a list
            else:
                bdry_circ = [QKParamCircuit(qk.QuantumCircuit(), []) for i in range(smax)]
                
            # make the MPS/tensor-train -- same qubits used by each tensor
            self.bdry_tensor = [IsoTensor('v_L'+str(i),
                                         [breg[i]],
                                         bdry_circ[i]) for i in range(smax)]
            def mlist(preg,x,y,z):
                if y == smax-1:
                    meas_list=[(preg,self.cregs[x][z],qk.QuantumCircuit())]
                else: 
                    meas_list=[]
                return meas_list
            self.sites= [[[IsoTensor('A'+str(x)+str(y)+str(z),
                                               [preg[z],breg[y]],
                                               pcircs[y][z],
                                               meas_list=mlist(preg[z],x,y,z) )
                           for z in range(2**(y))]#label the nodes on each layer
                          for y in range(smax)]#label the layers
                         for x in range(self.L)]
            
            # setup IsoNetwork
            # make a flat list of nodes
            self.nodes = self.bdry_tensor
            for x in range(self.L): 
                for y in range(smax):
                    self.nodes += self.sites[x][y]
            self.edges = [(self.bdry_tensor[i],self.sites[0][i][0],{'qreg':breg[i]}) for i in range(smax)]
            self.edges+=[(self.sites[x][y][z],self.sites[x][y][z+1],{'qreg':breg[y]}) for x in range(self.L) for y in range(smax) for z in range (int(2**(y)-1))]
            self.edges+=[(self.sites[x][y][z],self.sites[x][y+1][int(2*z)],{'qreg':preg[z]}) for x in range(self.L) for y in range(int(smax-1)) for z in range(int(2**(y)))]
            self.edges+=[(self.sites[x][y][int(2**(y-1)-1)],self.sites[x+1][y][0],{'qreg':breg[y]})for x in range(self.L-1) for y in range(int(smax-1))]
            self.qregs = breg+preg   
            # construct graph and check that is a DAG
            # check for repeated node names
            self.graph = nx.DiGraph()
            self.graph.add_nodes_from(self.nodes)
            self.graph.add_edges_from(self.edges)
            # check that graph is directed & acyclic (DAG)
            if nx.algorithms.dag.is_directed_acyclic_graph(self.graph) != True:
                raise RuntimeError('Graph must be directed and acyclic')
                
            # store node information    
            # self.creg_dict = creg_dict
            self.node_names = [node.name for node in self.nodes]
            if len(self.node_names) != len(set(self.node_names)):
                raise ValueError('Tensor nodes must have unique names')
    
            # store variational parameter info
            self.param_assignments = {}
            for node in self.nodes:
                self.param_assignments[node]=node.param_names
            
            # topologically sort nodes in order of execution
            self.sorted_nodes = [node for node in nx.topological_sort(self.graph)]

        else:
            raise NotImplementedError('only qiskit implemented')
            
    ## cpu simulation ##  
    def left_bdry_vector(self,params):
        """
        computes full unitaries for each state (any initial state for physicalqubit)
        inputs:
            params, dictionary of parameters {'name':numerical-value}
        returns:
            bdry_vec, unitary correspond to boundary
            ulist, list of unitaries for tensors in unit cell
        """
        bvec_l = self.bdry_tensor.unitary(params)[:,0] # boundary circuit tensor 
        return bvec_l
    
    def unitaries(self,params):
        """
        computes full unitaries for each state (any initial state for physicalqubit)
        inputs:
            params, dictionary of parameters {'name':numerical-value}
        returns:
            ulist, list of rank-4 tensors for each site in unit cell
        """
        ulist = [self.sites[j].unitary(params) for j in range(self.l_uc)]
        return ulist
    
    def tensors(self,params):
        """
        computes tensors for fixed initial state of physical qubit = |0>
        inputs:
            params, dictionary of parameters {'name':numerical-value}
        returns:
            tensors, list of rank-3 tensors for each site in unit cell
        """
        tensors = [self.sites[j].unitary(params)[:,:,0,:] for j in range(self.l_uc)]
        return tensors
    
    ## Convert to other format(s) ##
    def to_tenpy(self,params,L=1):
        """
        inputs:
            params, dictionary of parameters {'name':numerical-value}
            L, int, number of repetitions of unit cell, 
                set to np.inf for iMPS
            TODO: add any other args needed to specify, symmetries, site-type etc...
        outputs:
            tenpy MPS object created from cirq description
        """
        site = tenpy.networks.site.SpinHalfSite(conserve=None)
        if (L==np.inf) and (self.l_uc==1) and (self.nphys==1):
            B = np.swapaxes(self.tensors(params)[0],1,2)
            psi = tenpy.networks.mps.MPS.from_Bflat([site], 
                                                [B], 
                                                bc='infinite', 
                                                dtype=complex, 
                                                form=None)
            
        else:
            B_arrs = [np.swapaxes(tensor,1,2) for tensor in self.tensors(params)]
            B_arrs[0] = B_arrs[0][:,0:1,:]
            B_arrs[-1] = B_arrs[-1][:,:,0:1]
            psi = tenpy.networks.mps.MPS.from_Bflat([site]*L,
                                                    B_arrs, 
                                                    bc = 'finite', 
                                                    dtype=complex, 
                                                    form=None)    
        psi.canonical_form()
        psi.convert_form(psi.form)
        return psi    
    
    def as_mps(self,params,L=1):
        """
        converts to custom MPS class object
        inputs:
            params, dictionary of parameters {'name':numerical-value}
            L, int, number of repetitions of unit cell, 
                set to np.inf for iMPS
        outputs:
            custom MPS object created from cirq description
        """
        tensors = self.tensors(params)
        bvecl = self.left_bdry_vector(params)
        state = mps.MPS(tensors,L=L,bdry_vecs=[bvecl,None], rcf = True)
        return state
    
    def as_mpo(self,params):
        """
        converts to custom MPO class object
        inputs:
            params, dictionary of parameters {'name':numerical-value}
        outputs:
            custom MPS object created from cirq description
        """
        tensors = self.compute_unitaries(params)
        bvecl = self.compute_left_bdry_vector(params)
        op = mps.MPO(tensors,L=self.L,bdry_vecs=[bvecl,None], rcf = True)
        return op
        
    ##  correlation function sampling ##
    def sample_correlations(self,L,bases,N_samples):
        """
        basis: measurement basis for each site
            possible formats: 
                - cirq circuit for physical qubits that maps physical qubits to measurement basis
                - string of 
        possible backends:  
            'tenpy' - uses 
            'qasm' - output qasm script to measure
            
        inputs:
            options: dictionary with entries specifying:
                burn-in length, 
                unit cell length, 
                basis to measure in for each site,
                number of samples to take (could be infinite for cpu-simulations)
                backend: whether to run as 
                
        """
        raise NotImplementedError
        
#%% 

