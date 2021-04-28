#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 14:20:36 2020
@author: acpotter
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
class IsoMPS(IsoNetwork):
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
                 **kwargs):
        """
        inputs:
            preg, list of bond qubit registers
            breg, list of bond qubit registers
                (for qiskit: register= quantum register)
            l_uc, int, number of sites in unit cell
            pcircs, list, of parameterized circuit objects:
                pcircs[0] - boundary circuit (acting only on bond-qubits)
                pcircs[1...l_uc] for each site in unit-cell
            param_names,list of sympy symbols, parameterized gate parameters (shared by all tensors)
            L, int (default=1), Length of System (number of times to repeat unit cell)
            bdry_circ, boundary vector circuit for prepping initial state of bond-qubits
            circuit_format, str, (default='cirq'), type of circuit editor/simulator used
            measurement_circuit, list of circuits to be performed on physical register
        """

        self.l_uc = len(pcircs) # length of unit cell

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
        #whether to prepare the state as a thermal distribution
        
        if 'name' in kwargs.keys():
            self.name = kwargs['name']
        else:
            self.name = ''
        
        if 'thermal' in kwargs.keys():
            self.thermal = kwargs['thermal']
            if 'thermal_prob' in kwargs.keys(): # this should be a prob. list in shape of l*l_uc*n_phys
                self.thermal_prob = kwargs['thermal_prob']
            else: raise ValueError('thermal state requires probability distribution')
        else:
            self.thermal = False
            self.thermal_prob =[[0 for i in range(self.l_uc)] for j in range(self.L)] 

        if self.circuit_format == 'qiskit':
            # setup classical registers for measurement outcomes
            self.creg = qk.ClassicalRegister(len(preg)*self.L*self.l_uc,name='c'+self.name)              
            self.nphys = len(preg) # number of physical qubits
            self.nbond = len(breg) # number of bond qubits
            self.qregs = [preg,breg]
            if 'boundary_circuit' in kwargs.keys():
                bdry_circ = kwargs['boundary_circuit']
            else:
                bdry_circ = QKParamCircuit(qk.QuantumCircuit(), []) 
            if 'bases' in kwargs.keys():
                if 'FH' in kwargs.keys():
                    self.FH = kwargs['FH']
                else:
                    self.FH = False
                self.measurement_circuit = self.measurement(kwargs['bases'], preg, self.FH)
            else:
                self.measurement_circuit = [[qk.QuantumCircuit() for i in range(self.l_uc)]for j in range(self.L)]
                
            # make the MPS/tensor-train -- same qubits used by each tensor
            self.bdry_tensor = IsoTensor('v_L',
                                         [breg],
                                         bdry_circ)
            self.sites= [[IsoTensor('A'+str(x)+str(y),
                                               [preg,breg],
                                               pcircs[y],
                                               meas_list=[(preg,
                                                           self.creg,
                                                            self.measurement_circuit[x][y],
                                                          self.creg[(x*self.l_uc+y)*self.nphys:(x*self.l_uc+y+1)*self.nphys])],
                                            thermal=self.thermal, 
                                    thermal_prob=self.thermal_prob[x][y])
                          for y in range(self.l_uc)]
                         for x in range(self.L)]                        # setup IsoNetwork
            # make a flat list of nodes
            self.nodes = [self.bdry_tensor]
            for x in range(self.L): self.nodes += self.sites[x]
            
            self.edges = [(self.nodes[i],self.nodes[i+1],{'qreg':breg}) for i in range(len(self.nodes)-1)]
            
            
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
            
            # extract list of all parameters
            self.param_list = []
            tmp_list = []
            for v in self.param_assignments.values():
                tmp_list += v
            # prune duplicates
            [self.param_list.append(item) for item in tmp_list if item not in self.param_list]
            self.n_params = len(self.param_list) # number of parameters
            
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
        # extract parameters just for boundary vector circuit
        bdry_param_keys = self.param_assignments[self.bdry_tensor]
        bdry_param_vals = [params[k] for k in bdry_param_keys]
        bdry_param_dict = dict(zip(bdry_param_keys,bdry_param_vals))
        bvec_l = self.bdry_tensor.unitary(bdry_param_dict)[:,0] # boundary circuit tensor 
        return bvec_l
    
    def unitaries(self,params):
        """
        computes full unitaries for each state (any initial state for physicalqubit)
        inputs:
            params, dictionary of parameters {'name':numerical-value}
        returns:
            ulist, list of rank-4 tensors for each site in unit cell
        """
        ulist=[]
        for j in range(self.l_uc): # start at one to skip boundary vector
            site = self.sites[0][j] 
            site_param_names = site.param_names
            site_param_vals = [params[k] for k in site_param_names]
            site_param_dict = dict(zip(site_param_names,site_param_vals))
            ulist += [site.unitary(site_param_dict)]
#        ulist = [self.sites[j][0].unitary(params) for j in range(self.l_uc)]
        return ulist
    
    def tensors(self,params):
        """
        computes tensors for fixed initial state of physical qubit = |0>
        inputs:
            params, dictionary of parameters {'name':numerical-value}
        returns:
            tensors, list of rank-3 tensors for each site in unit cell
        """
        tensors=[]
        for j in range(self.l_uc): # start at one to skip boundary vector
            site = self.sites[0][j] 
            site_param_names = site.param_names
            site_param_vals = [params[k] for k in site_param_names]
            site_param_dict = dict(zip(site_param_names,site_param_vals))
            tensors += [site.unitary(site_param_dict)[:,:,0,:]]
        #tensors = [self.sites[j][0].unitary(params)[:,:,0,:] for j in range(self.l_uc)]
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
        if self.nphys == 1:
            site = tenpy.networks.site.SpinHalfSite(conserve = None)
        elif self.nphys == 2:
            site = tenpy.networks.site.SpinHalfFermionSite(cons_N=None, cons_Sz=None, filling=1.0)

        if (L==np.inf) and (self.l_uc==1) and (self.nphys==1):
            B = np.swapaxes(self.tensors(params)[0],1,2)
            psi = tenpy.networks.mps.MPS.from_Bflat([site], 
                                                [B], 
                                                bc='infinite', 
                                                dtype=complex, 
                                                form=None)
        
        elif (L == np.inf) and (self.l_uc != 1) and (self.nphys == 1):            
             B_arrs = [np.swapaxes(tensor,1,2) for tensor in self.tensors(params)]
             psi = tenpy.networks.mps.MPS.from_Bflat([site]*self.l_uc,
                                                     B_arrs, 
                                                     bc = 'infinite',
                                                     dtype=complex, 
                                                     form=None) 
        
        elif (L != np.inf) and (self.nphys == 1):
            if L != self.l_uc * self.L:
                raise ValueError('MPS must have the same length as IsoMPS object')
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
    
    def as_mps(self,params,include_left_bdry=False):
        """
        converts to custom MPS class object
        inputs:
            params, dictionary of parameters {'name':numerical-value}
            L, int, number of repetitions of unit cell, 
                set to np.inf for iMPS
        outputs:
            custom MPS object created from circuit unitaries
            mps length is equal to l_uc
        """
        tensors = self.tensors(params)
        if include_left_bdry:
            bvecl = self.left_bdry_vector(params)
            state = mps.MPS(tensors,L=self.l_uc,bdry_vecs=[bvecl,None], rcf = True)
        else:
            state = mps.MPS(tensors,L=self.l_uc,bdry_vecs=[None,None], rcf = True)
        return state
    
    def as_mpo(self,params,include_left_bdry=False,L=1):
        """
        converts to custom MPO class object
        inputs:
            params, dictionary of parameters {'name':numerical-value}
        outputs:
            custom MPS object created from cirq description
        """
        tensors = self.unitaries(params)
        # Note: bond index ordering rearranged in going from unitaries => tensors
        tensors = [np.swapaxes(tensor,1,3) for tensor in self.unitaries(params)]
        if include_left_bdry:
            bvecl = self.compute_left_bdry_vector(params)
            op = mps.MPO(tensors,L=self.L,bdry_vecs=[bvecl,None])
        else:
            op = mps.MPO(tensors,L=self.L,bdry_vecs=[None,None])
        return op
    
    def as_mpdo(self,params,thermal_probs,include_left_bdry=False,L=1):
        """
        thermal probs: list of should be for unit cell only
        WARNING: CURRENT IMPLEMENTATION ONLY WORKS FOR NPHYS=1 (SPIN-CHAINS)
        NEED TO GENERALIZE PROBABILITIES TO >2 VALUES
        """
        psi_mpo = self.as_mpo(params) # psi_mps as custom mps object
        Ws = [] # tensors for mpdo
        for x in range(self.l_uc):
            # thermal probabilities
            weights = np.array([thermal_probs[x],1-thermal_probs[x]])
            A = psi_mpo.tensors[x].copy() # site tensor (rank-4)
            # contract into bond-transfer operator
            W = np.einsum('sirj,r,ukrl->sikujl',A.conj(),weights,A) 
            # reshape into transfer-matrix and add to list
            Ws += [W.reshape(2**(self.nphys),
                             2**(2*self.nbond),
                             2**(self.nphys),
                            2**(2*self.nbond))] 

        # thermal state matrix product density operator
        mpdo = mps.MPO(Ws,L=np.inf,bdry_vecs=[None,None]) 
        return mpdo
    
        
    ##  correlation function sampling ##
    def measurement(self, bases, preg, FH=False):
        """
        let's aim at generating a measurement circuit here
        basis: measurement basis for each site
            possible formats: 
                - cirq circuit for physical qubits that maps physical qubits to measurement basis
                - string of basis as 
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
                pauli basis or fermi?
                
        process: first specify the circuit according to the measurement needed
        need to add the choice of measurement circuit in the 
        shots specified in the qsam thing?
        we need to qsam each time we sample?
        """
        if self.circuit_format == 'qiskit':
            mc_total = []
            if FH == False:
                 #measurement circuit
        #check whether the input string is a list 
                if self.L != len(bases):
                    raise ValueError('bases must have same length as L')
                for base_uc in bases:
                    if len(base_uc) != self.l_uc:
                        raise ValueError('base must be a string with same length as l_uc ')   
                    mc_uc = []
                    for base in base_uc:
                        qc = qk.QuantumCircuit()
                        for reg in self.qregs: qc.add_register(reg)
                        if base == 'x':
                            for i in range(len(preg)):
                                qc.h(preg[i])
                        if base == 'y':
                            for i in range(len(preg)):    
                                qc.rx(-np.pi/2,preg[i])
                                #qc.sdg(preg[i])
                        mc_uc.append(qc)
                    mc_total.append(mc_uc)
            else:
                # now bases is a string with total length L * l_uc * len(preg)
                # explicitly write the pauli string for each site (already consider the JW-string outside)
                for k in range(self.L):
                    mc1 = []
                    for j in range(self.l_uc):
                        qc = qk.QuantumCircuit()
                        for reg in self.qregs: qc.add_register(reg)
                        # loop over all the qubit in a preg
                        for i in range(len(preg)):
                            # for each qubit extract pauli basis for a single qubit from the string
                            base = bases[k * self.l_uc * len(preg) + j * len(preg) + i]
                            if base == 'x':
                                qc.h(preg[i])
                            elif base == 'y':
                                qc.rx(-np.pi/2,preg[i])
                        mc1.append(qc)
                    mc_total.append(mc1)     
            return mc_total
        else:
            raise NotImplementedError('only qiskit implemented')
