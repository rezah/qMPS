

import numpy as np
import random
from tenpy.networks.mps import MPS
from tenpy.networks.mpo import MPO
from tenpy.networks.site import SpinHalfSite

class thermal_state(object):
    
    """
    Represents thermal states (in the forms of Density Matrix Product Operator (DMPO)
    and (random) hologrphic Matrix Product State (random-holoMPS)) and is used for
    finite-temperature simulations.   
    """
    
    def __init__(self, tensor, N):
        """
        Parameters
        --------------
        N: int
            Number of sites in the main network chain (= L * l_uc, where L is number of 
            repetitions of the unit cell and l_uc is the length of unit-cell).
        tensor: numpy.ndarray
            Bulk rank-4 tensors of the main chain.
            tensor index ordering: physical-out, bond-out, physical-in, bond-in
            (with "in/out" referring to the right canonical form ordering)               
        """
        
        self.N = N
        self.tensor = tensor
        # tensor dimensions (consistent with rank-4 structure)
        self.d = tensor[:,0,0,0].size # physical leg dimension (assumes rank-4 structures)
        self.chi = tensor[0,:,0,0].size # bond leg dimension (assumes rank-4 structures)

    def network_from_cells(self, network_type, N, chi_MPO=None, params=None, bdry_vecs=None, method=None, T=None):      
        """
        Returns network of finite thermal-holographic Matrix Product State (random-holoMPS), finite 
        holo-MPS, finite holographic Matrix Product Operator (holoMPO), or MPO of a given model.
        --------------
        Inputs:
          --the input assumes either circuit structure or rank-4 numpy.ndarray and/or list of bulk tensors--       
          network_type: str
             One of "random_state", "circuit_MPS", "circuit_MPO", or "MPO" options.
          N: int
            Number of sites in the main network chain (= L * l_uc, where L is number of 
            repetitions of the unit cell and l_uc is the length of unit-cell). 
          chi_MPO: int
             Bond leg dimension for MPO-based structures. 
          params: numpy.ndarray
             Parameters of circuit structure.
          bdry_vecs: list
            List of left (first element) and right (second element) boundary vectors.
            (must be set to [None,None] by default, which gives left and right boundary vectors = |0> 
            for MPO-based structures. For holoMPS-based structures, the default [None,None]
            would give left boundary = |0> while the right boundary is traced over). 
          method: str 
            One of "thermal_state_class" or "tenpy" options. (if set to "tenpy", the returned structure
            would be one of physics-TenPy networks). This option is currently only available for 
            "random_state", "circuit_MPS", and "MPO" options. 
           T: float
            Tempreture (for thermal-holoMPS option).
        Note:
          -For random_state, circuit_MPS and circuit_MPO options, the original circuit with 
           parameters must be inserted as args. In this case, the returned list of bulk tensors
           includes rank-3 numpy.ndarray for random_state/circuit_MPS and rank-4 numpy.ndarray for
           circuit_MPO.
          -For holoMPS-based structures, the index ordering is: site, physical_out, bond-in, bond-out
           while for holoMPO-based structures, the index ordering is: physical-out, bond-out,
           physical-in, bond-in (with "in/out" referring to right canonical form ordering).
          -For MPO structures constructed by "thermal_state_class method", the unit cell tensor of MPO 
           network must be inserted as arg (e.g. Hamiltonian unit cell). In this case, the bulk tensors 
           would be rank-4 numpy.ndarray (consistent with final structure of MPO). For "tenpy"-method-based
           structures, the list of bulk tensors must be inserted (see TeNPy docs for more detail).     
          -Tracing over right boundary for holoMPS-based structures is appropriate for 
           holographic simulations. 
          -Set bdry_vecs to None by default for "tenpy" method. Set method to None for holoMPO-based structures.    
        """
        
        # for circuit-based structures:
        # both circuit and params must be included
        if network_type == 'random_state' or network_type == 'circuit_MPS' or network_type == 'circuit_MPO':
            
            # evaluating the circuit (rank-4 (p_out, b_out, p_in, b_in) unitary)
            unitary = self.get_tensor(params[:self.n_params])
            
            # if network_type is set to random-holoMPS:
            if network_type == 'random_state': 
            
                # defining tensor dimensions
                tensor = np.swapaxes(unitary[:,:,0,:],1,2) # change to MPS-based structure
                d = tensor[:,0,0].size # physical leg dimension (for random state)
                chi = tensor[0,:,0].size # bond leg dimension (for random state)
                
                # list of variational probability weights and random selection of each site
                prob_list = thermal_state.prob_list(self,params,T) 
                random_list = [random.choice(p) for p in prob_list]
                index_list = [prob_list[j].index(random_list[j]) for j in range(N)]
                tensor_list1 = [np.swapaxes(unitary[:,:,j,:],1,2) for j in index_list]

            # if network_type is set to holoMPS:
            elif network_type == 'circuit_MPS':
            
                # defining tensor dimensions
                # change the order of indices to (p_out, b_in, b_out) 
                # (with p_in = 0 to go from unitary to isometry)
                tensor = np.swapaxes(unitary[:,:,0,:],1,2) 
                d = tensor[:,0,0].size # physical leg dimension (for holoMPS)
                chi = tensor[0,:,0].size # bond leg dimension (for holoMPS)
                
                # bulk tensors of holoMPS structure
                tensor_list1 = [tensor]*N  

            # if network_type is set to circuit_MPO 
            # this option assumes original, circuit-based MPO structures (e.g. holoMPO)
            elif network_type == 'circuit_MPO':
                
                # defining tensor dimensions (consistent with rank-4 structures)
                # index ordering consistent with holographic-based MPO structures
                d = unitary[:,0,0,0].size # physical leg dimension (for MPO)
                chi = unitary[0,:,0,0].size # bond leg dimension (for MPO)
                tensor_list1 = [unitary]*N
            
            # testing boundary conditions 
            
            if network_type == 'random_state' or network_type == 'circuit_MPS': # specific to holoMPS-based structures
                
                if method == 'tenpy':
                    # based on previous circuit file
                    tensor_list1[0] = tensor_list1[0][:,0:1,:]
                    tensor_list1[-1] = tensor_list1[-1][:,:,0:1]
                    site = SpinHalfSite(None) 
                    M = MPS.from_Bflat([site]*N, tensor_list1, bc='finite', dtype=complex, form=None)
                    MPS.canonical_form_finite(M,renormalize=True,cutoff=0.0)
                
                elif method == 'thermal_state_class':
                    bdry = []
                    # if boundary vectors are not specified for holoMPS-based structures:     
                    # checking left boundary vector
                    # if left boundary vector not specified, set to (1,0,0,0...)
                    if np.array(bdry_vecs[0] == None).all():
                        bdry += [np.zeros(chi)]
                        bdry[0][0] = 1
                    else:
                        if bdry_vecs[0].size != chi:
                            raise ValueError('left boundary vector different size than bulk tensors')
                        bdry += [bdry_vecs[0]]
                
                    # checking right boundary vector (special to holoMPS-based structures)
                    if np.array(bdry_vecs[1] == None).all():
                        bdry += [None]
                    else:
                        if bdry_vecs[1].size != chi:
                            raise ValueError('right boundary vector different size than bulk tensors')
                        bdry += [bdry_vecs[1]]
                    
                    # if both boundary vectors are specified
                    for j in range(2):
                        if np.array(bdry_vecs[j] != None).all() and bdry_vecs[j].size == chi:
                            bdry.append(bdry_vecs[j]) 
                    
                    M = [[bdry[0]],tensor_list1,[bdry[1]]] # final state structure
                    
                else: 
                    raise ValueError('only one of "thermal_state_class" or "tenpy" options')
                        
            elif network_type == 'circuit_MPO': # specific to holoMPO-based structures
                bdry = []
                for j in range(2):
                    # if both boundary vectors are specified 
                    if np.array(bdry_vecs[j] != None).all() and bdry_vecs[j].size == chi:
                        bdry.append(bdry_vecs[j])
                    
                    # if boundary vectors not specified, set to (1,0,0,0...)
                    elif np.array(bdry_vecs[j] == None).all():
                        bdry += [np.zeros(chi)]
                        bdry[j][0] = 1
                    else:
                        if bdry_vecs[j].size != chi:
                            raise ValueError('boundary vectors different size than bulk tensors')
                        bdry += [bdry_vecs[j]]         
            
                M = [[bdry[0]],tensor_list1,[bdry[1]]] # final state structure
                      
        # if network_type is set to MPO: 
        # this option assumes genuine MPO_based structures (e.g. Hamiltonian MPO)  
        elif network_type == 'MPO':  
            
            if method == 'tenpy': # tenpy-based MPO
                site = SpinHalfSite(None)
                M = MPO.from_grids([site]*N, self, bc = 'finite', IdL=0, IdR=-1)  
                
            elif method == 'thermal_state_class':               
                # only bulk tensors of the main chain must be included (w/out params)
                tensor_list1 = [self]*N
                # testing boundary conditions
                bdry = []
                for j in range(2):
                    # if both boundary vectors are specified 
                    if np.array(bdry_vecs[j] != None).all() and bdry_vecs[j].size == chi_MPO:
                        bdry.append(bdry_vecs[j])
                
                    # if boundary vectors not specified, set to (1,0,0,0...)
                    elif np.array(bdry_vecs[j] == None).all():
                        bdry += [np.zeros(chi_MPO)]
                        bdry[j][0] = 1
                    else:
                        if bdry_vecs[j].size != chi_MPO:
                            raise ValueError('boundary vectors different size than bulk tensors')
                        bdry += [bdry_vecs[j]]
                
                M = [[bdry[0]],tensor_list1,[bdry[1]]] # final state structure
            else: 
                raise ValueError('only one of "thermal_state_class" or "tenpy" options')              
        else:
            raise ValueError('only one of "random_state", "circuit_MPS", "circuit_MPO", "MPO" options')
            
        return M
  
    def prob_list(self, params, T):
        """  
        Returns list of variational probability weights (based on Boltzmann ditribution)
        at each site for thermal-holographic matrix product state or thermal density matrix 
        product operator.
        --------------
        Inputs:
          --the input accepts holographic-based circuit structures--
          params: numpy.ndarray
             Parameters of circuit structure. This could also be any randomly generated 
             numpy.ndarray structure consistent with bulk tensor physical leg dimension.
          T: float
             Tempreture.
        """
        # evaluating the circuit (rank-4 (p_out, b_out, p_in, b_in) unitary)
        unitary = self.get_tensor(params[:self.n_params])
        prob_params = params[self.n_params:]  
        
        # tensor dimensions (consistent with rank-4 structure)
        d = unitary[:,0,0,0].size # physical leg dimension
        chi = unitary[0,:,0,0].size # bond leg dimension
        
        exc_list= [np.exp(-k/T) for k in prob_params] # list of boltzmann weights
        site_list = [exc_list[j:j+d] for j in range(0,len(exc_list),d)] # grouping weights for each site
        z_list = [sum(j) for j in site_list] # list of partition functions 
        norm_list = []
        for j in range(len(site_list)): # normalizing weights
            for k in site_list[j]:
                norm_list.append(k/z_list[j])
        prob_list = [norm_list[j:j+d] for j in range(0,len(norm_list),d)] # final probs for each site
        return prob_list
  
    def density_matrix(self, params, N, T, bdry_vecs=[None,None]):      
        """
        Returns thermal Density Matrix Product Operator (DMPO).
        --------------
        Inputs:
          --the input accepts holographic-based circuit structures--
          params: numpy.ndarray
             Parameters of circuit structure.
          N: int
            Number of sites in the main network chain (= L * l_uc, where L is number of 
            repetitions of the unit cell and l_uc is the length of unit-cell).
          T: float
             Tempreture.
          prob_list: list 
             List of probability weights of each physical state (the length of prob_list 
             should match the physical leg dimension). If set to None, it would call
             thermal_based prob_list fuction to compute probability weights for density
             matrix.
          bdry_vecs: list
            List of left (first element) and right (second element) boundary vectors.
            (set to [None,None] by default which gives left and right boundary vectors = |0>).   
        """
        # evaluating the circuit (rank-4 (p_out, b_out, p_in, b_in) unitary)
        unitary = self.get_tensor(params[:self.n_params])  
        
        # tensor dimensions (consistent with rank-4 structure)
        # index ordering consistent with holographic-based MPO structures
        d = unitary[:,0,0,0].size # physical leg dimension
        chi = unitary[0,:,0,0].size # bond leg dimension
        
        # constructing state and probability weights matrix chain 
        state = thermal_state.network_from_cells(self,'circuit_MPO',N,None,params,bdry_vecs,None,T)
        prob_list = thermal_state.prob_list(self,params,T)
        p_matrix_chain = [np.diag(p) for p in prob_list]
        
        # contractions of density matrix: 
        contractions = []
        for j in range(N):
            # contracting the probability weights chain with state
            W1 = np.tensordot(p_matrix_chain[j],state[1][j],axes=[1,0]) 
            W2 = np.tensordot(state[1][j].conj(),W1,axes=[2,0]) 
            # changing index ordering to: p_out, b_out, p_in, b_in
            W3 = np.swapaxes(np.swapaxes(W2,2,4),2,3)
            contractions.append(np.reshape(W3,[d,chi**2,d,chi**2]))
           
        # boundary contractions
        # state boundary contractions
        bvecl = np.kron(state[0][0].conj(),state[0][0]) 
        bvecr = np.kron(state[2][0].conj(),state[2][0])
        
        density_matrix = [[bvecl],contractions,[bvecr]]
        
        return density_matrix

    def network_site_contraction(self, state_type, chi_MPO=None, MPO=None):
        """
        Returns a list of contractions of the newtork at each site for <MPS|MPS> 
        (tranfer-matrix-like structures), <MPS|MPO|MPS>, or <MPO|DMPO> networks.
        MPS: holographic-matrix-product-state-based structures.
        MPO: matrix product operator.
        DMPO: density matrix product operator.
        --------------
        Inputs:
          --the input assumes thermal_state_class-based holoMPS and DMPO structures--
          state_type: str
             One of "random_state", "circuit_MPS", or "density_matrix" options. 
          chi_MPO: int
             Bond leg dimension for MPO-based structure. 
          MPO: thermal_state_class-based MPO structure.  
             Set to None for pure wave function simulations for MPS states.
        Note:
          -If MPO is not inserted for holoMPS states, the function computes transfer matrices 
           for the state wave fucntion at each site.
          -Length of MPO structure might be less than length of state.
          -The output would be returned as a list of contraction matrices computed for each 
           unit cell at each site.
        """     
        contraction_list = []
        # for holoMPS and random holoMPS-based structures: 
        if state_type == 'random_state' or state_type == 'circuit_MPS':
            
            # tensor dimensions (consistent with rank-3 structures)
            # index ordering consistent with holoMPS-based structures
            tensor = self[1][0]
            d = tensor[:,0,0].size # physical leg dimension 
            chi = tensor[0,:,0].size # bond leg dimension 
            N = len(self[1]) # number of sites (L * l_uc) in the main network chain (for holoMPS state).
            
            # contracted (transfer) matrices for the wave function: 
            # (w/out MPO inserted)
            if MPO == None: 
                # site contractions for state and its dual
                for j in range(N):
                    # contraction state/dual state
                    tensor1 = np.tensordot(self[1][j].conj(),self[1][j],axes=[0,0])
                    # reshaping into matrix 
                    # contraction (transfer) matrix at each site
                    tensor2 = np.reshape(np.swapaxes(np.swapaxes(np.swapaxes(tensor1,0,3),0,1),2,3),[chi**2,chi**2]) 
                    contraction_list.append(tensor2)                 
     
            # contracted matrices w/ MPO inserted
            else: 
                N_MPO = len(MPO[1]) # number of sites for squeezed MPO structure.
                # site contractions for state/MPO/dual-state
                for j in range(N_MPO):
    
                    # contractions with state
                    chi_s = chi_MPO * chi
                    tensor1 = np.tensordot(MPO[1][j],self[1][j],axes=[2,0])
                    tensor2 = np.reshape(np.swapaxes(np.swapaxes(tensor1,1,2),2,3),[d,chi_s,chi_s])

                    # contractions with dual state
                    chi_tot = chi_s * chi # total bond dimension
                    tensor3 = np.tensordot(self[1][j].conj(),tensor2,axes=[0,0])
                    # contracted matrices at each site
                    tensor4 = np.reshape(np.swapaxes(np.swapaxes(np.swapaxes(tensor3,0,3),0,1),2,3),[chi_tot,chi_tot])
                    contraction_list.append(tensor4)
                
                # contraction of rest of state and its dual if N_MPO different than N
                for j in range(N-N_MPO):
                    # contraction state/dual state
                    tensor5 = np.tensordot(self[1][j].conj(),self[1][j],axes=[0,0])
                    # reshaping into matrix 
                    # contraction (transfer) matrix at each site
                    tensor6 = np.reshape(np.swapaxes(np.swapaxes(np.swapaxes(tensor5,0,3),0,1),2,3),[chi**2,chi**2]) 
                    contraction_list.append(tensor6)    

        
        # contracted network structure at each site for density matrix
        # must include MPO (e.g. Hamiltonian MPO)
        elif state_type == 'density_matrix':
            
            N = len(self[1]) # number of sites (L * l_uc) in the main network chain (for density matrix).
            N_MPO = len(MPO[1]) # number of sites for inserted MPO structure.
            # tensor dimensions
            d_s = self[1][0][:,0,0,0].size # state physical leg dimension
            d_MPO = MPO[1][0][:,0,0,0].size # MPO physical leg dimension
            chi_s = self[1][0][0,:,0,0].size # state bond leg dimension
            chi_MPO = MPO[1][0][0,:,0,0].size # MPO bond leg dimension
            chi_tot = chi_s * chi_MPO # total bond leg dimension
            
            contraction_list = []
            for j in range(N):
                # MPO and density matrix constractions
                tensor1 = np.tensordot(self[1][j],MPO[1][j],axes=[2,0])
                # changing index ordering to: p_out, b_out, p_in, b_in
                tensor2 = np.swapaxes(np.swapaxes(tensor1,2,4),2,3)
                tensor3 = np.reshape(tensor2,[d_MPO,chi_tot,d_s,chi_tot])
                # tracing over p_out and p_in
                tensor4 = np.trace(tensor3,axis1=0,axis2=2)
                contraction_list.append(np.reshape(tensor4,[chi_tot,chi_tot]))
                
        else:
            raise ValueError('only one of "random_state", "circuit_MPS", or "density_matrix" options')
            
        return contraction_list

    def expectation_value(self, state_type, chi_MPO=None, MPO=None):
        """
        Returns the numerical result of full contractions of <MPS|MPS>,
         <MPS|MPO|MPS>, or <MPO|DMPO> networks.
         MPS: holographic-matrix-product-state-based structures.
         MPO: matrix product operator.
         DMPO: density matrix product operator.
        --------------
        Inputs:
          --the input assumes thermal_state_class-based holoMPS and DMPO structures--
          state_type: str
             One of "random_state", "circuit_MPS", or "density_matrix" options. 
          chi_MPO: int
             Bond leg dimension for MPO-based structures.
          MPO: thermal_state_class-based MPO structure.
             Set to None for pure wave function simulations.
        Note:
          -Left boundary condition is set by the given holoMPS boundary vectors, and the right 
           boundary is averaged over (as consistent with holographic-based simulations).
          -If MPO is not inserted (for MPS structures), the function computes the expectation value 
           for the state wave fucntion (<MPS|MPS>).
        """
        con_mat = thermal_state.network_site_contraction(self,state_type,chi_MPO,MPO) # list of contracted matrices
        # accumulation of contracted matrices defined at each site
        con_mat0 = con_mat[0]
        for j in range(1,len(con_mat)):
            con_mat0 = con_mat[j] @ con_mat0
        #con_mat0 = np.linalg.matrix_power(con0,L)
        
        # for holoMPS and random holoMPS-based structures:
        if state_type == 'random_state' or state_type == 'circuit_MPS':
           
            # tensor dimensions (consistent with rank-3 structure)
            # index ordering consistent with holoMPS-based structures
            tensor = self[1][0]
            d = tensor[:,0,0].size # physical leg dimension (for holoMPS)
            chi = tensor[0,:,0].size # bond leg dimension (for holoMPS)
    
            # w/out MPO inserted
            if MPO == None:
                bvecl = np.kron(self[0][0].conj(),self[0][0]) # left boundary contraction
                # right boundary contraction:
                if np.array(self[2][0] == None).all():
                    # summing over right vector if right boundary condition is not specified 
                    con_mat_on_rvec = np.reshape(con_mat0 @ bvecl,[chi**2])
                    rvec = np.reshape(np.eye(chi),[chi**2])
                    expect_val = np.dot(rvec,con_mat_on_rvec)
                else:
                    bvecr = np.kron(self[2][0].conj(),self[2][0])
                    expect_val = bvecr.conj().T @ con_mat0 @ bvecl
        
           # w/ MPO inserted
            else:
                bvecl = np.kron(self[0][0].conj(),np.kron(MPO[0][0],self[0][0])) # left boundary contraction
                # right boundary constraction:
                if np.array(self[2][0] == None).all():
                    # summing over right vectors if right boundary condition is not specified
                    # employ the specified right boundary vector of MPO.                 
                    con_vleft = np.reshape((con_mat0 @ bvecl),[chi,chi_MPO,chi]) # con_mat on left vector
                    MPO_rvec_contracted = np.reshape(np.tensordot(MPO[2][0],con_vleft,axes=[0,1]),[chi**2])
                    rvec = np.reshape(np.eye(chi),[chi**2])
                    expect_val = np.dot(rvec,MPO_rvec_contracted)
                else:
                    bvecr = np.kron(self[2][0].conj(),np.kron(MPO[2][0],self[2][0]))
                    expect_val = bvecr.conj().T @ con_mat0 @ bvecl

        # for density-matrix-based structures:
        # must include MPO (e.g. Hamiltonian MPO)
        elif state_type == 'density_matrix':      
            # boundary vector contractions
            bvecl = np.kron(self[0][0],MPO[0][0])
            bvecr = np.kron(self[2][0],MPO[2][0])
            expect_val = bvecr.conj().T @ con_mat0 @ bvecl
        
        else:
            raise ValueError('only one of "random_state", "circuit_MPS", or "density_matrix" options')
          
        return (expect_val).real

    def entropy(prob_list, N):
        """
        Returns the von Neumann entropy (per site) of a given list 
        probability weight list (the form of Shannon entropy).
        --------------
        --the input assumes thermal_state_class-based prob_list--
        N: int
            Number of sites in the main network chain (= L * l_uc, where L is number of 
            repetitions of the unit cell and l_uc is the length of unit-cell).   
        """
        new_prob_list = [np.array(j)[np.array(j) > 1.e-30] for j in prob_list] # avoiding NaN in numpy.log() function
        s_list1 = []
        for j in range(len(new_prob_list)):
            for p in new_prob_list[j]:
                s_list1.append(-p*np.log(p)) # converting to form of Shannon entropy
                
        s_list2 = [sum(s_list1[j:j+d]) for j in range(0,len(s_list1),d)]
        s_tot = sum(s_list2)/N # average entropy of chain
        return s_tot
 
    def free_energy(self, params, state_type, N, Hamiltonian, T, chi_H=None, bdry_vecs1=None, bdry_vecs2=None, method=None):
        """
        Returns the Helmholtz free energy of a thermal density matrix structure 
         or thermal holographic matrix product state.
        --------------
        Inputs:
        --the input accepts holographic-based circuit structures--
        state_type: str
           One of "density_matrix" or "random_state" options.
        N: int
            Number of sites in the main network chain (= L * l_uc, where L is number of 
            repetitions of the unit cell and l_uc is the length of unit-cell).
        params: numpy.ndarray
           Parameters of circuit structure.
        Hamiltonian: numpy.ndarray 
           The unit cell of the Hamiltonian MPO of model.  
        T: float
           Tempreture.
        chi_H: int
           Bond leg dimension for Hamiltonian MPO structure.
        bdry_vecs1 and bdry_vecs2: list
            List of left (first element) and right (second element) boundary vectors for 
            state and Hamiltonian networks, respectively (set to [None,None] by default).
        method: str 
            One of "thermal_state_class" or "tenpy" options. The option is currently available
            for "random_state".
        """        
        # for density-matrix-based structures:
        if state_type == 'density_matrix':
            
            S = thermal_state.entropy(thermal_state.prob_list(self,params,T),N) # entropy
            density_mat = thermal_state.density_matrix(self,params,N,T,bdry_vecs1) # density matrix            
             
            MPO_Hamiltonian = thermal_state.network_from_cells(Hamiltonian,'MPO',N,chi_H,None,bdry_vecs2,'thermal_state_class',T) # Hamiltonian MPO      
            E = thermal_state.expectation_value(density_mat,'density_matrix',chi_H,MPO_Hamiltonian) # energy of system    
            F = (E/N) - T*S # Helmholtz free energy 
            
        # for random-holoMPS-based structures:
        elif state_type == 'random_state':
            
            random_state = thermal_state.network_from_cells(self,'random_state',N,None,params,bdry_vecs1,method,T) # random_state MPS
            MPO_Hamiltonian = thermal_state.network_from_cells(Hamiltonian,'MPO',N,chi_H,None,bdry_vecs2,method,T) # Hamiltonian MPO
            S = thermal_state.entropy(thermal_state.prob_list(self,params,T),N) # entropy
            
            if method == 'thermal_state_class':   
                E = thermal_state.expectation_value(random_state,'random_state',chi_H,MPO_Hamiltonian) # energy of system
                F = (E/N) - T*S # Helmholtz free energy   
            
            elif method == 'tenpy':
                E = (MPO_Hamiltonian.expectation_value(random_state)).real # energy of system
                F = (E/N) - T*S # Helmholtz free energy
            else: 
                raise ValueError('only one of "thermal_state_class" or "tenpy" options')
                
        else:
            raise ValueError('only one of "random_state" or "density_matrix" options')
        return F
