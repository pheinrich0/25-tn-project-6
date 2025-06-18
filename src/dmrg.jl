function DMRG_2site(Hs,Minit,alpha,Nkeep,Nsweep;Econv=-Inf)
# < Description >
#
# M,E0,Eiter,dw = DMRG_2site(Hs,Minit,Nkeep,Nsweep [;Econv])
#
# Single-site density-matrix renormalization group (DMRG) 
# calculation to search for the ground state & its energy
# of a one-dimensional system; whose Hamiltonian is given 
# by the matrix product operator Hs.
#
# < Input >
# Hs : [1 x N cell array] Matrix product operator (MPO) of the
#	Hamiltonian. Each Hs[n] acts on site n, & is a rank-4
#	tensor. The order of legs of Hs[n] is
#	left-bottom-right-top, where bottom (top) leg is to be 
#	contracted with bra (ket) tensor. The length N of Hs, 
#	i.e., length(Hs), determines the chain length.
# Minit : [1 x N cell array] Initial MPS from which to start the 
#	ground state search
# alpha : [numeric] bond growth factor. The bond dimension is 
#	grown from its initial value Di to its final value 
#	Df = max(\alpha*Di,Nkeep)
# Nkeep : [numeric] Maximum bond dimension of the matrix product 
#	state (MPS) to consider.
# Nsweep : [numeric] Number of sweeps will be 2*Nsweep, as there 
#	are Nsweep times of round trip 
#	(right -> left, left -> right).
#
# <Option>
# 'Econv',.. : [numeric] Convergence criterion for energy. If 
#	Einit - Efin < Econv, stop sweeping even if less than
#	Nsweep sweeps have been done so far. Here, Einit and
#	Efin are the energies before and after one 
#	(right -> left, left -> right) round trip, respectively.
#	(Default: -inf, i.e. no energy convergence criterion.)
#
# < Output >
# M : [1 x N cell array] The result MPS which is obtained 
#	variationally to have the minimum expectation value of the
#	Hamiltonian H. It is in *left-canonical* form; since the 
#	last sweep is from left to right.
# E0 : [numeric] The energy of M.
# Eiter : [N x [2*Nsweep] numeric array] Each element Eiter[m,n] 
#	means the variational energy in the m-th iteration within 
#	the n-th sweep. Odd n is for right-to-left sweep & even n 
#	for left-to-right sweep. Note that the iteration index m 
#	matches with the site index for the left-to-right sweep; 
#	the iteration m corresponds to the site (N+1-m) for 
#	the right-to-left sweep.
# dw : [N x (2*Nsweep) numeric array] Each element dw(m,n) is the
#	discarded weight in the m-th iteration within
#       the n-th sweep. Odd n is for right-to-left sweep and even n
#       for left-to-right sweep. Note that the iteration index m
#       matches with the site index for the left-to-right sweep;
#       the iteration m corresponds to the site (N+1-m) for
#       the right-to-left sweep.
#
# Written by S.Lee [May 28,2019]
# Updated by S.Lee [May 23,2020]: Revised for SoSe2020.
# Update by J.Shim [May 25.2022]: Revised for SoSe2022.
# Transformed to Julia by Changkai Zhang [June 8, 2022].
    
# # sanity check for input & option
N = length(Hs)

if N < 2
    error("ERR: chain is too short.")
end

for itN = (1:N)
    if size(Hs[itN],2) != size(Hs[itN],4)
        error("ERR: The second & fourth legs of Hs{$itN} have different dimensions.")
    end
end
# # #

# show message
print("Two-site DMRG: search for the ground state\n")
print("# of sites = $N, Nkeep = $Nkeep, # of sweeps = $Nsweep x 2\n")

# initialize MPS by Minit
M = Minit;
# bring M into canonical form in place
rightcanonical!(M, Nkeep=Nkeep, tolerance=1e-8)
leftcanonical!(M,  Nkeep=Nkeep, tolerance=1e-8)

# ground-state energy for each iteration, Changed for 
# two-site DMRG; 1st dimension is N-1
Eiter = zeros(N-1,2*Nsweep)
# later, Eiter[end,end] will be taken as the final result E0

# discarded weight for each iteration
# Changed for two-site DMRG; 1st dimension is N-1
dw = zeros(N-1,2*Nsweep);

# # Hamiltonian for the left/right parts of the chain
Hlr = Array{Any}(undef,1,N+2);
Hlr[1] = reshape([1],(1,1,1));
Hlr[end] = reshape([1],(1,1,1));

# Since M is in left-canonical form by now, Hlr[..] are the left parts of
# the Hamiltonian. That is, Hlr[n+1] is the left part of Hamiltonian which
# is obtained by contracting M[1:n] with Hs[1:n]. (Note the index for Hlr
# is n+1, not n, since Hlr[1] is dummy.)
# initialize Hlr
for itN in 1:N
    Hlr[itN+1] = updateLeft(Hlr[itN], M[itN], Hs[itN], M[itN])
end

for itS = (1:Nsweep)

    # right -> left
    for itN = (N:-1:2) # Changed for two-site DMRG; different range of index
        # # # Changed for two-site DMRG [start] # # #
        # merge two rank-4 tensors for the two lattice sites to a 
        # single rank-4 tensor
        # merge the physical legs of Hloc by identity
        I_HH = identity(Hs[itN-1],2,Hs[itN],2)
        Hloc = contract(Hs[itN-1], [2], I_HH, [1])
        Hloc = contract(Hloc, [2,4], Hs[itN], [1,2])
        Hloc = contract(Hloc, [2,5], I_HH, [1,2])
        
        # merge two rank-3 ket tensors to a single rank-3 tensor
        # merge the physical legs of Ain by identity
        I_MM = identity(M[itN-1],2,M[itN],2)
        Cin = contract(M[itN-1], [2], I_MM, [1])
        Cin = contract(Cin, [2,3], M[itN], [1,2])
        # # # Changed for two-site DMRG [end] # # #
        
        # Use eigens_1site to obtain the variationally chosen ket tensor
        # Ceff & energy expectation value Eeff
        Ceff,Eeff = lanczos_1site(Hlr[itN-1],Hloc,Hlr[itN+2],Cin)
        # Changed for two-site DMRG; different indexing for the 3rd input
        
        Eiter[N+1-itN,2*itS-1] = Eeff
        
        # # # Changed for two-site DMRG [start] # # #
        # re-split the local leg by identity()
        Ceff = contract(Ceff, [2], I_MM, [3], [1,3,4,2])
        # update M[itN-1] & M[itN] by using Ceff, via SVD
        # decompose Ceff
        Di = size(M[itN-1],3); # initial bond-dimension
        Df = Int64(ceil(min(alpha*Di,Nkeep))); # final bond-dimension
        UT, ST, M[itN], dw[N+1-itN,2*itS-1] = svd(Ceff, [1,2], Nkeep=Df)
        # two additional features Df & dw are introudced
        M[itN-1] = contract(UT, 3, diagm(ST), 1)
        # # # Changed for two-site DMRG [end] # # #
        
        # update Hlr after right->left SVD
        T = permutedims(M[itN], (3,2,1))
        if itN == 1
            Hlr[itN+1] = []
        else
            # permute left<->right for Hs[itN] as well, to make use of updateLeft
            H2 = permutedims(Hs[itN],(3,2,1,4)); # right-bottom-left-top
            Hlr[itN+1] = updateLeft(Hlr[itN+2], T, H2, T)
        end
    end
    
    # display informaiton of the sweep
    @printf("Sweep #%i,%i (right -> left) : Energy = %.7g\n",
        2*itS-1,2*Nsweep,Eiter[N-1,2*itS-1])
    
    # left -> right
    for itN = (1:(N-1)) # Changed for two-site DMRG; different range of index
        # # # Changed for two-site DMRG [start] # # #
        # merge two rank-4 tensors for the two lattice sites to a
        # single rank-4 tensor
        # merge the physical legs of Hloc by identity
        I_HH = identity(Hs[itN],2,Hs[itN+1],2)
        Hloc = contract(Hs[itN], [2], I_HH, [1])
        Hloc = contract(Hloc, [2,4], Hs[itN+1], [1,2])
        Hloc = contract(Hloc, [2,5], I_HH, [1,2])
    
        # merge two rank-3 ket tensors to a single rank-3 tensor
        # merge the physical legs of Ain by identity
        I_MM = identity(M[itN],2,M[itN+1],2)
        Cin = contract(M[itN], [2], I_MM, [1])
        Cin = contract(Cin, [2,3], M[itN+1], [1,2])
        # # # Changed for two-site DMRG [end] # # #
        
        Ceff,Eeff = lanczos_1site(Hlr[itN],Hloc,Hlr[itN+3],Cin)
        # Changed for two-site DMRG; different indexing for the 3rd input
    
        Eiter[itN,2*itS] = Eeff
        
        # # # Changed for two-site DMRG [start] # # #
        # re-split the local leg by identity()
        Ceff = contract(Ceff, [2], I_MM, [3], [1,3,4,2])
        # update M[itN-1] & M[itN] by using Aeff, via SVD
        # decompose Aeff
        Di = size(M[itN],3); # initial bond-dimension
        Df = Int64(ceil(min(alpha*Di,Nkeep))); # final bond-dimension
        M[itN], ST, VT, dw[itN,2*itS] = svd(Ceff, [1,2], Nkeep=Df)
        # two additional features Df & dw are introudced
        M[itN+1] = contract(diagm(ST), 2, VT, 1)
        # # # Changed for two-site DMRG [end] # # #
        
        # update Hlr after left->right SVD
        if itN == N
            Hlr[itN+1] = []
        else
            Hlr[itN+1] = updateLeft(Hlr[itN], M[itN], Hs[itN], M[itN])
        end
    end
    
    # display informaiton of the sweep
    @printf("Sweep #%i,%i (left -> right) : Energy = %.7g\n",
        2*itS,2*Nsweep,Eiter[N-1,2*itS])

    global E0 = Eiter[N-1,2*itS]; # take the last value

    if itS > 1
        if abs(Eiter[N-1,2*itS] - Eiter[N-1,2*(itS-1)]) < Econv
            break # if ((itS-1)th energy - (itS)th energy), stop DMRG sweep
        end
    end
end

return M,E0,Eiter,dw

end



function DMRG_1site(Hs, Minit, Nkeep, Nsweep; Econv=-Inf)
# < Description >
#
# M,E0,Eiter = DMRG_1site(Hs,Minit,Nkeep,Nsweep [;Econv])
#
# Single-site density-matrix renormalization group (DMRG)
# calculation to search for the ground state & its energy
# of a one-dimensional system; whose Hamiltonian is given 
# by the matrix product operator Hs.
#
# < Input >
# Hs : [1 x N cell array] Matrix product operator (MPO) of the
#	Hamiltonian. Each Hs[n] acts on site n, & is a rank-4
#	tensor. The order of legs of Hs[n] is 
#	left-bottom-right-top, where bottom (top) leg is to be 
#	contracted with bra (ket) tensor. The length N of Hs, 
#	i.e., length(Hs), determines the chain length.
# Minit : [1 x N cell array] Initial MPS from which to start the 
#	ground state search
# Nkeep : [numeric] Maximum bond dimension of the matrix product 
#	state (MPS) to consider.
# Nsweep : [numeric] Number of sweeps will be 2*Nsweep, as there 
#	are Nsweep times of round trip 
#	(right -> left, left -> right).
#
# <Option>
# 'Econv',.. : [numeric] Convergence criterion for energy. If 
#	Einit - Efin < Econv, stop sweeping even if less than
#	Nsweep sweeps have been done so far. Here, Einit and
#	Efin are the energies before and after one 
#	(right -> left, left -> right) round trip, respectively.
#	(Default: -inf, i.e. no energy convergence criterion.)
#
# < Output >
# M : [1 x N cell array] The result MPS which is obtained 
#	variationally to have the minimum expectation value of the
#	Hamiltonian H. It is in *left-canonical* form; since the 
#	last sweep is from left to right.
# E0 : [numeric] The energy of M.
# Eiter : [N x [2*Nsweep] numeric array] Each element Eiter[m,n] 
#	means the variational energy in the m-th iteration within 
#	the n-th sweep. Odd n is for right-to-left sweep & even n 
#	for left-to-right sweep. Note that the iteration index m 
#	matches with the site index for the left-to-right sweep; 
#	the iteration m corresponds to the site (N+1-m) for 
#	the right-to-left sweep.
#
# Written by S.Lee [May 28,2019]
# Updated by S.Lee [May 23,2020]: Revised for SoSe2020.
# Update by J.Shim [May 25.2022]: Revised for SoSe2022.
# Transformed to Julia by Changkai Zhang [June 8, 2022].
    
# # sanity check for input & option
N = length(Hs)

if N < 2
    error("ERR: chain is too short.")
end

for itN = (1:N)
    if size(Hs[itN],2) != size(Hs[itN],4)
        error("ERR: The second & fourth legs of Hs{$itN} have different dimensions.")
    end
end
# # #

# show message
print("Single-site DMRG: search for the ground state\n")
print("# of sites = $N, Nkeep = $Nkeep, # of sweeps = $Nsweep x 2\n")

# initialize MPS by Minit
M = Minit;
# M,_,_ = canonForm(M,0); # bring into right-canonical form
# M,_,_ = canonForm(M,N); # bring into left-canonical form
rightcanonical!(M, Nkeep=Nkeep, tolerance=1e-8); # right-canonical form
leftcanonical!(M, Nkeep=Nkeep, tolerance=1e-8); # left-canonical form

# ground-state energy for each iteration
Eiter = zeros(N,2*Nsweep)
# later, Eiter[end,end] will be taken as the final result E0

# # Hamiltonian for the left/right parts of the chain
Hlr = Array{Any}(undef,1,N+2);
Hlr[1] = reshape([1],(1,1,1));
Hlr[end] = reshape([1],(1,1,1));

# Since M is in left-canonical form by now, Hlr[..] are the left parts of
# the Hamiltonian. That is, Hlr[n+1] is the left part of Hamiltonian which
# is obtained by contracting M[1:n] with Hs[1:n]. (Note the index for Hlr
# is n+1, not n, since Hlr[1] is dummy.)
for itN in (1:N)
    Hlr[itN+1] = updateLeft(Hlr[itN], M[itN], Hs[itN], M[itN])
end

for itS = (1:Nsweep)
    # right -> left
    for itN = (N:-1:1)
        # Use lanczos_1site to obtain the variationally chosen ket tensor
        # Ceff & energy expectation value Eeff
        Ceff,Eeff = lanczos_1site(Hlr[itN],Hs[itN],Hlr[itN+2],M[itN])
        
        Eiter[N+1-itN,2*itS-1] = Eeff
        
        # update M[itN] & M[itN-1] by using Ceff, via SVD
        # decompose Ceff
        US, M[itN] = svdright(Ceff, Nkeep=Nkeep)
        # contract UT*ST with M[itN], to update M[itN]
        if itN > 1
            M[itN-1] = contract(M[itN-1], [3], US, [1])
        end
        
        # update the Hamiltonian in effective basis
        T = permutedims(M[itN],(3,2,1)); # permute left<->right, to make use of updateLeft()
        if itN == 1
            Hlr[itN+1] = []
        else
            # permute left<->right for Hs[itN] as well, to make use of updateLeft()
            H2 = permutedims(Hs[itN],(3,2,1,4)); # right-bottom-left-top
            Hlr[itN+1] = updateLeft(Hlr[itN+2], T, H2, T)
        end
    end
    
    # display informaiton of the sweep
    @printf("Sweep #%i,%i (right -> left) : Energy = %.7g\n",
        2*itS-1,2*Nsweep,Eiter[N,2*itS-1])
    
    # left -> right
    for itN = (1:N)
        Ceff,Eeff = lanczos_1site(Hlr[itN],Hs[itN],Hlr[itN+2],M[itN])
        
        Eiter[itN,2*itS] = Eeff
        
        # update M[itN] & M[itN+1] by using Ceff, via SVD
        # decompose Ceff
        M[itN], SV = svdleft(Ceff, Nkeep=Nkeep)
        # contract UT*ST with M[itN], to update M[itN]
        if itN < N
            M[itN+1] = contract(SV, [2], M[itN+1], [1])
        end
        
        if itN == N
            Hlr[itN+1] = []
        else
            Hlr[itN+1] = updateLeft(Hlr[itN], M[itN], Hs[itN], M[itN])
        end
    end
    
    # display informaiton of the sweep
    @printf("Sweep #%i,%i (left -> right) : Energy = %.7g\n",
        2*itS,2*Nsweep,Eiter[N,2*itS])

    global E0 = Eiter[N-1,2*itS]; # take the last value

    if itS > 1
        if abs(Eiter[N-1,2*itS] - Eiter[N-1,2*(itS-1)]) < Econv
            break # if ((itS-1)th energy - (itS)th energy), stop DMRG sweep
        end
    end
end

return M,E0,Eiter

end



function lanczos_1site(Hleft,Hloc,Hright,Cinit;N=5,minH=1e-10)
# < Description >
#
# Ceff,Eeff = lanczos_1site(Hleft,Hloc,Hright,Cinit [; option])
#
# Obtain the ground state & its energy for the effective 
# Hamiltonian for the site-canonical MPS; by using the Lanczos 
# method.
#
# < Input >
# Hleft, Hloc, Hright: [tensors] The Hamiltonian for the left, site
#	 & right parts of the chain. They form the effective 
#	 Hamiltonian in the site-canonical basis.
# Cinit : [tensor] Ket tensor at a lattice site. It becomes an 
#	initial vector for the Lanczos method.
#
# The input tensors can be visualized as follows:
# (numbers are the order of legs, * means contraction)
#
#
#      	    1 -->-[ Cinit ]-<-- 3
#                     |
#                     ^ 2
#                     |
#
#
#     /--->- 3        | 4        3 -<---\
#     |               ^                 |
#     |    2     1    |    3     2      |
#   Hleft-->- * -->- Hloc->-- * ->-- Hright
#     |               |                 |
#     |               ^                 |
#     \---<- 1        | 2        1 ->---/
#
# < Option >
# N = .. : [numeric] Maximum number of Lanczos vectors [in 
#	addition to those given by Cinit] to be considered 
#	for the Krylov subspace.
#       (Default: 5)
# minH = .. : [numeric] Minimum absolute value of the 1st diagonal 
#	(i.e., superdiagonal) element of the Hamiltonian in the 
#	Krylov subspace. If a 1st-diagonal element whose absolute 
#	value is smaller than minH is encountered; the iteration 
#	stops. Then the ground-state vector & energy is obtained 
#	from the tridiagonal matrix constructed so far.
#       (Default: 1e-10)
#
# < Output >
# Ceff : [tensor] A ket tensor as the ground state of the 
#	effective Hamiltonian.
# Eeff : [numeric] The energy eigenvalue corresponding to Ceff.
# Written by S.Lee [May 31,2017]
# Documentation updated by S.Lee [Jun.8,2017]
# Updated by S.Lee [May 28,2019]: Revised for SoSe 2019.
# Updated by S.Lee [May 23,2020]: Revised for SoSe 2020.
# Updated by S.Lee [Jun.08,2020]: Typo fixed.
# Updated by J.Shim [May 25,2020]: Revised for SoSe 2022.
# Transformed to Julia by Changkai Zhang [June 8, 2022].

# size of ket tensor
Csz = (size(Cinit,1),size(Cinit,2),size(Cinit,3))

# initialize Cinit
Cinit = Cinit/norm(Cinit[:]); # normalize Cinit

# Krylov vectors [vectorized tensors]
Ckr = zeros(length(Cinit),N+1)
Ckr[:,1] = Cinit[:]

# In the Krylov basis; the Hamiltonian becomes tridiagonal
ff = zeros(N); # 1st diagonal
gg = zeros(N+1); # main diagonal

for itN = (1:(N+1))
    # contract Hamiltonian with ket tensor
    Ctmp = lanczos_1site_HC(Hleft,Hloc,Hright,reshape(Ckr[:,itN],Csz))
    Ctmp = Ctmp[:]; # vectorize
    
    gg[itN] = Ckr[:,itN]'*Ctmp; # diagonal element; "on-site energy"
    
    if itN < (N+1)
        # orthogonalize Atmp w.r.t. the previous ket tensors
        Ctmp = Ctmp - Ckr[:,(1:itN)]*(Ckr[:,(1:itN)]'*Ctmp)
        # twice; to reduce numerical noise
        Ctmp = Ctmp - Ckr[:,(1:itN)]*(Ckr[:,(1:itN)]'*Ctmp)
        
        # norm
        ff[itN] = norm(Ctmp)
        
        if ff[itN] > minH
            Ckr[:,itN+1] = Ctmp/ff[itN]
        else
            # stop iteration; truncate ff; gg
            ff = ff[1:itN-1]
            gg = gg[1:itN]
            Ckr = Ckr[:,1:itN]
            break
        end
    end
end

# Hamiltonian in the Krylov basis
Hkr = diagm(1 => ff)
Hkr = Hkr + Hkr' + diagm(gg)
Ekr,Vkr = eigen((Hkr+Hkr')/2)
_,minid = findmin(diagm(Ekr))

# ground state
Ceff = Ckr*Vkr[:,minid]
Ceff = Ceff/norm(Ceff); # normalize
Ceff = reshape(Ceff,Csz); # reshape to rank-3 tensor

# ground-state energy; measure again
Ctmp = lanczos_1site_HC(Hleft,Hloc,Hright,Ceff)
Eeff = Ceff[:]'*Ctmp[:]

return Ceff,Eeff

end



function lanczos_1site_HC(Hleft,Hloc,Hright,Cin)
# < Description >
#
# Cout = lanczos_1site_HC(Hleft,Hloc,Hright,Cin)
#
# Apply the effective Hamitonian for the site-canonical MPS.
# 
# < Input >
# Hleft, Hloc, Hright: [tensors] Refer to the description of the variables
#       with the same names; in "DMRG_1site_eigs".
# Cin : [tensor] A ket tensor at a lattice site, to be applied by the
#       effective Hamiltonian.
#
# < Output >
# Cout : [tensor] A ket tensor at a lattice site, after the application of
#   the effective Hamiltonian to Cin.
#
# Written by S.Lee [May 23,2020]
# Updated by S.Lee [May 27,2020]: Minor change.
# Updated by J.Shim [May 25,2022]: Revised for SoSe 2022.
# Transformed to Julia by Changkai Zhang [June 8, 2022].

# set empty tensors as 1; for convenience
if isempty(Hleft)
    Hleft = 1
end
if isempty(Hright)
    Hright = 1
end

Cout = contract(Hleft, [3], Cin, [1])
Cout = contract(Cout, [2,3], Hloc, [1,4])
Cout = contract(Cout, [2,4], Hright, [3,2])

return Cout

end