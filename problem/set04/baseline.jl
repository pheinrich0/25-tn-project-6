## T04: Two Site DMRG 
using LinearAlgebra
using Printf
using JLD2

using tn_julia: tightbindinghamiltonian, exactffenergy, extendhamiltonian_ff, iterativediagonalization, ffchainmpo, updateLeft

## a) Warm-up 
##      i) exact reference energy 

L = 100; # chain length

H = tightbindinghamiltonian(L);
Eex = exactffenergy(L);

##      ii) Iterative diagonalization 
##      iii) MPS

# maximal # of states to keep
Nkeep = 50;

extendhamiltonian(H::AbstractMatrix, A::AbstractArray{<:Number,3}) = extendhamiltonian_ff(H, A; t=1.0)
energies, MPS = iterativediagonalization(extendhamiltonian, 2, L, Nkeep)
Eiter = minimum(energies);

@printf("Eiter: %.6f\n",Eiter)
# err: Relative error
err = (Eiter-Eex)/abs(Eiter)
@printf("err: %.6f %%\n",100*err)

##      iv) MPO

W = ffchainmpo(L);

E_MPO = reshape([1],(1,1,1));
for itL in (1:L)
    global E_MPO = updateLeft(E_MPO, MPS[itL], W[itL], MPS[itL]);
end

@printf("Eiter - E_MPO : %.6f", Eiter - E_MPO[1])
@save "problem/set04/baseline.jld2" MPS W Eex Eiter L
