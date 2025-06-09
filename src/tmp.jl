# Exact GS energy Eex of Hff from numerical diagonalization
# row indices: |1>;|2>;..
# column indces: <1|;<2|;..

using LinearAlgebra
import tn_julia: iterativediagonalization, extendhamiltonian_ff, exactffenergy, sitecanonical!, computeleftenvironment, computerightenvironment, ffchainmpo

L = 100 # chain length
Nkeep = 50

extendhamiltonian(H::AbstractMatrix, A::AbstractArray{<:Number, 3}) = 
    extendhamiltonian_ff(H, A; t=1.0)
energies, MPS = iterativediagonalization(extendhamiltonian, 2, L, Nkeep)

Eiter = energies[end]

Eex = exactffenergy(L)

println("Relative error: $((Eiter-Eex)/abs(Eex))")

println([size(W[ell]) for ell in 1:length(W)])
println([size(MPS[ell]) for ell in 1:length(MPS)])

W = ffchainmpo(L)
ell = 50
sitecanonical!(MPS, ell)
Lell = computeleftenvironment(W, MPS, ell-1)
Rell = computerightenvironment(W, MPS, ell+1)
Cell = MPS[ell]

E = contract(HC, [1, 2, 3], conj(Cℓ), [1, 2, 3])

println(size(W[ell]))
println(size(Lell))
println(size(Rell))
println(size(Cell))

