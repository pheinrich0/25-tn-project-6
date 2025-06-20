using Arpack
using LinearAlgebra
using Statistics
using Printf
using JLD2

import tn_julia: contract, identity, updateLeft
using tn_julia: identity as tn_identity
using tn_julia: svd as tn_svd 

@load "problem/set04/iTEBD_iii.jld2"

function canonIMPS_domVec(T, rankT, isconj)
# Find the tensor X where X*X' originates from the dominant right
# eigenenvector of the transfer operator which is
#
#  1                 n+2
#  -->-[     T     ]->--
#      2|   ...   |n+1
#       ^         ^
#       *   ...   *
#       ^         ^
#      2|   ...   |n+1
#  --<-[     T'    ]-<--
#  1                 n+2
#
# isconj == 1 if transfer operator from left; == 0 from right

    D = size(T, 1)
    idT = collect(2:rankT-1)
    T = contract(T, idT, conj(T), idT, [1, 3, 2, 4])
    T = reshape(T, (D * D, D * D))
    if isconj
        T = transpose(T)
    end
    _, V = eigs(T, nev=1, which=:LM)
    V = real(V[:, end])
    V = reshape(V, (D, D))
    V = (V + V') / 2
    D1, V1 = eigen(V)
    if sum(D1) < 0
        D1 = -D1
    end
    oks = D1 .> 0
    D1 = D1[oks]
    V1 = V1[:, oks]
    X = V1 * Diagonal(sqrt.(D1))
    if isconj
        X = transpose(X)
    end
    return X
end

for itD in 1:length(Ds)
    Ao = contract(diagm(Le[itD]), [2], Go[itD], [1])
    AAdag = contract(Ao, [1, 2], conj(Ao), [1, 2])
    Id = tn_identity(Ao, 3)
    @printf("norm(|AAdag - I|) at odd site (D=%d): %.2e\n", Ds[itD], norm(abs.(Id - AAdag)))
    Ae = contract(diagm(Lo[itD]), [2], Ge[itD], [1])
    AAdag = contract(Ae, [1, 2], conj(Ae), [1, 2])
    Id = tn_identity(Ae, 3)
    @printf("norm(|AAdag - I|) at even site (D=%d): %.2e\n", Ds[itD], norm(abs.(Id - AAdag)))

    Bo = contract(Go[itD], [3], diagm(Lo[itD]), [1])
    BBdag = contract(Bo, [2, 3], conj(Bo), [2, 3])
    Id = tn_identity(Bo, 1)
    @printf("norm(|BBdag - I|) at odd site (D=%d): %.2e\n", Ds[itD], norm(abs.(Id - BBdag)))
    Be = contract(Ge[itD], [3], diagm(Le[itD]), [1])
    BBdag = contract(Be, [2, 3], conj(Be), [2, 3])
    Id = tn_identity(Be, 1)
    @printf("norm(|BBdag - I|) at even site (D=%d): %.2e\n", Ds[itD], norm(abs.(Id - BBdag)))
end

Gocan = Vector{Any}(undef, length(Ds))
Gecan = Vector{Any}(undef, length(Ds))
Locan = Vector{Any}(undef, length(Ds))
Lecan = Vector{Any}(undef, length(Ds))

# Orthogonalize Vidal's Gamma-Lambda representation of infinite MPS
# following the method given in [R. Orus & G. Vidal, Phys. Rev. B 78
# 155117 [2008]]. Here the goal of the orthogonalization is to make the ket
# tensors of Lambda*Gamma type (Gamma*Lambda type) be left-normalized
# (right-normalized), i.e., to bring them into canonical forms.
#
# ->-diagm(Lo)->-*->-Ge->-*->-diagm(Le)->-*->-Go->-*->-diagm(Lo)->-
#  1          2   1 ^  3   1          2   1 ^  3   1          2
#                   |2                      |2
for itD = (1:length(Ds))
    @printf("## ## ## ## Canonicalization: START (D=%d) ## ## ## ##\n", Ds[itD])
    # # "Coarse grain" the tensors: contract the tensors for the unit cell
    # # altogether. Then the coarse-grained tensor will be orthogonalized.

    # # # # STEP 1 in [iTEBD.5]: START # # # #
    # T=Ge*Le*Go: coarsed-grained tensor
    T = Ge[itD]
    T = contract(T, [3], diagm(Le[itD]), [1])
    T = contract(T, [3], Go[itD], [1])
    # # # STEP 1 in [iTEBD.5]: END # # #

    # # # # STEP 2 in [iTEBD.5]: START # # # #
    # ket tensor to compute transfer operator from right
    TR = contract(T, [4], diagm(Lo[itD]), [1])
    # # step [i] in PRB 78,155117 [2008]: START # #
    # find the dominant eigenenvector for the transfer operator from right
    XR = canonIMPS_domVec(TR, 4, false)
    # ket tensor to compute transfer operator from left
    TL = contract(diagm(Lo[itD]), [2], T, [1])
    # find the dominant eigenenvector for the transfer operator from left
    XL = canonIMPS_domVec(TL, 4, true)
    # # step [i] in PRB 78,155117 [2008]: END # #

    # # step [ii] in PRB 78,155117 [2008]: START # #
    # do SVD in Fig. 2[ii] of [R. Orus & G. Vidal, Phys. Rev. B 78, 155117 [2008]]
    U, SXdiag, Vd = tn_svd(XL * diagm(Lo[itD]) * XR, [1]; Nkeep=Ds[itD])
    SX = diagm(SXdiag)
    # # step [ii] in PRB 78,155117 [2008]: END # #

    # # step [iii] in PRB 78,155117 [2008]: START # #
    # orthogonalize the coarse-grained tensor
    T = contract(Vd / XR, [2], T, [1])
    T = contract(T, [4], XL \ U, [1])
    Locan[itD] = SXdiag / norm(SXdiag)
    # # step [iii] in PRB 78,155117 [2008]: END # #
    # # # # STEP 2 in [iTEBD.5]: END # # # #

    # # # # STEP 3 in [iTEBD.5]: START # # # #
    # decompose the orthogonalized coarse-grained tensor into the tensors
    # for individual sites
    # contract singular value tensors to the left & right ends; before
    # doing SVD.
    T = contract(diagm(Locan[itD]), [2], T, [1])
    T = contract(T, [4], diagm(Locan[itD]), [1])
    U2, S2diag, V2 = tn_svd(T,[1,2],Nkeep=Ds[itD])
    S2 = diagm(S2diag)
    Lecan[itD] = S2diag / norm(S2diag)
    Gecan[itD] = contract(diagm(1 ./ Locan[itD]), [2], U2, [1])
    Gocan[itD] = contract(V2, [3], diagm(1 ./ Locan[itD]), [1])
    # # # # STEP 3 in [iTEBD.5]: END # # # #
    @printf("## ## ## ## Canonicalization: END (D=%d) ## ## ## ##\n", Ds[itD])
end

# Check that the GS is canonicalized
for itD in 1:length(Ds)
    # check left isometry at odd site
    Ao = contract(diagm(Lecan[itD]), [2], Gocan[itD], [1])
    AAdag = contract(Ao, [1, 2], conj(Ao), [1, 2])
    Id = tn_identity(Ao, 3)
    @printf("norm(|AAdag - I|) at odd site (D=%d; canonicalized): %.2e\n",
            Ds[itD], norm(abs.(Id - AAdag)))

    # check left isometry at even site
    Ae = contract(diagm(Locan[itD]), [2], Gecan[itD], [1])
    AAdag = contract(Ae, [1, 2], conj(Ae), [1, 2])
    Id = tn_identity(Ae, 3)
    @printf("norm(|AAdag - I|) at even site (D=%d; canonicalized): %.2e\n",
            Ds[itD], norm(abs.(Id - AAdag)))

    # check right isometry at odd site
    Bo = contract(Gocan[itD], [3], diagm(Locan[itD]), [1])
    BBdag = contract(Bo, [2, 3], conj(Bo), [2, 3])
    Id = tn_identity(Bo, 1)
    @printf("norm(|BBdag - I|) at odd site (D=%d; canonicalized): %.2e\n",
            Ds[itD], norm(abs.(Id - BBdag)))

    # check right isometry at even site
    Be = contract(Gecan[itD], [3], diagm(Lecan[itD]), [1])
    BBdag = contract(Be, [2, 3], conj(Be), [2, 3])
    Id = tn_identity(Be, 1)
    @printf("norm(|BBdag - I|) at even site (D=%d; canonicalized): %.2e\n",
            Ds[itD], norm(abs.(Id - BBdag)))
end

@save "problem/set04/iTEBD_iv.jld2" Ds Gecan Gocan Lecan Locan