"""
    position_mpo(N)

Build the MPO for the operator X = ∑_{itL=1}^N ℓ n_ℓ on a length-N spinless fermion chain,
with n_ℓ = (1–σ^z_ℓ)/2.  Returns a tuple (W, bL, bR) where

- `W` is a Vector of length N of 4-index tensors
  W[ℓ] of size
    • (1,2,2,2) for ℓ=1
    • (2,2,2,2) for 1<ℓ<N
    • (2,1,2,2) for ℓ=N

- `bL` is the left boundary row-vector of length 2: [0, 1]
- `bR` is the right boundary column-vector of length 2: [1; 0]

Such that  
    bL * W[1] * W[2] * … * W[N] * bR  
yields the MPO for X.
"""

function position_mpo(N::Int)
    # physical dimension d=2, bond-dimension D=2
    F, Z, Id = tn.spinlessfermionlocalspace()

    W = Vector{Array{Float64,4}}(undef, N)
    for itL in 1:N
        # choose the right shape for each site
        if itL == 1
            dims = (1, 2, 2, 2)
            W_L = zeros(Float64, dims...)
            W_L[1, :, 1, :] = itL*Diagonal([1,0])
            W_L[1, :, 2, :] = Id
        elseif itL == N
            dims = (2, 2, 1, 2)
            W_L = zeros(Float64, dims...)
            W_L[1, :, 1, :] = Id
            W_L[2, :, 1, :] = itL*Diagonal([1,0])
        else
            dims = (2, 2, 2, 2)
            W_L = zeros(Float64, dims...)
            W_L[1, :, 1, :] = Id
            W_L[2, :, 1, :] = itL*Diagonal([1,0])
            W_L[2, :, 2, :] = Id       
        end
        W[itL] = W_L

    end
    return W
end


# test: for the fermionic_mps the pos operator should be zero

