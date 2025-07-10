"""
    dm_alpha_mpo(m, N; α=1)

Build the MPO representation of

    d†_{m,α} = N^(-1/2) ∑_{j=1}^N e^(-imj) c†_{j,α}

for a chain of length N.  Returns a Vector of length N of 4-index
tensors W[1],…,W[N] with shapes

  size(W[1]) == (1, 2, 2, 2),
  size(W[itL]) == (2, 2, 2, 2)  for itL=2:N-1,
  size(W[N]) == (2, 2, 2, 1).

Keyword argument
- `α`  is the spin-flavor index (1 or 2) that picks out which c† to emit.
"""
# m arent the momenta but integers from 1:N/2
function dm_alpha_mpo(m::Int, spin, N::Int, coeffMat)
    # pre-allocate vector of MPO tensors
    W = Vector{Array{ComplexF64,4}}(undef, 2*N)

    if size(coeffMat, 2) != 2N
        error("coeffMat must have 2N = $(2N) columns, but has $(size(coeffMat, 2)) columns.")
    end
    # by convention, the odd sites correspond to spin up, the even sites to spin down
    if spin == 1/2  # spin up, ie. nonzero coeffient A_ml at even site
        coeffs = coeffMat[2*m-1, :]
    elseif spin==-1/2 # spin down, A_ml has entry at odd sites
        coeffs = coeffMat[2*m, :]
    else
        error("Invalid spin index: spin = $spin.")
    end
    
    F, Z, Id = tn.spinlessfermionlocalspace()
    # loop combined indices l = j,alpha
    for itL in 1:2N
        # choose correct tensor shape
        if itL == 1
            dims = (1, 2, 2, 2)
            W_L = zeros(ComplexF64, dims...)
            W_L[1, :, 1, :] = F*coeffs[itL]
            W_L[1, :, 2, :] = Z
        elseif itL == 2N
            dims = (2, 2, 1, 2)
            W_L = zeros(ComplexF64, dims...)
            W_L[1, :, 1, :] = Id
            W_L[2, :, 1, :] = F*coeffs[itL]
        else
            dims = (2, 2, 2, 2)
            W_L = zeros(ComplexF64, dims...)
            W_L[1, :, 1, :] = Id
            W_L[2, :, 1, :] = F*coeffs[itL]
            W_L[2, :, 2, :] = Z          
        end
        W[itL] = W_L
    
    end
    
    return W
end


"""
    momenta(N)

Return the length=N/2 vector of momenta m = (2π/N) * s
with s = {0, ±1, …, ±(N/4−1), ± N/4} if N % 4 == 0,
     and s = {0, ±1, …, ±((N-2)/4)} if N % 4 == 2.
Throws an error if N is not even.
"""
function momenta(N::Int)
    @assert N % 2 == 0 "N must be even"
    if N % 4 == 0
        p = collect(1:(N÷4)-1)
        s = vcat(-reverse(p), 0,  p,  N÷4)
    elseif N % 4 == 2
        maxs = (N - 2) ÷ 4
        p = collect(1:maxs)
        s = vcat(0,  p,  -reverse(p))
    else
        error("Shouldn’t happen: N is even but not ≡ 0 or 2 mod 4")
    end
    return (2π/N) * s   # return a list of N/2 momenta
end



## MPO representation of the position operator