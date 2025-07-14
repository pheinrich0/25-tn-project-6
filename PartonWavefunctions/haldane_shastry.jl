"""
    ground_state_energy(N)

Compute the ground‐state energy 
E₀(N) = -π^2 * (N + 5/N) / 24
for a (real or integer) system size `N`.
"""
function exactGS_energy(N::Real)
    return -π^2 * (N + 5/N) / 24
end

"""
    spin_spin_corr(N, q)

Compute the ground‐state spin–spin correlation
⟨Sₚ·Sₚ₊q⟩ for an even chain length `N` and separation `q`
(with 1 ≤ q < N), using

  ⟨S·S⟩ = ( ∑_{a=1}^{N/2} 3*(-1)^q/(2a-1) * sin(π*(2a-1)*q/N) )
           / ( 2N * sin(π*q/N) ).

By translation invariance, the result is independent of the site `p`.
"""

function spin_spin_corr(N::Integer, q::Integer)
    @assert iseven(N)        "N must be even"
    @assert 1 ≤ q < N        "q must satisfy 1 ≤ q < N"
    
    # numerator: sum over a = 1..N/2
    num = zero(Float64)
    for a in 1:(N ÷ 2)
        num += 3 * (-1)^q / (2a - 1) * sin(pi * (2a - 1) * q / N)
    end

    # denominator
    den = 2*N * sin(pi * q / N)

    return num / den
end

# ──────────────────────────────────────────────────────────────
# Abrikosov-fermion (spinon) representation on a single site j
# ------------------------------------------------------------
# S_j^a = ½ c†_{jα} τ^a_{αβ} c_{jβ}     (α,β ∈ {↑,↓}; a = x,y,z)
#
# ──────────────────────────────────────────────────────────────
# Two-site Heisenberg scalar product  𝐒_p · 𝐒_q
# ------------------------------------------------------------
# 𝐒_p·𝐒_q = -¼ n_p n_q  + ½ c†_{pα} c_{pβ} c†_{qβ} c_{qα}
#         = -¼ n_p n_q  – ½ c†_{pα} c_{qα} c†_{qβ} c_{pβ}
#           (with n_j = c†_{jα} c_{jα})
# ──────────────────────────────────────────────────────────────



"""
    mpo_SdotS(N, p, q)

Return a vector of `Array{ComplexF64,4}` containing an MPO for Sₚ·S_q
on a spin-½ chain of length `N` (open boundaries).

"""
function mpo_SdotS(N::Int, p::Int, q::Int)
    @assert 1 ≤ p < q ≤ N "Require 1 ≤ p < q ≤ N"

    # Local operators
    Sx = ComplexF64[0 1; 1 0]
    Sy = ComplexF64[0 -im; im 0]
    Sz = ComplexF64[1 0; 0 -1]
    Sx, Sy, Sz = 0.25Sx, 0.25Sy, 0.25Sz      # prefactor ¼ absorbed here
    I₂ = Matrix{ComplexF64}(I,2,2)

    mpo = Vector{Array{ComplexF64,4}}(undef, N)

    for j in 1:N
        # Decide bond dimensions for this site
        leftdim  = (j == 1      || j < p) ? 1 : 5
        rightdim = (j == N      || j > q) ? 1 :
                   (j == p      ? 5 :
                   (j == q      ? 1 : 5))

        W = zeros(ComplexF64, leftdim, rightdim, 2, 2)

        # Convenient helper: write an operator only if indices in range
        function put!(A, l, r, op)
            if l ≤ size(A,1) && r ≤ size(A,2)
                A[l,r,:,:] .= op
            end
        end

        # Identity always allowed
        put!(W, 1, 1, I₂)

        if j == p
            put!(W, 1, 2, Sx)
            put!(W, 1, 3, Sy)
            put!(W, 1, 4, Sz)
        elseif p < j < q
            put!(W, 2, 2, I₂)
            put!(W, 3, 3, I₂)
            put!(W, 4, 4, I₂)
        elseif j == q
            put!(W, 2, 1, Sx)     # closes Sx channel
            put!(W, 3, 1, Sy)     # closes Sy channel
            put!(W, 4, 1, Sz)     # closes Sz channel
        end

        mpo[j] = W
    end
    return mpo
end


## approach via large tensor and qr decomposition.
