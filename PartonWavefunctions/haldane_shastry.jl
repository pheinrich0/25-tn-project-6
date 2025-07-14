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

using ITensors

"""
    two_spin_mpo(N, p, q; shift_constant = false)

Return an MPO representing the operator S_p · S_q on an S=1/2 chain
of length `N`.  Indices `p` and `q` must be different.

Set `shift_constant = true` if you want to include the –¼ identity
coming from the single-occupancy constraint.
"""
function two_spin_mpo(N::Int, p::Int, q::Int; shift_constant::Bool=false)
    @assert p != q "p and q have to be different sites."
    @assert 1 ≤ p ≤ N && 1 ≤ q ≤ N "p and/or q out of bounds."

    sites = siteinds("S=1/2", N)          # local physical indices
    ampo  = AutoMPO()                     # helper that builds MPOs

    # --- Heisenberg form (★) ------------------------------------
    for op in ("Sx", "Sy", "Sz")
        # coefficient ¼ because S = σ/2
        coeff = 0.25
        push!(ampo, coeff, op, p, op, q)
    end

    # Optional constant shift –¼ (identity on the two sites)
    if shift_constant
        push!(ampo, -0.25, "Id", p, "Id", q)
    end

    return MPO(ampo, sites)
end

