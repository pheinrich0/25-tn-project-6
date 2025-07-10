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


