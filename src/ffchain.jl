function tightbindinghamiltonian(L::Int)
    H = zeros(L,L); # Single-particle Hamiltonian
    H = H - diagm(1=>ones(L-1)); # |l><l+1| terms
    H = H - diagm(-1=>ones(L-1)); # |l+1><l| terms
    return H
end

function exactffenergy(L::Int)
    H = tightbindinghamiltonian(L)
    E,_ = eigen(H) # E[i]: ith lowest eigenenergy
    Eex = sum(E[E.<=0])
    return Eex
end

function extendhamiltonian_ff(
    H_ell::AbstractMatrix, A_ell::AbstractArray{<:Number, 3};
    t::Float64 = 1.0
)
    F, Z, Id = spinlessfermionlocalspace()
    
    # Create the left fermion operator from the previous site
    AID = identity(A_ell, 1)
    Fprev = updateLeft(reshape(AID, (size(AID,1), 1, size(AID,1))), A_ell, reshape(F, (1, size(F,2), 1, size(F,1))), A_ell)
    
    # Create the new MPS tensor for the current site (equivalent to getIdentity)
    A_ellplus1 = identity(H_ell, 2, Id, 2)
    
    # Propagate previous Hamiltonian through identity on new site
    H_propagated = updateLeft(reshape(H_ell, (size(H_ell,1), 1, size(H_ell,2))), A_ellplus1, reshape(Id, (1, size(Id,2), 1, size(Id,1))), A_ellplus1)
    H_propagated = dropdims(H_propagated, dims=2)  # Drop the dummy dimension
    
    # Build hopping term: -F†_{l-1} F_l - F†_l F_{l-1}
    Hhop = -updateLeft(Fprev, A_ellplus1, 
        reshape(permutedims(F, (2, 1)), (1, size(F,2), 1, size(F,1))), A_ellplus1)
    Hhop = dropdims(Hhop, dims=2)
    Hhop = Hhop + Hhop'  # Make Hermitian to include both hopping directions
    
    H_ellplus1 = H_propagated + t * Hhop
    
    return H_ellplus1, A_ellplus1
end

function ffchainmpo(L::Int)
    F, Z, Id = spinlessfermionlocalspace()
    d = size(F, 1)

    W = Vector{Array{ComplexF64, 4}}(undef, L)

    # Site 1: shape (1, d, 4, d)
    W[1] = zeros(ComplexF64, 1, d, 4, d)
    W[1][1, :, 2, :] .= -F'
    W[1][1, :, 3, :] .= -F
    W[1][1, :, 4, :] .= Id

    # Bulk sites: shape (4, d, 4, d)
    for l in 2:L-1
        W[l] = zeros(ComplexF64, 4, d, 4, d)
        W[l][1, :, 1, :] .= Id
        W[l][2, :, 1, :] .= F
        W[l][3, :, 1, :] .= F'
        W[l][4, :, 2, :] .= -F'
        W[l][4, :, 3, :] .= -F
        W[l][4, :, 4, :] .= Id
    end

    # Last site: shape (4, d, 1, d)
    W[L] = zeros(ComplexF64, 4, d, 1, d)
    W[L][1, :, 1, :] .= Id
    W[L][2, :, 1, :] .= F
    W[L][3, :, 1, :] .= F'

    return W
end