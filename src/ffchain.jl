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
    Dleft = size(A_ell, 1)
    Ileft = reshape(I(Dleft), (Dleft, 1, Dleft))

    F, Z, Id = spinlessfermionlocalspace()
    Fdag = F'  # creation operator
    d = size(F, 1)

    A_ellplus1 = identity(A_ell, 3, F, 1)  # you might want to use Id here instead

    # Combine F† and F into a hopping interaction tensor
    hop = cat(Fdag, F, dims=3)

    # First contraction over A_ell
    hop1 = reshape(permutedims(hop, (2, 3, 1)), (1, d, 2, d))
    H1 = updateLeft(Ileft, A_ell, hop1, A_ell)

    # Second contraction over A_ellplus1
    hop2 = reshape(permutedims(hop, (3, 2, 1)), (2, d, 1, d))
    H2 = updateLeft(H1, A_ellplus1, hop2, A_ellplus1)

    # Propagate previous H_ell through identity
    D = size(H_ell, 1)
    H3 = updateLeft(
        reshape(H_ell, (D, 1, D)),
        A_ellplus1,
        reshape(Id, (1, d, 1, d)),
        A_ellplus1
    )

    H_ellplus1 = reshape(H3 - t * H2, (d * D, d * D))
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