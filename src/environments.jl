function computeLeftEnvironment(MPO, MPS, ell_max)
    Lenv = ones((eltype(MPO[1])), (1, 1, 1))  # identity 3-leg tensor

    for ell in 1:ell_max
        Lenv = updateLeft(Lenv, MPS[ell], MPO[ell], MPS[ell])
    end

    return Lenv
end

function computeRightEnvironment(MPO, MPS, ell_min)
    L = length(MPO)
    Renv = ones((eltype(MPO[1])), (1, 1, 1))  # identity 3-leg tensor

    for ell in reverse(ell_min:L)
        Renv = updateLeft(Renv, permutedims(MPS[ell], (3,2,1)), permutedims(MPO[ell], (3,2,1,4)), permutedims(MPS[ell], (3,2,1)))
    end

    return Renv
end

function applyHtoC(W, MPS, ell)
    Lenv = computeleftenvironment(W, MPS, ell-1)
    Renv = computerightenvironment(W, MPS, ell+1)
    Cell = MPS[ell]
    Well = W[ell]

    HC = contract(Lenv,[3], Cell, [1])
    HC = contract(HC, [2,3], Well,[1,4])
    HC = contract(HC,[2,4], Renv,[3,2])
end