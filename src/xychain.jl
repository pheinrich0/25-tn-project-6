function xypairhamiltonian()
    Splus, Sminus, Sz, Id = spinlocalspace(1 // 2)
    A = identity(Splus, 1, Splus, 1)

    # add a third leg that sums over + and -
    Splusminus = cat(Splus, Sminus, dims=3)
    # leg ordering for left tensor in updateLeft
    Sleft = permutedims(Splusminus, (1, 3, 2))
    # leg ordering for right tensor in updateLeft
    Sright = reshape(
        permutedims(Splusminus, (3, 1, 2)),
        (2, 2, 1, 2)
    )

    StimesS = updateLeft(Sleft, A, Sright, A)

    # remove the dummy leg that was needed for updateLeft
    Hxy = reshape(-StimesS, (4, 4))
    return Hxy
end

function extendhamiltonian_xy(
    H_ell::AbstractMatrix, A_ell::AbstractArray{<:Number, 3};
    spin::Rational=1//2, J::Float64=1.0
)
    Dleft = size(A_ell, 1)
    Ileft = reshape(I(Dleft), (Dleft, 1, Dleft))

    Splus, Sminus, Sz, Id = spinlocalspace(spin)
    A_ellplus1 = identity(A_ell, 3, Splus, 1)

    d = size(Sz, 1)

    Sx = (Splus + Sminus) / (sqrt(2))
    Sy = (Splus - Sminus) / (1im * sqrt(2))
    Sxy = cat(Sx, Sy, dims=3)
    # leg ordering for right tensor in first updateLeft
    Sright1 = reshape(
        permutedims(Sxy, (2, 3, 1)),
        (1, 2, 2, 2)
    )
    S1 = updateLeft(Ileft, A_ell, Sright1, A_ell)

    # leg ordering for right tensor in second updateLeft
    Sright2 = reshape(
        permutedims(Sxy, (3, 2, 1)),
        (2, 2, 1, 2)
    )
    S2 = updateLeft(S1, A_ellplus1, Sright2, A_ellplus1)

    D = size(H_ell, 1)
    H2 = updateLeft(
        reshape(H_ell, (D, 1, D)),
        A_ellplus1,
        reshape(I(2), (1, 2, 1, 2)),
        A_ellplus1
    )

    H_ellplus1 = reshape(H2 + J * S2, (d * D, d * D))
    return H_ellplus1, A_ellplus1
end

function xychainmpo(L::Int, J::Float64)
    Splus, Sminus, Sz, Id = spinlocalspace(1 // 2)
    Sx = (Splus + Sminus) / (sqrt(2))
    Sy = (Splus - Sminus) / (1im * sqrt(2))

    W = zeros(Complex{Float64}, 4, 2, 4, 2) # ordering: left bottom right top
    W[1, :, 1, :] = Id
    W[2, :, 1, :] = Sx
    W[3, :, 1, :] = Sy
    W[4, :, 2, :] = J*Sx
    W[4, :, 3, :] = J*Sy
    W[4, :, 4, :] = Id

    return [
        W[[4], :, :, :], # first site
        [W for _ in 2:L-1]...,
        W[:, :, [1], :] # last site
    ]
end
