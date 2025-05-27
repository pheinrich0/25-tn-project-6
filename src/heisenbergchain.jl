
function extendhamiltonian_heisenberg(
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
    Sxyz = cat(Sx, Sy, Sz, dims=3)
    # leg ordering for right tensor in first updateLeft
    Sright1 = reshape(
        permutedims(Sxyz, (2, 3, 1)),
        (1, d, 3, d)
    )
    S1 = updateLeft(Ileft, A_ell, Sright1, A_ell)

    # leg ordering for right tensor in second updateLeft
    Sright2 = reshape(
        permutedims(Sxyz, (3, 2, 1)),
        (3, d, 1, d)
    )
    S2 = updateLeft(S1, A_ellplus1, Sright2, A_ellplus1)

    D = size(H_ell, 1)
    H2 = updateLeft(
        reshape(H_ell, (D, 1, D)),
        A_ellplus1,
        reshape(Id, (1, d, 1, d)),
        A_ellplus1
    )

    H_ellplus1 = reshape(H2 + J * S2, (d * D, d * D))
    return H_ellplus1, A_ellplus1
end
