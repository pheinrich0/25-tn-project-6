
function xypairhamiltonian()
    Splus, Sminus, Sz, Id = spinlocalspace(1 // 2)
    A = identity(Splus, 1, Splus, 1)

    SplusSplus = updateLeft(
        reshape(Splus, (2, 1, 2)),
        A,
        reshape(Splus, (1, 2, 1, 2)),
        A
    )

    SminusSminus = updateLeft(
        reshape(Sminus, (2, 1, 2)),
        A,
        reshape(Sminus, (1, 2, 1, 2)),
        A
    )

    Hxy = reshape(
        -(SplusSplus + SminusSminus),
        (4, 4)
    )
    return Hxy
end
