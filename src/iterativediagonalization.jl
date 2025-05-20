function iterativediagonalization(L::Int, Nkeep::Int; truncationtolerance::Float64=1e-8)
    energies = zeros(L)

    Splus, Sminus, Sz, Id = spinlocalspace(1 // 2)
    H_ell = zeros(size(Id))
    A_ell = identity(ones(1, 1), 2, H_ell, 1)
    for ell in 2:L
        H_new, A_new = extendhamiltonian(H_ell, A_ell)
        # Diagonalize the Hamiltonian
        eigvals, eigvecs = eigen((H_new' + H_new) / 2)
        # Sort eigenvalues and eigenvectors
        sorted_indices = sortperm(eigvals)
        eigvals = eigvals[sorted_indices]
        eigvecs = eigvecs[:, sorted_indices]
        # Keep the Nkeep lowest eigenvalues and corresponding eigenvectors
        truncationenergy = eigvals[min(length(eigvals), Nkeep)] + truncationtolerance
        keep = eigvals .<= truncationenergy
        eigvals = eigvals[keep]
        eigvecs = eigvecs[:, keep]

        energies[ell] = eigvals[1]

        # Update the Hamiltonian and tensor
        A_ell = contract(A_new, [3], eigvecs, [1])
        H_ell = Diagonal(eigvals)
    end

    return energies
end
