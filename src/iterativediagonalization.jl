function iterativediagonalization(
    extendhamiltonian::Function,
    localdimension::Int,
    L::Int, Nkeep::Int;
    truncationtolerance::Float64=1e-12
)
    H_ell = zeros(localdimension, localdimension)
    A_ell = identity(ones(1, 1), 2, H_ell, 1)

    energies = Float64[0.0]
    MPS = Array{ComplexF64, 3}[A_ell]

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

        # Update the Hamiltonian and tensor
        A_ell = contract(A_new, [3], eigvecs, [1])
        H_ell = Diagonal(eigvals)

        push!(energies, eigvals[1])
        push!(MPS, A_ell)
    end

    return energies, MPS
end
