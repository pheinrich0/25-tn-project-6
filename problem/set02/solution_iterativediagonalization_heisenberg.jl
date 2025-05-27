using Plots
using LaTeXStrings
using LinearAlgebra

import tn_julia: extendhamiltonian_heisenberg, iterativediagonalization, contract

L = 50
Nkeep = 300
spin = 1//1
J = 1.0

localdimension = Int(2 * spin + 1)

extendhamiltonian(H::AbstractMatrix, A::AbstractArray{<:Number, 3}) = extendhamiltonian_heisenberg(H, A; J=J, spin=spin)
energies_iterdiag, mps = iterativediagonalization(extendhamiltonian, localdimension, L, Nkeep)

# Plot energy per site vs. chain length
plot(
    1:L, energies_iterdiag ./ (1:L), label="Iterative diagonalization",
    title=L"Iterative diagonalization, $S = %$(spin.num)/%$(spin.den)$", xlabel="L", ylabel="Energy per site"
)

if spin == 1//2
    hline!([0.25 - log(2)], label=L"Infinite chain (exact), $S = 1/2$", linestyle=:dash)
elseif spin == 1
    hline!([-1.401484039], label=L"Infinite chain (DMRG), $S = 1$", linestyle=:dash)
end

savefig("iterativediagonalization-energies-heisenberg.pdf")
