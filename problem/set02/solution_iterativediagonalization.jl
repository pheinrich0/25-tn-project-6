using Plots
using LaTeXStrings
using LinearAlgebra

import tn_julia: extendhamiltonian_xy, iterativediagonalization, contract

L = 200
Nkeep = 300
spin = 1//2

exactenergy(L) = 0.5 - 0.5 / sin(0.5pi / (L+1))

extendhamiltonian(H::AbstractMatrix, A::AbstractArray{<:Number, 3}) = extendhamiltonian_xy(H, A; J=-1.0, spin=spin)
energies_iterdiag, mps = iterativediagonalization(extendhamiltonian_xy, 2, L, Nkeep)

# Part (c): Plot energy per site vs. chain length
p1 = plot(
    1:L, exactenergy.(1:L) ./ (1:L),
    label="Exact",
    title="Iterative diagonalization", xlabel="L", ylabel="Energy per site",
    ylim=(-0.32, -0.25))
plot!(p1, 2:2:L, energies_iterdiag[2:2:L] ./ (2:2:L), label="Iter. diag., even")

p2 = plot(
    2:2:L, abs.(exactenergy.(2:2:L) .- energies_iterdiag[2:2:L]) ./ (2:2:L),
    label="iter. diag.",
    xlabel="L", ylabel="Energy difference per site")

plot(p1, p2, layout=(2,1))
savefig("iterativediagonalization-energies.pdf")

# Part (d) (iv): Check the MPS norm
mpscontraction = ones(1, 1)
for M in mps
    mpscontraction = contract(mpscontraction, [1], M, [1])
    mpscontraction = contract(mpscontraction, [1, 2], M, [1, 2])
end
println("Squared MPS norm: $(opnorm(mpscontraction))")
