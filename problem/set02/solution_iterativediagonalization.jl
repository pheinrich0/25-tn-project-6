using Plots
using LaTeXStrings

import tn_julia: extendhamiltonian, iterativediagonalization

L = 200
Nkeep = 300

exactenergy(L) = 0.5 - 0.5 / sin(0.5pi / (L+1))
energies_iterdiag = iterativediagonalization(L, Nkeep)

p1 = plot(
    2:L, exactenergy.(2:L) ./ (2:L),
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
