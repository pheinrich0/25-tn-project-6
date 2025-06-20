using LinearAlgebra
using Printf
using JLD2
using CairoMakie

using tn_julia: sitecanonical, applyHtoC

# 1s DMRG vs iterative diagonalization: Local variance
var_it = zeros(L,1); # local variace of MPSiso
var_1s = zeros(L,1); # local variance from MPS1s

@load "problem/set03/solution_a.jld2" mps W L
@load "problem/set03/solution_d.jld2" MPS1s

for Lc = (1:L)
    mps_it = sitecanonical(mps, Lc; tolerance=1e-8)
    mps_1s = sitecanonical(MPS1s, Lc; tolerance=1e-8)

    # Apply H^{1s}_ell to C_ell
    HC_it = applyHtoC(W, mps_it, Lc)
    HC_1s = applyHtoC(W, mps_1s, Lc)

    # Compute energy
    E_it = contract(HC_it, [1,2,3], mps_it[Lc], [1,2,3])
    E_1s = contract(HC_1s, [1,2,3], mps_1s[Lc], [1,2,3])
    CHHC_it = contract(HC_it, [1,2,3], conj(HC_it), [1,2,3])
    CHHC_1s = contract(HC_1s, [1,2,3], conj(HC_1s), [1,2,3])
    var_it[Lc] = CHHC_it[1] - E_it[1]^2
    var_1s[Lc] = CHHC_1s[1] - E_1s[1]^2
end

# Visualize the variance of 1s DMRG and iterative diagonalization
publication_theme() = Theme(
    fontsize=16, font="CMU Serif",
    Axis=(xlabelsize=20, ylabelsize=20, xgridstyle=:dash, ygridstyle=:dash,
        xtickalign=1, ytickalign=1, yticksize=10, xticksize=10),
    Legend=(framecolor=(:black, 0.5), backgroundcolor=(:white, 0.5), labelsize=20),
    Colorbar=(ticksize=16, tickalign=1, spinewidth=0.5),
)

with_theme(publication_theme()) do

fig = Figure(size=(500, 400)) 
ax = Axis(fig[1,1],xlabel=L"\ell",ylabel=L"\mathrm{Local\,\,Variance}")

lines!(ax,(1:L),var_1s[:,1],label=L"\Delta^\mathrm{loc,1s}_E",linestyle=:dash)
lines!(ax,(1:L),var_it[:,1],label=L"\Delta^\mathrm{loc,iter}_E")
axislegend(ax,position=:cc)

save("problem/set03/solution_e.pdf",fig,pt_per_unit=2)

end
