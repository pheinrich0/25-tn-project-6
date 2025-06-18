## T04 Two Site DMRG (a)
using LinearAlgebra
using Printf
using JLD2
using CairoMakie

using tn_julia: sitecanonical, sitecanonical!, contract,
                computeLeftEnvironment, computeRightEnvironment,
                lanczos_1site
using tn_julia: identity as tn_identity
using tn_julia: svd as tn_svd

@load "problem/set04/baseline.jld2" MPS W Eex Eiter L


## i) Applying 2-site Hamiltonian
Lc = 50; # isometry center
MPS_iso = sitecanonical(MPS, Lc; tolerance=1e-8)

# left and right environments
Lenv = computeLeftEnvironment(W, MPS_iso, Lc-1)
Renv = computeRightEnvironment(W, MPS_iso, Lc+2)

# Compare GS energies from iterative diagmonalization & 2s Hamiltonian
Cinit = MPS_iso[Lc]; # C at isometry center
Binit = MPS_iso[Lc+1]; # B at the site Lc+1

# # # # Following the Hint: START # # # #
# Merging physical legs of C & B
I_CB = tn_identity(Cinit, 2, Binit, 2)
CB = contract(Cinit, [2], I_CB, [1])
CB = contract(CB, [2,3], Binit, [1,2])
# Merging physcial legs of W & W
I_WW = tn_identity(W[Lc], 2, W[Lc+1], 2)
WW = contract(W[Lc], [2], I_WW, [1])
WW = contract(WW, [2,4], W[Lc+1], [1,2])
WW = contract(WW, [2,5], I_WW, [1,2])
# # # # Following the Hint: END # # # #

# HCB: contracting Lenv; Renv; WW; & CB
HCB = contract(Lenv, [3], CB, [1])
HCB = contract(HCB, [2,3], WW, [1,4])
HCB = contract(HCB, [2,4], Renv, [3,2])
# E: energy expectation value
E = contract(HCB, [1,2,3], conj(CB), [1,2,3])
# display result
@printf("E : %e\n", E[1])
@printf("Eiter - E : %e\n", Eiter - E[1])


## ii) Lanczos method
NLZs = (1:10); # number of Lanczos steps
ELZs = zeros(size(NLZs)); # energy specfic number of steps
for itN = (1:length(NLZs))
    local NLZ = NLZs[itN]
    # # # # Input CB & WW with merged physical legs: START # # # #
    local CBeff, Eeff = lanczos_1site(Lenv, WW, Renv, CB; N=NLZ)
    # # # # Input CB & WW with merged physical legs: END # # # #
    # energy
    ELZs[itN] = Eeff
end

publication_theme() = Theme(
    fontsize = 16, font = "CMU Serif",
    Axis = (xlabelsize = 20, ylabelsize = 20, 
            xgridstyle = :dash, ygridstyle = :dash,
            xtickalign = 1, ytickalign = 1, 
            yticksize = 10, xticksize = 10),
    Legend = (framecolor = (:black, 0.5), 
              backgroundcolor = (:white, 0.5), labelsize = 20),
    Colorbar = (ticksize = 16, tickalign = 1, spinewidth = 0.5),
)

with_theme(publication_theme()) do

    fig = Figure(size = (500, 400))
    ax = Axis(fig[1,1], xlabel = L"N_\mathrm{Lanczos}",
        ylabel = L"\mathrm{Relative\,\,Error\,\,(%)}")
    scatterlines!(ax, NLZs, (ELZs .- Eex) ./ abs(Eex) .* 100,
        label = L"\frac{E_\mathrm{LZ} - E_\mathrm{ex}}{|E_\mathrm{ex}|}")
    axislegend(ax)
    save("problem/set04/solution_a_ii.pdf", fig, pt_per_unit = 2)
end


## iii) Decomposition and discarded weights
NLZ = 10; # number of Lanczos steps
Dfs = (50:10:110); # final bond dimension

# Although the maximal value of Df is 100; we study Df up to 110 to
# explicitly check the maximal value
EDfs = zeros(size(Dfs)); # energies
dws = zeros(size(Dfs)); # discarded weigenhts

# # # # Input CB & WW with merged physical legs: START # # # #
CBeff, Eeff = lanczos_1site(Lenv, WW, Renv, CB; N=NLZ)
# # # # Input CB & WW with merged physical legs: END # # # #

# re-split the merged physical legs
CBeff = contract(CBeff, [2], I_CB, [3], [1,3,4,2])
for itD = (1:length(Dfs))
    U, S, Vd, dws[itD] = tn_svd(CBeff, [1,2], Nkeep=Dfs[itD])
    CB_Df = contract(U, [3], diagm(S), [1])
    CB_Df = contract(CB_Df, [3], Vd, [1])
    CB_Df = contract(CB_Df, [2,3], I_CB, [1,2], [1,3,2])
    # energy
    HCB_Df = contract(Lenv, [3], CB_Df, [1])
    HCB_Df = contract(HCB_Df, [2,3], WW, [1,4])
    HCB_Df = contract(HCB_Df, [2,4], Renv, [3,2])
    # E: energy expectation value
    EDfs[itD] = first(contract(HCB_Df, [1,2,3], conj(CB_Df), [1,2,3]))
end

publication_theme() = Theme(
    fontsize = 16, font = "CMU Serif",
    Axis = (xlabelsize = 20, ylabelsize = 20,
            xgridstyle = :dash, ygridstyle = :dash,
            xtickalign = 1, ytickalign = 1, yticksize = 10, xticksize = 10),
    Legend = (framecolor = (:black, 0.5), 
              backgroundcolor = (:white, 0.5), labelsize = 20),
    Colorbar = (ticksize = 16, tickalign = 1, spinewidth = 0.5),
)

with_theme(publication_theme()) do
    fig = Figure(size = (1000, 400))
    
    ax1 = Axis(fig[1,1], xlabel = L"D_\mathrm{f}",
               ylabel = L"\mathrm{Energy\,\,Difference}",
               ytickformat = "{:.6f}")
    
    ax2 = Axis(fig[1,2], yscale = log10, xlabel = L"D_\mathrm{f}", 
               ylabel = L"\mathrm{Discarded\,\,Weight}")
    
    ax3 = Axis(fig[1,3], xlabel = L"D_\mathrm{f}", 
               ylabel = L"\mathrm{Relative\,\,Error\,\,(%)}", 
               ytickformat = "{:.7f}")
    
    scatterlines!(ax1, Dfs, (EDfs .- Eex), label = L"E - E_\mathrm{ex}")
    axislegend(ax1)
    
    scatterlines!(ax2, Dfs[1:5], dws[1:5])
    s1 = scatter!(ax2, [80], [5e-10], color = :white)
    s2 = scatter!(ax2, [80], [5e-10], color = :white)
    xlims!(ax2, high = 110)
    axislegend(ax2, [s1, s2], 
               [L"\xi = 0", L"\mathrm{for}\,\,D_\mathrm{f}\geq100"],
               framevisible = false, position = :rt, labelsize = 21)
    
    scatterlines!(ax3, Dfs, (EDfs .- Eex) ./ abs.(Eex) .* 100,
                  label = L"\frac{E - E_\mathrm{ex}}{|E_\mathrm{ex}|}")
    axislegend(ax3)
    
    save("problem/set04/solution_a_iii.pdf", fig, pt_per_unit = 2)
end