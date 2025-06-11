using LinearAlgebra
using Printf
using JLD2
using CairoMakie

using tn_julia: DMRG_1site

MPS = load("problem/set03/solution_a.jld2", "mps")
W = load("problem/set03/solution_a.jld2", "W")
Eex = load("problem/set03/solution_a.jld2", "Eex")
Eiter = load("problem/set03/solution_a.jld2", "Eiter")

# 1s DMRG vs iterative diagonalization: Energy error
Esweep = []
varsweep = []
Einit = Inf
Efin = -Inf
loopcnt = 0; # count DMRG loop
looplim = 50; # limit loop # for safety
iscvg = 1e-10; # criterion of energy convergence
Nkeep = 50;

while true
    global loopcnt = loopcnt + 1
    global Einit, Efin, Nkeep, Mfin
    if loopcnt == 1
        Minit = MPS
        Einit = Efin
    else
        Minit = Mfin
        Einit = Efin
    end
    Mfin,Efin,_ = DMRG_1site(W, Minit, Nkeep, 1)
    append!(Esweep, Efin)
    @printf("DMRG [step %d] Efin - Einit : %e\n", loopcnt, Efin-Einit)
    if (abs(Efin-Einit) < iscvg) || (loopcnt > looplim)
        break
    end
end
MPS1s = Mfin;

@save "problem/set03/solution_d.jld2" MPS1s Esweep Eex

publication_theme() = Theme(
    fontsize=16, font="CMU Serif",
    Axis=(xlabelsize=20, ylabelsize=20, xgridstyle=:dash, ygridstyle=:dash,
        xtickalign=1, ytickalign=1, yticksize=10, xticksize=10),
    Legend=(framecolor=(:black, 0.5), backgroundcolor=(:white, 0.5), labelsize=20),
    Colorbar=(ticksize=16, tickalign=1, spinewidth=0.5),
)

with_theme(publication_theme()) do

fig = Figure(size=(500, 400)) 
ax = Axis(fig[1,1],yscale=log10,xlabel=L"#\,\,\mathrm{of\,\,Iteration}",
          ylabel=L"\mathrm{Relative\,\, Error}")

lines!(ax,(1:loopcnt),(Esweep[:,1].-Eex)./abs(Eex),
       label=L"\frac{E_\mathrm{sweep}-E_\mathrm{ex}}{|E_\mathrm{ex}|}")
lines!(ax,(1:loopcnt),ones(loopcnt).*(Eiter-Eex)/abs(Eex),linestyle=:dash,
      color=:black,label=L"\frac{E_\mathrm{iter}-E_\mathrm{ex}}{|E_\mathrm{ex}|}")
ylims!(ax,low=1e-8)
axislegend(ax,position=:rc)

save("problem/set03/solution_d.pdf",fig,pt_per_unit=2)

end