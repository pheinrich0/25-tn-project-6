using LinearAlgebra
using Printf
using JLD2
using CairoMakie

using tn_julia: contract, lanczos_1site

@load "problem/set03/solution_b.jld2" mps_iso W Eex Lc Lenv Renv

# Diagonalizing H1s
# construct H1s
H1s = contract(Lenv, [2], W[Lc], [1])
H1s = contract(H1s, [4], Renv, [2])
H1s = permutedims(H1s, (1,3,5,2,4,6))
# Hdir: H1s in matrix form [to diagonalize]
Hdir = reshape(H1s, (size(H1s,1)*size(H1s,2)*size(H1s,3),
                     size(H1s,4)*size(H1s,5)*size(H1s,6)))
# Edir: GS energy & Cdir: GS
Edir, Cdir = eigen((Hdir + Hdir') / 2)
_, minid = findmin(diagm(Edir))
Edir = Edir[minid]
Cdir = Cdir[:,minid]
Cdir = reshape(Cdir, (size(H1s,1),size(H1s,2),size(H1s,3)))
# vardir : variance from Cdir
HC = contract(Lenv, [3], Cdir, [1])
HC = contract(HC, [2,3], W[Lc], [1,4])
HC = contract(HC, [2,4], Renv, [3,2])
HEC = HC - Edir * Cdir; # HC - EC
vardir = contract(HEC, [1,2,3], conj(HEC), [1,2,3]); # var: |HC-EC|^2
# display result
ratio = (Edir - Eex) / abs(Eex)
@printf("(Edir - Eex) / |Eex| : %.6f %%\n", 100*ratio)
@printf("vardir : %e\n", vardir[1])

# Lanczos method
NLZs = (0:10); # number of Lanczos steps
ELZs = zeros(size(NLZs)); # energy specfic number of stpeps
varLZs = zeros(size(NLZs)); # variance at specfic number of steps
Cinit = mps_iso[Lc] # initial vector for Lanczos method
for itN = (1:length(NLZs))
    NLZ = NLZs[itN]
    Ceff, Eeff = lanczos_1site(Lenv, W[Lc], Renv, Cinit, N=NLZ)
    # energy
    ELZs[itN] = Eeff
    # variance
    local HC = contract(Lenv, [3], Ceff, [1])
    HC = contract(HC, [2,3], W[Lc], [1,4])
    HC = contract(HC, [2,4], Renv, [3,2])
    local CHHC = contract(HC, [1,2,3], conj(HC), [1,2,3])
    varLZs[itN] = CHHC[1] - ELZs[itN]^2
end

# Visualize the energy and variance convergence
publication_theme() = Theme(
    fontsize=16, font="CMU Serif",
    Axis=(xlabelsize=20, ylabelsize=20, xgridstyle=:dash, ygridstyle=:dash,
        xtickalign=1, ytickalign=1, yticksize=10, xticksize=10),
    Legend=(framecolor=(:black, 0.5), backgroundcolor=(:white, 0.5), labelsize=20),
    Colorbar=(ticksize=16, tickalign=1, spinewidth=0.5),
)

with_theme(publication_theme()) do

fig = Figure(size=(1000, 400))
ax1 = Axis(fig,xlabel=L"N_{\mathrm{Lanczos}}",ylabel=L"\mathrm{GS\,\, Energy}")
ax2 = Axis(fig,yscale=log10,xlabel=L"N_{\mathrm{Lanczos}}",ylabel=L"\mathrm{Local\,\, Variance}")
ax3 = Axis(fig,yscale=log10,xlabel=L"N_{\mathrm{Lanczos}}",ylabel=L"\mathrm{Relative\,\, Error}")

scatterlines!(ax1,NLZs,ELZs,label=L"E_\mathrm{LZ}")
lines!(ax1,NLZs,ones(size(NLZs)).*Edir,linestyle=:dash,
      color=:black,label=L"E_\mathrm{dir}")
axislegend(ax1)

scatterlines!(ax2,NLZs,varLZs,label=L"\Delta^\mathrm{loc,LZ}_E")
s=scatter!(ax2,[1],[5e-5],color=:white)
ylims!(ax2,low=1e-6)
axislegend(ax2,labelsize=23)
axislegend(ax2,[s],[L"\Delta^\mathrm{loc,dir}_E=0"],
           framevisible=false,position=:lb,labelsize=26)

scatterlines!(ax3,NLZs,(ELZs.-Edir)./abs.(ELZs),
        label=L"\frac{E_\mathrm{LZ}-E_\mathrm{dir}}{|E_\mathrm{LZ}|}")
scatterlines!(ax3,collect(NLZs),(ELZs.-Eex)./abs.(ELZs),
         label=L"\frac{E_\mathrm{LZ}-E_\mathrm{ex}}{|E_\mathrm{LZ}|}")
lines!(ax3,collect(NLZs),ones(size(NLZs)).*((Edir-Eex)./abs(Eex)),
      label=L"\frac{E_\mathrm{dir}-E_\mathrm{ex}}{|E_\mathrm{ex}|}",
      linestyle=:dash,color=:black)
ylims!(ax3,low=1e-10)
axislegend(ax3,position=:lb)

fig[1,1] = ax1; fig[1,2] = ax2; fig[1,3] = ax3;

save("problem/set03/solution_c.pdf",fig,pt_per_unit=2)

end