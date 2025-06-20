using CairoMakie
using Polynomials

@load "problem/set04/baseline.jld2" L Eex
@load "problem/set04/solution_c.jld2" n dws E0s

## Figure for Exercise (c)(i): initial particle distribution 
publication_theme() = Theme(
    fontsize = 16, font = "CMU Serif",
    Axis = (
        xlabelsize = 20, ylabelsize = 20,
        xgridstyle = :dash, ygridstyle = :dash,
        xtickalign = 1, ytickalign = 1,
        yticksize = 10, xticksize = 10
    ),
    Legend = (
        framecolor = (:black, 0.5),
        backgroundcolor = (:white, 0.5),
        labelsize = 20
    ),
    Colorbar = (
        ticksize = 16,
        tickalign = 1,
        spinewidth = 0.5
    )
)

with_theme(publication_theme()) do
    fig = Figure(size = (800, 250))
    ax = Axis(fig[1, 1], xlabel = L"\ell", ylabel = L"n_\ell")
    scatterlines!(ax, 1:L, vec(collect([x[1] for x in n]')))
    save("problem/set04/solution_c_i.pdf", fig, pt_per_unit = 2)
end


## Figure for Exercise (c)(iii): ground state energy
p = fit(Vector{Float64}(dws[2:end]),(E0s[2:end].-Eex)./abs(Eex),1)
print("Linear fitting: $p\n")

publication_theme() = Theme(
    fontsize = 16, font = "CMU Serif",
    Axis = (
        xlabelsize = 20, ylabelsize = 20,
        xgridstyle = :dash, ygridstyle = :dash,
        xtickalign = 1, ytickalign = 1,
        yticksize = 10, xticksize = 10
    ),
    Legend = (
        framecolor = (:black, 0.5),
        backgroundcolor = (:white, 0.5),
        labelsize = 20
    ),
    Colorbar = (
        ticksize = 16,
        tickalign = 1,
        spinewidth = 0.5
    )
)

with_theme(publication_theme()) do
    fig = Figure(size = (500, 400))
    ax = Axis(
        fig[1, 1],
        xscale = log10, yscale = log10,
        xlabel = L"\xi_\mathrm{tot}",
        ylabel = L"(E_\mathrm{GS} - E_\mathrm{ex}) / |E_\mathrm{ex}|"
    )

    lines!([2e-5, 9e-9], p.([2e-5, 9e-9]), linestyle = :dash, color = :black)
    scatter!(ax, Point2f(dws[1], (E0s[1] - Eex) / abs(Eex)), label = L"D_\mathrm{max}=40")
    scatter!(ax, Point2f(dws[2], (E0s[2] - Eex) / abs(Eex)), label = L"D_\mathrm{max}=60")
    scatter!(ax, Point2f(dws[3], (E0s[3] - Eex) / abs(Eex)), label = L"D_\mathrm{max}=80")
    scatter!(ax, Point2f(dws[4], (E0s[4] - Eex) / abs(Eex)), label = L"D_\mathrm{max}=100")
    s1 = scatter!(ax, Point2f(1e-7, 1e-7), color = :white)
    s2 = scatter!(ax, Point2f(1e-7, 1e-7), color = :white)

    axislegend(ax, position = :rb, labelsize = 19)
    axislegend(
        ax, [s1, s2],
        [L"\mathrm{linear\,\,fit:}", L"0.0427\xi_\mathrm{tot} -6.87e^{-11}"],
        position = :lt, framevisible = false, labelsize = 19
    )

    save("problem/set04/solution_c_iii.pdf", fig, pt_per_unit = 2)
end