
using LaTeXStrings
using PyPlot
using Polynomials

"""
    DMRG2s_plot_a_ii(N,E_lanczos,E_ex)

Plots relative error between Lanczos eigenvalues and exact eigenvalues
against the number of Lanczos steps 

# Input
- `N` : numeric, represents the number of Lanczos steps.
- `E_lanczos` : N-dimensional vector containing energy values obtained from 
              the Lanczos scheme for different # of iterations 
- `E_ex` : numeric, the exact ground state energy obtain from 
           diagonalizing the single-particle Hamiltonian

# Output
- `fig`: a figure that plots the relative error vs # of iterations 
"""
function DMRG2s_plot_a_ii(E_lanczos,E_ex,N)

N_lanczos = (1:N) #array containing Lanczos iterations
fig, ax = subplots(figsize=(246, 200) ./ 72)

ax.set_xlabel(L"N_{\mathrm{Lanczos}}")
ax.set_ylabel("Relative error")

ax.plot(
    N_lanczos, (E_lanczos .- E_ex) ./ abs(E_ex), ".-",
    label=L"\frac{E_\mathrm{LZ}-E_\mathrm{ex}}{|E_\mathrm{ex}|}"
)

ax.legend()
tight_layout()

return fig

end

"""
    DMRG2s_plot_a_iii(E_trunc,E_ex,D_f,disc_weight)

Plots:
(1) The difference between truncated and exact energy vs.
    bond dimension.
(2) Bond dimension vd discarded weight. 
(3) Relative error between truncated and exact energies
    vs. bond dimension.

# Input
- `E_trunc` : a vector containing energies after SVD truncation
- `E_ex` : numeric, the exact ground state energy obtain from 
           diagonalizing the single-particle Hamiltonian
- `D_f` : a vector of considered final bond dimensions
- `disc_weight` : a vector containing discarded weights 


# Output
- `fig`: a figure containing all three comparison plots 
"""

function DMRG2s_plot_a_iii(E_trunc,E_ex,D_f,disc_weight)

fig, axs = subplots(ncols=3, figsize=(510, 300) ./ 72)
for ax in axs
    ax.set_xlabel(L"D_{\mathrm{f}}")
end

axs[1].set_ylabel("Energy difference")
axs[1].plot(D_f, E_trunc.-E_ex, ".-", label=L"E-E_\mathrm{ex}")
axs[1].legend()

axs[2].set_ylabel("Discarded weight")
axs[2].semilogy(D_f, disc_weight, ".-")

axs[3].set_ylabel("Relative error")
axs[3].plot(D_f,(E_trunc.-E_ex)./abs.(E_ex), ".-", label=L"\frac{E-E_\mathrm{ex}}{|E_\mathrm{ex}|}")
axs[3].legend()

tight_layout()

return fig

end


"""
    DMRG2s_plot_c_i(L,n)

Plots local occupation vs. site number to verify the charge distribution 
in the system. 

# Input
- 'n' : a vector containing local occupation numbers for each site.
- `L` : numeric, system size. 

# Output
- `fig` 
"""

function DMRG2s_plot_c_i(L,n)

fig, ax = subplots(figsize=(510, 120) ./ 72)
ax.set_xlabel(L"\ell")
ax.set_ylabel(L"n_\ell")
plot(1:L, only.(n)', ".-")
tight_layout()

return fig

end

"""
    DMRG2s_plot_c_ii(E0s,E_ex,disc_weight)

Plots relative error of the DMRG g.s. energy w.r.t the exact 
value vs. the discarded weight for each value of D_max 
(40:20:100). 

# Input
- `E0s` : a vector containing ground state energies obtained with 
          'DMRG_2site'routine for different values of D_max
- `E_ex` : numeric, the exact ground state energy obtain from 
           diagonalizing the single-particle Hamiltonian
- `disc_weight` : a vector containing discarded weights 

# Output
- `fig` 
"""

function DMRG2s_plot_c_ii(E0s,E_ex,disc_weight)

fig, ax = subplots(figsize=(300, 200) ./ 72)
ax.set_xlabel(L"\xi_\mathrm{tot}")
ax.set_ylabel(L"(E_\mathrm{GS}-E_\mathrm{ex})/\;\;\:|E_\mathrm{ex}|")

ax.loglog(disc_weight[1], (E0s[1]-E_ex)/abs(E_ex), ".", label=L"D_\mathrm{max}=40")
ax.loglog(disc_weight[2], (E0s[2]-E_ex)/abs(E_ex), ".", label=L"D_\mathrm{max}=60")
ax.loglog(disc_weight[3], (E0s[3]-E_ex)/abs(E_ex), ".", label=L"D_\mathrm{max}=80")
ax.loglog(disc_weight[4], (E0s[4]-E_ex)/abs(E_ex), ".", label=L"D_\mathrm{max}=100")
ax.legend()
tight_layout()

return fig

end

"""
    DMRG2s_plot_c_iii_fit(E0s,E_ex,disc_weight)

Plots relative error of the DMRG g.s. energy w.r.t the exact 
value vs. the discarded weight for each value of D_max 
(40:20:100). Fits a linear fit to the best points in the 
dataset. (Please adjust accordingly). 

# Input
- `E0s` : a vector containing ground state energies obtained with 
          'DMRG_2site'routine for different values of D_max
- `E_ex` : numeric, the exact ground state energy obtain from 
           diagonalizing the single-particle Hamiltonian
- `disc_weight` : a vector containing discarded weights 

# Output
- `fig` 
"""

function DMRG2s_plot_c_iii_fit(E0s,E_ex,disc_weight)

p = fit(Vector{Float64}(disc_weight[2:end]),(E0s[2:end].-E_ex)./abs(E_ex),1)
print("Linear fitting: $p\n")

fig, ax = subplots(figsize=(300, 200) ./ 72)
ax.set_xlabel(L"\xi_\mathrm{tot}")
ax.set_ylabel(L"(E_\mathrm{GS}-E_\mathrm{ex})/\;\;\:|E_\mathrm{ex}|")

ax.loglog(
    disc_weight, p.(disc_weight), color="gray", linewidth=1,
    label="linear fit")

ax.annotate(
    @sprintf("\$%.4f\\,\\xi_{\\mathrm{tot}} + %.3e\$", p.coeffs[2], p.coeffs[1]),
    xy=(disc_weight[3], p(disc_weight[3])), xytext=(-5, -20), textcoords="offset points", color="gray")

ax.loglog(disc_weight[1], (E0s[1]-E_ex)/abs(E_ex), ".", label=L"D_\mathrm{max}=40")
ax.loglog(disc_weight[2], (E0s[2]-E_ex)/abs(E_ex), ".", label=L"D_\mathrm{max}=60")
ax.loglog(disc_weight[3], (E0s[3]-E_ex)/abs(E_ex), ".", label=L"D_\mathrm{max}=80")
ax.loglog(disc_weight[4], (E0s[4]-E_ex)/abs(E_ex), ".", label=L"D_\mathrm{max}=100")

ax.legend(loc="upper left")
tight_layout()

return fig 

end