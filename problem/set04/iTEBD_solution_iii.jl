using LinearAlgebra
using Statistics
using Printf
using JLD2

import tn_julia: contract, identity, updateLeft
using tn_julia: identity as tn_identity
using tn_julia: svdleftright as tn_svd 

@load "problem/set04/iTEBD_i.jld2"

# iTEBD: GS search

N = 2000; # # of imaginary time steps
Ds = [4,10,100]; # bond dimensions
# Es[itD](2*m-1,1) and Es[itD](2*m-1,2): odd & even bond energies at the
# m-th odd bond update for maximum bond dimension Ds[itD]
# Es[itD](2*m,1) and Es[itD](2*m,2): odd & even bond energies at the m-th
# even bond update for maximum bond dimension Ds[itD]
Es = Array{Any}(undef,length(Ds),1)
Go = Array{Any}(undef,length(Ds),1)
Ge = Array{Any}(undef,length(Ds),1)
Lo = Array{Any}(undef,length(Ds),1)
Le = Array{Any}(undef,length(Ds),1)

# # # # iTEBD: START # # # #
for itD = (1:length(Ds))
    @printf("\n## ## ## ## iTEBD: START (D=%d) ## ## ## ##\n",Ds[itD])

    # initialize energy storage
    Es[itD] = zeros(N*2,2)

    # intial state of even and odd sites: spin up & down
    # initialize Gamma for even & odd sites
    Go[itD] = zeros(1,2,1); Go[itD][1,1,1] = 1; # spin up
    Ge[itD] = zeros(1,2,1); Ge[itD][1,2,1] = 1; # spin down
    Lo[itD] = [1]; Le[itD] = [1]; # initalize Lambda for odd & even bonds

    # check left isometry
    Ao = contract(diagm(Le[itD]), [2], Go[itD], [1])
    AAdag = contract(Ao, [1,2], conj(Ao), [1,2])
    Id = tn_identity(Ao, 3)
    @printf("norm(|AAdag - I|) at odd site (D=%d): %.2e\n", Ds[itD], norm(abs.(Id - AAdag)))

    Ae = contract(diagm(Lo[itD]), [2], Ge[itD], [1])
    AAdag = contract(Ae, [1,2], conj(Ae), [1,2])
    Id = tn_identity(Ae, 3)
    @printf("norm(|AAdag - I|) at even site (D=%d): %.2e\n", Ds[itD], norm(abs.(Id - AAdag)))

    # check right isometry
    Bo = contract(Go[itD], [3], diagm(Lo[itD]), [1])
    BBdag = contract(Bo, [2,3], conj(Bo), [2,3])
    Id = tn_identity(Bo, 1)
    @printf("norm(|BBdag - I|) at odd site (D=%d): %.2e\n", Ds[itD], norm(abs.(Id - BBdag)))

    Be = contract(Ge[itD], [3], diagm(Le[itD]), [1])
    BBdag = contract(Be, [2,3], conj(Be), [2,3])
    Id = tn_identity(Be, 1)
    @printf("norm(|BBdag - I|) at even site (D=%d): %.2e\n", Ds[itD], norm(abs.(Id - BBdag)))

    # iTEBD iteration
    for itN = 1:N
        # --- odd bond update: START ---
        # To: Le*Go*Lo*Ge*Le
        To = contract(diagm(Le[itD]), [2], Go[itD], [1])
        To = contract(To, [3], diagm(Lo[itD]), [1])
        To = contract(To, [3], Ge[itD], [1])
        To = contract(To, [4], diagm(Le[itD]), [1])

        # contract exp(-β/N * Ho) with To
        eHTo = contract(expHo, [3,4], To, [2,3], [3,1,2,4])

        # SVD: reshape and truncate
        U,S,V = tn_svd(eHTo,Nkeep=Ds[itD])
        S = diag(S)
        # normalize singular values
        S = S / norm(S)
        # update Λo (Lo)
        Lo[itD] = S

        # update Γo and Γe (Go, Ge)
        Go[itD] = contract(diagm(1 ./ Le[itD]), [2], U, [1])
        Ge[itD] = contract(V, [3], diagm(1 ./ Le[itD]), [1])
        # measure energy; for the bra/ket states of:
        # Le*Go*Lo*Ge*Le*Go*Lo
        # ----- -- -----
        #  =U   =S   =V

        # Build USV = U * S * V
        US = contract(U, [3], diagm(S), [1])
        USV = contract(US, [3], V, [1])
        USV = reshape(USV, (size(USV,1), size(So,1) * size(Se,1), size(USV,4)))
        H2 = updateLeft([],[],USV,Homat,2,USV)

        # Build GL = Go ⋅ Λo
        GL = contract(Go[itD], [3], diagm(Lo[itD]), [1])

        # Compute ⟨GL| H2 |GL⟩
        Es[itD][2*itN-1,1] = tr(updateLeft(H2,2,GL,[],[],GL))

        # energy at the bond for Le [the 5th tensor in the network]
        # Compute H2 = ⟨US|US⟩
        H2 = updateLeft([],[],US,[],[],US)

        # Build VGL = V ⋅ GL
        VGL = contract(V, [3], GL, [1])
        VGL = reshape(VGL, (size(VGL,1), size(So,1) * size(Se,1), size(VGL,4)))

        # Compute ⟨VGL| Hemat |VGL⟩
        Es[itD][2*itN-1,2] = tr(updateLeft(H2,2,VGL,Hemat,2,VGL))
        
        # normalize by ⟨ψ|ψ⟩
        T = updateLeft([],[],US,[],[],US)
        T = updateLeft(T,2,V,[],[],V)
        T = updateLeft(T,2,GL,[],[],GL)
        Es[itD][2*itN-1,:] = Es[itD][2*itN-1,:]/tr(T)

        if mod(itN, 100) == 0 && itN < N
            @printf("# %d/%d (odd bond), E = %.8g\n",
                    itN, N, first(mean(Es[itD][2*itN-1, :], dims=1)))
        end
        # --- odd bond update: END ---

        # --- even bond update: START ---
        # Te: Lo*Ge*Le*Go*Lo
        Te = contract(diagm(Lo[itD]), [2], Ge[itD], [1])
        Te = contract(Te, [3], diagm(Le[itD]), [1])
        Te = contract(Te, [3], Go[itD], [1])
        Te = contract(Te, [4], diagm(Lo[itD]), [1])

        # apply exp(-β/N * He)
        eHTe = contract(expHe, [3,4], Te, [2,3], [3,1,2,4])

        # SVD: decompose and truncate
        U,S,V = tn_svd(eHTe,Nkeep=Ds[itD])
        S = diag(S)
        # normalize S
        S = S / norm(S)
        # update Λe (Le)
        Le[itD] = S

        # update Go, Ge
        Go[itD] = contract(V, [3], diagm(1 ./ Lo[itD]), [1])
        Ge[itD] = contract(diagm(1 ./ Lo[itD]), [2], U, [1])
        # measure energy; for the bra/ket states of:
        # Lo*Ge*Le*Go*Lo*Ge*Le
        # ----- -- -----
        #  =U   =S   =V

        # energy at the bond for Le
        US = contract(U, [3], diagm(S), [1])
        USV = contract(US, [3], V, [1])
        USV = reshape(USV, (size(USV,1), size(So,1)*size(Se,1), size(USV,4)))
        H2 = updateLeft([],[],USV,Hemat,2,USV)
        GL = contract(Ge[itD], [3], diagm(Le[itD]), [1])
        Es[itD][2*itN,1] = tr(updateLeft(H2,2,GL,[],[],GL))


        # energy at the bond for Lo
        H2 = updateLeft([],[],US,[],[],US)
        VGL = contract(V, [3], GL, [1])
        VGL = reshape(VGL, (size(VGL,1), size(So,1)*size(Se,1), size(VGL,4)))
        Es[itD][2*itN,2] = tr(updateLeft(H2,2,VGL,Homat,2,VGL))

        # normalize by the norm of the bra/ket states
        T = updateLeft([],[],US,[],[],US)
        T = updateLeft(T,2,V,[],[],V)
        T = updateLeft(T,2,GL,[],[],GL)
        Es[itD][2*itN,:] = Es[itD][2*itN,:]/tr(T)

        if mod(itN, 100) == 0 && itN < N
            @printf("# %d/%d (even bond), E = %.8g\n",
                    itN, N, first(mean(Es[itD][2*itN-1, :], dims=1)))
        end
        # --- even bond update: END ---
    end
end
# # # # iTEBD: END # # # #

@save "problem/set04/iTEBD_iii.jld2" Ds Es Go Ge Lo Le

using PyPlot

Esite = 1/4 - log(2);

fig, axs = subplots(ncols=2, figsize=(510, 200)./72)
axs[1].set_xlabel(L"D")
axs[1].set_ylabel("GS energy")
axs[1].axhline(Esite, linewidth=0.5, color="gray")
axs[1].plot(Ds[1],first(mean(Es[1][end,:],dims=1)), ".", label=L"D=4")
axs[1].plot(Ds[2],first(mean(Es[2][end,:],dims=1)), ".", label=L"D=10")
axs[1].plot(Ds[3],first(mean(Es[3][end,:],dims=1)), ".", label=L"D=100")
axs[1].annotate(L"E_\mathrm{site}=1/4-\log2", xy=(20,-0.443), color="gray")
# axs[1].set_ylim(-0.445, -0.44)
axs[1].legend()

axs[2].set_xlabel(L"D")
axs[2].set_ylabel(L"(E(D)-E_\mathrm{site})/|E_\mathrm{site}|")
axs[2].semilogy(Ds[1],(first(mean(Es[1][end,:],dims=1))-Esite)/abs(Esite), ".", label=L"D=4")
axs[2].semilogy(Ds[2],(first(mean(Es[2][end,:],dims=1))-Esite)/abs(Esite), ".", label=L"D=10")
axs[2].semilogy(Ds[3],(first(mean(Es[3][end,:],dims=1))-Esite)/abs(Esite), ".", label=L"D=100")
axs[2].set_ylim(5e-4, 1e-2)
axs[2].legend()

tight_layout()
fig.savefig("problem/set04/iTEBD_iii.pdf")
