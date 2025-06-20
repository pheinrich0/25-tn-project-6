using Polynomials
using LinearAlgebra
using Statistics
using Printf
using JLD2
using PyPlot

import tn_julia: contract, identity, updateLeft
using tn_julia: identity as tn_identity


@load "problem/set04/iTEBD_i.jld2"
@load "problem/set04/iTEBD_iii.jld2"
@load "problem/set04/iTEBD_iv.jld2"

# iTEBD: Correlation function (OPTIONAL)
k = 100
C = Array{Any}(undef,k,length(Ds))
C1 = 0.147715726853315; # Benckmark value for k=1
C2 = 0.060679769956435; # Benckmark value for k=2
# Using canonicalized Gammas amd Lambdas
Ge = Gecan
Go = Gocan
Le = Lecan
Lo = Locan
for itD = (1:length(Ds))
    Ao = contract(diagm(Le[itD]), [2], Go[itD], [1])
    Ae = contract(diagm(Lo[itD]), [2], Ge[itD], [1])
    for itk = (1:k)
        # compute correlator <Sz*Sz>
        Co = tn_identity(Ae, 3)
        Co = updateLeft(Co, 2, Ao, So[:,3,:], 2, Ao)
        Ce = tn_identity(Ao, 3)
        Ce = updateLeft(Ce, 2, Ae, Se[:,3,:], 2, Ae)
        for itk2 = (1:k)
            if itk2 < itk
                if mod(itk2,2) == 1
                    Co = updateLeft(Co, 2, Ae, [], 0, Ae)
                    Ce = updateLeft(Ce, 2, Ao, [], 0, Ao)
                else
                    Co = updateLeft(Co, 2, Ao, [], 0, Ao)
                    Ce = updateLeft(Ce, 2, Ae, [], 0, Ae)
                end
            elseif itk2 == itk
                if mod(itk2,2) == 1
                    Co = updateLeft(Co, 2, Ae, Se[:,3,:], 2, Ae)
                    Co = contract(Co, [2], diagm(Le[itD]), [1])
                    Co = contract(Co, [2], diagm(Le[itD]), [1])
                    Ce = updateLeft(Ce, 2, Ao, So[:,3,:], 2, Ao)
                    Ce = contract(Ce, [2], diagm(Lo[itD]), [1])
                    Ce = contract(Ce, [2], diagm(Lo[itD]), [1])
                else
                    Co = updateLeft(Co, 2, Ao, So[:,3,:], 2, Ao)
                    Co = contract(Co, [2], diagm(Lo[itD]), [1])
                    Co = contract(Co, [2], diagm(Lo[itD]), [1])
                    Ce = updateLeft(Ce, 2, Ae, Se[:,3,:], 2, Ae)
                    Ce = contract(Ce, [2], diagm(Le[itD]), [1])
                    Ce = contract(Ce, [2], diagm(Le[itD]), [1])
                end
            end
        end

        To = tn_identity(Ae, 3)
        To = updateLeft(To, 2, Ao, [], 0, Ao)
        Te = tn_identity(Ao, 3)
        Te = updateLeft(Te, 2, Ae, [], 0, Ae)
        for itk2 = (1:itk)
            if mod(itk2,2) == 1
                To = updateLeft(To, 2, Ae, [], 0, Ae)
                Te = updateLeft(Te, 2, Ao, [], 0, Ao)
            else
                To = updateLeft(To, 2, Ao, [], 0, Ao)
                Te = updateLeft(Te, 2, Ae, [], 0, Ae)
            end
            if itk2 == itk
                if mod(itk,2) == 1
                    To = contract(To, [2], diagm(Le[itD]), [1])
                    To = contract(To, [2], diagm(Le[itD]), [1])
                    Te = contract(Te, [2], diagm(Lo[itD]), [1])
                    Te = contract(Te, [2], diagm(Lo[itD]), [1])
                else
                    To = contract(To, [2], diagm(Lo[itD]), [1])
                    To = contract(To, [2], diagm(Lo[itD]), [1])
                    Te = contract(Te, [2], diagm(Le[itD]), [1])
                    Te = contract(Te, [2], diagm(Le[itD]), [1])
                end
            end
        end

        So_ell = tn_identity(Ae, 3)
        So_ell = updateLeft(So_ell, 2, Ao, So[:,3,:], 2, Ao)
        Se_ell = tn_identity(Ao, 3)
        Se_ell = updateLeft(Se_ell, 2, Ae, Se[:,3,:], 2, Ae)
        So_ell_k = tn_identity(Ae, 3)
        So_ell_k = updateLeft(So_ell_k, 2, Ao, [], 0, Ao)
        Se_ell_k = tn_identity(Ao, 3)
        Se_ell_k = updateLeft(Se_ell_k, 2, Ae, [], 0, Ae)
        for itk2 = (1:k)
            if itk2 < itk
                if mod(itk2,2) == 1
                    So_ell = updateLeft(So_ell, 2, Ae, [], 0, Ae)
                    Se_ell = updateLeft(Se_ell, 2, Ao, [], 0, Ao)
                    So_ell_k = updateLeft(So_ell_k, 2, Ae, [], 0, Ae)
                    Se_ell_k = updateLeft(Se_ell_k, 2, Ao, [], 0, Ao)
                else
                    So_ell = updateLeft(So_ell, 2, Ao, [], 0, Ao)
                    Se_ell = updateLeft(Se_ell, 2, Ae, [], 0, Ae)
                    So_ell_k = updateLeft(So_ell_k, 2, Ao, [], 0, Ao)
                    Se_ell_k = updateLeft(Se_ell_k, 2, Ae, [], 0, Ae)
                end
            elseif itk2 == itk
                if mod(itk,2) == 1
                    So_ell = updateLeft(So_ell, 2, Ae, [], 0, Ae)
                    So_ell = contract(So_ell, [2], diagm(Le[itD]), [1])
                    So_ell = contract(So_ell, [2], diagm(Le[itD]), [1])
                    Se_ell = updateLeft(Se_ell, 2, Ao, [], 0, Ao)
                    Se_ell = contract(Se_ell, [2], diagm(Lo[itD]), [1])
                    Se_ell = contract(Se_ell, [2], diagm(Lo[itD]), [1])
                    So_ell_k = updateLeft(So_ell_k, 2, Ae, Se[:,3,:], 2, Ae)
                    So_ell_k = contract(So_ell_k, [2], diagm(Le[itD]), [1])
                    So_ell_k = contract(So_ell_k, [2], diagm(Le[itD]), [1])
                    Se_ell_k = updateLeft(Se_ell_k, 2, Ao, So[:,3,:], 2, Ao)
                    Se_ell_k = contract(Se_ell_k, [2], diagm(Lo[itD]), [1])
                    Se_ell_k = contract(Se_ell_k, [2], diagm(Lo[itD]), [1])
                else
                    So_ell = updateLeft(So_ell, 2, Ao, [], 0, Ao)
                    So_ell = contract(So_ell, [2], diagm(Lo[itD]), [1])
                    So_ell = contract(So_ell, [2], diagm(Lo[itD]), [1])
                    Se_ell = updateLeft(Se_ell, 2, Ae, [], 0, Ae)
                    Se_ell = contract(Se_ell, [2], diagm(Le[itD]), [1])
                    Se_ell = contract(Se_ell, [2], diagm(Le[itD]), [1])
                    So_ell_k = updateLeft(So_ell_k, 2, Ao, So[:,3,:], 2, Ao)
                    So_ell_k = contract(So_ell_k, [2], diagm(Lo[itD]), [1])
                    So_ell_k = contract(So_ell_k, [2], diagm(Lo[itD]), [1])
                    Se_ell_k = updateLeft(Se_ell_k, 2, Ae, Se[:,3,:], 2, Ae)
                    Se_ell_k = contract(Se_ell_k, [2], diagm(Le[itD]), [1])
                    Se_ell_k = contract(Se_ell_k, [2], diagm(Le[itD]), [1])
                end
            end
        end
        C[itk,itD] = (-1)^itk*(tr(Co)/tr(To)+tr(Ce)/tr(Te))/2
        if mod(itk,2) == 1
            C[itk,itD] = C[itk,itD] - (-1)^itk*
                (tr(So_ell)/tr(To) * tr(So_ell_k)/tr(Te)
                + tr(Se_ell)/tr(Te) * tr(Se_ell_k)/tr(To))/2
        else
            C[itk,itD] = C[itk,itD] - (-1)^itk*
                (tr(So_ell)/tr(To) * tr(So_ell_k)/tr(To)
                + tr(Se_ell)/tr(Te) * tr(Se_ell_k)/tr(Te))/2
        end
    end
end

pl = fit(log.(1:10),log.(vec(abs.([x[1] for x in C[1:10,3]]))),1)
pr = fit(30:100,log.(vec(abs.([x[1] for x in C[30:100,3]]))),1)

fig, ax = subplots(figsize=(510, 400) ./ 72)
ax.set_xlabel(L"k")
ax.set_ylabel("Correlation function")
ax.loglog(1:k, vec(abs.([x[1] for x in C[:,1]])), "k^-", markersize=3, label=L"D=4")
ax.loglog(1:k, vec(abs.([x[1] for x in C[:,2]])), "ks-", markersize=3, label=L"D=10")
ax.loglog(1:k, vec(abs.([x[1] for x in C[:,3]])), "ko-", markersize=3, label=L"D=100")

ax.loglog([1, 2], [C1, C2], "o", label="literature values")
ax.loglog(1:20, exp(pl[0]).*(1:20).^pl[1], color="tab:orange", label=L"0.14\times k^{-1.18}")
ax.loglog(10:100,exp(pr[0]).*exp(pr[1]).^(10:100), color="tab:pink", label=L"0.02\times 0.90^{k}")

ax.set_ylim(1e-8, 1)
ax.legend()

fig.savefig("problem/set04/iTEBD_v.pdf")
