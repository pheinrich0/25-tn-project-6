## T04 Two Site DMRG (c)
using LinearAlgebra
using Printf
using JLD2

using tn_julia: updateLeft, DMRG_2site

@load "problem/set04/baseline.jld2" W L

# i) Initial state of D=1
MPS = Vector{Array{Float64,3}}(undef, L); # initial MPS
for itL = (1:L)
    MPS[itL] = zeros(1,2,1);
    if mod(itL,2) == 1
        MPS[itL][1,1,1] = 1; # occupied states for 1,3,5
    else
        MPS[itL][1,2,1] = 1; # empty states for 2,4,6
    end
end
# verify MPS by local occupations
n = Array{Any}(undef, L);
for itL = (1:L)
    n[itL] = reshape([1],(1,1,1));
    # electron number opeartor at site itL
    n_itL = zeros(1,2,1,2); # ordering: left down right up
    n_itL[1,1,1,1] = 1
    for itL2 = (1:L)
        if itL2 == itL
            n[itL] = updateLeft(n[itL], MPS[itL2], n_itL, MPS[itL2])
        else
            # Create identity operator when X is empty
            I_op = zeros(1,2,1,2)
            I_op[1,1,1,1] = 1
            I_op[1,2,1,2] = 1
            n[itL] = updateLeft(n[itL], MPS[itL2], I_op, MPS[itL2])
        end
    end
end

println("Local occupations:")
for itL = 1:L
    @printf("Site %d: %.6f\n", itL, n[itL][1,1,1])
end


# ii) 2s DMRG: GS energy error versus discarded weight for different Dmax
alpha = sqrt(2)
Dmax = [40,60,80,100]
Nsweep = 100; # number of DMRG sweeps will be 2*Nsweep
thresh = 1e-12; # Convergence criterion for energy
E0s = zeros(size(Dmax)); # GS energy for different Dfs
dws = Array{Any}(undef,size(Dmax)); # discarded weight for different Dfs
MPS2s = Array{Any}(undef,size(Dmax)); # GS MPS for different Dfs

for itD = (1:length(Dmax))
    @printf("## ## ## ## 2s DMRG: START [Dmax=%d] ## ## ## ##\n",Dmax[itD])
    # DMRG sweep until reaching convergence
    MPS2s[itD],E0s[itD],_,_ =
        DMRG_2site(W,MPS,alpha,Dmax[itD],Nsweep;Econv=thresh)
    # one more DMRG sweep for collecting discarded weights
    _,_,_,dw =
        DMRG_2site(W,MPS2s[itD],alpha,Dmax[itD],1)
    dws[itD] = sum(dw[:])
    @printf("## ## ## ## 2s DMRG: END [Dmax=%d] ## ## ## ##\n",Dmax[itD])
end

@save "problem/set04/solution_c.jld2" MPS2s n dws E0s Dmax
