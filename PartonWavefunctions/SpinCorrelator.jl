using LinearAlgebra, Plots, JLD2 
using Dates, Test
import tn_julia as tn
include("checkoccupation.jl")

# Parameters
N = 32;
Dmax = 100;

# haldane_shastry.jl contains the exact 
# ⟨S_p·S_q⟩ = ( ∑_{a=1}^{N/2} 3*(-1)^q/(2a-1) * sin(π*(2a-1)*q/N) ) / ( 2N * sin(π*q/N) )
include("haldane_shastry.jl")

qvals = 1:(N-1);
yvals = [spin_spin_corr(N, q) for q in qvals]

# Plot with thicker line (lw=3) and boxed frame
plot(
    qvals, yvals;
    lw = 2,
    framestyle = :box,
    xlabel = "q",
    ylabel = "⟨Sₚ·Sₚ₊q⟩",
    title = "Spin–spin correlation for N = $N",
    legend = false
)

# #############################################################################
## calculate the spin-spin correlator S_p*S_q for the projected fermi sea     #
# #############################################################################

# imports: Fermiseas and gutzwiller projectors
@load "PartonWavefunctions/NormalizedMPS.jld2" MPS_iter_Norm MPS_wannier_Norm normMPS_lmr
@load "PartonWavefunctions/gutzwillerProj.jld2" gutzwiller_mpo applyPG_fermion_Normalized



# ## MPO for S_q*S_p in spin space:
Sp, Sm, Sz, _ = tn.spinlocalspace();

function spinCorrelator(p::Int, q::Int)
    @assert 1 ≤ p < q ≤ N "Require 1 ≤ p < q ≤ N"

    tn
end
