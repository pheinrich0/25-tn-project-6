## T03: Single Site DMRG (b)
using LinearAlgebra
using Printf
using JLD2

using tn_julia: leftcanonical,  rightcanonical,  sitecanonical,
                leftcanonical!, rightcanonical!, sitecanonical!,
                computeLeftEnvironment, computeRightEnvironment,
                applyHtoC, updateLeft, spinlessfermionlocalspace,
                contract

@load "problem/set03/share.jld2" mps W Eex Eiter L

## i) shift isometry center
Liso = 50; # isometry center
Nkeep = 50; # maximal # of states to keep
mps_iso = sitecanonical(mps, Liso; Nkeep=100, tolerance=1e-8)

## iii) apply H^{1s}_ell to C_ell
HC = applyHtoC(W, mps_iso, Liso)

## iv) compute energy
E = contract(HC, [1,2,3], mps_iso[Liso], [1,2,3])
CHHC = contract(HC, [1,2,3], conj(HC), [1,2,3])
var = CHHC[1] - E[1]^2

# display result
@printf("E : %.4f\n", E[1])
@printf("Eiter - E : %.4f\n", Eiter - E[1])
@printf("var : %.4f\n", var)