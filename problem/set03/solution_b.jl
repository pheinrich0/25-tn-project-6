## T03: Single Site DMRG (b)
using LinearAlgebra
using Printf
using JLD2

using tn_julia: leftcanonical,  rightcanonical,  sitecanonical,
                leftcanonical!, rightcanonical!, sitecanonical!,
                computeLeftEnvironment, computeRightEnvironment,
                applyHtoC, updateLeft, spinlessfermionlocalspace,
                contract

@load "problem/set03/solution_a.jld2" mps W Eex Eiter L

## i) shift isometry center
Lc = 50; # isometry center
Nkeep = 50; # maximal # of states to keep
mps_iso = sitecanonical(mps, Lc; Nkeep=100, tolerance=1e-8)

## ii) compute left and right environments
Lenv = computeLeftEnvironment(W, mps_iso, Lc-1)
Renv = computeRightEnvironment(W, mps_iso, Lc+1)

## iii) apply H^{1s}_ell to C_ell
HC = applyHtoC(W, mps_iso, Lc)

## iv) compute energy
E = contract(HC, [1,2,3], mps_iso[Lc], [1,2,3])
CHHC = contract(HC, [1,2,3], conj(HC), [1,2,3])
var = CHHC[1] - E[1]^2

# display result
@printf("E : %.4f\n", E[1])
@printf("Eiter - E : %.4f\n", Eiter - E[1])
@printf("var : %.4f\n", var)

@save "problem/set03/solution_b.jld2" mps mps_iso W Eex Lc Lenv Renv