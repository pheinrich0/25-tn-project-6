using LinearAlgebra

"""
    spinlocalspace(spin::Rational=1//2)

Construct the spin local space operators for a given spin value.

Parameters:
- `spin::Rational`: The spin value, which should be a positive (half-)integer.
Returns:
- `Splus`: The spin raising operator.
- `Sminus`: The spin lowering operator.
- `Sz`: The spin-z operator.
- `Id`: The identity operator.

The spin local space operators are represented as matrices of size `(2spin + 1, 2spin + 1)`.
"""
function spinlocalspace(spin::Rational=1 // 2)
    if !isinteger(2 * spin)
        error("Spin should be positive (half-)integer.")
    end

    spinvals = collect(spin:-1:-spin)
    Splus = diagm(1 =>
        sqrt.((spin .- spinvals[2:end]) .* (spin .+ spinvals[1:end-1]))
    ) / sqrt(2)

    Sminus = diagm(-1 =>
        sqrt.((spin .+ spinvals[1:end-1]) .* (spin .- spinvals[2:end]))
    ) / sqrt(2)

    Sz = diagm(Float64.(spinvals))
    Id = identity(Sz, 1)

    return Splus, Sminus, Sz, Id
end

"""
    spinlessfermionlocalspace()

Construct the spin local space operators for a given spin value.

Returns:
- `F`: The fermion annihilation operator.
- `Z`: Jordan-Wigner string operator for anti-commutation sign of fermions.
- `Id`: The identity operator.

All operators are represented as matrices of size (2, 2).
"""
function spinlessfermionlocalspace()
    Id = I(2)

    F = [0 1; 0 0]
    Z = [1 0; 0 -1]

    return F, Z, Id
end

using LinearAlgebra

struct LocalSpace
    F
    Z
    S
    I
end

function getLocalSpace(Mode::String, s::Float64=1/2)
# < Description >
#
# [S,I] = getLocalSpace("Spin",s)         # spin
# [F,Z,I] = getLocalSpace("Fermion")      # spinless fermion
# [F,Z,S,I] = getLocalSpace("FermionS")   # spinful [spin-1/2] fermion 
#
# Generates the local operators as tensors. The result operators F & S
# are rank-3; whose 1st and 3rd legs are to be contracted with bra & ket
# tensors; respectively. The 2nd legs of F & S encode the flavors of the
# operators; such as spin raising/lowering/z | particle flavor.
# Basis of the output tensors depend on the input as follows:
#   * "Spin";s: +s; +s-1; ...; -s
#   * "Fermion': |vac>; c"|vac>
#   * "FermionS': |vac>; c'_down|vac>; c'_up|vac>; c'_down c"_up|vac>
# Here c' means fermion creation operator.
#
# < Input >
# s : [integer | half-integer] The value of spin [e.g., 1/2, 1, 3/2, ...].
#
# < Output >
# 
# A LocalSpace-type struct containing the following data:
# 
# S : [rank-3 tensor] Spin operators.
#       S[:,1,:] : spin raising operator S_+ multiplied with 1/sqrt(2)
#       S[:,2,:] : spin lowering operator S_- multiplied with 1/sqrt(2)
#       S[:,3,:] : spin-z operator S_z
#       Then we can construct the Heisenberg interaction ($\vec[S] \cdot
#       \vec[S]$) by: contract[S,3,2,conj(S),3,2] that results in
#       (S^+ * S^-)/2 + (S^- * S^+)/2 + (S^z * S^z) = (S^x * S^x) + (S^y *
#       S^y) + (S^z * S^z).
#       There are two advantages of using S^+ and S^- rather than S^x &
#       S^y: (1) more compact. For spin-1/2 case for example, S^+ and S^-
#       have only one non-zero elements while S^x & S^y have two. (2) We
#       can avoid complex number which can induce numerical error & cost
#       larger memory; a complex number is treated as two double numbers.
# I : [rank-2 tensor] Identity operator.
# F : [rank-3 tensor] Fermion annihilation operators. For spinless fermions
#       ("Fermion"), the 2nd dimension of F is singleton, & F[:,1,:] is()
#       the annihilation operator. For spinful fermions ["FermionS"]
#       F[:,1,:] & F[:,2,:] are the annihilation operators for spin-up
#       & spin-down particles; respectively.
# Z : [rank-2 tensor] Jordan-Wigner string operator for anticommutation
#        sign of fermions.
#
# Written originally by S.Lee in 2017 in terms of MATLAB.
# Transformed by Changkai Zhang in 2022 into Julia.

# # parsing input()
if (length(Mode) == 0) || ~(Mode in ["Spin","Fermion","FermionS"])
    error("ERR: Input #1 should be either ''Spin'', ''Fermion'', | ''FermionS''.")
end

if Mode == "Spin"
    if (abs(2*s - round(2*s)) .> 1e-14) || (s <= 0)
        error("ERR: Input #2 for ''Spin'' should be positive [half-]integer.")
    end
    s = round(2*s)/2
    isFermion = false
    isSpin = true; # create S tensor
    Id = I(Int64(round(2s+1)));
elseif Mode == "Fermion"
    isFermion = true; # create F & Z tensors
    isSpin = false
    Id = I(2);
elseif Mode == "FermionS"
    isFermion = true
    isSpin = true
    s = 0.5
    Id = I(4);
end
# # #

if isFermion
    if isSpin # spinful fermion
        # basis: empty, down, up, two [= c_down^+ c_up^+ |vac>]
        F = zeros(4,2,4)
        # spin-up annihilation
        F[1,1,3] = 1; 
        F[2,1,4] = -1; # -1 sign due to anticommutation
        # spin-down annihilation
        F[1,2,2] = 1; 
        F[3,2,4] = 1

        Z = diagm([1, -1, -1, 1])

        S = zeros(4,3,4)
        S[3,1,2] = 1/sqrt(2); # spin-raising operator [/sqrt(2)]
        S[2,2,3] = 1/sqrt(2); # spin-lowering operator [/sqrt(2)]
        # spin-z operator
        S[3,3,3] = 1/2; 
        S[2,3,2] = -1/2
    else # spinless fermion
        # basis: empty; occupied
        F = zeros(2,1,2)
        F[1,1,2] = 1

        Z = diagm([1, -1])
    end
else # spin
    # basis: (
    Sp = (s-1:-1:-s)
    Sp = diagm(1 => sqrt.((s.-Sp).*(s.+Sp.+1))); # spin raising operator

    Sm = (s:-1:-s+1); 
    Sm = diagm(-1 => sqrt.((s.+Sm).*(s.-Sm.+1))); # spin lowering operator

    Sz = diagm(s:-1:-s); # spin-z operator

    S = permutedims(cat(Sp/sqrt(2),Sm/sqrt(2),Sz,dims=3),(1,3,2))
end

# assign the tensors into varargout()
if isFermion
    if isSpin # spinful fermion
        LSpace = LocalSpace(F,Z,S,Id);
    else # spinless fermion
        LSpace = LocalSpace(F,Z,nothing,Id);
    end
else # spin
    LSpace = LocalSpace(nothing,nothing,S,Id);
end

return LSpace

end

