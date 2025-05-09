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
    Id = I(Int(2spin + 1))

    return Splus, Sminus, Sz, Id
end
