using LinearAlgebra, Plots, JLD2
import tn_julia as tn

# Setup: N spin lattice: index j = 1, ..., N
N=32;
# parton model: at each lattice site there is a fermionic degree of freedom for each spin
# there are two possibilities for modeling the parton space; either combined index l = (j, alpha) that goes from 1 to 2N
#    -> leg dim 2 at each site l
# or N sites labeled by j = 1, ..., N where a 4-dimensional space sits

# set up the fermionic ground state (length 2N mps)
# from sheet 4: empty = [0,1] and occupied = [1,0] 

fermionic_mps = Vector{Array{ComplexF64, 3}}(undef, 2N)

unoccupied = zeros(1, 2, 1);
unoccupied[1, 2, 1] = 1.0; # unoccupied site
occupied = zeros(1, 2, 1);
occupied[1, 1, 1] = 1.0; 

# some test for 1 site / spin up + spin down partons
test1s = tn.contract( tn.contract(unoccupied, 3, occupied, 1), [2,3], Id_site, [1,2], [1,3,2])
reshape(tn.contract(unoccupied,3, occupied, 1),(1,4,1))
test1s[1,:,1]
# ie. unoccupied × occupied = (0,1,0,0)
# doesnt agree with
kron([0,1], [1,0]) # = (0,0,1,0)

test1s_2 = tn.contract( tn.contract(occupied,3, occupied, 1), [2,3], Id_site, [1,2], [1,3,2])
test1s_2[1,:,1]
# Calculate the Kronecker product of two basis vectors to obtain the corresponding basis vector in the enlarged basis
reshape(tn.contract(occupied, 3, unoccupied, 1), (1,4,1))
#vs 
kron([1,0], [0,1]) # = (0,1,0,0)

tn.contract( tn.contract(unoccupied,3, unoccupied, 1), [2,3], Id_site, [1,2], [1,3,2])
# so the basis vectors in C^4 apparently correspond to {double occupied, spin down, spin up, empty}
# the disagreement between the kronecker product and tensor contraction is confusing

for i in 1:2N
    # easiest representation of a single site MPS for spinless fermions
    fermionic_mps[i] = unoccupied;
end

filled_mps = Vector{Array{ComplexF64, 3}}(undef, 2N)
for i in 1:2N
    # easiest representation of a single site MPS for spinless fermions
    filled_mps[i] = occupied;
end

totalOccupation(filled_mps)
calculate_n_j_alpha(filled_mps, 4)
## contract with 2-leg identities to get the parton gs-MPS
# by convention, the odd sites correspond to spin up, the even sites to spin down
Id_site = tn.identity(occupied, 2, occupied, 2)
parton_mps = Vector{Array{ComplexF64, 3}}(undef, N)
for itN in 1:N
    # contract spin up with neighboring spin down
    updown = tn.contract(fermionic_mps[(2*itN-1)], 3, fermionic_mps[2*itN], 1) # 1x2x2x1 
    updown = tn.contract( updown, [2,3], Id_site, [1,2], [1,3,2]) # 1x1x4 -> 1x4x1
    parton_mps[itN] = updown
end
fermionic_mps[1] 
fermionic_mps[2]
parton_mps[1]

## number operators, as a double check and for constructing the projector later, parton number operators n(mps[i]) ∈ [0,1,2]
# fermion number operator for the single spin space
n_op_fermion = zeros(1,2,1,2);
n_op_fermion[1, :, 1, :] = Diagonal([1,0]);

function calculate_n_j_alpha(MPS, l)
    C = reshape([1], (1,1,1))

    for i in eachindex(MPS)
        if i==l
            C= tn.updateLeft(C, MPS[i], n_op_fermion, MPS[i])
        else
            C = tn.updateLeft(C, MPS[i],reshape(I(2), (1,2,1,2)), MPS[i])
        end
    end

    # At the end, C is (1,1,1), so extract scalar value
    return real(C[1,1,1])
end

l_vals = 1:2N
nl_vals = [calculate_nl(fermionic_mps, l) for l in 1:2N]
Plots.plot(l_vals, nl_vals)

# the number operator in the full parton space has the "diagonal" [2,1,1,0]
n_op_parton = zeros(1,4,1,4)
n_op_parton[1,:, 1,:] = Diagonal([2,1,1,0])

function calculate_n_j(mps, j)
    one = reshape([1.0], (1,1,1))
    var = tn.updateLeft(one, mps[j], n_op_parton, mps[j])
    return tn.contract(var, [1,2,3], one, [1,2,3])[1]
end

calculate_n_j(parton_mps, 4)
fermionic_mps
@save "PartonWavefunctions/groundstates.jld2" parton_mps fermionic_mps

r_snake = collect(Iterators.flatten(((i, N - i + 1) for i in 1:div(N,2))))