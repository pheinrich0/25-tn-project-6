using LinearAlgebra, Plots, JLD2 
using Dates, Test
import tn_julia as tn
# settings 
N=32;
Dmax = 100;  # bond dimension cutoff

## a) Half-filled Fermi sea construction
# Import the fermionic ground state
@load "PartonWavefunctions/groundstates.jld2" parton_mps fermionic_mps

# import functions for d_m
include("dm_mpo.jl")
include("checkoccupation.jl")
momentaN32 = momenta(32);

# generate the matrix containing the coefficients A_m_l = A_m_{j, α}, to construct the mpo for d_{m,α} =  N^(-1/2) ∑_{j=1}^N e^(-i*m*j) c'_{j,α}
# Matrix of size N x 2N     elements given by (A)_{k, spin}, {l, spin} = N^(-1/2) exp(-i*momenta[k] * l) if either both row and column index are even or both odd, else zero
# d_m (alpha) has only nonzero contributions in the length 2N mps for indices, where the spin matches alpha (either up or down)
# the i'th row corresponds to the combined index {m, α} ∈ [1,..., N]

A = zeros(ComplexF64, N, 2N);
norm = 1/sqrt(N);
@assert length(momentaN32) == N/2 "length(momenta) must be N"

for k in 1:Int(N/2) # iterator over momenta "rows*1/2"
    for l in 1:N    #iterator over site indices "columns*1/2"
        A[2*k-1, 2*l-1] = norm * exp(-im * momentaN32[k] * l)  # odd row, odd column -> d_up operator, spin up site
        A[2*k,2*l] = norm * exp(-im * momentaN32[k] * l)   # eve row, even column -> d_down operator, spin down site

        # odd + even combinations remain zero
    end
end

A[1,3]
copymps = [ComplexF64.(T) for T in deepcopy(fermionic_mps)]
test = dm_alpha_mpo(1, 1/2, 32, A)
test3 = tn.apply_mpo(test, copymps, Dmax)
test3[1]
copymps[1] # copymps is modified when using this function

## loop to apply the dm mpos 
MPS_iter = Array{Array{ComplexF64, 3}, 2}(undef, 2 * length(momentaN32), 2N)
mps = [ComplexF64.(T) for T in deepcopy(fermionic_mps)]   # start with the Jordan-Wigner MPS, make sure its datatype is correct

for itN in 1:length(momentaN32)
    for alpha in [1/2, -1/2]  # alpha = 0 for spin up, 1 for spin down
        mpo_dk_k = dm_alpha_mpo(itN, alpha, N, A)
        mps = tn.applyMPO(mps, mpo_dk_k, Dmax)
        if alpha==1/2   # spin up -> odd row index
            MPS_iter[(2*itN-1), :] = mps
        else    # spin down -> even row index
            MPS_iter[2*itN, :] = mps
        end
        println("Iteration $alpha for k = $itN completed.", " Time: ", now())
    end
end

for i in eachindex(mps)
    print(size(mps[i]))
end

#create list of mps for single occupation mps, ie. 32x64 matrix of mps tensor
dk_singleOccStates = Array{Array{ComplexF64, 3}, 2}(undef, 2 * length(momentaN32), 2N)
vac = [ComplexF64.(T) for T in deepcopy(fermionic_mps)]  
# check whether the |0> state is normalized
println("contracting <0|0> gives: ", tn.mps_product(vac, vac))

for itN in 1:length(momentaN32)
    for alpha in [1/2, -1/2]  # alpha = 0 for spin up, 1 for spin down
        mpo_dk_k = dm_alpha_mpo(itN, alpha, N, A)
        varmps = tn.applyMPO(vac, mpo_dk_k, Dmax)
        println("for j=$itN α=$alpha contract <0|dk dk'|0> gives: ", tn.mps_product(varmps, varmps))
        println("the occupation number is: ", check_occupation(varmps))
        if alpha==1/2   # spin up -> odd row index
            dk_singleOccStates[(2*itN-1), :] = varmps
        else    # spin down -> even row index
            dk_singleOccStates[2*itN, :] = varmps
        end
    end
end


## cross check with occupation number after each dm application  

n_op_fermion = zeros(1,2,1,2);
n_op_fermion[1, :, 1, :] = Diagonal([1,0]);

# plot occupation number vs steps
MPS_iter[1,:]
check_occupation(MPS_iter[end,:])

# Store occupation numbers for each MPS in MPS_iter
occ_vals = zeros(length(MPS_iter) ÷ size(MPS_iter, 2))
for i in 1:length(occ_vals)
    occ_val = check_occupation(MPS_iter[i, :])
    occ_vals[i] = real(occ_val)  # Discard small imaginary part due to numerical error
end

# Plot the occupation numbers
plotOccN_standard = plot(occ_vals, xlabel="State index", ylabel="Occupation number", title="Occupation Number per State", legend=false)

## function for entanglement_entropy
include("entanglementEntropy.jl")

@show length(MPS_iter[1,:])
entanglement_entropy(MPS_iter[14,:], 32) # entanglement, for mps split in the middle

ee_vals = zeros(length(MPS_iter) ÷ size(MPS_iter, 2))
for i in 1:length(ee_vals)
    ee_val = entanglement_entropy(MPS_iter[i, :], 32)
    ee_vals[i] = real(ee_val)  # Discard small imaginary part due to numerical error
end
# errors occur for some stupid reason for later -> fixed by using solution for bondcanonical

# Plot the ee
pEE_standard = plot(ee_vals, xlabel="Iteration", ylabel="Entanglement entropy", title="Entanglement entropy in the center", legend=false)


## b) Wannier Orbitals: ToDo: Diagonalize the matrix given by 
# function X_mpo(L) returns a mpo for the position operator for this system:
# L=64
include("positionOperatorMPO.jl")
x_op64 = position_mpo(64)

# generate the 32x32 matrix containing  Xtilde[m,n] = ⟨0| d_m_alpha X (d_n_alpha)† |0⟩
# the expectation value can be calculated using the function mpo expectation, the mps for d_m_alpha |0> are generated with applyMPO
Xtilde = zeros(ComplexF64, 32, 32);
vaccuum =[ComplexF64.(T) for T in deepcopy(fermionic_mps)];

# ⟨0| d_m_alpha X (d_n_alpha)† |0⟩ remark: chatgpt states it is more efficient to contract dm|0> first, then we have two mps and a mpo
# -> apply the newly added function mpo_transition(W, MPS2, MPS2)
for itN in 1:length(momentaN32)
    for itM in 1:length(momentaN32)
    # for each combination of itN and itM, there are up to four matrix elements to fill in, since all combinations of alpha are also possible
    
    phi_nUp = tn.applyMPO( vaccuum, dm_alpha_mpo(itN, 1/2, N, A), Dmax)
    phi_nDown = tn.applyMPO( vaccuum, dm_alpha_mpo(itN, -1/2, N, A), Dmax)
    phi_mUp = tn.applyMPO( vaccuum, dm_alpha_mpo(itM, 1/2, N, A), Dmax)
    phi_mDown = tn.applyMPO( vaccuum, dm_alpha_mpo(itM, -1/2, N, A), Dmax)
    
    Xtilde[2*itN-1, 2*itM-1] = tn.mpo_transition(x_op64, phi_nUp, phi_mUp) # up-up /odd-odd
    Xtilde[2*itN-1, 2*itM] = tn.mpo_transition(x_op64, phi_nUp, phi_mDown)  # up-down / odd-even
    Xtilde[2*itN, 2*itM-1] = tn.mpo_transition(x_op64, phi_nDown, phi_mUp)  # down-up / even-odd
    Xtilde[2*itN, 2*itM] = tn.mpo_transition(x_op64, phi_nDown, phi_mDown)  # down-down / even-even
    end
end

# double check wether construction is right, Conclusion: yes
@test Xtilde[3,5] ≈ tn.mpo_transition(x_op64, dk_singleOccStates[3,:], dk_singleOccStates[5,:])
@test Xtilde[22,31] ≈ tn.mpo_transition(x_op64, dk_singleOccStates[22,:], dk_singleOccStates[31,:])

# taking complex conjugate: (X̃_mn)^* =  X̃_nm
# since X is hermitian ⟨0|(d_m X d_n')'|0⟩ = ⟨0| d_n * X' * d_m' |0⟩ = ⟨0| d_n * X * d_m† |0⟩
# → ie. the matrix X_tilde is hermitian → we expect real eigenvalues
eigvals, B = eigen(Xtilde)#

eigvals[end]
pWanniereigvals = plot(real(eigvals));     # the eigenvalues are the even numbers 


B[:,1]'*Xtilde*B[:,1]≈eigvals[1]

# since the function for dm_alpha_mpo takes the matrix A as an input and then dm_alpha is determined by the m_alpha'th row of a
# we can simply pass the matrix B^T A since
 # ζ_r† = ∑_ℓ (Bᵀ * A)[r, ℓ] * c_ℓ†
#       = ∑_ℓ (∑_m B[m, r] * A[m, ℓ]) * c_ℓ† =  ∑_ℓ (B^T A)[r,l] * c_ℓ†

# calculate B^T * A and check dimensions
# calculate B^T * A and check dimensions
transpose(B)[1,:]==B[:,1]
BT_A = transpose(B) * A
@test size(BT_A) == size(A) 

## now do the same loop applying the dks with truncation as for A
# Initialize storage and starting MPS for BT_A
MPS_iter_wannier = Array{Array{ComplexF64, 3}, 2}(undef, 2 * length(momentaN32), 2N)
mps_wannier = [ComplexF64.(T) for T in deepcopy(fermionic_mps)]

# Apply dm_alpha_mpo with BT_A coefficients
for itN in 1:length(momentaN32)
    for alpha in (1/2, -1/2)
        mpo_dk_wannier = dm_alpha_mpo(itN, alpha, N, BT_A)
        mps_wannier = tn.applyMPO(mps_wannier, mpo_dk_wannier, Dmax)
        if alpha==1/2   # spin up -> odd row index
            MPS_iter_wannier[(2*itN-1), :] = mps_wannier
        else    # spin down -> even row index
            MPS_iter_wannier[2*itN, :] = mps_wannier
        end
        println("Iteration $alpha for k = $itN completed.", " Time: ", now())
    end
end

# Compute occupation numbers for each state
occ_vals_wannier = zeros(size(MPS_iter_wannier, 1))
for i in 1:length(occ_vals_wannier)
    occ_vals_wannier[i] = real(check_occupation(MPS_iter_wannier[i, :]))
    println("Occupation after application #$i ", occ_vals_wannier[i])
end

# Plot the occupation numbers for BT_A
plot(occ_vals_wannier,
     xlabel="State index",
     ylabel="Occupation number",
     title="Occupation Number per State; Wannier Orbitals",
     legend=false)

# Compute entanglement entropy for each state
ee_vals_BT = zeros(size(MPS_iter_wannier, 1))
for i in 1:length(ee_vals_BT)
    ee_vals_BT[i] = real(entanglement_entropy(MPS_iter_wannier[i, :], N))
    println("E.e. after application #$i ", ee_vals_BT[i])
end
# Plot the entanglement entropy for BT_A
pWannierEE =plot(ee_vals_BT,
     xlabel="Iteration",
     ylabel="Entanglement entropy",
     title="Entanglement entropy in the center; Wannier Orbitals",
     legend=false)


# Combined plot of entanglement entropy for Standard vs Wannier Orbitals
pCombined = plot(ee_vals, label="Standard", xlabel="Iteration", ylabel="Entanglement entropy", title="Entanglement entropy comparison")
plot!(pCombined, ee_vals_BT, label="Wannier Orbitals")

# check whether the wannier states are correctly normalized and are eigenstates of 
wannierStates = Array{Array{ComplexF64, 3}, 2}(undef, 2 * length(momentaN32), 2N)
vac = [ComplexF64.(T) for T in deepcopy(fermionic_mps)]  

for itN in 1:length(momentaN32)
    for alpha in [1/2, -1/2]  # alpha = 0 for spin up, 1 for spin down
        mpo_dk_k = dm_alpha_mpo(itN, alpha, N, BT_A)
        varmps = tn.applyMPO(vac, mpo_dk_k, Dmax)
        println("for j=$itN α=$alpha contract <0|dk dk'|0> gives: ", tn.mps_product(varmps, varmps))
        println("the occupation number is: ", check_occupation(varmps))
        println("The position operator exp. value is ", tn.mpo_expectation(x_op64, varmps))
        if alpha==1/2   # spin up -> odd row index
            wannierStates[(2*itN-1), :] = varmps
        else    # spin down -> even row index
            wannierStates[2*itN, :] = varmps
        end
    end
end

## Apply Wannier orbitals with left meets right strategy
# Apply MPOs in the alternating sequence
# Generate alternating index order: first, last, second, second-last, ...

# Build flattened list of (itN, alpha) pairs in j-up, j-down order
flat_pairs = [(itN, alpha) for itN in 1:length(momentaN32) for alpha in (1/2, -1/2)]
L = length(flat_pairs)
order = Int[]
for k in 1:ceil(Int, L/2)
    push!(order, k)
    if k != L - k + 1
        push!(order, L - k + 1)
    end
end
order

MPS_iter_LmeetsR = Array{Array{ComplexF64, 3}, 2}(undef, 2 * length(momentaN32), 2N)
mps_LmeetsR = [ComplexF64.(T) for T in deepcopy(fermionic_mps)]
n=1
for idx in 1:length(order)
    println(idx)
    println(flat_pairs[order[idx]])
end
Dmax
for idx in 1:19
    itN, alpha = flat_pairs[order[idx]]
    mpo_dk = dm_alpha_mpo(itN, alpha, N, BT_A)
    mps_LmeetsR = tn.applyMPO(mps_LmeetsR, mpo_dk, 100) # had to use DMax=101 to avoid lapackerror wtf wtf bro fuck programming......
    # Store to the correct row: up -> odd, down -> even
    MPS_iter_LmeetsR[idx, :] = mps_LmeetsR
    println("Iteration $n for s=$alpha, k = $itN completed.", " Time: ", now())
    n=n+1
end
# also plot occupation, and entanglement_entropy
occ_vals_LmR = zeros(size(MPS_iter_LmeetsR, 1))
for i in 1:length(occ_vals_LmR)
    occ_vals_LmR[i] = real(check_occupation(MPS_iter_LmeetsR[i, :]))
    println("Occupation after application #$i ", occ_vals_LmR[i])
end

# Plot the occupation numbers for BT_A
plot(occ_vals_LmR,
     xlabel="State index",
     ylabel="Occupation number",
     title="Occupation Number per State; Left meets Right",
     legend=false)

# Compute entanglement entropy for each state
ee_vals_LmR = zeros(size(MPS_iter_LmeetsR, 1))
for i in 1:length(ee_vals_LmR)
    ee_vals_LmR[i] = real(entanglement_entropy(MPS_iter_LmeetsR[i, :], N))
    println("E.e. after application #$i ", ee_vals_LmR[i])
end
# Plot the entanglement entropy for BT_A
pLmR =plot(ee_vals_LmR,
     xlabel="Iteration",
     ylabel="Entanglement entropy",
     title="Entanglement entropy in the center; Left meets right",
     legend=false)

# Combined plot of entanglement entropy for Standard, Wannier, and Left meets right
pCombined = plot(ee_vals, label="Standard", xlabel="Iteration", ylabel="Entanglement entropy", title="Entanglement entropy comparison")
plot!(pCombined, ee_vals_BT, label="Wannier Orbitals")
plot!(pCombined, ee_vals_LmR, label="Left meets right")


## wannier test: do we have eigenfcts of location operator
Testmps = [ComplexF64.(T) for T in deepcopy(fermionic_mps)]

testWannierMpo = dm_alpha_mpo(5, 1/2, N, BT_A)
check_occupation(mps_test)
mps_test = tn.applyMPO(Testmps, testWannierMpo, Dmax)

Testmps[4]=occupied
tn.mpo_expectation(x_op64, Testmps)

size(B)
@test B'*B ≈ I(size(B, 1))  # is a unitary matrix

size(A)
@test A*A' ≈ I(size(A,1)) # also unitary