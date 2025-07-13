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

# check whether the coefficients of the single occ states are correct
dk_singleOccStates[1,:][3]
A[1,3]

mpo_dk_k = dm_alpha_mpo(5, 1/2, N, A)[10][2,1,1,2]==A[9,10]
mpo_dk_k = dm_alpha_mpo(5, -1/2, N, A)[10][2,1,1,2]==A[10,10]

## cross check with occupation number after each dm application  

n_op_fermion = zeros(1,2,1,2);
n_op_fermion[1, :, 1, :] = Diagonal([1,0]);


# Store occupation numbers for each MPS in MPS_iter
occ_vals = zeros(length(MPS_iter) ÷ size(MPS_iter, 2))
for i in 1:length(occ_vals)
    occ_val = check_occupation(MPS_iter[i, :])
    occ_vals[i] = real(occ_val)  # Discard small imaginary part due to numerical error
end

# Plot the occupation numbers
plotOccN_standard = plot(occ_vals, xlabel="State index", ylabel="Occupation number", title="Occupation Number per State", legend=false)

# for the fermi sea, also plot the local occupation at each chain site
# Plot the local occupation at each chain site for the Fermi sea MPS
local_occ = zeros(2N)
for i in 1:length(local_occ)
    local_occ[i] = real(check_localoccupation(MPS_iter[end,:], i))  # Discard small imaginary part due to numerical error
end

pLocalOccFermisea = plot(1:2N, local_occ, xlabel="Site", ylabel="Local occupation", title="Local occupation per site (Fermi sea)", legend=false)
sum(local_occ)

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


# Compute all Xtilde matrix elements by looping over single-occupation MPS states
XtildeCompare = zeros(ComplexF64, 32, 32)
for i in 1:size(dk_singleOccStates, 1)
    for j in 1:size(dk_singleOccStates, 1)
        XtildeCompare[i, j] = tn.mpo_transition(x_op64, dk_singleOccStates[i, :], dk_singleOccStates[j, :])
    end
end

XtildeCompare ≈ Xtilde

# taking complex conjugate: (X̃_mn)^* =  X̃_nm
# since X is hermitian ⟨0|(d_m X d_n')'|0⟩ = ⟨0| d_n * X' * d_m' |0⟩ = ⟨0| d_n * X * d_m† |0⟩
# → ie. the matrix X_tilde is hermitian → we expect real eigenvalues
eigvals, B = eigen(Xtilde)#

eigvals[end]
pWanniereigvals = scatter(1:length(eigvals), real(eigvals);
    marker=:circle, markersize=6, color=:blue,
    xlabel="Index", ylabel="Eigenvalue",
    title="Eigenvalues of Position Operator",
    label="Eigenvalues")

@testset
for j in 1:size(B, 2)
    print(@test B[:,j]' * Xtilde * B[:,j] ≈ eigvals[j])
end

eigvals
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

# local occupations for the fermi sea constructed via Wannier orbitals
# for the fermi sea, also plot the local occupation at each chain site
# Plot the local occupation at each chain site for the Fermi sea MPS
local_occWannier = zeros(2N)
for i in 1:length(local_occWannier)
    local_occWannier[i] = real(check_localoccupation(MPS_iter_wannier[end,:], i))  # Discard small imaginary part due to numerical error
end


# Create two separate plots
p2 = plot(
    1:2N, local_occWannier,
    linewidth=2,
    xlabel="Site",
    ylabel="Local occupation",
    title="Local occupation (Fermi sea via Wannier)",
    label="Wannier orbitals",
    legend=:topright
)
p1 = plot(
    1:2N, local_occ;
    linewidth=2,
    xlabel="Site",
    ylabel="Local occupation",
    title="Local occupation (Fermi sea via d_k orbitals)",
    label="d_k orbitals",
    legend=:topright
)

# Combine the two plots vertically
plot(p1, p2, layout = @layout([a; b]), size=(700, 600))

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
xExpVal  = [];

for itN in 1:length(momentaN32)
    for alpha in [1/2, -1/2]  # alpha = 0 for spin up, 1 for spin down
        mpo_dk_k = dm_alpha_mpo(itN, alpha, N, BT_A)
        varmps = tn.applyMPO(vac, mpo_dk_k, Dmax)
        println("for j=$itN α=$alpha contract <0|dk dk'|0> gives: ", tn.mps_product(varmps, varmps))
        println("the occupation number is: ", check_occupation(varmps))
        expVal = tn.mpo_expectation(x_op64, varmps)
        println("The position operator exp. value is ", expVal)
        push!(xExpVal, expVal)
        if alpha==1/2   # spin up -> odd row index
            wannierStates[(2*itN-1), :] = varmps
        else    # spin down -> even row index
            wannierStates[2*itN, :] = varmps
        end
    end
end
xExpVal

pxExpVal = scatter(1:length(xExpVal), real(xExpVal);
    marker=:circle, markersize=6, color=:blue,
    xlabel="# wannier mps", ylabel="<X>",
    title="Exp. value of Position Operator",
    label="expVal")

# ## ## Conclusion: The expectation values do resemble the eigenvalues, but the ordering seems to be off.

# ## ## Show how the Wannier states look like in terms of local occupation
local_occWannier3 = zeros(2N)
for i in 1:length(local_occWannier)
    local_occWannier3[i] = real(check_localoccupation(wannierStates[15,:], i))  # Discard small imaginary part due to numerical error
end

plot(
    1:2N, local_occWannier3;
    xlabel="Site ℓ",
    ylabel="n_ℓ",
    title="Local occ. n_ℓ; Wannier state ζ'(k=15)|0⟩",
    legend=false,
    framestyle=:box,
    linewidth=2,
    linecolor=:blue,
    grid=false,
    size=(600,350),
    tickfont=font(12),
    guidefont=font(14),
    titlefont=font(14)
)
sum(local_occWannier3)

# other approach to try to get the correct expvalues from wannier construction
function wannier(r::Int)
    # r is the index of the Wannier-orbital
    # B is the matrix of eigenvectors of Xtilde
    # The Wannier function is given by the r-th column of B
    W_r = Vector{Array{ComplexF64,4}}(undef, 2N)
    F, Z, Id = tn.spinlessfermionlocalspace()
    # loop combined indices l = j,alpha
    for itL in 1:2N
        # choose correct tensor shape
        if itL == 1
            dims = (1, 2, 2, 2)
            W_L = zeros(ComplexF64, dims...)
            W_L[1, :, 1, :] = F* transpose(B[:,r])* A[:, itL] # sum over m, α
            W_L[1, :, 2, :] = Z
        elseif itL == 2N
            dims = (2, 2, 1, 2)
            W_L = zeros(ComplexF64, dims...)
            W_L[1, :, 1, :] = Id
            W_L[2, :, 1, :] = F*transpose(B[:,r])* A[:, itL]
        else
            dims = (2, 2, 2, 2)
            W_L = zeros(ComplexF64, dims...)
            W_L[1, :, 1, :] = Id
            W_L[2, :, 1, :] = F*transpose(B[:,r])* A[:, itL]
            W_L[2, :, 2, :] = Z          
        end
        W_r[itL] = W_L
    end
    return W_r
end

transpose(B)[1,:]==B[:,1]
transpose(B[:,1]) # row vector
transpose(B[:,1])* A[:, 2]
BT_A[1,2] # should be equal to the first element of the first row of BT_A
# Test whether simply passing BT_A works

# Test: dm_alpha_mpo ≈ wannier for k=1:maxk

@testset "dm_alpha_mpo ≈ wannier for k=1:32" begin
    for k in 1:32
        itN, alpha = getSiteSpinIndex(k)
        mpo_k = dm_alpha_mpo(itN, alpha, N, BT_A)
        w_k   = wannier(k)
        @test mpo_k ≈ w_k
        end
end

savedstates = Array{Array{ComplexF64, 3}, 2}(undef, 2 * length(momentaN32), 2N)
expVals2 = []
for itN in 1:2*length(momentaN32)

        mpo_dk_k = wannier(itN)
        varmps = tn.applyMPO(vac, mpo_dk_k, Dmax)
        println("for j=$itN contract <0|dk dk'|0> gives: ", tn.mps_product(varmps, varmps))
        println("the occupation number is: ", check_occupation(varmps))
        expVal = tn.mpo_expectation(x_op64, varmps)
        println("The position operator exp. value is ", expVal)
        push!(expVals2, expVal)
        savedstates[itN, :] = varmps
end
expVals2
plot(expVals2)

savedstates[1,:][1]

wannierStates[1,:][1]
# these do agree
# the reason why the eigenvalues cant be properly reproduced doesnt make sense
f
# ###############################################################################
# #                                                                             #
# #    Ex. c).: Apply Wannier Orbitals with Left-Meets-Right Strategy           #
# #                                                                             #
# ###############################################################################

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
for idx in 1:32
    itN, alpha = flat_pairs[order[idx]]
    mpo_dk = dm_alpha_mpo(itN, alpha, N, BT_A)
    # at step 21 the svd function from the package linear algebra fails, with a higher bond dimension for this one step, this can be fixed
    if idx == 21
        mps_LmeetsR = tn.applyMPO(mps_LmeetsR, mpo_dk, 101) # use DMax=101 to avoid lapackerror
    else
        mps_LmeetsR = tn.applyMPO(mps_LmeetsR, mpo_dk, 100) # had to use DMax=101 to avoid lapackerror wtf wtf bro fuck programming......
    end        # Store to the correct row: up -> odd, down -> even
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

# Plot the occupation numbers for left meets right
pOccNLeftMeetsRight = plot(occ_vals_LmR,
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

# ## ## RESULT: SAVE DIFFERENT FERMI savedstates
dk_fermisea = MPS_iter[end,:];
wannier_fermisea = MPS_iter_wannier[end, :];
lmr_fermisea = MPS_iter_LmeetsR[end,:];
@save "PartonWavefunctions/fermiseas.jld2" dk_fermisea wannier_fermisea lmr_fermisea

# Combined plot of entanglement entropy for Standard, Wannier, and Left meets right
pCombined = plot(
    ee_vals,
    label="Standard",
    xlabel="Iteration",
    ylabel="Entanglement entropy",
    title="Entanglement entropy comparison",
    framestyle=:box,
    linewidth=2.5
)
plot!(pCombined, ee_vals_BT, linewidth =2.5,label="Wannier Orbitals")
plot!(pCombined, ee_vals_LmR, linewidth =2.5, label="Left meets right")

# Combined plot of occupation numbers for Standard, Wannier, and Left meets Right
pOccCombined = plot(
    occ_vals,
    framestyle=:box,
    linewidth=1.5,
    label="Standard",
    xlabel="Iteration",
    ylabel="Occupation number",
    title="Occupation Number Comparison",
    aspect_ratio=:equal
)
plot!(pOccCombined, occ_vals_wannier, linewidth=1.5, label="Wannier Orbitals")
plot!(pOccCombined, occ_vals_LmR, linewidth=1.5, label="Left meets right")
display(pOccCombined)