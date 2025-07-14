# if you combine two neighboring spin sites into a 4dim leg, the gullwitzer projector looks like
P2 = zeros(ComplexF64, 4, 4);
P2[2, 2] = 1.0;  # |01>
P2[3, 3] = 1.0;  # |10>
# and the mpo would look like 
N=32;
Proj_parton = Vector{Array{ComplexF64,4}}(undef, N)
for i in eachindex(Proj_parton)
    Proj_parton[i] = reshape(P2, (1,4,1,4))
end

## this part doesnt produce the right result, skip to end
function gullwitzer_mpo()
    W1 = zeros(ComplexF64, 1, 2, 2, 2)
    W2 = zeros(ComplexF64, 2, 2, 2, 1)

    # |01> → virtual index 1
    W1[1, 1, 2, 1] = 1.0 
    W2[1, 1, 2, 1] = 1.0 

    # |10> → virtual index 2
    W1[1, 2, 1, 2] = 1.0 
    W2[2, 2, 1, 1] = 1.0 

    return W1, W2
end

Proj1, Proj2 = gullwitzer_mpo()
Id_site = tn.identity(Proj1, 2, Proj2, 2)
PG = tn.contract(Proj1, 3, Proj2, 1)
PG2 = tn.contract(PG, [2,4], Id_site, [2,1])
ProjG= tn.contract(PG2, [2,3], Id_site,[1,2], [1,3,2,4])

ProjG[1,:,1,:]


# this doesnt match the the two leg projector, it is thus propably better to instead manipulate the mps to have combined legs

# different approach the Gullwitzer proj can be written as n_1 + n_2 - 2*n_1*n_2
# this can be factorized to a mpo rep as (1, n1, n1) (n2, 1 , -2*n2)
nOp = diagm([1,0])
P_up = zeros(1,2,3,2)
P_down = zeros(3,2,1,2)

P_up[1,:,1,:] = I(2);
P_up[1,:,2,:] = nOp;
P_up[1,:,3,:] = nOp;
P_down[1,:,1,:] = nOp;
P_down[2,:,1,:] = I(2);
P_down[3,:,1,:] = -2*nOp;

reshape(tn.contract(P_up, 3, P_down, 1)[1,:,:,:,1,:], (4,4))
# also doesnt really look right but ok
testmps = Vector{Array{ComplexF64, 3}}(undef, 2)
unoccupied = zeros(1, 2, 1);
unoccupied[1, 2, 1] = 1.0; 
occ = zeros(1, 2, 1);
occ[1, 1, 1] = 1.0; 
testmps = [occ, unoccupied]
#check_occupation(testmps)

res=tn.applyMPO_float(testmps, [P_up, P_down], 100)

# check_occupation(res)
# the double occupation gets removed, i guess thats a good sign

# create a L=64 mpo, that has for each odd-even pair P_up and P_down
Gullwitzer_mpo = Vector{Array{ComplexF64,4}}(undef, 2N)
for i in 1:32
    Gullwitzer_mpo[2*i-1] = P_up
    Gullwitzer_mpo[2*i] = P_down
end

@save "gullwitzerProj.jld2" Proj_parton Gullwitzer_mpo



# other option, always contract two spin subsites, then apply the c^4 operator
testes = tn.contract(testmps[1], 3, testmps[2], 1)
mps4 = Vector{Array{ComplexF64, 3}}(undef, 1)
mps4[1] = tn.contract(testes, [2,3], Id_site, [1,2], (1,3,2))
ss_proj_parton = Vector{Array{ComplexF64, 4}}(undef, 1)
ss_proj_parton[1] =Proj_parton[1]
@test tn.applyMPO(mps4, ss_proj_parton, 10)[1]==mps4[1]

# this works as expected, so maybe it is better to first create combined legs and then apply the gullwitzer proj

# Function to create N site parton mps from 2N site fermionic mps
# HOW TO USE: add include() statement

function combine_spins_mps(mps::Vector{Array{ComplexF64,3}})
    if isodd(length(mps))
        error("Length of input mps must be even, got $(length(mps))")
    end
    L = Int(length(mps) ÷ 2)
    new_mps = Vector{Array{ComplexF64,3}}(undef, L)
    Id_site = tn.identity(mps[1], 2, mps[2], 2)
    for i in 1:L
        # contract the right leg of spin up with the left leg of spin down
        contractBond = tn.contract(mps[2i-1], 3, mps[2i], 1) 
        new_mps[i] = tn.contract(contractBond, [2,3], Id_site, [1,2], (1,3,2))
    end
    return new_mps
end

length(Proj_parton)
Dmax

#applies the projector to the 2N chain in C^4
function applyPG_parton(mps::Vector{Array{ComplexF64,3}})
    if length(mps)!= 2*length(Proj_parton)
        error("Length of input mps doesnt match projector")
    end
    return_mps = Vector{Array{ComplexF64,3}}(undef, length(mps))
    projected4leg = Vector{Array{ComplexF64,4}}(undef, length(mps))
    partonmps = combine_spins_mps(mps)  # Dx4xD tensors
    projected = tn.applyMPO(partonmps, Proj_parton, Dmax)
    for i in eachindex(projected)
        projected4leg[i] = tn.contract(projected[i], 2, Id_site, 3, [1,3,4,2]) # Dx2x2xD tensors
        # do SVD to obtainn two Dx2xD tensors
        U, S, Vd, _ = tn.svd(projected4leg[i], [1, 2], Nkeep=Dmax)
        return_mps[2*i-1] = U
        return_mps[2*i] = tn.contract(Diagonal(S), 2, Vd, 1) # DxD tensors
    end

    return return_mps
end

# applies the projector using the n1 + n2 - 2*n1*n2 mpo representation
function applyPG_fermion(mps::Vector{Array{ComplexF64,3}})
    if length(mps)!= length(Gullwitzer_mpo)
        error("Length of input mps doesnt match projector")
    end

    projected = tn.applyMPO(mps, Gullwitzer_mpo, Dmax)
    return projected
end

## two ways to apply gullwitzer to 
@load "PartonWavefunctions/fermiseas.jld2" dk_fermisea wannier_fermisea lmr_fermisea

testcombine = combine_spins_mps(lmr_fermisea)
applyPG_fermion(lmr_fermisea)
applyPG_parton(lmr_fermisea)

