## New approach:

# ## a) conctract two neighboring Spin sites to get Dmax ⨯ 4 × Dmax tensors
# ## b) delete sitetensor[:,1,:] → double occupation and sitetensor[:,4,:] → empty
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

function projectToSpinMPS(mps::Vector{Array{ComplexF64,3}})
    @assert length(mps) == 2N "Input must be fermionic MPS"
    contractmps = combine_spins_mps(mps)
    spinMPS  = Vector{Array{ComplexF64,3}}(undef, N)
    for (i, A) in pairs(contractmps)

        Dl, _, Dr = size(A)

        # allocate the projected tensor (Dl × 2 × Dr)
        B = Array{ComplexF64}(undef, Dl, 2, Dr)

        B[:,1,:] = A[:,2,:]   # keep |↑⟩
        B[:,2,:] = A[:,3,:]   # keep |↓⟩
        
        spinMPS[i] = B
    end
    # add normalization
    norm = sqrt(real(tn.mps_product(spinMPS, spinMPS)))
    spinMPS[1]*=1/norm
    return spinMPS, norm
end

@load "PartonWavefunctions/NormalizedMPS.jld2" MPS_iter_Norm MPS_wannier_Norm normMPS_lmr
testfermisea = normMPS_lmr[end, :];
testres, normres = projectToSpinMPS(testfermisea);
normres
sqrt(real(tn.mps_product(testres, testres)))

showcaseOccupationN = zeros(N)
for i in eachindex(showcaseOccupationN)
    showcaseOccupationN[i] = real(check_localoccupation(testres, i))  # Discard small imaginary part due to numerical error
end

plot(1:N, showcaseOccupationN, xlabel="Site ℓ", ylabel="n_ℓ", title="Local occupation per site (Fermi sea)", legend=false, framestyle=:box)
sum(showcaseOccupationN)

spin