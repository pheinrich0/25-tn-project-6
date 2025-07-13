function check_occupation(MPS::Vector{Array{ComplexF64, 3}})
    L = length(MPS)
    n = 0
    for i in 1:L
        n_i = zeros(1, 2, 1, 2)  # left and right bond dimensions are 1
        n_i[1, :, 1, :] = (I(2) - [-1 0; 0 1]) ./2
        I_i = reshape(I(2), (1, 2, 1, 2))  # Identity for i≠l
        n_tensor = reshape([1], (1,1,1)) # n[l] reshaped to a tensor
        for l in 1:L
            n_tensor = tn.updateLeft(n_tensor, MPS[l], if i==l n_i else I_i end, MPS[l])
        end
        n = n + n_tensor[1,1,1]
        end
    return n
end

function check_localoccupation(MPS::Vector{Array{ComplexF64, 3}}, l::Int)
    L = length(MPS)
    n_i = zeros(1, 2, 1, 2)
    n_i[1, :, 1, :] = (I(2) - [-1 0; 0 1]) ./ 2
    I_i = reshape(I(2), (1, 2, 1, 2))
    n_tensor = reshape([1], (1,1,1))
    for i in 1:L
        n_tensor = tn.updateLeft(n_tensor, MPS[i], if i==l n_i else I_i end, MPS[i])
    end
    return n_tensor[1,1,1]
end

vaccuum =[ComplexF64.(T) for T in deepcopy(fermionic_mps)];
check_localoccupation(vaccuum, 4)


function check_occupation_float(MPS::Vector{Array{Float64, 3}})
    L = length(MPS)
    n = 0
    for i in 1:L
        n_i = zeros(1, 2, 1, 2)  # left and right bond dimensions are 1
        n_i[1, :, 1, :] = (I(2) - [-1 0; 0 1]) ./2
        I_i = reshape(I(2), (1, 2, 1, 2))  # Identity for i≠l
        n_tensor = reshape([1], (1,1,1)) # n[l] reshaped to a tensor
        for l in 1:L
            n_tensor = tn.updateLeft(n_tensor, MPS[l], if i==l n_i else I_i end, MPS[l])
        end
        n = n + n_tensor[1,1,1]
        end
    return n
end