# bond dimensions
Da = 10; # alpha()
Db = 12; # beta()
Dc = 14; # gamma()
Dd = 17; # delta
Dm = 20; # mu

A = rand(Dc,Dd);   # tensor A[gamma,delta]
B = rand(Da,Dm,Dc); # tensor B[alpha,mu,gamma]
C = rand(Db,Dm,Dd); # tensor C[beta,mu,delta]

B1 = permutedims(B,(1,3,2)); # B[alpha,mu,gamma] -> B[alpha,gamma,mu]
B1 = reshape(B1,(Da*Dc,Dm)); # B[alpha,gamma,mu] -> B[alpha*gamma,mu]
C1 = permutedims(C,(2,1,3)); # C[beta,mu,delta] -> C[mu,beta,delta]
C1 = reshape(C1,(Dm,Db*Dd)); # C[mu,beta,delta] -> C[mu;beta*delta]
BC = B1*C1;  # \sum_[mu] B[alpha*gamma,mu] * C[mu,beta*delta]
             # = BC[alpha*gamma,beta,delta]

# BC[alpha*gamma,beta*delta] -> BC[alpha,gamma,beta,delta]
BC = reshape(BC,(Da,Dc,Db,Dd));
# BC[alpha,gamma,beta,delta] -> BC[alpha,beta,gamma,delta]
BC = permutedims(BC,(1,3,2,4));
# BC[alpha,beta,gamma,delta] -> BC[alpha*beta;gamma*delta]
BC = reshape(BC,(Da*Db,Dc*Dd));
A1 = A[:];  # A[gamma,delta] -> A[gamma*delta]

# \sum_(gamma,delta) BC[alpha*beta,gamma*delta] * A[gamma*delta]
#        = ABC[alpha,beta]
ABC1 = BC*A1;
ABC1 = reshape(ABC1,(Da,Db));  # ABC[alpha*beta] -> ABC[alpha,beta]

using BenchmarkTools

function matrix_multiplication()
    B1 = permutedims(B,(1,3,2)); # B[alpha,mu,gamma] -> B[alpha,gamma,mu]
    B1 = reshape(B1,(Da*Dc,Dm));# B[alpha,gamma,mu] -> B[alpha*gamma,mu]
    C1 = permutedims(C,(2,1,3)); # C[beta,mu,delta] -> C[mu,beta,delta]
    C1 = reshape(C1,(Dm,Db*Dd));# C[mu,beta,delta] -> C[mu,beta*delta]
    # \sum_[mu] B[alpha*gamma,mu] * C[mu,beta*delta]
    #       = BC[alpha*gamma,beta*delta]
    BC = B1*C1;
    # BC[alpha*gamma,beta*delta] -> BC[alpha,gamma,beta,delta]
    BC = reshape(BC,(Da,Dc,Db,Dd));
    return BC
end

@benchmark MM = matrix_multiplication()

function for_loops()
    # create an 4D-array initialized with zeros()
    BC = zeros(Da,Dc,Db,Dd)
    for it1 = (1:size(BC,1)) # alpha()
        for it2 = (1:size(BC,2)) # gamma()
            for it3 = (1:size(BC,3)) # beta()
                for it4 = (1:size(BC,4)) # delta
                    for it5 = (1:size(B,2)) # mu
                        BC[it1,it2,it3,it4] = BC[it1,it2,it3,it4] + B[it1,it5,it2]*C[it3,it5,it4]
                    end
                end
            end
        end
    end
    return BC
end

@benchmark MM = for_loops()
