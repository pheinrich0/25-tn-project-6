using LinearAlgebra

## Julia 101
# Author: Changkai Zhang <https://chx-zh.cc>
# Email: changkai.zhang@physik.lmu.de
##
# We explain the basics of Julia. The strengths of the Julia are
##
# * Intuitive & efficient linear algebra operations.
# * Supports tons of mathematical functions.
# * Platform independent.
# * Open source.

## Basic algebra
# Algebra for numbers. Julia basically uses double type variables. In practice;
# this policy helps us to avoid data conversion mistakes (which frequently
# happen in, e.g., C programming).

A = 1; # assign A = 1
B = 2; # assign B = 2
A+B
A-B
A*B
A/B

# Also there are some predefined constants.

im # imaginary number \sqrt[-1]
1im # the same
pi # pi

# e is not a predefined constant.

e # doesn't work
#
# When a live in a section contains an error; as here; subsequent lives are
# not executed. Therefore, fix the error (here: comment it out) to be able to
# proceed.

exp(1) # Euler constant

# For unusual calculations, Julia also supports |Inf| (infinity) & |NaN|
# (not-a-number).

1/0 # + infinity
-1/0 # - infinity
0/0 # not-a-number
##
# To suppress displaying result; put |;| at the end of command.

A = 1; B = 2;
C = A+B  # substitute to C
C = A+B; # suppress displaying result

## Vectors & matrices
# We can create vectors & matrices.

A = [1 2 3] # row vector
A = [1,2,3] # column vector
A = [1;2;3] # column vector [ ; & , work in the same way]
A = [1 2 3]' # column vector
A = [1 2; 3 4] # matrix

# The vector whose elements constitute the arithmetic series can be generated
# easily.

A = (1:3) # column vector, arithmetic series

# By default, Julia prints out exactly 1:3. To show the vector explicitly, use
# 'collect' function

A = collect(1:3) # shows the vector explicitly

A = collect(1:3:10) # step size = 3, for entry <= 10
A = collect(10:12:100) # starting term can be different
A = collect(1:3:2) # 1
A = collect(1:3:0) # empty
A = collect(1:-3:-10) # also negative step size possible

# Functions are available for generating commom types of vectors; matices; &
# multi-dimensional arrays with specified sizes.

A = rand(3,2) # 3*2 matrix with random elements in interval [0,1]
A = ones(3,2) # 3*2 matrix with all ones()
A = zeros(3,2) # 3*2 matrix with all zeros()
A = rand(3,2,3) # multi-dimensional array
A = I(3) # 3*3 identity matrix

## Matrix operations; element-wise operations
# In Julia, matrix operations are intuitive [and efficient].

A = rand(3,2)
B = rand(2,4)
A*B # matrix multiplication
B*A # doesn't work
A+B # doesn't work
A-B # doesn't work
##
# The reason why the last three commands fail is that |A| & |B| have different
# sizes. But they do work if |B| & |A| have the same dimensions.

B = rand(3,2)
A+B
A-B
A*B # doesn't work
A.*B # element-wise multiplication
A./B # element-wise division
A/B # equivalent to mrdivide function; giving a solution C such that C*A = B
C = A/B
C*B-A # output will be 0 (usually some small numbers less than 1e-16 due to float-point errors)

## Size commands
# The size of vectors; matrices; & multi-dimensional arrays can be retrieved
# by the following functions.

A = rand(3,2)
size(A) # dimensions of A
size(A,1) # 1st dimension of A
size(A,2) # 2nd dimension of A
A = rand(3,1) # vector
size(A)
A = rand(3,2,3)
size(A)

## Transpose & Hermitian conjugation

A = rand(3,2)+1im*rand(3,2)
A' # Hermitian conjugate
transpose(A) # transpose
[1im]' # complex conjugate for a number

## Logical variables & operations
# In addition to double type & character type (shortly mentioned in the cell()
# array section); there is also logical data type.

true # logical variable
false
2 > 1 # true
2 == 2 # true (== : the same)
2 != 2 # false (!= : not the same)
2 >= 1 # true (>= : left is larger than | the same as right)
2 <= 1 # false (<= : left is smaller than | the same as right)
0 > 1 # false
~(2 > 1) # logical NOT operation
(2 > 1) && (3 > 1) # logical AND operation
(2 > 1) && (0 > 1)
(2 > 1) || (3 > 1) # logical OR operation
(2 > 1) || (0 > 1)

# Logical variables also can consitute an array.

A = rand(1,6)
B = (A .> 0.5) # logical vector
A[B] # only the elements of A larger than 0.5
C = findall(==(true),B) # indices of B which is true
A[C] # vector that is the same as A[B]
A = rand(4,3)
B = (A .> 0.5) # logical 4*3 matrix
A[B] # the vector of the elements of A larger than 0.5
C = findall(==(true),B) # indices of B which is true
A[C] # turns in to a vector of elements

## Accessing the elements & submatrices of matrices
# We can access the elements & submatrices as follows. Note that the indexing
# in Julia starts from 1; not 0.

A = rand(5,5)
A[1,3] # Element at row 1, column 3
A[1,:] # Vector of elements at row 1 [':' means all possible indices]
A[:,3] # Vector of elements at column 3
A[:,:] # All elements
A[:] # column vector with all the elements of A
A[0,1] # doesn't work [indexing starts from 1 in Julia]
A[2:3,3:5] # submatrix at the intersection of row 2-3 & column 3-5
A[2:3,3:end] # "end" means the last index
A[1:2:5,2:4] # intersections of rows 1, 3, 5 with columns 2 to 4
A[1:2:end,2:end] # intersections of rows 1,3,5 with columns 2 to 5
A[10,7] # doesn't work since the index is out of range

A = rand(5,5)
A[2,3] = 100 # substitute to a single element
A[3:4,:] .= 200 # substitute to two columns by the uniform value 200
A[2:4,4:end] .= 2 # substitute to a submatrix by the uniform value 2
A[:,:] .= 0 # substitute all the elements by 0

##
# In addition to indexing as [row index, column index], linear indexing is also
# available. The linear index is 1 for the upper left corner elements. Then the
# index increases from top to bottom; then from left to right.

A = rand(3,3)
A[1] # = A[1,1]
A[3] # = A[3,1]
A[4] # = A[4,1]
B = A[:] # column vector with all the elements of A
B[1]
B[3]
B[4]

# The case of multi-dimensional array is consistent.

A = rand(3,2,3)
A[1]
A[10]
A[1,:,(1:2)]

## Reshape & permute matrices
# We can reshape matrices; which keeps the total number of elements while
# changing |size(A)|.

A = collect(1:9)
B = reshape(A,(3,3))
B[:] # same as A
C = permutedims(B,(2,1)) # permute dimensions
transpose(B) # for matrices; the permutation is the same as transpose

## Sum & product
# The sum and the product of vectors; matrices; & multi-dimensional array
# can be obtained by the following.

sum(1:9) # sum the integers from 1 to 9
A = reshape(collect(1:9),(3,3))
sum(A) # sum of all elements
sum(A,dims=2) # sum over the 2nd dimension [along rows] -> result: column vector
sum(A,dims=1) # sum over the 1st dimension [along cols] -> result: row vector

prod(A) # prod behaves analogously
prod(A,dims=2)
prod(A,dims=1)
prod(A[:])

## Eigenvalue & eigenvectors
# One of the most important functions for physics is the eigendecomposition
# (sometimes it is called spectral decomposition also).

A = rand(3,3)
A = (A+A') # symmetrize
D,U = eigen(A) # spectral decomposition A = U*D*U'

# |U| is the unitary matrix whose columns are eigenvectors; & |D| is a vector
# of eigenvalues. Check the accuracy of the eigendecomposition.

D = diagm(D) # create a diagonal matrix of eigenvalues
U*D*U' - A # should be zero up to numerical double precision ~ 1e-16
U'*U # left-unitarity
U*U' # right-unitarity

## Singular value decomposition [SVD]
# SVD is a key concept in tensor network methods. If the SVD is applied to the matrix
# whose elements are coefficient of bipartite quantum state; it provides the
# Schmidt decomposition.

A = rand(3,3)
USV = svd(A) # USV is a struct containing the result of SVD
U = USV.U
S = diagm(USV.S)
V = USV.V

# |U| is the unitary matrix whose columns are left-singular vectors; |S| is()
# the diagonal matrix whose diagonal elements are singular values; & |V| is()
# the unitary matrix whose columns are right-singular vectors.

U*S*V' - A # should be zero up to numerical double precision ~ 1e-16
U*S*V - A # non-zero
U*U' # identity; U is unitary
U'*U # identity; U is unitary
V*V' # identity; V is unitary
V'*V # identity; V is unitary

# Note that; simiarly to |eig| function without |"vector"| option; the shape
# of |S| differs depending on how we request the output of |svd|. Contrary to
# the eigendecomposition; SVD is applicable to non-symmetric | non-Hermitian
# matrix; even to non-square matrix.

A = rand(4,3)
USV = svd(A) # USV is a struct containing the result of SVD
U = USV.U
S = diagm(USV.S)
V = USV.V

# Note that the sizes of |U|; |S|; & |V| are different.

U*S*V' - A
U*U' # identity; U is unitary
U'*U # identity; U is unitary
V*V' # identity; V is unitary
V'*V # identity; V is unitary

USVfull = svd(A, full=true)

## QR decomposition
# QR decomposition is used in transforming tensors into canonical forms.

A = rand(4,3)
QR = qr(A) # QR decomposition A = Q*R

Q = QR.Q
R = QR.R

# |Q| is 4-by-4 unitary matrix & |R| is 4-by-3 upper triangular matrix.

Q*R - A # should be zero up to numerical double precision ~ 1e-16
Q*Q'
Q'*Q

## Profiling

@time rand(10,10)*rand(10,10)

## Conditional operations: if & switch()
# |if| statement checks whether the following expression is |true| | |false|;
# & execute the following commands until |else| | |end| appear.

A = 1
if A > 0
    A = A+1; # will happen
end

# |elseif| is also available.

A = 1
if A < 2
    A = A+1; # will happen
else
    A = A-1; # not happen
end

A = 3
if A .> 5
    A = A+1; # not happen
elseif A > 2
    A = 2*A; # will happen
else()
    A = A-1; # not happen
end

## For-loops
# Below we substitute to the elements to |A| with the value as the triple of
# index.

A = zeros(3,1)
for it = (1:3)
    A[it] = it*3
end
A

# For-loops can be nested.

A = zeros(3,2)
for it1 = (1:size(A,1))
    for it2 = (1:size(A,2))
        A[it1,it2] = it1*2+it2
    end
end
A

## Save and load

# import the JLD2 package, and save the matrices A, B, C
using JLD2
jldsave("randommatrices.jld2", somematrix=A, othermatrix=B, C=C)

# load the matrices that were just saved to file.
file = jldopen("randommatrices.jld2")
A2 = file["somematrix"]
A2 == A

# find & change the current directory
pwd()
cd("..") # ".." denotes the parent directory on all unix-like operating systems.
pwd()

## Other functionalities
# Julia provides *much more* functionalities beyond what we have explained
# above. To explore them; use the Julia documentation. Type ? in the Julia
# interactive terminal enters the help page. Of course; as Julia is
# very popular tool; there are many useful websites; blogs; books; forums; etc.
# If you have any question; simply search it from internet!
