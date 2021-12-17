# Importing necessary libraries
import numpy as np

# Calculate the eigenvalues of covariance matrix of X using Numpy for comparison
def calc_numpy_eig(X):
    p,q = np.linalg.eig(np.dot(np.transpose(X),X))
    return (p,q)

# Define utlity function, we will take grad of this in the
# update step, v is the current eigenvector being calculated
# X is the design matrix and V1 holds the previously computed eigenvectors

# Update rule to be used for calculating eigenvectors
# For first eigenvector use riemannian_projection = False (update rule given in the paper doesn't work without the penalty term)
# For all others, use riemannian_projection = True to be aligned with the paper
# But using riemannian_projection = False also works and in the tests that I did it converges much faster than including the
# Riemannian Projection

def diff(V,i,X) :
    penalties = np.zeros((X @ V[:,0]).shape)
    for j in range(i) : 
        penalties += (np.dot(X @ V[:,i], X @ V[:,j]) / np.dot(X @ V[:,j], X @ V[:,j])) * (X @ V[:,j])
        #print(sum.shape)
    return 2 * X.T @ (X @ V[:,i] -  penalties)

def update(i,X,V,lr=0.1):
    
    diff_v = diff(V,i,X)
    #diff_v_R = diff_v - np.dot(diff_v, V[:,i]) * V[:,i]
    #print(np.linalg.norm(diff_v_R,2))
    v_i = V[:,i] + lr * diff_v
    V[:,i] = v_i / np.linalg.norm(v_i,2)
    return V[:,i]

# Run the update step iteratively across all eigenvectors
def calc_eigengame_eigenvectors(X,n,iterations=100):
    v = np.array([[1.0],[1.0],[1.0],[1.0]])
    v = v/np.linalg.norm(v)
    v0 = np.array([[1.0],[1.0],[1.0],[1.0]])
    v0 = v0/np.linalg.norm(v0)
    V1 = np.zeros_like(X)
    V1[:,0] = v.T

    for k in range(n):
        print ("Finding the eigenvector ",k)
        for i in range(iterations):
            if k==0:
                v = update(k,X,V1)
            else:
                #v = update(v,X,V1,riemannian_projection=True)
                v = update(k,X,V1)
        V1[:,k] = v
        v = v0
        if k<n-1:
            V1[:,k+1] = v0.T
    return V1

# Calculate eigenvalues once the eigenvectors have been computed
def calc_eigengame_eigenvalues(X,V1):
    M = np.dot(np.transpose(X),X)
    n = np.size(V1,axis=1)
    eigvals = np.zeros((1,n))
    for k in range(n):
        eigvals[:,k] = np.dot(V1[:,k],np.dot(M,V1[:,k].reshape(-1,1)))
    return eigvals

# Matrix X for which we want to find the PCA
X = np.array([[7.,4.,5.,2.],
            [2.,19.,6.,13.],
            [34.,23.,67.,23.],
            [1.,7.,8.,4.]])

# X = np.array([[9.,0.,0.,0.],
#             [0.,8.,0.,0.],
#             [0.,0.,7.,0.],
#             [0.,0.,0.,1.]])

# Centre the data
# X = X-np.mean(X,axis=0)
# print(X)

p,q = calc_numpy_eig(X)
V1 = calc_eigengame_eigenvectors(X,4, iterations = 100)
print("\n Eigenvalues calculated using numpy are :\n",p)
print("\n Eigenvectors calculated using numpy are :\n",q)
print("\n Eigenvalues calculate using the Eigengame are :\n",calc_eigengame_eigenvalues(X,V1))
print("\n Eigenvectors calculated using the Eigengame are :\n",V1)
print("\n Squared error in estimation of eigenvectors as compared to numpy :\n",np.sum((np.abs(q)-np.abs(V1))**2,axis=0))
