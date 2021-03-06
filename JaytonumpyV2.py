# Importing necessary libraries
import numpy as np

# Calculate the eigenvalues of covariance matrix of X using Numpy for comparison
def calc_numpy_eig(X):
    p,q = np.linalg.eig(np.dot(np.transpose(X),X))
    return (p,q)

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

def normalize_data(X) : 
    for i in range(len(X)) : 
        X[i] -= np.mean(X[i])
        X[i] /= np.linalg.norm(X[i],2)
    return X

def create_matrix(n,d) : 
    return np.random.normal(size = [n,d])

def update(i,X,V,lr=0.1):
    
    diff_v = diff(V,i,X)
    #diff_v_R = diff_v - np.dot(diff_v, V[:,i]) * V[:,i]
    #print(np.linalg.norm(diff_v_R,2))
    v_i = V[:,i] + lr * diff_v
    V[:,i] = v_i / np.linalg.norm(v_i,2)
    return V[:,i]

# Run the update step iteratively across all eigenvectors
def calc_eigengame_eigenvectors(X,n,iterations=100):
    d = len(X[0])
    v = np.ones(d, dtype = float)
    v = v/np.linalg.norm(v,2)
    v0 = np.ones(d, dtype = float)
    v0 = v0/np.linalg.norm(v0,2)
    #print(X)
    V1 = np.zeros([d,n])
    V1[:,0] = v

    for k in range(n):
        print ("Finding the eigenvector ",k)
        for i in range(iterations):
            if k==0:
                v = update(k,X,V1)
            else:
                #v = update(v,X,V1,riemannian_projection=True)
                v = update(k,X,V1)
        V1[:,k] = v.T
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

def order_np_eigvectors(p, q) : #Just a simple reordering, works perfect with small dimensions, could be optimizes for higher ones
    d = len(p)
    indexes = np.zeros(d, dtype = int)
    sort_p = np.sort(p)[::-1]
    #print(p, sort_p)
    for i in range(d) : 
        #print(p, sort_p[i])
        indexes[i] = np.where(p == sort_p[i])[0]
    res = np.zeros_like(q)
    for i in range(d) : 
        res[:,i] = q[:,indexes[i]]
    return res

def Oja_algo(X, T, V, nu) : 
    for t in range(T) : 
        V = V + nu * X.T @ X @ V 
        Q,R = np.linalg.qr(V)
        S = np.sign(np.sign(np.sum(np.diag(R))) + 0.5)
        V = Q * S
    return V
# Matrix X for which we want to find the PCA
# X = np.array([[7.,4.,5.,2.],
#             [2.,19.,6.,13.],
#             [34.,23.,67.,23.],
#             [1.,7.,8.,4.]])

n = 1000
d = 20
iterations = 200
X = create_matrix(n,d)

Oja = Oja_algo(X, iterations, create_matrix(d,d), 0.1)
p,q = calc_numpy_eig(X)
q = order_np_eigvectors(p, q)
EigenGame = calc_eigengame_eigenvectors(X,d, iterations)
#print("\n Eigenvalues calculated using numpy are :\n",p)
#print("\n Eigenvectors calculated using numpy are :\n",q)
#print("\n Eigenvalues calculate using the Eigengame are :\n",calc_eigengame_eigenvalues(X,V1))
#print("\n Eigenvectors calculated using the Eigengame are :\n",V1)
print("\n EigenGame : Squared error in estimation of eigenvectors as compared to numpy :\n",(np.sum((np.abs(q)-np.abs(EigenGame))**2, axis=0)))
print("\n EigenGame : Biggest squared error in estimation of eigenvectors as compared to numpy :\n",np.max(np.sum((np.abs(q)-np.abs(EigenGame))**2, axis=0)))
print("\n Oja's Algorithm : Squared error in estimation of eigenvectors as compared to numpy :\n",(np.sum((np.abs(q)-np.abs(Oja))**2, axis=0)))
print("\n Oja's Algorithm : Biggest squared error in estimation of eigenvectors as compared to numpy :\n",np.max(np.sum((np.abs(q)-np.abs(Oja))**2, axis=0)))
