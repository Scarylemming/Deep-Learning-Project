import math
#from keras.datasets import mnist
import numpy as np 
import matplotlib.pyplot as plt

def diff(V,i,X) :
    penalties = np.zeros((X @ V[:,0]).shape)
    for k in range(i) : 
        penalties += (np.dot(X @ V[:,i], X @ V[:,k]) / np.dot(X @ V[:,k], X @ V[:,k])) * (X @ V[:,k])
        #print(sum.shape)
    
    return 2 * X.T @ (X @ V[:,i] -  penalties)
    

def solve_EigenGame_R(X,V,i,tol,alpha) : 
    t = math.ceil(5 / 4 * min(np.linalg.norm(diff(V,i,X),2) / 2, tol)**(-2))
    print(t)
    t = 10
    for j in range(t) : 
        diff_v = diff(V,i,X)
        diff_v_R = diff_v - np.dot(diff_v, V[:,i]) * V[:,i]
        print(np.linalg.norm(diff_v_R,2))
        v_i = V[:,i] + alpha * diff_v_R
        V[:,i] = v_i / np.linalg.norm(v_i,2)
    return V[:,i]

def normalize_data(X) : 
    for i in range(len(X)) : 
        X[i] -= np.mean(X[i])
        X[i] /= np.linalg.norm(X[i],2)
    return X

def create_matrix(n,d) : 
    return np.random.normal(size = [n,d])


n = 1000 #On a 10 vecteurs
d = 10 #On va être en dimension 5
k = 5 #Topk eigenvectors
alpha = 0.1 #Step size
tol = 0.5


def play_EigenGame(n,d,k) : 
    tol = 0.1
    alpha = 0.5
    iter = 5
    #n is the number of points
    #d is the dimension of each point
    #k is the number of Principal Components to get
    data = create_matrix(n,d) #We create some data
    plt.scatter(data[:,0], data[:,1])
    plt.show()
    X = normalize_data(data) #We normalize the data so it's in the right form to apply EigenGame on
    
    M = X.T @ X #M is the covariance Matrix
    #Now, the important part is to calculate the Eigenvalues and the Eigenvectors of the Covariance Matrix
    #To calculte these Eigenvectors, we use the EigenGame !
    V = normalize_data(create_matrix(d,k)) #We create a random normalized matrix with k starting Eigenvectors.
    for j in range(iter) : 
        for i in range(k) : 
            V[:,i] = solve_EigenGame_R(X, V, i, tol, alpha)
    return V,M

def find_Lambda(V,M) : 
    return (V**(-1)).T @ M @ V

V,M = play_EigenGame(n,d,k)
a = find_Lambda(V,M)
