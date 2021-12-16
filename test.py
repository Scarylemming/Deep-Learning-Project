import math

import numpy as np 

def diff(V,i,X) :
    sum = np.zeros((X @ V[:,0]).shape)
    for k in range(i) : 
        sum += (np.dot(X @ V[:,i], X @ V[:,k]) / np.dot(X @ V[:,k], X @ V[:,k])) * (X @ V[:,k])
        #print(sum.shape)
    
    return 2 * X.T @ (X @ V[:,i] -  sum)
    

def solve_EigenGame_R(X,V,i,tol,alpha) : 
    t = math.ceil(5 / 4 * min(np.linalg.norm(diff(V,i,X),2) / 2, tol)**(-2))
    
    print(t)
    
    for j in range(t) : 
        diff_v = diff(V,i,X)
        diff_v_R = diff_v - np.dot(diff_v, V[:,i]) * V[:,i]
        v_i = V[:,i] + alpha * diff_v_R
        V[:,i] = v_i / np.linalg.norm(v_i,2)
    return V
def create_matrix(n,d) : 
    return np.random.random(size = [n,d])


n = 10 #On a 10 vecteurs
d = 5 #On va être en dimension 5
k = 2 #Topk eigenvectors
alpha = 0.1 #Step size
tol = 0.5


X = create_matrix(n,d) 


M = X.T @ X

v0 = M[:,0]

V = create_matrix(d,k)

print(V)

a = diff(V,1,X)
print(a)

b = solve_EigenGame_R(X,V,1,tol,alpha)
