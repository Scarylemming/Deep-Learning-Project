import numpy as np

def Oja_algo(X, T, V, nu) : 
    for t in range(T) : 
        V = V + nu * X.T @ X @ V 
        Q,R = np.linalg.qr(V)
        S = np.sign(np.sign(np.sum(np.diag(R))) + 0.5)
        V = Q * S
    return V

def create_matrix(n,d) : 
    return np.random.normal(size = [n,d])

def calc_numpy_eig(X):
    p,q = np.linalg.eig(np.dot(np.transpose(X),X))
    return (p,q)
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

n = 10 
d = 4


X = create_matrix(n,d)

p,q = calc_numpy_eig(X)
q = order_np_eigvectors(p, q)

a = Oja_algo(X, 100, create_matrix(d,d), 0.2)

print("Eigenvectors with Oja's rule :\n", a)
print("Eigenvectors with numpy :\n", q)


print("\n Squared error in estimation of eigenvectors as compared to numpy :\n",(np.sum((np.abs(q)-np.abs(a))**2, axis=0)))

