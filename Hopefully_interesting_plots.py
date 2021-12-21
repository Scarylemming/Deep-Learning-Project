# Importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt
import time
import math
import pandas as pd

# Calculate the eigenvalues of covariance matrix of X using Numpy for comparison
def calc_numpy_eig(X):
    p,q = np.linalg.eig(np.dot(np.transpose(X),X))
    q = order_np_eigvectors(p, q)
    return (p,q)

#Gradient of the utility function of the players
def diff(V,i,X) :
    penalties = np.zeros((X @ V[:,0]).shape)
    for j in range(i) : 
        penalties += (np.dot(X @ V[:,i], X @ V[:,j]) / np.dot(X @ V[:,j], X @ V[:,j])) * (X @ V[:,j])
        #print(sum.shape)
    return 2 * X.T @ (X @ V[:,i] -  penalties)

#Function to normalize data to be centered around 0 on all axis and of variance 1. 
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
def calc_eigengame_eigenvectors(X,n,iterations=100, lr = 0.1):
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
                v = update(k,X,V1,lr)
            else:
                #v = update(v,X,V1,riemannian_projection=True)
                v = update(k,X,V1,lr)
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

#Reordering of the eigenvectors with their respective eigenvalues, works perfect with small dimensions, could be optimizes for higher ones
def order_np_eigvectors(p, q) : 
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

#Oja's algorithm as it is used in the Paper. They have an additional variable mask that is never used, so I didn't copy it here
def Oja_algo(X, T, V, nu) : 
    for t in range(T) : 
        V = V + nu * X.T @ X @ V 
        Q,R = np.linalg.qr(V)
        S = np.sign(np.sign(np.sum(np.diag(R))) + 0.5)
        V = Q * S
    return V

def create_plots(n,d,iterations, learning_rate,numbers) :
    plot_size = len(numbers)
    Oja_times = [0 for i in range(plot_size)]
    EigenGame_times = [0 for i in range(plot_size)]
    Oja_max_eig_error = [0 for i in range(plot_size)]
    Oja_sum_eig_error = [0 for i in range(plot_size)]
    EigenGame_max_eig_error = [0 for i in range(plot_size)]
    EigenGame_sum_eig_error = [0 for i in range(plot_size)]
    
    for i in range(plot_size) : 
        print(i)
        n = int(numbers[i])
        d = min(n,10)
        X = create_matrix(n,d)
        p,q = calc_numpy_eig(X)
        V = np.ones([d,d])
        t0 = time.time()
        Oja = Oja_algo(X,iterations,V,learning_rate)
        t1 = time.time()
        Oja_times[i] = t1-t0
        Oja_sum_eig_error[i] = np.sum((np.abs(q)-np.abs(Oja))**2)
        Oja_max_eig_error[i] = np.max(np.sum((np.abs(q)-np.abs(Oja))**2, axis=0))
        
        t0 = time.time()
        EigenGame = calc_eigengame_eigenvectors(X,d, iterations)
        t1 = time.time()
        EigenGame_times[i] = t1-t0
        EigenGame_sum_eig_error[i] = np.sum((np.abs(q)-np.abs(EigenGame))**2)
        EigenGame_max_eig_error[i] = np.max(np.sum((np.abs(q)-np.abs(EigenGame))**2, axis=0))
    
    plt.plot(list(range(plot_size)), EigenGame_max_eig_error)
    plt.plot(list(range(plot_size)), Oja_max_eig_error)
    plt.show()
    
    pass
def get_angle(u,v) : 
    return math.acos(np.dot(u,v))
def angle_threshold(u,v,threshold) : 
    if (abs(get_angle(u,v) - math.pi / 2) > threshold) and (abs(get_angle(u,v) + math.pi / 2) > threshold) : 
        return False
    else : 
        return True
def longest_streak(V,threshold) : 
    streak = 0
    for i in range(len(V)) : 
        good = True
        for j in range(i) : 
            if not angle_threshold(V[:,i],V[:,j],threshold) : 
                good = False
                break
        if not good :
            break
        else : 
            streak += 1
    return streak
def Oja_plot(X, T, V, nu = 0.1,threshold = math.pi / 8) : 
    streaks = []
    p,q = calc_numpy_eig(X)
    for t in range(T) : 
        V = V + nu * X.T @ X @ V 
        Q,R = np.linalg.qr(V)
        S = np.sign(np.sign(np.sum(np.diag(R))) + 0.5)
        V = Q * S
        streaks.append(np.max(np.sum((np.abs(q)-np.abs(V))**2, axis=0)))
    return V, streaks
def EigenGame_plot(X,d,iterations=100, lr = 0.1, threshold = math.pi / 8):
    streaks = []
    d = len(X[0])
    p,q = calc_numpy_eig(X)
    v = np.ones(d, dtype = float)
    v = v/np.linalg.norm(v,2)
    v0 = np.ones(d, dtype = float)
    v0 = v0/np.linalg.norm(v0,2)
    #print(X)
    V1 = np.zeros([d,d])
    V1[:,0] = v

    for k in range(d):
        #print ("Finding the eigenvector ",k)
        for i in range(iterations):
            if k==0:
                v = update(k,X,V1,lr)
            else:
                #v = update(v,X,V1,riemannian_projection=True)
                v = update(k,X,V1,lr)
        V1[:,k] = v.T
        v = v0
        if k<d-1:
            V1[:,k+1] = v0.T
            #print("q", q.shape,"V", V1.shape)
        streaks.append(np.max(np.sum((np.abs(q)-np.abs(V1))**2, axis=0)))
    return V1, streaks
def EigenGame_plot_all(X,d,iterations = 100, lr = 0.1, threshold = math.pi / 8) : 
    res = []
    for i in range(iterations) : 
        print(i)
        V, streaks = EigenGame_plot(X,d,i,lr, threshold)
        res.append(streaks[-1])
    
    
    return V, res
def plot_all(n,d,lr,iterations,threshold) : 
    V = np.ones([d,d])
    V_Oja, streaks_Oja = Oja_plot(X,iterations,V,lr,threshold)
    print("End Oja's algorithm")
    V_EigenGame, streaks_EigenGame = EigenGame_plot_all(X,d,iterations,lr,threshold)
    # plt.plot(list(range(len(streaks_Oja)-1)),streaks_Oja[1:],label = "Oja")
    # plt.plot(list(range(len(streaks_EigenGame)-1)),streaks_EigenGame[1:], label = "EigenGame")
    # plt.title("Biggest eigenvector error based on number of iterations performed")
    # plt.xlabel("Number of iterations")
    # plt.ylabel("Biggest eigenvector error")
    # plt.legend(loc='upper right')
    # plt.yscale("log")
    # plt.show()
    df = pd.read_csv("Oja.csv")
    df[len(df.columns) + 1] = streaks_Oja
    df.to_csv("Oja.csv", index = False)
    df = pd.read_csv("EigenGame.csv")
    df[len(df.columns) + 1]= streaks_EigenGame
    df.to_csv("EigenGame.csv", index = False)
    
    return V_Oja, V_EigenGame
n = 100
d = 50
iterations = 10


# Oja = Oja_algo(X, iterations, create_matrix(d,d), 0.1)
# p,q = calc_numpy_eig(X)
# EigenGame = calc_eigengame_eigenvectors(X,d, iterations)
#print("\n Eigenvalues calculated using numpy are :\n",p)
#print("\n Eigenvectors calculated using numpy are :\n",q)
#print("\n Eigenvalues calculate using the Eigengame are :\n",calc_eigengame_eigenvalues(X,V1))
#print("\n Eigenvectors calculated using the Eigengame are :\n",V1)
# print("\n EigenGame : Squared error in estimation of eigenvectors as compared to numpy :\n",(np.sum((np.abs(q)-np.abs(EigenGame))**2, axis=0)))
# print("\n EigenGame : Biggest squared error in estimation of eigenvectors as compared to numpy :\n",np.max(np.sum((np.abs(q)-np.abs(EigenGame))**2, axis=0)))
# print("\n Oja's Algorithm : Squared error in estimation of eigenvectors as compared to numpy :\n",(np.sum((np.abs(q)-np.abs(Oja))**2, axis=0)))
# print("\n Oja's Algorithm : Biggest squared error in estimation of eigenvectors as compared to numpy :\n",np.max(np.sum((np.abs(q)-np.abs(Oja))**2, axis=0)))

# numbers = 2**np.linspace(0,10)
#create_plots(n,d,iterations, 0.1, numbers)

#print(angle_threshold(Oja[:,0], Oja[:,1], math.pi / 8))
for name in ["Oja.csv", "EigenGame.csv"] : 
    df = {1 : [0 for i in range(iterations)]}
    df = pd.DataFrame(df)
    df.to_csv(name, index = False)

for i in range(4) :
    X = create_matrix(n,d)
    a,b = plot_all(n,d,0.1,iterations,math.pi / 32)

for name in ["Oja.csv", "EigenGame.csv"] : 
    df = pd.read_csv(name)
    df = df.drop(["1"], axis = 1)
    print(df)
    df.to_csv(name, index = False)

Oja_df = pd.read_csv("Oja.csv")
streaks_Oja = Oja_df.mean(axis = 1)
EigenGame_df = pd.read_csv("EigenGame.csv")
streaks_EigenGame = EigenGame_df.mean(axis = 1)

plt.plot(list(range(len(streaks_Oja)-1)),streaks_Oja[1:],label = "Oja")
plt.plot(list(range(len(streaks_EigenGame)-1)),streaks_EigenGame[1:], label = "EigenGame")
plt.title("Biggest eigenvector error based on number of iterations performed")
plt.xlabel("Number of iterations")
plt.ylabel("Biggest eigenvector error")
plt.legend(loc='upper right')
plt.yscale("log")
plt.show()