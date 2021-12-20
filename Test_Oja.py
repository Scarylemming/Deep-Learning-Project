import numpy as np


class Oja(object):
    """ A method for doing dimensionality reduction by transforming the feature
    space to a lower dimensionality, removing correlation between features and
    maximizing the variance along each feature axis.
    
    The Oja learning rule is an enhancement of Hebbian learning rule, which 
    based on normalized weights. The artificial neuron learns to compute 
    a principal component of its input stream.

    http://www.scholarpedia.org/article/Oja_learning_rule
    
    Attributes:
        principal_components: list of principal components arrays
        eigenvectors: list of eigenvectors arrays
        feature_n: An integer number of a dataset features
        sample_n: An integer size of a dataset
    """
    
    def __init__(self):
        self.principal_components = []
        self.eigenvectors = []
        self.feature_n = 0
        self.sample_n = 0

    def fit(self, X):
        """ Fit the dataset to the number of principal components.
        
        Args:
            2d array, where a horizontal axis is equal to the number of 
            features, a vertical axis is equal to the number of dataset samples.
        
        Returns: 
            lists of arrays of the principal components and  the eigenvectors.
        """
        
        X = self.normalize_data(X)
        
        self.sample_n = X.shape[0]
        self.feature_n = X.shape[1]
        
        # Set the initial vector of weight coefficients in the range [-1, 1]
        w = np.random.uniform(-1, 1, (1, self.feature_n))
        w = w / np.linalg.norm(w)

        learning_rate = 1 / self.sample_n
        for k in range(1, self.feature_n + 1):
            PC = np.array([0.0 for _ in range(X.shape[0])], ndmin=2).T
            # 10^k is the number of iterations necessary to find the eigenvector
            for _ in range(10 ** k):
                for i, x in enumerate(X):
                    # Recalculation of the principal component element in accordance with the updated weights.
                    y = np.dot(w, x)
                    # Recalculation of the eigenvector by Oya's recurrent formula
                    w = w + learning_rate * y * (x - y * w)
                    w = w / np.linalg.norm(w)
                    PC[i] = y
                    
            # After each next eigenvector and principal component need to 
            # subtract the principal components and eigenvector of the sample.
            X = X - PC * w
            self.principal_components.append(PC)
            self.eigenvectors.append(w)

        return self.principal_components, self.eigenvectors
    def normalize_data(self, X) : 
        for i in range(len(X)) : 
            X[i] -= np.mean(X[i])
            X[i] /= np.linalg.norm(X[i],2)
        return X

    def decompress(self):
        """ A method for decompressing principal components. 
            
            Returns:
                2d array which represents approximate data after compression 
                and decompression. A horizontal axis is equal to the number of 
                features, a vertical axis is equal to the number of dataset samples.
        """
        
        X_apx = np.array([[0.0] * self.feature_n for _ in range(self.sample_n)])

        for w, PC in zip(self.eigenvectors, self.principal_components):
            for i in range(PC.shape[0]):
                X_apx[i] = X_apx[i] + w * PC[i]

        return X_apx

n = 20 
d = 5

obj = np.random.normal(size = [n,d])