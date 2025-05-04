import numpy as np

class LinearRegression():
    
    def __init__(self, lr = 0.001, n_iter = 1000):
        # NOTE: Feel free to add any hyperparameters 
        # (with defaults) as you see fit
        self.lr = lr
        self.n_iter = n_iter
        self.weights = None
        self.bias = None


    def fit(self, X, y):
        """
        Estimates parameters for the classifier
        
        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
            y (array<m>): a vector of floats
        """

        arr = X.to_numpy()
        X = arr.reshape(-1, 1)

        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iter):
            y_pred = np.dot(X, self.weights)+ self.bias

            dw = (1/n_samples) * np.dot(X.T, (y_pred-y))
            db = (1/n_samples) * np.sum(y_pred - y)

            self.weights -= self.lr*dw
            self.bias -= self.lr*db

    
    def predict(self, X):
        """
        Generates predictions
        
        Note: should be called after .fit()
        
        Args:
            X (array<m,n>): a matrix of floats with 
                m rows (#samples) and n columns (#features)
            
        Returns:
            A length m array of floats
        """
        arr = X.to_numpy()
        X = arr.reshape(-1, 1)
        
        y_pred = np.dot(X, self.weights)+ self.bias
        return y_pred


        raise NotImplementedError("The predict method is not implemented yet.")
