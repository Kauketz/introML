import numpy as np

class LogisticRegression():
    
    def __init__(self, lr = 0.1, n_iter = 1000):
        # NOTE: Feel free to add any hyperparameters 
        # (with defaults) as you see fit
        self.lr = lr
        self.n_iter = n_iter
        self.weights = None
        self.bias = None
        self.losses, self.train_accuracies = [], []
        pass
            
 
    def _compute_loss(self, y_true, y_pred):
        y_zero_loss = y_true * np.log(y_pred + 1e-9)
        y_one_loss = (1-y_true) * np.log(1 - y_pred + 1e-9)
        return -np.mean(y_zero_loss + y_one_loss)
    
    # def compute_gradients(self, X, y, y_pred):

    #     n = X.shape[0]
    #     error = y_pred - y
        
    #     dw = (1/n) * np.dot(X.T, error)
        
    #     db = (1/n) * np.sum(error)
        
    #     return dw, db

    def compute_gradients(self, x, y_true, y_pred):
        difference =  y_pred - y_true
        gradient_b = np.mean(difference)
        gradients_w = np.matmul(x.transpose(), difference)
        gradients_w = np.array([np.mean(grad) for grad in gradients_w])

        return gradients_w, gradient_b
    
    def accuracy(self, y_true, y_pred):
        return np.mean(y_true == y_pred)

    def _sigmoid(self, x):
        return np.array([self._sigmoid_function(value) for value in x])

    def _sigmoid_function(self, x):
        if x >= 0:
            z = np.exp(-x)
            return 1 / (1 + z)
        else:
            z = np.exp(x)
            return z / (1 + z)
    
    def update_parameters(self, dw, db):
        # self.weights -= self.lr * dw
        # self.bias -= self.lr * db
        self.weights = self.weights - 0.1 * dw
        self.bias = self.bias - 0.1 * db

    def fit(self, X, y):
        """
        Estimates parameters for the classifier
        
        Args:
            X (array<m,n>): a matrix of floats with
                m rows (#samples) and n columns (#features)
            y (array<m>): a vector of floats
        """


        self.weights = np.zeros(X.shape[1])
        self.bias = 0

        for _ in range(self.n_iter):
            lin_model = np.matmul(self.weights, X.transpose()) + self.bias
            y_pred = self._sigmoid(lin_model)
            loss = self._compute_loss(y, y_pred)

            grad_w, grad_b = self.compute_gradients(X, y, y_pred)
            self.update_parameters(grad_w, grad_b)

            pred_to_class = [1 if _y > 0.5 else 0 for _y in y_pred]
            self.train_accuracies.append(self.accuracy(y, pred_to_class))
            self.losses.append(loss)
        

        # raise NotImplementedError("The fit method is not implemented yet.")

    
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

        x_dot_weights = np.matmul(X, self.weights.transpose()) + self.bias
        probabilities = self._sigmoid(x_dot_weights)
        return [1 if p > 0.5 else 0 for p in probabilities]

        raise NotImplementedError("The predict method is not implemented yet.")
