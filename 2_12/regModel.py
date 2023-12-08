import numpy as np

class HWRegression():

    def __init__(self, learning_rate: float = 1e-2, max_iter: int = 100, early_stop_eps: float = 1e-2):

        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.iter = 0
        self.early_stop_eps = early_stop_eps
        self.loss = []
        self.cur_loss = np.inf

    def fit(self, data: np.ndarray, target: np.ndarray):
        """fit with MSE loss

        Args:
            data (np.ndarray): data samples (phi)
            target (np.ndarray): xy coordinates

        Returns:
            CircleRegression: fitted regressor
        """

        self.data = data
        self.target = target

        self.m, self.n = data.shape

        self.R = 0
        self.C = np.zeros((self.n,))

        self.xy = np.empty_like(target)
        self.xy[:, 0] = np.cos(data).flatten()
        self.xy[:, 1] = np.sin(data).flatten()

        for _ in range(self.max_iter):
            self.update_weights()
            if abs(self.cur_loss - self.loss[-1]) < self.early_stop_eps:
                break
            self.cur_loss = self.loss[-1]
        self.cur_loss = self.loss[-1]

        return self

    def update_weights(self):

        Y_pred = self.predict(self.data)

        dW = -(2 * np.sum((self.xy.T).dot(self.target - Y_pred)) / self.m)
        db = -2 * np.sum(self.target - Y_pred, axis=0) / self.m

        self.R = self.R - self.learning_rate * dW
        self.C = self.C - self.learning_rate * db

        self.loss.append(np.sum((self.target - Y_pred)**2))
        self.iter += 1

        return self

    def predict(self, X):
        xy = np.empty_like(self.target)
        xy[:, 0] = np.sin(X.flatten())
        xy[:, 1] = np.cos(X.flatten())

        return self.R*self.xy + self.C