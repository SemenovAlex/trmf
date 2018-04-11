"""
Temporal Regularized Matrix Factorization
"""

# Author: Alexander Semenov <alexander.s.semenov@yandex.ru>

import numpy as np



class trmf:
    """Temporal Regularized Matrix Factorization.

    Parameters
    ----------
    
    lags : array-like, shape (n_lags,)
        Set of lag indices to use in model.
    
    K : int
        Length of latent embedding dimension
    
    lambda_f : float
        Regularization parameter used for matrix F.
    
    lambda_x : float
        Regularization parameter used for matrix X.
    
    lambda_w : float
        Regularization parameter used for matrix W.

    alpha : float
        Regularization parameter used for make the sum of lag coefficient close to 1.
        That helps to avoid big deviations when forecasting.
    
    eta : float
        Regularization parameter used for X when undercovering autoregressive dependencies.

    Attributes
    ----------

    F : ndarray, shape (n_timeseries, K)
        Latent embedding of timeseries.

    X : ndarray, shape (K, n_timepoints)
        Latent embedding of timepoints.

    W : ndarray, shape (K, n_lags)
        Matrix of autoregressive coefficients.
    """

    def __init__(self, lags, K, lambda_f, lambda_x, lambda_w, alpha, eta):
        self.lags = lags
        self.L = len(lags)
        self.K = K
        self.lambda_f = lambda_f
        self.lambda_x = lambda_x
        self.lambda_w = lambda_w
        self.alpha = alpha
        self.eta = eta
        
        self.W = None
        self.F = None
        self.X = None


    def fit(self, data, max_iter=10, resume=False,
            F_step=0.0001, F_max_iter=1, X_step=0.0001, X_max_iter=1, W_step=0.0001, W_max_iter=1):
        """Fit the TRMF model according to the given training data.

        Model fits through sequential updating three matrices:
            -   matrix self.F;
            -   matrix self.X;
            -   matrix self.W.
            
        Each matrix updated with gradient descent.

        Parameters
        ----------
        data : ndarray, shape (n_timeseries, n_timepoints)
            Training data.

        F_step : float
            Step of gradient descent when updating matrix F.

        F_max_iter : int
            Number of gradient steps to be made to update F.

        X_step : float
            Step of gradient descent when updating matrix X.

        X_max_iter : int
            Number of gradient steps to be made to update X.

        W_step : float
            Step of gradient descent when updating matrix W.

        W_max_iter : int
            Number of gradient steps to be made to update W.

        n_epoch : int
            Number of iterations of updating matrices F, X and W.

        resume : bool
            Used to continue fitting.

        Returns
        -------
        self : object
            Returns self.
        """

        if not resume:
            self.Y = data
            self.N, self.T = data.shape
            self.W = np.random.randn(self.K, self.L) / self.L
            self.F = np.random.randn(self.N, self.K)
            self.X = np.random.randn(self.K, self.T)

        for _ in range(max_iter):
            self._update_F(step=F_step, n_iter=F_max_iter)
            self._update_X(step=X_step, n_iter=X_max_iter)
            self._update_W(step=W_step, n_iter=W_max_iter)


    def predict(self, T):
        """Predict each of timeseries T timepoints ahead.

        Model evaluates matrix X with the help of matrix W,
        then it evaluates prediction by multiplying it by F.

        Parameters
        ----------
        T : int
            Number of timepoints to forecast.

        Returns
        -------
        preds : ndarray, shape (n_timeseries, T)
            Predictions.
        """

        X_preds = self._predict_X(T)
        return np.dot(self.F, X_preds)


    def _predict_X(self, T):
        """Predict X T timepoints ahead.

        Evaluates matrix X with the help of matrix W.

        Parameters
        ----------
        T : int
            Number of timepoints to forecast.

        Returns
        -------
        X_preds : ndarray, shape (self.K, T)
            Predictions of timepoints latent embeddings.
        """

        X_preds = np.zeros((self.K, T))
        X_adjusted = np.hstack([self.X, X_preds])
        for t in range(self.T, self.T + T):
            for l in range(self.L):
                lag = self.lags[l]
                X_adjusted[:, t] += X_adjusted[:, t - lag] * self.W[:, l]
        return X_adjusted[:, self.T:]


    def _update_F(self, step, n_iter):
        """Gradient descent of matrix F.

        n_iter steps of gradient descent of matrix F.

        Parameters
        ----------
        step : float
            Step of gradient descent when updating matrix.

        n_iter : int
            Number of gradient steps to be made.

        Returns
        -------
        self : objects
            Returns self.
        """

        for _ in range(n_iter):
            self.F -= step * self._grad_F()


    def _update_X(self, step, n_iter):
        """Gradient descent of matrix X.

        n_iter steps of gradient descent of matrix X.

        Parameters
        ----------
        step : float
            Step of gradient descent when updating matrix.

        n_iter : int
            Number of gradient steps to be made.

        Returns
        -------
        self : objects
            Returns self.
        """

        for _ in range(n_iter):
            self.X -= step * self._grad_X()


    def _update_W(self, step, n_iter):
        """Gradient descent of matrix W.

        n_iter steps of gradient descent of matrix W.

        Parameters
        ----------
        step : float
            Step of gradient descent when updating matrix.

        n_iter : int
            Number of gradient steps to be made.

        Returns
        -------
        self : objects
            Returns self.
        """

        for _ in range(n_iter):
            self.W -= step * self._grad_W()


    def _grad_F(self):
        """Gradient of matrix F.

        Evaluating gradient of matrix F.

        Parameters
        ----------

        Returns
        -------
        self : objects
            Returns self.
        """

        return - 2 * np.dot(self.Y - np.dot(self.F, self.X), self.X.T) + 2 * self.lambda_f * self.F


    def _grad_X(self):
        """Gradient of matrix X.

        Evaluating gradient of matrix X.

        Parameters
        ----------

        Returns
        -------
        self : objects
            Returns self.
        """

        for l in range(self.L):
            lag = self.lags[l]
            W_l = self.W[:, l].repeat(self.T, axis=0).reshape(self.K, self.T)
            X_l = self.X * W_l
            z_1 = self.X - np.roll(X_l, lag, axis=1)
            z_1[:, :max(self.lags)] = 0.
            z_2 = - (np.roll(self.X, -lag, axis=1) - X_l) * W_l
            z_2[:, -lag:] = 0.

        grad_T_x = z_1 + z_2
        return - 2 * np.dot(self.F.T, self.Y - np.dot(self.F, self.X)) + self.lambda_x * grad_T_x + self.eta * self.X


    def _grad_W(self):
        """Gradient of matrix W.

        Evaluating gradient of matrix W.

        Parameters
        ----------

        Returns
        -------
        self : objects
            Returns self.
        """

        grad = np.zeros((self.K, self.L))
        for l in range(self.L):
            lag = self.lags[l]
            W_l = self.W[:, l].repeat(self.T, axis=0).reshape(self.K, self.T)
            X_l = self.X * W_l
            z_1 = self.X - np.roll(X_l, lag, axis=1)
            z_1[:, :max(self.lags)] = 0.
            z_2 = - (z_1 * np.roll(self.X, lag, axis=1)).sum(axis=1)
            grad[:, l] = z_2
        return grad + self.W * 2 * self.lambda_w / self.lambda_x -\
               self.alpha * 2 * (1 - self.W.sum(axis=1)).repeat(self.L).reshape(self.W.shape)

