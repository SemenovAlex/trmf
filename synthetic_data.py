import numpy as np


class SyntheticData:
    """Time series synthesizer.

    Creates random weghts of autoregressive dependency.
    Creates random latent embeding X of timepoints.
    Creates random latent embeding F of timeseries.
    Creates timeseries by multiplying F and X.
    
    Parameters
    ----------

    N : int
        Number of timeseries.
        
    T : int
        Number of timepoints.
        
    K : int
        Size of latent dimension.
    
    lags : array-like, shape (n_lags,)
        Set of lag indices to be used for autoregressive behavior.
    
    sigma_w : float
        Standard deviation of noise in matrix W.
    
    sigma_x : float
        Standard deviation of noise in matrix X.
        
    sigma_t : float
        Standard deviation of noise in timeseries.
    
    Attributes
    ----------

    """

    def __init__(self, N, T, K, lags, sigma_w, sigma_x, sigma_t):

        self.N = N
        self.T = T
        self.K = K
        self.lags = lags
        self.L = len(lags)
        self.sigma_w = sigma_w
        self.sigma_x = sigma_x
        self.sigma_t = sigma_t


    def get_data(self):
        return self.Y


    def synthesize_data(self):
        self.generate_W()
        self.generate_F()
        self.generate_X()
        self.Y = np.dot(self.F, self.X) + np.random.randn(self.N, self.T) * self.sigma_t


    def generate_W(self):
        W = np.random.randn(self.K, self.L) * self.sigma_w + 1
        W /= (W.sum(axis=1)).repeat(W.shape[1]).reshape(W.shape)
        self.W = W


    def generate_F(self):
        self.F = np.random.randn(self.N, self.K)


    def generate_X(self):
        self.X = np.random.randn(self.K, self.T) * self.sigma_x
        for t in range(max(self.lags), self.T):
            for l in range(self.L):
                lag = self.lags[l]
                self.X[:, t] += self.X[:, t - lag] * self.W[:, l]
