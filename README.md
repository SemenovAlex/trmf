# Temporal Regularized Matrix Factorization

Project was inspired by the paper:

Yu, H. F., Rao, N., & Dhillon, I. S. (2016). Temporal regularized matrix factorization for high-dimensional time series prediction. In Advances in neural information processing systems (pp. 847-855).

Which can be found there: http://www.cs.utexas.edu/~rofuyu/papers/tr-mf-nips.pdf


## 1. Problem description

We have N timeseries of length T which are presented by matrix Y. We want to factorize it <a href="https://www.codecogs.com/eqnedit.php?latex=$Y&space;=&space;F\times&space;X$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$Y&space;=&space;F\times&space;X$" title="$Y = F\times X$" /></a>.

To solve this problem we will minimize:

<a href="https://www.codecogs.com/eqnedit.php?latex=$$\min\limits_{F,X}\sum\limits_{(i,t)\in\Omega}\left(Y_{it}-f_i^Tx_t\right)^2&plus;\lambda_fR_f(F)&plus;\lambda_xR_x(X).$$" target="_blank"><img src="https://latex.codecogs.com/gif.latex?$$\min\limits_{F,X}\sum\limits_{(i,t)\in\Omega}\left(Y_{it}-f_i^Tx_t\right)^2&plus;\lambda_fR_f(F)&plus;\lambda_xR_x(X).$$" title="$$\min\limits_{F,X}\sum\limits_{(i,t)\in\Omega}\left(Y_{it}-f_i^Tx_t\right)^2+\lambda_fR_f(F)+\lambda_xR_x(X).$$" /></a>

By doing that we will find latent embedding vectors for timeseries and latent temporal embeddings for timepoints.
One can further use this embeddings to forecast new data or to impute missings.

## 2. Package description
Package consists of:
- trmf : time series modelling
- synthetic_data : data generation for experiments

additionaly:
- Metrics : metrics for experiments and validation
- Forecast : simple models for testing experiments
- RollingCV : rolling cross-validation implementation

## 3. Experiments

In experiments_[something].ipynb you can find some experiments on the package:

1) experiments_synthetic.ipynb: testing trmf model against other simple model on synthetic data


2) experiments_electricity.ipynb: testing trmf model against other simple model on electricity data


3) experiments_crypto.ipynb: testing trmf model against other simple model on crypto-currency data


4) experiments_missings.ipynb: testing trmf model against other simple model on missing data imputation

| mp=5% | mp=10% | mp=25% |
|------|------|------|
| Naive | 0.367/0.574 | 0.373/0.584 | 0.391/0.613 |
| Mean | 129.506/150.367 | 108.291/125.712 | 89.242/103.586 |
| TRMF | **0.359/0.516** | **0.36/0.519** | **0.361/0.52** |


## 4. Conclusion

1) TRMF model needs additional regularization on matrix W (sum of the row elements must be close to one). Otherwise, predictions for long periods will be unstable;

2) Every timeseries is better to be normalized before using TRMF;

3) TRMF is good on data imputation;

4) TRMF is good when data has missings.

## 5. Plan

1) Article analysis // done
2) Synthetic Data Generator // done
3) Basic realization of trmf with gradient descent // done
4) Documentation and help functions // done
5) Experiments on synthetic data (vs other models) // done
6) Rolling CV functionality // done
7) Experiments on electricity data (vs other models) // done
8) CryptoCurrency forecasting (vs other models) // done
9) Missing data handling // done
10) Missing data imputation experiments (vs other models) // done
