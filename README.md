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

## 3. Experiments

In workbook.ipynb you can find some experiments on the package:

1) Testing convergence of model on synthetic data
2) Testing stability of model to overfitting on synthetic data
3) ... (in progress)

## 4. Plan

1) Article analysis // done
2) Synthetic Data Generator // done
3) Basic realization of trmf with gradient descent // done
4) Documentation and help functions // todo
5) Experiments on synthetic data // done
6) Rolling CV functionality // todo
7) Experiments on electricity data (vs autoregressive model) // todo
8) CryptoCurrency forecasting (vs autoregressive model) // todo
9) Missing data handling // todo
10) Missing data imputation experiments // todo
