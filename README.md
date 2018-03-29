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

## 3. Examples

## 4. Experiments


