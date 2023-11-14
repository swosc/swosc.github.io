
// The input data is a vector 'y' of length 'N'.
data {
  int<lower=0> N;
  int<lower=0> K;
  vector[N] y;
  matrix[N, K] X;
  
  // hyperparameters
  real beta_0;
  real<lower=0> lambda_0;
  real<lower=0> nu_0;
  real<lower=0> s_02;
}

transformed data {
    matrix[N, K+1] X_mat = append_col(rep_vector(1, N), X);
    vector[K+1] beta_0_vec = rep_vector(beta_0, K+1);
    matrix[K+1, K+1] Lambda_0 = lambda_0 * identity_matrix(K+1);
}

// The parameters accepted by the model. Our model
// accepts two parameters 'beta' and 'sigma'.
parameters {
  vector[K+1] beta;
  real<lower=0> sigma2;
}

// The model to be estimated. We model the output
// 'y' to be normally distributed with mean 'X_mat * beta'
// and standard deviation 'sigma'.
model {
  real sigma = sqrt(sigma2);
  
  beta ~ multi_normal(beta_0_vec, sigma2 * Lambda_0);
  sigma2 ~ scaled_inv_chi_square(nu_0, s_02);
  
  y ~ normal(X_mat * beta, sigma);
}

generated quantities {
    real sigma = sqrt(sigma2);
    vector[N] y_ppd;
    
    for (i in 1:N) {
        y_ppd[i] = normal_rng(X_mat[i,] * beta, sigma);
    }
}
