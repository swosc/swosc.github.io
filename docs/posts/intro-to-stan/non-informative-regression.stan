// The input data is two vectors 'y' and 'X' of length 'N'.
data {
  int<lower=0> N;
  vector[N] y;
  vector[N] x;
}

transformed data {
    matrix[N, 2] X_c = append_col(rep_vector(1, N), x);
    matrix[2,2] XtX_inv = inverse(X_c' * X_c);

    vector[2] beta_hat = XtX_inv * X_c' * y;
    vector[N] y_hat = X_c * beta_hat;
    
    real<lower=0> s_2 = 1.0 / (N - 2) * (y - y_hat)' * (y - y_hat);
}

// The parameters accepted by the model. Our model
// accepts two parameters 'beta' and 'sigma'.
parameters {
  vector[2] beta;
  real<lower=0> sigma2; // Note that this is the variance
}

// The model to be estimated. We model the output
// 'y' ~ N(x beta, sigma) by specifying the analytic
// posterior defined above.
model {
  beta ~ multi_normal(beta_hat, sigma2 * XtX_inv);
  sigma2 ~ scaled_inv_chi_square(N-2, sqrt(s_2));
}

generated quantities {
    real sigma = sqrt(sigma2);
    vector[N] y_ppd;
    
    for (i in 1:N) {
        y_ppd[i] = normal_rng(X_c[i,] * beta, sigma);
    }
}
