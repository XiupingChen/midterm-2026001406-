#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

// [[Rcpp::export]]
Rcpp::List powerIterationDense(const arma::mat& A, const arma::vec& v0,
                               int iterations) {
    if (A.n_rows == 0 || A.n_cols == 0 || A.n_cols != v0.n_elem ||
        !A.is_finite() || !v0.is_finite() || arma::norm(v0, 2) == 0 ||
        iterations < 1) {
        Rcpp::stop("Check matrix, initial vector and iteration count.");
    }
    arma::vec v = v0 / arma::norm(v0, 2);
    for (int step = 0; step < iterations; ++step) {
        arma::vec z = A.t() * (A * v);
        const double length = arma::norm(z, 2);
        if (!(length > 0.0) || !std::isfinite(length))
            Rcpp::stop("Power iteration reached a zero or nonfinite vector.");
        v = z / length;
    }
    arma::vec Av = A * v;
    const double sigma = arma::norm(Av, 2);
    if (!(sigma > 0.0) || !std::isfinite(sigma))
        Rcpp::stop("Unable to compute a positive singular value.");
    arma::vec u = Av / sigma;
    return Rcpp::List::create(Rcpp::Named("sigma") = sigma,
                              Rcpp::Named("u") = u,
                              Rcpp::Named("v") = v);
}
