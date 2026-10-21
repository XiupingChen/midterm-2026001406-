#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

// [[Rcpp::export]]
Rcpp::List weightedLeastSquares(const arma::mat& X, const arma::vec& y,
                                const arma::vec& weights) {
    if (X.n_rows == 0 || X.n_cols == 0 || X.n_rows < X.n_cols ||
        X.n_rows != y.n_elem || X.n_rows != weights.n_elem ||
        !X.is_finite() || !y.is_finite() || !weights.is_finite() ||
        arma::any(weights <= 0)) {
        Rcpp::stop("Check dimensions, finiteness and strictly positive weights.");
    }
    const arma::vec rootWeights = arma::sqrt(weights);
    arma::mat weightedX = X;
    weightedX.each_col() %= rootWeights;
    const arma::vec weightedY = rootWeights % y;
    const arma::vec coefficients = arma::solve(weightedX, weightedY);
    const arma::vec fitted = X * coefficients;
    const arma::vec residuals = y - fitted;
    const double weightedRss = arma::dot(weights, residuals % residuals);
    return Rcpp::List::create(
        Rcpp::Named("coefficients") = coefficients,
        Rcpp::Named("fitted") = fitted,
        Rcpp::Named("weighted_rss") = weightedRss);
}
