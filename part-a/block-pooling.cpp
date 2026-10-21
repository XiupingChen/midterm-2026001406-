#include <RcppArmadillo.h>
// [[Rcpp::depends(RcppArmadillo)]]

// [[Rcpp::export]]
arma::mat blockPool(const arma::mat& img, int blockSize) {
    if (img.n_rows == 0 || img.n_cols == 0 || !img.is_finite() || blockSize < 1) {
        Rcpp::stop("Use a nonempty finite image and a positive block size.");
    }
    const arma::uword b = static_cast<arma::uword>(blockSize);
    arma::mat pooled((img.n_rows + b - 1) / b,
                     (img.n_cols + b - 1) / b, arma::fill::zeros);
    for (arma::uword row = 0; row < pooled.n_rows; ++row) {
        const arma::uword rowBegin = row * b;
        const arma::uword rowEnd = std::min(rowBegin + b, img.n_rows);
        for (arma::uword col = 0; col < pooled.n_cols; ++col) {
            const arma::uword colBegin = col * b;
            const arma::uword colEnd = std::min(colBegin + b, img.n_cols);
            double sum = 0.0;
            for (arma::uword i = rowBegin; i < rowEnd; ++i)
                for (arma::uword j = colBegin; j < colEnd; ++j)
                    sum += img(i, j);
            pooled(row, col) = sum /
                static_cast<double>((rowEnd - rowBegin) * (colEnd - colBegin));
        }
    }
    return pooled;
}
