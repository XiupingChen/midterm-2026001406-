#include <RcppEigen.h>
#include <vector>
// [[Rcpp::depends(RcppEigen)]]

// [[Rcpp::export]]
Eigen::SparseMatrix<double> buildSparseMatrix(
        int rows, int cols, const Eigen::VectorXi& rowIndex,
        const Eigen::VectorXi& colIndex, const Eigen::VectorXd& values) {
    if (rows < 1 || cols < 1 || rowIndex.size() == 0 ||
        rowIndex.size() != colIndex.size() || rowIndex.size() != values.size() ||
        !values.allFinite()) Rcpp::stop("Invalid triplet dimensions or values.");
    for (int k = 0; k < rowIndex.size(); ++k) {
        if (rowIndex(k) < 0 || rowIndex(k) >= rows ||
            colIndex(k) < 0 || colIndex(k) >= cols) {
            Rcpp::stop("Triplet indices must be zero-based and in range.");
        }
    }
    std::vector<Eigen::Triplet<double>> triplets;
    triplets.reserve(values.size());
    for (int k = 0; k < values.size(); ++k)
        triplets.emplace_back(rowIndex(k), colIndex(k), values(k));
    Eigen::SparseMatrix<double> result(rows, cols);
    result.setFromTriplets(triplets.begin(), triplets.end());
    result.makeCompressed();
    return result;
}

// [[Rcpp::export]]
Rcpp::List powerIterationSparse(const Eigen::SparseMatrix<double>& A,
                                Eigen::VectorXd v0, int iterations) {
    if (A.rows() == 0 || A.cols() == 0 || A.cols() != v0.size() ||
        !v0.allFinite() || v0.norm() == 0 || iterations < 1) {
        Rcpp::stop("Check sparse matrix, initial vector and iteration count.");
    }
    Eigen::VectorXd v = v0 / v0.norm();
    for (int step = 0; step < iterations; ++step) {
        Eigen::VectorXd z = A.transpose() * (A * v);
        const double length = z.norm();
        if (!(length > 0.0) || !std::isfinite(length))
            Rcpp::stop("Power iteration reached a zero or nonfinite vector.");
        v = z / length;
    }
    Eigen::VectorXd Av = A * v;
    const double sigma = Av.norm();
    if (!(sigma > 0.0) || !std::isfinite(sigma))
        Rcpp::stop("Unable to compute a positive singular value.");
    Eigen::VectorXd u = Av / sigma;
    return Rcpp::List::create(Rcpp::Named("sigma") = sigma,
                              Rcpp::Named("u") = u,
                              Rcpp::Named("v") = v);
}
