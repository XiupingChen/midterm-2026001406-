library(Matrix)
Rcpp::sourceCpp("power-iteration-dense.cpp")
Rcpp::sourceCpp("power-iteration-sparse.cpp")
source("experiment-utils.R")

iterations <- 1000L
small <- rbind(diag(c(3, 2, 1)), c(0, 0, 0))
initial <- rep(1, ncol(small))
dense <- powerIterationDense(small, initial, iterations)
checkPowerResult(dense, small)
reference <- svd(small, nu = 1L, nv = 1L)
stopifnot(abs(dense$sigma - reference$d[1L]) < 1e-8,
          vectorError(dense$v, reference$v[, 1L]) < 1e-8)

# Two entries at (0, 0) must be added, not overwritten.
duplicate <- buildSparseMatrix(2L, 2L, c(0L, 0L, 1L), c(0L, 0L, 1L), c(1, 2, 1))
stopifnot(max(abs(as.matrix(duplicate) - diag(c(3, 1)))) < 1e-12)
sparse <- powerIterationSparse(duplicate, c(1, 1), iterations)
checkPowerResult(sparse, duplicate)
stopifnot(abs(sparse$sigma - 3) < 1e-8, vectorError(sparse$v, c(1, 0)) < 1e-8)
cat("Small dense and sparse checks, including duplicate triplets, passed.\n")

img <- loadImage()
v0 <- rep(1, ncol(img))
reference <- svd(img, nu = 1L, nv = 1L)
dense <- powerIterationDense(img, v0, iterations)
checkPowerResult(dense, img)
dense_sigma_error <- abs(dense$sigma - reference$d[1L]) / reference$d[1L]
dense_vector_error <- vectorError(dense$v, reference$v[, 1L])
stopifnot(dense_sigma_error < 1e-6, dense_vector_error < 1e-5)
print(c(rows = nrow(img), cols = ncol(img),
        dense_relative_sigma_error = dense_sigma_error,
        dense_vector_error = dense_vector_error))

set.seed(2026)
draws <- as.integer(0.05 * length(img))
entries <- sampleEntries(img, draws)
approximation <- buildSparseMatrix(nrow(img), ncol(img),
    entries$rowIndex, entries$colIndex, entries$values)
sparse <- powerIterationSparse(approximation, v0, iterations)
checkPowerResult(sparse, approximation)
sparse_reference <- svd(as.matrix(approximation), nu = 1L, nv = 1L)
sigma_error <- abs(sparse$sigma - sparse_reference$d[1L]) / sparse_reference$d[1L]
direction_error <- vectorError(sparse$v, sparse_reference$v[, 1L])
stopifnot(sigma_error < 1e-6, direction_error < 1e-5)
print(c(draws = draws, stored_nonzeros = Matrix::nnzero(approximation),
        sparse_relative_sigma_error = sigma_error,
        sparse_algorithm_vector_error = direction_error,
        approximation_vector_error = vectorError(sparse$v, reference$v[, 1L])))
