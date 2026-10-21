Rcpp::sourceCpp("weighted-least-squares.cpp")

checkFit <- function(fit, X, y, weights) {
    required <- c("coefficients", "fitted", "weighted_rss")
    stopifnot(is.list(fit), all(required %in% names(fit)))
    beta <- fit[["coefficients"]]
    fitted <- fit[["fitted"]]
    rss <- fit[["weighted_rss"]]
    stopifnot(is.numeric(beta), length(beta) == ncol(X), all(is.finite(beta)),
              is.numeric(fitted), length(fitted) == nrow(X), all(is.finite(fitted)),
              is.numeric(rss), length(rss) == 1L, is.finite(rss),
              max(abs(as.vector(fitted) - as.vector(X %*% beta))) < 1e-8,
              abs(rss - sum(weights * (y - as.vector(fitted))^2)) < 1e-8)
    invisible(NULL)
}

X <- cbind(intercept = 1, x = c(0, 1, 2, 3, 4))
y <- c(1.1, 1.8, 3.2, 3.5, 5.1)
w <- c(1, 2, 1, 3, 2)
fit <- weightedLeastSquares(X, y, w)
checkFit(fit, X, y, w)
beta <- as.vector(fit$coefficients)
fitted <- as.vector(fit$fitted)
reference <- lm(y ~ X - 1, weights = w)
stopifnot(length(beta) == ncol(X), length(fitted) == nrow(X),
          all(is.finite(beta)), all(is.finite(fitted)),
          max(abs(beta - coef(reference))) < 1e-8,
          max(abs(fitted - as.vector(X %*% beta))) < 1e-8,
          abs(fit$weighted_rss - sum(w * (y - fitted)^2)) < 1e-8)

ols <- weightedLeastSquares(X, y, rep(1, length(y)))
scaled <- weightedLeastSquares(X, y, 7 * w)
checkFit(ols, X, y, rep(1, length(y)))
checkFit(scaled, X, y, 7 * w)
stopifnot(max(abs(as.vector(ols$coefficients) - coef(lm(y ~ X - 1)))) < 1e-8,
          max(abs(as.vector(scaled$coefficients) - beta)) < 1e-8,
          abs(scaled$weighted_rss - 7 * fit$weighted_rss) < 1e-8)
print(list(coefficients = beta, weighted_rss = fit$weighted_rss))
cat("R reference, all-one weights and rescaled-weight checks passed.\n")
