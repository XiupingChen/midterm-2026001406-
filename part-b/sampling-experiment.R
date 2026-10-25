# This also compiles and validates the Task 1 and Task 2 functions outside timing.
source("power-iteration.R")

sample_counts <- as.integer(length(img) * c(0.01, 0.02, 0.05, 0.10))
repetitions <- 3L

runExperiment <- function(img, v0, sample_counts, repetitions, iterations) {
    dense_seconds <- measureTime(function() powerIterationDense(img, v0, iterations))
    reference_v <- svd(img, nu = 0L, nv = 1L)$v[, 1L]
    records <- vector("list", length(sample_counts) * repetitions)
    record_number <- 0L
    for (draws in sample_counts) {
        for (repetition in seq_len(repetitions)) {
            build_seconds <- system.time({
                entries <- sampleEntries(img, draws)
                approximation <- buildSparseMatrix(nrow(img), ncol(img),
                    entries$rowIndex, entries$colIndex, entries$values)
            })[["elapsed"]]
            iteration_seconds <- measureTime(function() {
                powerIterationSparse(approximation, v0, iterations)
            })
            estimate <- powerIterationSparse(approximation, v0, iterations)
            record_number <- record_number + 1L
            records[[record_number]] <- data.frame(
                draws = draws,
                repetition = repetition,
                nonzeros = Matrix::nnzero(approximation),
                build_seconds = build_seconds,
                iteration_seconds = iteration_seconds,
                vector_error = vectorError(estimate$v, reference_v)
            )
        }
    }
    list(dense_seconds = dense_seconds, raw = do.call(rbind, records))
}

set.seed(2026)
results <- runExperiment(img, v0, sample_counts, repetitions, iterations)
print(results)

summary <- aggregate(results$raw[c("nonzeros", "build_seconds",
                                   "iteration_seconds", "vector_error")],
                     by = list(draws = results$raw$draws), FUN = median)
print(summary)
dir.create("results", showWarnings = FALSE)
write.csv(results$raw, "results/raw-results.csv", row.names = FALSE)
write.csv(summary, "results/median-results.csv", row.names = FALSE)
writeLines(sprintf("%.9g", results$dense_seconds), "results/dense-seconds.txt")

png("results/iteration-time.png", width = 1000, height = 700, res = 130)
plot(summary$draws, summary$iteration_seconds, type = "b", pch = 19,
     log = "y",
     ylim = c(0.8 * min(summary$iteration_seconds),
              1.2 * max(summary$iteration_seconds, results$dense_seconds)),
     xlab = "Number of sampled entries (q)",
     ylab = "Power iteration time (seconds, log scale)",
     main = "Sparse power iteration by sample count")
abline(h = results$dense_seconds, col = "firebrick", lty = 2, lwd = 2)
legend("bottomright", c("Sparse median", "Dense baseline"),
       col = c("black", "firebrick"), lty = c(1, 2), pch = c(19, NA), bty = "n")
dev.off()

png("results/vector-error.png", width = 1000, height = 700, res = 130)
plot(summary$draws, summary$vector_error, type = "b", pch = 19,
     xlab = "Number of sampled entries (q)",
     ylab = "Sign-invariant right singular vector error",
     main = "Approximation error by sample count")
dev.off()
