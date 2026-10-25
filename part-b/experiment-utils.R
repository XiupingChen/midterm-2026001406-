loadImage <- function(path = "assets/image.jpg", max_size = 512L) {
    if (!file.exists(path)) stop("Image not found: ", path)
    if (!requireNamespace("jpeg", quietly = TRUE)) stop("Install the R package 'jpeg'.")
    img <- jpeg::readJPEG(path)
    if (length(dim(img)) == 3L) {
        img <- 0.299 * img[, , 1L] + 0.587 * img[, , 2L] + 0.114 * img[, , 3L]
    }
    stride <- max(1L, ceiling(max(dim(img)) / max_size))
    img[seq.int(1L, nrow(img), by = stride),
        seq.int(1L, ncol(img), by = stride), drop = FALSE]
}

# The input is a nonzero dense image. Returned indices are zero-based.
sampleEntries <- function(A, draws) {
    if (!is.matrix(A) || length(A) == 0L || any(!is.finite(A)) ||
        sum(abs(A)) == 0 || length(draws) != 1L || draws < 1 || draws != floor(draws)) {
        stop("Use a finite nonzero matrix and a positive integer draw count.")
    }
    probability <- as.vector(0.5 * (A^2 / sum(A^2) + abs(A) / sum(abs(A))))
    probability <- probability / sum(probability)
    indices <- sample.int(length(A), draws, replace = TRUE, prob = probability)
    list(rowIndex = as.integer((indices - 1L) %% nrow(A)),
         colIndex = as.integer((indices - 1L) %/% nrow(A)),
         values = as.vector(A)[indices] / (draws * probability[indices]))
}

vectorError <- function(estimate, reference) {
    estimate <- as.vector(estimate)
    reference <- as.vector(reference)
    min(sqrt(sum((estimate - reference)^2)), sqrt(sum((estimate + reference)^2)))
}

checkPowerResult <- function(result, A, tolerance = 1e-8) {
    u <- as.vector(result$u)
    v <- as.vector(result$v)
    sigma <- result$sigma
    stopifnot(length(sigma) == 1L, is.finite(sigma), sigma > 0,
              length(u) == nrow(A), length(v) == ncol(A),
              all(is.finite(u)), all(is.finite(v)),
              abs(sum(u^2) - 1) < tolerance, abs(sum(v^2) - 1) < tolerance,
              max(abs(as.vector(A %*% v) - sigma * u)) < tolerance * max(1, sigma))
}

# Reports seconds per full function call; compilation is outside this helper.
measureTime <- function(fun, min_seconds = 0.15, repetitions = 3L) {
    invisible(fun())
    batch <- 1L
    repeat {
        elapsed <- system.time(for (i in seq_len(batch)) fun())[["elapsed"]]
        if (elapsed >= min_seconds || batch >= 1024L) break
        batch <- batch * 2L
    }
    times <- replicate(repetitions, {
        system.time(for (i in seq_len(batch)) fun())[["elapsed"]] / batch
    })
    median(times)
}
