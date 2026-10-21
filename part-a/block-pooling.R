Rcpp::sourceCpp("block-pooling.cpp")

small <- matrix(seq_len(20) / 20, 4L, 5L)
expected <- matrix(c(3.5, 5.5, 11.5, 13.5, 17.5, 19.5) / 20, 2L, 3L)
result <- blockPool(small, 2L)
stopifnot(identical(dim(result), dim(expected)), max(abs(result - expected)) < 1e-12,
          max(abs(blockPool(small, 1L) - small)) < 1e-12,
          max(abs(blockPool(small, 10L) - mean(small))) < 1e-12)
cat("Small matrix, partial edge, identity and whole-image checks passed.\n")

x <- seq(-1, 1, length.out = 384L)
y <- seq(-1, 1, length.out = 256L)
img <- outer(y, x, function(y, x) {
    0.15 + 0.5 * (x^2 + y^2 < 0.5) + 0.25 * (abs(x) < 0.15)
})
sizes <- c(2L, 4L, 8L)
draw <- function(values, caption) {
    plot.new()
    plot.window(xlim = c(0, ncol(values)), ylim = c(0, nrow(values)), asp = 1)
    rasterImage(as.raster(values), 0, 0, ncol(values), nrow(values), interpolate = FALSE)
    title(main = caption)
}
old_par <- par(mfrow = c(2, 2), mar = c(1, 1, 3, 1))
draw(img, "Original")
for (size in sizes) draw(blockPool(img, size), paste("Block size =", size))
par(old_par)
