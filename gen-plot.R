library(Hmisc)

## Global configurations
plot.path <- "images/"
last.power <- 8

## Reading the data from the CSV files.
perf.data <- read.csv("output_clean.csv")
row.names <- c("gen_proj_mat_ach",
               "gen_proj_mat_gauss",
               "sparse_UNK0",
               "sparse_UNK1",
               "proj_ach",
               "proj_gauss",
               "sparse_UNK2",
               "sparse_UNK3",
               "sparse_UNK4",
               "sparse_UNK5",
               "proj_pdist_ach",
               "proj_pdist_gauss",
               "distortion_ach",
               "distortion_gauss",
               "p_distortion_ach",
               "p_distortion_gauss")

rownames(perf.data) <- row.names

key.names <- read.csv("keynames.csv", header = FALSE)
pdist.times <- read.csv("pdist-times.csv")

## Plots of the projection times
times.idx <- c(1, 2, 5, 6, 11, 12)
times.xval <- numeric(length = last.power)

for (i in 1:last.power)
    times.xval[i] <- 4 ** i

y.up <- max(perf.data[times.idx, 1:7], na.rm = T)


old.wd <- getwd()
dir.create(plot.path, showWarnings = FALSE, recursive = TRUE)
setwd(plot.path)

png("projection-times_log-scale.png", width = 800, height = 800)

plot(x = 0,
     y = 0,
     type = 'n',
     xlim = c(0, last.power),
     ylim = c(0, y.up),
     xlab = "Dimension log_4(n)",
     ylab = "Operation Time(s)",
     main = "Projection times using Achlioptas and Gaussian methods")

minor.tick(nx = 2, ny = 10)

for (i in times.idx) {
    lines(x = 1:7, y = perf.data[i, 1:7], col = i)
    points(x = 1:7, y = perf.data[i, 1:7], col = i, pch = i)
}

legend(x = "topleft",
       legend = key.names$V1[times.idx],
       col = times.idx,
       pch = times.idx)

dev.off()

## Plots of the maximum distortions
max.dist.idx <- c(13, 14)
y.up <- max(perf.data[max.dist.idx, 1:7], na.rm = T)

png("maximum-distortions_log-scale.png", width = 800, height = 800)

plot(x = 0,
     y = 0,
     type = 'n',
     xlim = c(0, last.power),
     ylim = c(0, y.up),
     xlab = "Dimension (log_4(n))",
     ylab = "Largest distortion value",
     main = "Maximum distortions using Achlioptas and Gaussian methods")

minor.tick(nx = 2, ny = 1)

for (i in 1:length(max.dist.idx)) {
    lines(x = 1:7, y = perf.data[max.dist.idx[i], 1:7], col = i)
    points(x = 1:7, y = perf.data[max.dist.idx[i], 1:7], col = i, pch = i)
}

legend(x = "topright",
       legend = key.names$V1[max.dist.idx],
       col = 1:2,
       pch = 1:2)

dev.off()

## Plots of the probability of such distortions
prob.dist.idx <- c(15, 16)
y.up <- max(perf.data[prob.dist.idx, 1:7], na.rm = T)

png("prob-maximum-distortions_log-scale.png", width = 800, height = 800)

plot(x = 0,
     y = 0,
     type = 'n',
     xlim = c(0, last.power),
     ylim = c(0, y.up),
     xlab = "Dimension (log_4(n))",
     ylab = "Probability of Maximum Distortion",
     main = "Probability of maximum distortions using Achlioptas and Gaussian methods")

minor.tick(nx = 2, ny = 10)

for (i in 1:length(prob.dist.idx)) {
    lines(x = 1:7, y = perf.data[prob.dist.idx[i], 1:7], col = i)
    points(x = 1:7, y = perf.data[prob.dist.idx[i], 1:7], col = i, pch = i)
}

legend(x = "topleft",
       legend = key.names$V1[prob.dist.idx],
       col = 1:2,
       pch = 1:2)

dev.off()

## Plots of the original pairwise distance times compared to the projected
## pairwise distance times.
mean.time <- mean(t(pdist.times))

ach.times.idx <- c(1, 5, 11)
gauss.times.idx <- c(2, 6, 12)

ach.total.times <- colSums(perf.data[ach.times.idx, 1:7])
gauss.total.times <- colSums(perf.data[gauss.times.idx, 1:7])
total.dist.times <- rbind(rep(mean.time, times = 7), ach.total.times, gauss.total.times)

y.up <- max(rbind(ach.total.times, gauss.total.times), mean.time)

png("proj-times-comparison.png", width = 800, height = 800)

plot(x = 0,
     y = 0,
     type = 'n',
     xlim = c(0, last.power),
     ylim = c(0, y.up),
     xlab = "Dimension (log_4(n))",
     ylab = "Time (s)",
     main = "Time to calculate the pairwise distances (original data and projections)")

minor.tick(nx = 2, ny = 10)

for (i in 1:nrow(total.dist.times)) {
    lines(x = 1:7, y = total.dist.times[i, ], col = i)
    points(x = 1:7, y = total.dist.times[i, ], col = i, pch = i)
}

legend(x = "topleft",
       legend = c("Original Pairwise Distances", "Pairwise Distances (ACH)", "Pairwise Distances (GAUSS)"),
       col = 1:3,
       pch = 1:3)

dev.off()

setwd(old.wd)
