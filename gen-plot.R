library(Hmisc)

## Global configurations
plot.path <- "images/"
last.power <- 8

## Reading the data from the CSV files.
perf.data <- read.csv("output_full.csv", header = F)
row.names <- c("gen_proj_mat_ach",
               "gen_proj_mat_gauss",
               "proj_ach",
               "proj_gauss",
               "proj_pdist_ach",
               "proj_pdist_gauss",
               "distortion_ach",
               "distortion_gauss",
               "p_99_error")

exponents <- 1:last.power
col.names <- sprintf("%s", 4 ** exponents)

rownames(perf.data) <- row.names
colnames(perf.data) <- col.names

key.names <- read.csv("keynames.csv", header = FALSE)

## Plots of the projection times
times.idx <- 1:last.power

y.up <- max(perf.data[times.idx, 1:length(times.idx)], na.rm = T)

old.wd <- getwd()
dir.create(plot.path, showWarnings = FALSE, recursive = TRUE)
setwd(plot.path)

pdf("projection-times_log-scale.pdf", width = 8, height = 8)

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
    lines(x = times.idx, y = perf.data[i, times.idx], col = i)
    points(x = times.idx, y = perf.data[i, times.idx], col = i, pch = i)
}

legend(x = "topleft",
       legend = key.names$V1[times.idx],
       col = times.idx,
       pch = times.idx)

dev.off()

## Plots of the maximum distortions
max.dist.idx <- 7:9
y.up <- max(perf.data[max.dist.idx, times.idx], na.rm = T)

pdf("maximum-distortions_log-scale.pdf", width = 8, height = 8)

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
    lines(x = 1:8, y = perf.data[max.dist.idx[i], 1:8], col = i)
    points(x = 1:8, y = perf.data[max.dist.idx[i], 1:8], col = i, pch = i)
}

legend(x = "topright",
       legend = key.names$V1[max.dist.idx],
       col = 1:3,
       pch = 1:3)

dev.off()

## Plots of the original pairwise distance times compared to the projected
## pairwise distance times.
mean.time <- 42.5662

ach.times.idx <- c(1, 3, 5)
gauss.times.idx <- c(2, 4, 6)

ach.total.times <- colSums(perf.data[ach.times.idx, times.idx])
gauss.total.times <- colSums(perf.data[gauss.times.idx,times.idx])
total.dist.times <- rbind(ach.total.times, gauss.total.times)

y.up <- max(rbind(ach.total.times, gauss.total.times), mean.time)

pdf("proj-times-comparison.pdf", width = 8, height = 8)

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
    lines(x = 1:8, y = total.dist.times[i, ], col = i+1)
    points(x = 1:8, y = total.dist.times[i, ], col = i+1, pch = i+1)
}

abline(h = mean.time)
points(rep(mean.time, times = 7))

text(x = 8, y = 50, labels = sprintf("%.4f", mean.time), col = 1)

legend(x = "topleft",
       legend = c("Original Pairwise Distances", "Pairwise Distances (ACH)", "Pairwise Distances (GAUSS)"),
       col = 1:3,
       pch = 1:3)

dev.off()

setwd(old.wd)

## Generating the tables.
## Generating a table with the times.
xtable(perf.data[1:6, ])
## Generating a table with the maximum distortions and the probability of such distortions.
xtable(perf.data[max.dist.idx, ])
