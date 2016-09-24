perf.data <- read.csv("output_clean.csv")
row.names <- c("gen_proj_mat_ach",
               "gen_proj_mat_gauss",
               "sparse_conv_proj_ach",
               "sparse_conv_proj_gauss",
               "proj_ach",
               "proj_gauss",
               "UNK1",
               "UNK2",
               "UNK3",
               "UNK4",
               "proj_pdist_ach",
               "proj_pdist_gauss",
               "distortion_ach",
               "distortion_gauss",
               "p_distortion_ach",
               "p_distortion_gauss")

key.names <- read.csv("keynames.csv", header = FALSE)

rownames(perf.data) <- row.names

last.power <- 8

## Plots of the projection times
times.idx <- c(1, 2, 5, 6, 11, 12)
y.up <- max(perf.data[times.idx, 1:6], na.rm = T)
png("projection-times.png", width = 800, height = 800)

plot(x = 0,
     y = 0,
     type = 'n',
     xlim = c(0, last.power),
     ylim = c(0, y.up),
     xlab = "log_4(n)",
     ylab = "Time(s)",
     main = "Projection times using Achlioptas and Gaussian methods")

for (i in times.idx) {
    lines(x = 1:6, y = perf.data[i, 1:6], col = i)
    points(x = 1:6, y = perf.data[i, 1:6], col = i, pch = i)
}

legend(x = "topleft",
       legend = key.names$V1[times.idx],
       col = times.idx,
       pch = times.idx)

dev.off()
