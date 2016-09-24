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

rownames(perf.data) <- row.names

last.power <- 8
y.up <- max(perf.data[c(1, 2, 5, 6, 11, 12), 1:6], na.rm = T)
plot(x = 0,
     y = 0,
     type = 'n',
     xlim = c(0, last.power),
     ylim = c(0, y.up),
     xlab = "log_4(n)",
     ylab = "Time(s)")

idx.lines <- c(1, 2, 5, 6)

for (i in idx.lines) {
    lines(x = 1:6, y = perf.data[i, 1:6], col = i)
    points(x = 1:6, y = perf.data[i, 1:6], col = i, pch = i)
}
