suppressPackageStartupMessages({
  library(alto)
  library(ggplot2)
})

parse_args <- function(argv) {
  args <- list(
    cohort = "dhaka",
    in_root = "output/align",
    method = "product",
    suffix = NULL,
    l_min = 2,
    l_max = 20
  )
  i <- 1
  while (i <= length(argv)) {
    key <- argv[[i]]
    value <- argv[[i + 1]]
    if (key == "--cohort") args$cohort <- value
    if (key == "--in-root") args$in_root <- value
    if (key == "--method") args$method <- value
    if (key == "--suffix") args$suffix <- value
    if (key == "--l-min") args$l_min <- as.integer(value)
    if (key == "--l-max") args$l_max <- as.integer(value)
    i <- i + 2
  }
  if (is.null(args$suffix)) {
    args$suffix <- if (args$method == "product") "" else paste0("_", args$method)
  }
  args
}

read_models <- function(in_dir, l_min, l_max) {
  models <- list()
  for (L in seq(l_min, l_max)) {
    stem <- sprintf("L%02d", L)
    beta_path <- file.path(in_dir, paste0("beta_", stem, ".csv"))
    theta_path <- file.path(in_dir, paste0("theta_", stem, ".csv"))
    beta <- read.csv(beta_path, row.names = 1, check.names = FALSE)
    theta <- read.csv(theta_path, row.names = 1, check.names = FALSE)
    beta <- as.matrix(beta)
    theta <- as.matrix(theta)
    beta <- beta / rowSums(beta)
    theta <- theta / rowSums(theta)
    models[[stem]] <- list(
      gamma = theta,
      beta = log(pmax(beta, 1e-300))
    )
  }
  models
}

argv <- commandArgs(trailingOnly = TRUE)
args <- parse_args(argv)
in_dir <- file.path(args$in_root, args$cohort)
models <- read_models(in_dir, args$l_min, args$l_max)

alignment <- align_topics(models, method = args$method)
saveRDS(models, file.path(in_dir, paste0("alto_models", args$suffix, ".rds")))
saveRDS(alignment, file.path(in_dir, paste0("alto_alignment", args$suffix, ".rds")))
write.csv(slot(alignment, "topics"), file.path(in_dir, paste0("alto_topics", args$suffix, ".csv")), row.names = FALSE)
write.csv(slot(alignment, "weights"), file.path(in_dir, paste0("alto_weights", args$suffix, ".csv")), row.names = FALSE)

pdf(file.path(in_dir, paste0("alto_alignment", args$suffix, ".pdf")), width = 11, height = 7)
print(plot_alignment(alignment))
dev.off()

pdf(file.path(in_dir, paste0("alto_beta", args$suffix, ".pdf")), width = 11, height = 7)
print(plot_beta(alignment, n_features = 12, threshold = 0.001))
dev.off()
