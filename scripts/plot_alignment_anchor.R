suppressPackageStartupMessages({
  library(alto)
  library(ggplot2)
})

parse_args <- function(argv) {
  args <- list(
    cohort = "dhaka",
    in_root = "output/align",
    anchor_l = 4,
    alignment_suffix = "",
    suffix = "anchor_first_tab10_all_edges",
    title = NULL
  )
  i <- 1
  while (i <= length(argv)) {
    key <- argv[[i]]
    value <- argv[[i + 1]]
    if (key == "--cohort") args$cohort <- value
    if (key == "--in-root") args$in_root <- value
    if (key == "--anchor-l") args$anchor_l <- as.integer(value)
    if (key == "--alignment-suffix") args$alignment_suffix <- value
    if (key == "--suffix") args$suffix <- value
    if (key == "--title") args$title <- value
    i <- i + 2
  }
  if (is.null(args$title)) {
    args$title <- c(
      dhaka = "Dhaka",
      diabimmune = "DIABIMMUNE",
      hmp = "HMP2"
    )[[args$cohort]]
  }
  args
}

tab10 <- c(
  "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
  "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"
)

argv <- commandArgs(trailingOnly = TRUE)
args <- parse_args(argv)
in_dir <- file.path(args$in_root, args$cohort)
alignment <- readRDS(file.path(in_dir, paste0("alto_alignment", args$alignment_suffix, ".rds")))
topic_table <- slot(alignment, "topics")
model_levels <- levels(factor(topic_table$m))

anchor_model <- sprintf("L%02d", args$anchor_l)
anchor_topics <- topic_table[as.character(topic_table$m) == anchor_model, ]
if (nrow(anchor_topics) == 0) {
  stop(sprintf("Anchor model %s not found.", anchor_model))
}
anchor_rank <- as.integer(sub(".*_T", "", as.character(anchor_topics$k_label)))
anchor_topics <- anchor_topics[order(anchor_rank), ]
anchor_rank <- anchor_rank[order(anchor_rank)]
if (any(is.na(anchor_rank))) {
  stop("Anchor topic labels must end with _T##.")
}
if (nrow(anchor_topics) > length(tab10)) {
  stop("Anchor model has more topics than tab10 colors.")
}

anchor_paths <- as.character(anchor_topics$path)
unique_anchor <- !duplicated(anchor_paths)

path_levels <- levels(factor(topic_table$path))
path_colors <- setNames(rep("#d9d9d9", length(path_levels)), path_levels)
path_colors[anchor_paths[unique_anchor]] <- tab10[anchor_rank[unique_anchor]]

path_mass <- aggregate(prop ~ path, topic_table, sum)
path_mass$path <- as.character(path_mass$path)
path_mass <- path_mass[order(-path_mass$prop), ]
if (max(anchor_rank) < length(tab10)) {
  remaining_colors <- tab10[(max(anchor_rank) + 1):length(tab10)]
} else {
  remaining_colors <- character()
}
additional_paths <- path_mass$path[!(path_mass$path %in% anchor_paths[unique_anchor])]
additional_paths <- head(additional_paths, length(remaining_colors))
path_colors[additional_paths] <- remaining_colors[seq_along(additional_paths)]

fig <- plot_alignment(alignment) +
  scale_fill_manual(values = path_colors, limits = names(path_colors), drop = FALSE) +
  scale_color_manual(values = path_colors, limits = names(path_colors), drop = FALSE) +
  scale_x_continuous(
    breaks = seq_along(model_levels),
    labels = sub("^L0*", "", model_levels)
  ) +
  labs(title = args$title, x = NULL) +
  theme(
    plot.title = element_text(hjust = 0.5, size = 22),
    axis.title.x = element_blank(),
    axis.text.x = element_text(size = 16),
    axis.text.y = element_blank(),
    axis.ticks.y = element_blank(),
    legend.position = "none",
    plot.margin = margin(4, 6, 4, 6)
  )

pdf(
  file.path(in_dir, paste0(args$cohort, "-alignment.pdf")),
  width = 8.5,
  height = 4.8
)
print(fig)
dev.off()
