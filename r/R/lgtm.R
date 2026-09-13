#' Fit LGTM from metadata and microbiome tables
#'
#' Calls the existing LGTM Python API. The backend matches samples by
#' `sample_id`, sorts them by subject and time, normalizes each abundance row
#' to sum to one, and encodes categorical covariates.
#'
#' @param metadata A data frame with one row per sample and required columns
#'   `sample_id`, `subject_id`, and numeric `time`. Time is continuous. Extra
#'   columns are categorical covariates modeled through interactions with
#'   time. Sample IDs must be unique.
#' @param microbiome A data frame with `sample_id` and one numeric abundance
#'   column per taxon. Sample IDs must be unique, and each abundance row must
#'   have a positive total. Preserve taxon names when reading the table.
#' @param latent_dim Number of latent topics. Default: 6.
#' @param n_epoch Number of training epochs. Default: 100.
#' @param batch_size Mini-batch size. Default: 64.
#' @param learning_rate Adam learning rate. Default: 0.05.
#' @param hidden_dim Encoder hidden dimension. Default: 64.
#' @param seed Random seed for Python, NumPy, and PyTorch. Default: 42.
#' @param n_basis_functions Number of Hilbert-space basis functions for the
#'   squared exponential kernel approximation. Default: 5.
#' @param kl_weight GP KL regularization weight. Default: 0.001.
#' @param mc_samples Number of Monte Carlo samples during training. Default: 1.
#' @param patience Early stopping patience. Default: 0.
#' @param early_stop Enable early stopping using the training data as validation
#'   data. This is a full-data fit, not cross-validation. Default: TRUE.
#' @param init_topics Initialize topic loadings using NNDSVD. Default: TRUE.
#' @param use_encoder Use the encoder pathway during training. Default: TRUE.
#'
#' @returns A list containing:
#'   - `sample_topic`: a sample-by-topic data frame of topic proportions, with
#'     sample IDs as row names and `topic-1`, `topic-2`, etc. as column names.
#'   - `topic_taxon`: a topic-by-taxon data frame of loadings, with topic names
#'     as row names and taxon names as column names.
#'   - `model`: the fitted Python `LGTM` estimator.
#'
#'   Both tables use the same topic order, sorted by decreasing mean
#'   sample-topic proportion. The Python model reference is valid only in the
#'   current R session; save result tables separately when using `saveRDS()`.
#' @examples
#' \dontrun{
#' metadata <- read.csv("metadata.csv", check.names = FALSE)
#' microbiome <- read.csv("microbiome.csv", check.names = FALSE)
#' fit <- lgtm_fit(
#'   metadata,
#'   microbiome,
#'   latent_dim = 6,
#'   n_epoch = 100,
#'   seed = 42
#' )
#' fit$sample_topic
#' fit$topic_taxon
#' }
#' @export
lgtm_fit <- function(
  metadata,
  microbiome,
  latent_dim = 6L,
  n_epoch = 100L,
  batch_size = 64L,
  learning_rate = 0.05,
  hidden_dim = 64L,
  seed = 42L,
  n_basis_functions = 5L,
  kl_weight = 0.001,
  mc_samples = 1L,
  patience = 0L,
  early_stop = TRUE,
  init_topics = TRUE,
  use_encoder = TRUE
) {
  lgtm <- reticulate::import("lgtm")
  config <- lgtm$LGTMConfig(
    latent_dim = as.integer(latent_dim),
    n_epoch = as.integer(n_epoch),
    batch_size = as.integer(batch_size),
    learning_rate = learning_rate,
    hidden_dim = as.integer(hidden_dim),
    seed = as.integer(seed),
    n_basis_functions = as.integer(n_basis_functions),
    kl_weight = kl_weight,
    mc_samples = as.integer(mc_samples),
    patience = as.integer(patience),
    early_stop = early_stop,
    init_topics = init_topics,
    use_encoder = use_encoder
  )
  model <- lgtm$LGTM(config)
  model$fit(metadata, microbiome)
  list(
    sample_topic = model$sample_topic_,
    topic_taxon = model$topic_taxon_,
    model = model
  )
}

#' Plot LGTM results
#'
#' Reuses the Python API's Matplotlib figures. When `show = TRUE`, a PNG preview
#' is drawn on the R graphics device. The original Figure and Axes remain
#' available for customization and vector export with `figure$savefig()`.
#'
#' @param fit A result from [lgtm_fit()].
#' @param type Plot type: `"si"` for covariate importance, `"topics"` for the
#'   topic overview, `"topic"` for one topic's taxon loadings, `"gp"` for GP
#'   components, or `"latent"` for observed sample-topic proportions.
#' @param topic One-based topic number, sorted by decreasing mean proportion.
#'   Used for `"topic"`, `"gp"`, and `"latent"`. Default: 1.
#' @param ... Arguments forwarded to the corresponding Python plotting method.
#'   See Details for supported arguments and their defaults.
#' @param show Draw a preview on the R graphics device. Default: TRUE.
#' @param dpi Resolution of the PNG preview. Default: 300.
#'
#' @details
#' Plot-specific arguments follow the Python API:
#' - `"si"`: `rotation = 45` sets the angle of covariate labels in degrees.
#' - `"topics"`: `top_k = 5` limits the taxa shown per topic; `threshold = 0.01`
#'   is the minimum absolute loading for display; `num_plot = NULL` shows all
#'   topics, or an integer limits the number shown.
#' - `"topic"`: `top_n = 20` sets the number of taxa considered;
#'   `threshold = 0.01` controls which taxa are annotated;
#'   `figsize = c(18, 2)` sets the figure size.
#' - `"gp"`: `repeats = 5` sets the number of stochastic GP draws used for
#'   uncertainty bands; `figsize = c(18, 3)` sets the figure size;
#'   `plot_id = FALSE` omits subject-ID GP components.
#' - `"latent"`: `figsize = c(18, 3)` sets the figure size;
#'   `plot_pointplot = FALSE` disables group mean points; `plot_id = FALSE`
#'   omits subject-ID panels.
#'
#' Use a numeric vector of length two for `figsize`, in inches.
#'
#' @returns Invisibly, a list containing the original Matplotlib `figure` and
#'   its `axes`. The Figure is removed from pyplot's registry after plotting
#'   but remains usable through this list.
#' @examples
#' \dontrun{
#' lgtm_plot(fit, type = "si")
#' plot_result <- lgtm_plot(fit, type = "gp", topic = 1)
#' plot_result$figure$savefig("gp.svg", bbox_inches = "tight")
#' }
#' @export
lgtm_plot <- function(
  fit,
  type = c("si", "topics", "topic", "gp", "latent"),
  topic = 1L,
  ...,
  show = TRUE,
  dpi = 300
) {
  type <- match.arg(type)
  method <- switch(
    type,
    si = fit$model$plot_si,
    topics = fit$model$plot_topics,
    topic = fit$model$plot_topic,
    gp = fit$model$plot_gp,
    latent = fit$model$plot_latent
  )
  args <- list(...)
  for (name in intersect(
    names(args),
    c("top_k", "num_plot", "top_n", "repeats")
  )) {
    if (!is.null(args[[name]])) {
      args[[name]] <- as.integer(args[[name]])
    }
  }
  if (type %in% c("topic", "gp", "latent")) {
    args$topic <- as.integer(topic)
  }
  result <- do.call(method, args)
  figure <- result[[1L]]
  plt <- reticulate::import("matplotlib.pyplot")
  on.exit(plt$close(figure), add = TRUE)

  if (show) {
    buffer <- reticulate::import("io")$BytesIO()
    on.exit(buffer$close(), add = TRUE)
    figure$savefig(buffer, format = "png", dpi = dpi, bbox_inches = "tight")
    preview <- png::readPNG(as.raw(buffer$getvalue()))
    grid::grid.newpage()
    grid::grid.raster(preview)
  }
  invisible(list(figure = figure, axes = result[[2L]]))
}
