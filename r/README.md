# LGTM for R

An R interface to the [LGTM Python package](https://yuanx749.github.io/lgtm/).

## Installation

```r
install.packages("remotes")
remotes::install_github("yuanx749/lgtm", ref = "v0.2.0", subdir = "r")
```

Before the first Python call in each R session:

```r
Sys.setenv(
  UV_INDEX = "https://download.pytorch.org/whl/cpu",
  UV_INDEX_STRATEGY = "unsafe-best-match"
)
library(lgtm)
```

Reticulate installs the Python backend on first use unless an existing Python
environment is selected.

## Usage

Prepare two data frames from CSV/TSV files, with one row per sample:

1. Metadata: required columns `sample_id`, `subject_id`, and `time`.
2. Microbiome profile: `sample_id` followed by taxa abundance columns.

`time` is continuous; additional metadata columns are categorical covariates
modeled through interactions with time. Sample IDs must be unique within each
table. The backend matches samples by ID and normalizes abundance rows.

```r
metadata <- read.csv("metadata.csv", check.names = FALSE)
microbiome <- read.csv("microbiome.csv", check.names = FALSE)

fit <- lgtm_fit(metadata, microbiome, latent_dim = 6, n_epoch = 100, seed = 42)
sample_topic <- fit$sample_topic
topic_taxon <- fit$topic_taxon

lgtm_plot(fit, type = "si")
lgtm_plot(fit, type = "topics")
plot_result <- lgtm_plot(fit, type = "gp", topic = 1)
plot_result$figure$savefig("gp.svg", bbox_inches = "tight")
```

Plots show a raster preview and return the original Matplotlib Figure/Axes
invisibly. Use `show = FALSE` to create a figure without displaying it.

See `?lgtm_fit` and `?lgtm_plot` for parameters and other plot types.
