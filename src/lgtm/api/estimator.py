import numpy as np

from lgtm.api.backend import (
    LGTMConfig,
    prepare_data,
    sample_topic_frame,
    topic_taxon_frame,
    train_and_analyze,
)


class LGTM:
    """Longitudinal Gaussian process modulated neural topic modeling for
    microbiome data analysis.

    This class provides a small Python API around the same preprocessing and
    training path used by the web app. The input is two pandas tables:
    metadata and microbiome profile.

    Metadata must contain `sample_id`, `subject_id`, and `time`. `time` is
    treated as a continuous covariate. Any extra metadata columns are treated
    as categorical covariates and modeled through interactions with time.

    The microbiome profile must contain `sample_id` plus one column per taxon.
    Samples are matched by `sample_id`, taxa abundances are converted to
    numeric values, and each sample is normalized to sum to one.

    Parameters
    ----------
    config : LGTMConfig or None, default=None
        Training configuration. When `None`, default `LGTMConfig()` is used.

    Attributes
    ----------
    sample_topic_ : pandas.DataFrame
        Sample-topic matrix sorted by decreasing mean topic proportion.
    topic_taxon_ : pandas.DataFrame
        Topic-taxon matrix sorted with the same topic order as `sample_topic_`.
    theta_ : numpy.ndarray
        Raw sample-topic matrix in the model's original topic order.
    beta_ : numpy.ndarray
        Raw topic-taxon matrix in the model's original topic order.
    topic_order_ : numpy.ndarray
        Indices mapping sorted topic order to the model's original topic order.
    model_ : torch.nn.Module
        Fitted PyTorch model.
    args_ : object
        Final training configuration passed to the underlying trainer.
    """

    def __init__(self, config=None):
        self.config = LGTMConfig() if config is None else config

    def fit(self, metadata, microbiome):
        """Fit LGTM from metadata and microbiome tables.

        Parameters
        ----------
        metadata : pandas.DataFrame
            Table with required columns `sample_id`, `subject_id`, and `time`.
            Extra columns are treated as categorical covariates.
        microbiome : pandas.DataFrame
            Table with `sample_id` and taxa abundance columns. Rows are
            automatically normalized to sum to one after matching samples.

        Returns
        -------
        LGTM
            The fitted estimator. Fitted outputs are stored in attributes
            ending with an underscore, following the scikit-learn convention.
        """
        self.prepared_ = prepare_data(metadata, microbiome)
        self.output_ = train_and_analyze(
            self.prepared_,
            config=self.config,
        )
        self.args_ = self.output_.args
        self.results_ = self.output_.results
        self.model_ = self.output_.model
        self.theta_ = self.output_.theta
        self.beta_ = self.output_.beta
        self.sobol_result_ = self.output_.sobol_result
        self.var_y_ = self.output_.var_y
        self.topic_order_ = np.argsort(np.mean(self.theta_, axis=0))[::-1]
        self.sample_topic_ = sample_topic_frame(
            self.theta_,
            self.prepared_.sample_ids,
            topic_order=self.topic_order_,
        )
        self.topic_taxon_ = topic_taxon_frame(
            self.beta_,
            self.prepared_.taxa,
            topic_order=self.topic_order_,
        )
        return self

    def plot_si(self, rotation=45):
        """Plot the covariate importance overview.

        Parameters
        ----------
        rotation : float, default=45
            Rotation angle for covariate labels on the heatmap.

        Returns
        -------
        tuple
            Matplotlib `(figure, axes)` returned by `plot_si_heatmap`.
        """
        from lgtm.utils_plot import plot_si_heatmap

        return plot_si_heatmap(
            self.theta_,
            self.sobol_result_,
            self.var_y_,
            self.prepared_.attrs.x_cols,
            rotation=rotation,
        )

    def plot_topics(self, top_k=5, threshold=0.01, num_plot=None):
        """Plot an overview of topic-taxon loadings.

        Parameters
        ----------
        top_k : int, default=5
            Maximum number of taxa shown for each topic.
        threshold : float, default=0.01
            Minimum absolute loading for a taxon to be eligible for display.
        num_plot : int or None, default=None
            Number of topics to show. `None` shows all topics.

        Returns
        -------
        tuple
            Matplotlib `(figure, axes)` for the topic overview.
        """
        from lgtm.utils_plot import plot_topics

        return plot_topics(
            self.beta_,
            self.theta_,
            taxa=self.prepared_.taxa,
            top_k=top_k,
            threshold=threshold,
            num_plot=num_plot,
        )

    def plot_topic(self, topic=1, top_n=20, threshold=0.01, figsize=(18, 2)):
        """Plot taxa loadings for one topic.

        Parameters
        ----------
        topic : int, default=1
            One-based topic id after sorting by mean sample proportion.
        top_n : int, default=20
            Number of taxa considered when selecting taxa for the plot.
        threshold : float, default=0.01
            Taxon label threshold. Taxa below this loading are not annotated.
        figsize : tuple, default=(18, 2)
            Matplotlib figure size.

        Returns
        -------
        tuple
            Matplotlib `(figure, axes)` for the selected topic.
        """
        from lgtm.utils_plot import plot_topic

        cnt, l = self._topic_position(topic)
        return plot_topic(
            self.beta_,
            self.theta_,
            cnt=cnt,
            l=l,
            taxa=self.prepared_.taxa,
            top_n=top_n,
            threshold=threshold,
            figsize=figsize,
        )

    def plot_gp(self, topic=1, repeats=5, figsize=(18, 3), plot_id=False):
        """Plot GP components for one topic.

        Parameters
        ----------
        topic : int, default=1
            One-based topic id after sorting by mean sample proportion.
        repeats : int, default=5
            Number of stochastic GP draws used for uncertainty bands.
        figsize : tuple, default=(18, 3)
            Matplotlib figure size.
        plot_id : bool, default=False
            Whether to include subject-id GP components.

        Returns
        -------
        tuple
            Matplotlib `(figure, axes)` for GP component plots.
        """
        from lgtm.utils_plot import plot_gp

        cnt, l = self._topic_position(topic)
        return plot_gp(
            self.model_,
            self.prepared_.dataset,
            self.prepared_.attrs,
            cnt=cnt,
            l=l,
            sobol_result=self.sobol_result_,
            repeats=repeats,
            figsize=figsize,
            plot_id=plot_id,
        )

    def plot_latent(
        self,
        topic=1,
        figsize=(18, 3),
        plot_pointplot=False,
        plot_id=False,
    ):
        """Plot observed sample-topic proportions for one topic.

        Parameters
        ----------
        topic : int, default=1
            One-based topic id after sorting by mean sample proportion.
        figsize : tuple, default=(18, 3)
            Matplotlib figure size.
        plot_pointplot : bool, default=False
            Whether to overlay group mean points.
        plot_id : bool, default=False
            Whether to include subject-id panels.

        Returns
        -------
        tuple
            Matplotlib `(figure, axes)` for latent sample-topic plots.
        """
        from lgtm.utils_plot import plot_latent

        _, l = self._topic_position(topic)
        return plot_latent(
            self.model_,
            self.prepared_.attrs,
            self.theta_,
            l=l,
            figsize=figsize,
            plot_pointplot=plot_pointplot,
            plot_id=plot_id,
        )

    def _topic_position(self, topic):
        cnt = int(topic) - 1
        if cnt < 0 or cnt >= len(self.topic_order_):
            raise IndexError(f"topic must be between 1 and {len(self.topic_order_)}.")
        return cnt, self.topic_order_[cnt]
