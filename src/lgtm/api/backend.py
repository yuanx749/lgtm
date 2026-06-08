import copy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from lgtm.config import args as base_args
from lgtm.data import MetagenomeDataset
from lgtm.train import train1
from lgtm.utils import get_sobol_indices


REQUIRED_METADATA_COLUMNS = ("sample_id", "subject_id", "time")


@dataclass
class Attrs:
    df: pd.DataFrame
    df_y: pd.DataFrame
    y: np.ndarray
    samples: np.ndarray
    subjects: np.ndarray
    timepoints: np.ndarray
    n_samples: int
    n_subjects: int
    n_steps: int
    features: np.ndarray
    n_features: int
    x: np.ndarray
    x_num: np.ndarray
    x_cat: np.ndarray
    x_cols: list
    n_covariates: int
    n_cat_lst: list
    se_idx: list
    ca_idx: list
    bin_idx: list
    interactions: list
    C: list
    id_covariate: int


@dataclass
class PreparedData:
    attrs: Attrs
    dataset: MetagenomeDataset
    sample_ids: np.ndarray
    taxa: np.ndarray


@dataclass
class TrainingOutput:
    args: object
    results: dict
    model: object
    theta: np.ndarray
    beta: np.ndarray
    sobol_result: object
    var_y: np.ndarray


@dataclass
class LGTMConfig:
    """Training configuration for the LGTM Python API.

    Parameters
    ----------
    latent_dim : int, default=6
        Number of latent topics.
    n_epoch : int, default=100
        Number of training epochs.
    batch_size : int, default=64
        Mini-batch size used by the PyTorch data loader.
    learning_rate : float, default=0.05
        Adam optimizer learning rate.
    hidden_dim : int, default=64
        Hidden dimension in the encoder network.
    seed : int, default=42
        Random seed used for PyTorch, NumPy, and Python random.
    n_basis_functions : int, default=5
        Number of Hilbert-space basis functions used for the SE kernel
        approximation.
    kl_weight : float, default=0.001
        Weight of the GP KL regularization term.
    mc_samples : int, default=1
        Number of Monte Carlo samples used during stochastic training.
    patience : int, default=0
        Early stopping patience. The full-data fit uses the training data as
        validation data when early stopping is enabled.
    early_stop : bool, default=True
        Whether to enable early stopping in the full-data training routine.
    init_topics : bool, default=True
        Whether to initialize topic loadings with NNDSVD.
    use_encoder : bool, default=True
        Whether to use the encoder pathway during training.
    """

    latent_dim: int = 6
    n_epoch: int = 100
    batch_size: int = 64
    learning_rate: float = 0.05
    hidden_dim: int = 64
    seed: int = 42
    n_basis_functions: int = 5
    kl_weight: float = 0.001
    mc_samples: int = 1
    patience: int = 0
    early_stop: bool = True
    init_topics: bool = True
    use_encoder: bool = True


def read_table(file_obj, filename):
    suffix = Path(filename).suffix.lower()
    if suffix == ".csv":
        sep = ","
    elif suffix == ".tsv":
        sep = "\t"
    else:
        raise ValueError(
            f"Unsupported file type: {filename}. Only CSV/TSV are allowed."
        )
    return pd.read_csv(file_obj, sep=sep)


def _validate_metadata(metadata):
    missing = [col for col in REQUIRED_METADATA_COLUMNS if col not in metadata.columns]
    if missing:
        missing_str = ", ".join(missing)
        raise ValueError(f"Metadata missing required columns: {missing_str}")

    df = metadata.copy()
    df["sample_id"] = df["sample_id"].astype(str)
    df["subject_id"] = df["subject_id"].astype(str)
    df["time"] = pd.to_numeric(df["time"], errors="raise")

    if df["sample_id"].duplicated().any():
        raise ValueError("Metadata sample_id must be unique.")
    if (
        df["sample_id"].isna().any()
        or df["subject_id"].isna().any()
        or df["time"].isna().any()
    ):
        raise ValueError(
            "Metadata sample_id/subject_id/time cannot contain missing values."
        )

    return df


def _validate_and_normalize_microbiome(microbiome):
    if "sample_id" not in microbiome.columns:
        raise ValueError("Microbiome profile must contain sample_id column.")

    df = microbiome.copy()
    df["sample_id"] = df["sample_id"].astype(str)
    if df["sample_id"].duplicated().any():
        raise ValueError("Microbiome sample_id must be unique.")

    taxa_cols = [col for col in df.columns if col != "sample_id"]
    if not taxa_cols:
        raise ValueError("Microbiome profile must include at least one taxa column.")

    for col in taxa_cols:
        df[col] = pd.to_numeric(df[col], errors="raise")

    values = df[taxa_cols].to_numpy(dtype=np.float32)
    if np.isnan(values).any():
        raise ValueError("Microbiome profile cannot contain missing values.")

    row_sums = values.sum(axis=1, keepdims=True)
    if (row_sums <= 0).any():
        raise ValueError("Each microbiome sample must have positive total abundance.")
    values = values / row_sums

    df[taxa_cols] = values
    return df, taxa_cols


def _build_attrs(merged, taxa_cols):
    num_cols = ["time"]
    base_cols = list(REQUIRED_METADATA_COLUMNS)
    extra_cat_cols = [c for c in merged.columns if c not in base_cols + taxa_cols]
    cat_cols = ["subject_id", *extra_cat_cols]

    for col in cat_cols:
        merged[col] = merged[col].astype("category")

    x_num = merged[num_cols].to_numpy(dtype=np.float32)
    x_cat = merged[cat_cols].apply(lambda s: s.cat.codes).to_numpy(dtype=np.float32)
    x = np.hstack((x_num, x_cat))
    x_cols = num_cols + cat_cols

    n_cat_lst = [len(merged[col].cat.categories) for col in cat_cols]
    time_idx = x_cols.index("time")
    id_covariate = x_cols.index("subject_id")
    interactions = []
    C = []
    for i, cat_col in enumerate(cat_cols):
        cat_idx = len(num_cols) + i
        if cat_idx == id_covariate:
            continue
        n_cat = n_cat_lst[i] + int(merged[cat_col].isna().any())
        interactions.append([time_idx, cat_idx])
        C.append(n_cat)

    samples = merged["sample_id"].to_numpy(copy=True)
    y = merged[taxa_cols].to_numpy(dtype=np.float32)
    subjects = merged["subject_id"].cat.categories.to_numpy(copy=True)
    timepoints = np.sort(merged["time"].unique())

    return Attrs(
        df=merged.copy(),
        df_y=merged[taxa_cols].copy(),
        y=y,
        samples=samples,
        subjects=subjects,
        timepoints=timepoints,
        n_samples=len(samples),
        n_subjects=len(subjects),
        n_steps=len(timepoints),
        features=np.asarray(taxa_cols),
        n_features=len(taxa_cols),
        x=x,
        x_num=x_num,
        x_cat=x_cat,
        x_cols=x_cols,
        n_covariates=len(x_cols),
        n_cat_lst=n_cat_lst,
        se_idx=[time_idx],
        ca_idx=[],
        bin_idx=[],
        interactions=interactions,
        C=C,
        id_covariate=id_covariate,
    )


def prepare_data(metadata, microbiome):
    meta = _validate_metadata(metadata)
    micro, taxa_cols = _validate_and_normalize_microbiome(microbiome)

    merged = meta.merge(micro, on="sample_id", how="inner")
    if merged.empty:
        raise ValueError("No shared sample_id between metadata and microbiome profile.")

    merged = merged.sort_values(["subject_id", "time", "sample_id"]).reset_index(
        drop=True
    )
    attrs = _build_attrs(merged, taxa_cols)
    dataset = MetagenomeDataset(attrs.y, attrs.x_num, attrs.x_cat, transform=None)
    return PreparedData(
        attrs=attrs,
        dataset=dataset,
        sample_ids=attrs.samples,
        taxa=attrs.features,
    )


def _build_train_args(config):
    args = copy.deepcopy(base_args)
    args.cohort = ""
    args.latent_dim = int(config.latent_dim)
    args.n_epoch = config.n_epoch
    args.batch_size = config.batch_size
    args.lr = config.learning_rate
    args.hidden_dim = config.hidden_dim
    args.seed = config.seed
    args.M = config.n_basis_functions
    args.b = config.kl_weight
    args.k = config.mc_samples
    args.patience = config.patience
    args.early_stop = config.early_stop
    args.init_b = config.init_topics
    args.encode_y = config.use_encoder
    args.name = f"lgtm-L{args.latent_dim}"
    return args


def _build_sobol_func(model, attrs):
    n_covariates = attrs.n_covariates

    def _func(x):
        x = np.asarray(x, dtype=np.float32)
        if x.shape[0] != n_covariates:
            raise ValueError(
                f"Sobol input dimension mismatch: expected {n_covariates}, got {x.shape[0]}"
            )
        x_mat = x.T
        x_tensor = torch.tensor(x_mat, dtype=torch.float32)
        with torch.no_grad():
            _, _, log_theta_x, *_ = model(x_tensor, stochastic_flag=False)
        theta = log_theta_x.exp().squeeze(1).detach().cpu().numpy().T
        return theta

    return _func


def train_and_analyze(
    prepared,
    latent_dim=None,
    n_epoch=None,
    batch_size=None,
    learning_rate=None,
    hidden_dim=None,
    seed=None,
    n_basis_functions=None,
    kl_weight=None,
    mc_samples=None,
    patience=None,
    early_stop=None,
    init_topics=None,
    use_encoder=None,
    config=None,
):
    if config is None:
        config = LGTMConfig() if latent_dim is None else LGTMConfig(latent_dim=latent_dim)
        overrides = {
            "n_epoch": n_epoch,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "hidden_dim": hidden_dim,
            "seed": seed,
            "n_basis_functions": n_basis_functions,
            "kl_weight": kl_weight,
            "mc_samples": mc_samples,
            "patience": patience,
            "early_stop": early_stop,
            "init_topics": init_topics,
            "use_encoder": use_encoder,
        }
        for name, value in overrides.items():
            if value is not None:
                setattr(config, name, value)
    args = _build_train_args(config)
    results = train1(prepared.attrs, args)
    model = results["artifacts"]["model"]
    model.eval()

    x_tensor = torch.tensor(prepared.dataset.x, dtype=torch.float32)
    y_tensor = torch.tensor(prepared.dataset.y, dtype=torch.float32)
    x_mask_tensor = torch.tensor(prepared.dataset.x_mask)
    theta = model.get_theta(x_tensor, y=y_tensor, x_mask=x_mask_tensor)
    if theta.ndim == 1:
        theta = theta[:, None]
    beta = model.get_beta()

    sobol_func = _build_sobol_func(model, prepared.attrs)
    sobol_result, var_y = get_sobol_indices(
        sobol_func, prepared.dataset, prepared.attrs
    )

    return TrainingOutput(
        args=args,
        results=results,
        model=model,
        theta=theta,
        beta=beta,
        sobol_result=sobol_result,
        var_y=var_y,
    )


def sample_topic_to_csv(theta, sample_ids, topic_order=None):
    df = sample_topic_frame(theta, sample_ids, topic_order=topic_order)
    return df.to_csv(index=True).encode("utf-8")


def topic_taxon_to_csv(beta, taxa, topic_order=None):
    df = topic_taxon_frame(beta, taxa, topic_order=topic_order)
    return df.to_csv(index=True).encode("utf-8")


def sample_topic_frame(theta, sample_ids, topic_order=None):
    """Return sample-topic proportions as a labeled table."""
    if topic_order is not None:
        theta = theta[:, topic_order]
    cols = [f"topic-{i + 1}" for i in range(theta.shape[1])]
    df = pd.DataFrame(theta, index=sample_ids, columns=cols)
    df.index.name = "sample_id"
    return df


def topic_taxon_frame(beta, taxa, topic_order=None):
    """Return topic-taxon loadings as a labeled table."""
    if topic_order is not None:
        beta = beta[topic_order, :]
    row_names = [f"topic-{i + 1}" for i in range(beta.shape[0])]
    df = pd.DataFrame(beta, index=row_names, columns=taxa)
    df.index.name = "topic"
    return df
