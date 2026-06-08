# LGTM

Code for LGTM: Gaussian Process Modulated Neural Topic Modeling for Longitudinal Microbiome.

Live demo: https://lgtm-web.streamlit.app/

## Installation

Linux is recommended.

- Clone this repository and enter the project folder.
- Install the package with [uv](https://docs.astral.sh/uv/):

    ```bash
    uv venv .venv
    source .venv/bin/activate
    uv pip install -e .
    ```

- For development, create and activate the environment with [Mamba](https://mamba.readthedocs.io/en/latest/):

    ```bash
    mamba env create -p ./env -f env-dev.yml
    mamba activate ./env
    ```

- For experiments with other methods, use `env-exp.yml`.

## Usage

As a demo, first download the public HMP2 dataset, then open `examples/hmp_gp.ipynb` for training and visualization.

```bash
chmod +x scripts/hmp_download.sh
./scripts/hmp_download.sh
```

Python API:

```python
import pandas as pd
from lgtm import LGTM, LGTMConfig

metadata = pd.read_csv("metadata.csv")
microbiome = pd.read_csv("microbiome.csv")

config = LGTMConfig(
    latent_dim=6,
    n_epoch=100,
    batch_size=64,
    learning_rate=0.05,
    seed=42,
)
model = LGTM(config).fit(metadata, microbiome)

sample_topic = model.sample_topic_
topic_taxon = model.topic_taxon_

fig, axes = model.plot_si()
fig, axes = model.plot_topics()
fig, axes = model.plot_gp(topic=1)
```
