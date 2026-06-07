# LGTM

Code for LGTM: Gaussian Process Modulated Neural Topic Modeling for Longitudinal Microbiome.

Live demo: https://lgtm-web.streamlit.app/

## Installation

Linux is recommended.

- Clone this repository and enter the project folder.
- Create and activate the environment with [Mamba](https://mamba.readthedocs.io/en/latest/):

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
