# LGTM

Code for LGTM: Gaussian Process Modulated Neural Topic Modeling for Longitudinal Microbiome.

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

As a demo, first download the public HMP2 dataset, then open `hmp_gp.ipynb` for training and visualization.

```bash
chmod +x hmp_download.sh
./hmp_download.sh
```

## Notes

- `hmp_data.py`: Data preprocessing and exploration.
- `data.py`: Data loading and splitting.
- `model.py`: Model implementation.
- `train.py`: Training functions; can be used as a command-line interface.
- `config.py`: Model and training configuration.
- `tune.py`: Hyperparameter tuning (recommended on clusters).
- `utils.py`: Utility functions.
- Other files implement baseline methods used for comparison.
