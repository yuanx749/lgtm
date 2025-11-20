class Args:
    id_embed_dim = 10
    M = 5
    se_idx = [0]
    ca_idx = []
    bin_idx = []
    interactions = list(zip([0] * 12, range(2, 14)))
    C = [3] + [2] * 11
    id_covariate = 1
    id_handler = "onehot"
    k = 1
    b = 0.001
    loss_function = "cce"
    n_epoch = 100
    patience = 0
    lr = 0.05
    batch_size = 64
    latent_dim = 6
    hidden_dim = 64
    normalize_latent = "mean"
    linear_decoded = True
    non_negative = True
    normalize_weight = True
    encode_y = True
    init_b = True
    transform = None
    seed = 42
    fold = 0
    val_split = False
    early_stop = True
    cohort = "hmp"
    task = "dr"
    method = "gp"
    save_pred = False
    save_model = False


args = Args()
