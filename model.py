# Built upon https://github.com/YigitBalik/DGBFGP

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class NonNegativeLinear(nn.Module):

    def __init__(
        self, in_features, out_features, normalize_weight=False, eta=1, b_init=None
    ):
        super().__init__()
        self.eta = eta
        if normalize_weight:
            self.act = nn.Softmax(dim=0)
        else:
            self.act = F.relu
        if b_init is None:
            std = np.sqrt(1.0 / eta * (1 - 1 / out_features))
            self.weight = nn.Parameter(
                torch.normal(0, std, size=(out_features, in_features))
            )
        else:
            self.weight = nn.Parameter(torch.from_numpy(b_init.T))

    def forward(self, x):
        weight_non_neg = self.act(self.weight)
        return F.linear(x, weight_non_neg)


class LinearDecoder(nn.Module):

    def __init__(
        self,
        latent_dim,
        y_num_dim,
        non_negative=False,
        normalize_weight=False,
        eta=1,
        b_init=None,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.y_num_dim = y_num_dim
        self.non_negative = non_negative
        self.normalize_weight = normalize_weight
        self.eta = eta
        if non_negative:
            self.linear = NonNegativeLinear(
                latent_dim, y_num_dim, normalize_weight, eta, b_init
            )
        else:
            self.linear = nn.Linear(latent_dim, y_num_dim, bias=False)

    def forward(self, z):
        y = self.linear(z)
        return y

    @torch.inference_mode()
    def get_loadings(self):
        loadings = self.linear.weight
        if self.non_negative:
            loadings = self.linear.act(loadings)
        if not self.normalize_weight:
            loadings = F.softmax(loadings, dim=0)
        return loadings.T.detach().cpu().numpy()


class MLPEncoder(nn.Module):
    def __init__(self, latent_dim, y_num_dim, hidden_dim, p_drop=0.0):
        super().__init__()
        self.latent_dim = latent_dim
        self.y_num_dim = y_num_dim
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(y_num_dim, hidden_dim)
        self.dropout1 = nn.Dropout(p_drop)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)

    def encode(self, y):
        h = self.fc1(y)
        h = F.relu(h)
        h = self.dropout1(h)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, y, stochastic_flag=True, k=1):
        mu, logvar = self.encode(y)
        mu = mu.unsqueeze(1).repeat(1, k, 1)
        logvar = logvar.unsqueeze(1).repeat(1, k, 1)
        if stochastic_flag:
            z = self.reparameterize(mu, logvar)
        else:
            z = mu
        return z, mu, logvar


class BayesianLinear(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(BayesianLinear, self).__init__()

        self.input_dim = in_dim
        self.output_dim = out_dim

        self.mu_weights = nn.Parameter(torch.zeros(self.input_dim, self.output_dim))
        self.rho_weights = nn.Parameter(
            torch.Tensor(self.input_dim, self.output_dim).uniform_(-4, -2)
        )

    def forward(self, x, stochastic_flag=True, k=1):
        if stochastic_flag:
            sigma_weights = torch.log1p(torch.exp(self.rho_weights))
            epsilon_weights = torch.randn(
                k, self.input_dim, self.output_dim, device=x.device
            )

            weights = self.mu_weights.unsqueeze(0)
            weights = weights + epsilon_weights * sigma_weights.unsqueeze(0)
            output = torch.einsum("bkm, kml -> bkl", x, weights)

        else:
            weights = self.mu_weights.unsqueeze(0)
            output = torch.matmul(x, weights)

        return output, weights

    def loss(self, sigma):
        eps = 1e-6
        sigma_weights = torch.log1p(torch.exp(self.rho_weights))
        sigma = torch.permute(sigma, (1, 0))
        a = torch.log((sigma + eps) / (sigma_weights + eps))
        b = (sigma_weights**2 + self.mu_weights**2) / (2 * sigma**2 + eps)
        KL_weights = torch.sum(-0.5 + a + b) / (self.input_dim * self.output_dim)
        KL = KL_weights
        return KL

    def log_prior(self, A_sample, sigma):
        sigma = sigma.unsqueeze(-1)
        log_p_A = (
            -torch.log(sigma) - 0.5 * np.log(2 * np.pi) - 0.5 * A_sample**2 / sigma**2
        )
        return torch.sum(log_p_A, dim=[1, 2]) / (self.input_dim * self.output_dim)

    def log_posterior(self, A_sample):
        sigma_weights = torch.log(1 + torch.exp(self.rho_weights))
        mu_weights = self.mu_weights
        log_q_A = (
            -torch.log(sigma_weights)
            - 0.5 * np.log(2 * np.pi)
            - 0.5 * (A_sample - mu_weights) ** 2 / sigma_weights**2
        )
        return torch.sum(log_q_A, dim=[1, 2]) / (self.input_dim * self.output_dim)

class MLPDecoder(nn.Module):
    def __init__(self, latent_dim, y_num_dim, hidden_dim, p_drop=0.0):
        super(MLPDecoder, self).__init__()
        self.latent_dim = latent_dim
        self.y_num_dim = y_num_dim
        self.hidden_dim = hidden_dim

        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.dropout1 = nn.Dropout(p_drop)
        self.fc2 = nn.Linear(hidden_dim, y_num_dim)

    def forward(self, z):
        h = self.fc1(z)
        h = F.relu(h)
        h = self.dropout1(h)
        y = self.fc2(h)
        return y


class OneHotEncoder(nn.Module):
    def __init__(self, dim):
        super(OneHotEncoder, self).__init__()
        self.dim = dim

    def forward(self, x):
        x = x.long()
        x = F.one_hot(x, self.dim).float()
        return x


def create_C_matrix_cs(num_cat, rho):
    return (1 - rho) * np.eye(num_cat) + rho * np.ones((num_cat, num_cat))


def decompose_cs(n_cat, rho):
    K = create_C_matrix_cs(n_cat, rho)
    evals, evecs = np.linalg.eigh(K)
    return evals.astype(np.float32), evecs.astype(np.float32)


class BasisFunction(nn.Module):
    def __init__(
        self,
        M,
        basis_func: str = "regff",
        type="SE",
        scale=1.0,
        alpha=1.0,
        alpha_fixed: bool = False,
        scale_fixed: bool = False,
        C=0,
        dim=1,
        **kwargs,
    ):
        super(BasisFunction, self).__init__()
        self.M = M
        self.type = type
        self.basis_func = basis_func

        se_scales = float(scale)
        se_alphas = float(alpha)
        self.dim = dim

        if type == "SE" or type == "PROD":

            self.register_buffer("scale_prior_mean", torch.tensor([0.0]))
            self.register_buffer("scale_prior_std", torch.tensor([1.0]))
            self.scale_posterior_mean = nn.Parameter(torch.tensor([se_scales] * dim))
            self.scale_posterior_log_std = nn.Parameter(torch.tensor([np.log(1)] * dim))

            self.register_buffer("alpha_prior_mean", torch.tensor([0.0]))
            self.register_buffer("alpha_prior_std", torch.tensor([1.0]))
            self.alpha_posterior_mean = nn.Parameter(torch.tensor([se_alphas] * dim))
            self.alpha_posterior_log_std = nn.Parameter(torch.tensor([np.log(1)] * dim))

            if alpha_fixed:
                self.alpha_posterior_log_std.requires_grad_(False)
                self.alpha_posterior_mean.requires_grad_(False)

            if scale_fixed:
                self.scale_posterior_mean.requires_grad_(False)
                self.scale_posterior_log_std.requires_grad_(False)

            if type == "PROD":
                self.C = C
                cat_eigval, cat_eigvec = decompose_cs(C, 0)  # diagonal kernel
                self.register_buffer("cat_eigval", torch.tensor(cat_eigval))
                self.register_buffer("cat_eigvec", torch.tensor(cat_eigvec))

        if basis_func == "regff" or basis_func == "hs":
            self.register_buffer("omega", torch.arange(0, self.M).unsqueeze(0))
            self.J = float(2.55)
        else:
            pass

    def forward(self, x, v=None, stochastic_flag=True):
        densities = torch.ones((self.dim, self.M), device=x.device)
        if self.type == "SE" or self.type == "PROD":
            scale_posterior_std = torch.exp(self.scale_posterior_log_std)
            alpha_posterior_std = torch.exp(self.alpha_posterior_log_std)
            if stochastic_flag:
                if self.scale_posterior_mean.requires_grad:
                    scales = (
                        torch.randn(self.dim, device=x.device) * scale_posterior_std
                        + self.scale_posterior_mean
                    )
                    scales = torch.exp(scales)
                else:
                    scales = torch.exp(
                        self.scale_posterior_mean + scale_posterior_std**2 / 2
                    )
                if self.alpha_posterior_log_std.requires_grad:
                    alphas = (
                        torch.randn(self.dim, device=x.device) * alpha_posterior_std
                        + self.alpha_posterior_mean
                    )
                    alphas = torch.exp(alphas)
                else:
                    alphas = torch.exp(
                        self.alpha_posterior_mean + alpha_posterior_std**2 / 2
                    )
            else:
                scales = torch.exp(
                    self.scale_posterior_mean + scale_posterior_std**2 / 2
                )
                alphas = torch.exp(
                    self.alpha_posterior_mean + alpha_posterior_std**2 / 2
                )
            omega = self.omega.repeat(self.dim, 1)
            alphas = alphas.view(-1, 1)
            scales = scales.view(-1, 1)
            if self.basis_func == "regff":
                densities = (
                    alphas**2
                    * scales
                    * np.sqrt(2 * torch.pi)
                    * torch.exp(-(scales**2) * omega**2 / 2)
                )
            elif self.basis_func == "hs":
                sqrt_lambdas = torch.pi * omega / (2 * self.J)
                densities = (
                    alphas**2
                    * scales
                    * np.sqrt(2 * torch.pi)
                    * torch.exp(-(scales**2) * sqrt_lambdas**2 / 2)
                )
        elif self.type == "BIN":
            phi_x = torch.zeros(x.shape[0], self.M, device=x.device)
            phi_x[:, 0] = (1 - x).squeeze(-1)
            phi_x[:, 1] = x.squeeze(-1)
            phi_x = 2 * phi_x - 1
            phi_x = phi_x.unsqueeze(1)
            densities = torch.ones((self.dim, self.M), device=x.device)
            return phi_x, densities
        elif self.type == "ID":
            phi_x = x
            densities = torch.ones((self.dim, self.M), device=x.device)
            return phi_x, densities
        elif self.type == "CA":
            x = x.long()
            x = torch.where(x == -1, self.M - 1, x)
            phi_x = F.one_hot(x, self.M).float()
            densities = torch.ones((self.dim, self.M), device=x.device)
            return phi_x, densities

        if self.basis_func == "regff":
            wx = x.unsqueeze(-1) * self.omega
            phi_x = torch.concat([torch.cos(wx), torch.sin(wx)], dim=-1)
            densities = torch.cat([densities, densities], dim=-1)
        elif self.basis_func == "hs":
            phi_x = (
                1
                / np.sqrt(self.J)
                * torch.sin(
                    torch.pi * self.omega * (x.unsqueeze(-1) + self.J) / (2 * self.J)
                )
            )

        if self.type == "PROD":
            vec = self.cat_eigvec[v.int()]
            phi_x = (phi_x.unsqueeze(-1) * vec.unsqueeze(-2)).view(
                *phi_x.shape[:-1], densities.shape[-1] * self.C
            )
            densities = (
                densities.unsqueeze(-1) * self.cat_eigval.view(1, 1, self.C)
            ).view(self.dim, densities.shape[-1] * self.C)

        densities = densities + 1e-5
        return phi_x, densities


class CovariateModule(nn.Module):
    def __init__(self, covar_info: dict):
        super(CovariateModule, self).__init__()
        self.covar_type = covar_info["type"]
        self.basis_func = covar_info["basis"]
        self.A = covar_info["A"]

        if self.covar_type in ["SE", "CA", "BIN"]:
            self.index = covar_info["index"]
        if self.covar_type == "ID":
            self.index = covar_info["index"]
            self.embed_model = covar_info["embed_model"]
        elif self.covar_type == "PROD":
            self.cts_covar = covar_info["cts_covar"]
            self.cat_covar = covar_info["cat_covar"]
            self.interaction_index = covar_info["interaction_index"]

    def forward(self, x, y=None, stochastic_flag=True, k=1):

        if self.covar_type in ["SE", "CA", "BIN"]:
            input_x = x[..., self.index]
            phi_x, densities_c = self.basis_func(
                input_x, stochastic_flag=stochastic_flag
            )
            gp_sample, A_sample = self.A(phi_x, stochastic_flag, k)
        elif self.covar_type == "ID":
            input_x = x[..., self.index]
            x_id = self.embed_model(input_x)
            phi_x, densities_c = self.basis_func(x_id)
            gp_sample, A_sample = self.A(phi_x, stochastic_flag, k)
        elif self.covar_type == "PROD":
            cts_covar = self.cts_covar
            cat_covar = self.cat_covar
            x_cts = x[..., cts_covar]
            x_cat = x[..., cat_covar]
            phi_x, densities_c = self.basis_func(
                x_cts, x_cat, stochastic_flag=stochastic_flag
            )
            gp_sample, A_sample = self.A(phi_x, stochastic_flag, k)

        return gp_sample, densities_c, A_sample

    def loss(self, sigma):
        KL = self.A.loss(sigma)
        if (
            self.covar_type in ["SE", "PROD"]
            and self.basis_func.scale_posterior_mean.requires_grad
        ):
            prior_mean = self.basis_func.scale_prior_mean
            prior_std = self.basis_func.scale_prior_std
            posterior_mean = self.basis_func.scale_posterior_mean
            posterior_std = torch.exp(self.basis_func.scale_posterior_log_std)
            KL_l = torch.sum(
                ((posterior_mean - prior_mean) ** 2 + posterior_std**2 - prior_std**2)
                / (2 * prior_std**2)
                + torch.log(prior_std / posterior_std)
            )
            KL = KL + KL_l
        if (
            self.covar_type in ["SE", "PROD"]
            and self.basis_func.alpha_posterior_log_std.requires_grad
        ):
            prior_mean = self.basis_func.alpha_prior_mean
            prior_std = self.basis_func.alpha_prior_std
            posterior_mean = self.basis_func.alpha_posterior_mean
            posterior_std = torch.exp(self.basis_func.alpha_posterior_log_std)
            KL_a = torch.sum(
                ((posterior_mean - prior_mean) ** 2 + posterior_std**2 - prior_std**2)
                / (2 * prior_std**2)
                + torch.log(prior_std / posterior_std)
            )
            KL = KL + KL_a
        return KL

    def log_prior(self, A_sample, sigma):
        log_p_A = self.A.log_prior(A_sample, sigma)
        if (
            self.covar_type in ["SE", "PROD"]
            and self.basis_func.scale_posterior_mean.requires_grad
        ):
            prior_mean = self.basis_func.scale_prior_mean
            prior_std = self.basis_func.scale_prior_std
            scales = self.basis_func.scales
            log_p_scale = (
                -torch.log(scales)
                - torch.log(prior_std)
                - 0.5 * np.log(2 * np.pi)
                - 0.5 * (torch.log(scales) - prior_mean) ** 2 / prior_std**2
            )
            log_p_A = log_p_A + torch.sum(log_p_scale, dim=1)
        if (
            self.covar_type in ["SE", "PROD"]
            and self.basis_func.alpha_posterior_log_std.requires_grad
        ):
            prior_mean = self.basis_func.alpha_prior_mean
            prior_std = self.basis_func.alpha_prior_std
            alphas = self.basis_func.alphas
            log_p_alpha = (
                -torch.log(alphas)
                - torch.log(prior_std)
                - 0.5 * np.log(2 * np.pi)
                - 0.5 * (torch.log(alphas) - prior_mean) ** 2 / prior_std**2
            )
            log_p_A = log_p_A + torch.sum(log_p_alpha, dim=1)
        return log_p_A

    def log_posterior(self, A_sample):
        log_q_A = self.A.log_posterior(A_sample)
        if (
            self.covar_type in ["SE", "PROD"]
            and self.basis_func.scale_posterior_mean.requires_grad
        ):
            posterior_mean = self.basis_func.scale_posterior_mean
            posterior_std = torch.exp(self.basis_func.scale_posterior_log_std)
            scales = self.basis_func.scales
            log_q_scale = (
                -torch.log(scales)
                - torch.log(posterior_std)
                - 0.5 * np.log(2 * np.pi)
                - 0.5 * (torch.log(scales) - posterior_mean) ** 2 / posterior_std**2
            )
            log_q_A = log_q_A + torch.sum(log_q_scale, dim=1)
        if (
            self.covar_type in ["SE", "PROD"]
            and self.basis_func.alpha_posterior_log_std.requires_grad
        ):
            posterior_mean = self.basis_func.alpha_posterior_mean
            posterior_std = torch.exp(self.basis_func.alpha_posterior_log_std)
            alphas = self.basis_func.alphas
            log_q_alpha = (
                -torch.log(alphas)
                - torch.log(posterior_std)
                - 0.5 * np.log(2 * np.pi)
                - 0.5 * (torch.log(alphas) - posterior_mean) ** 2 / posterior_std**2
            )
            log_q_A = log_q_A + torch.sum(log_q_alpha, dim=1)
        return log_q_A

class DGBFGP(nn.Module):

    def __init__(
        self,
        y_num_dim,
        x_num_dim,
        latent_dim,
        hidden_dim,
        P,
        id_embed_dim,
        id_handler,
        M,
        C,
        id_covariate,
        se_idx,
        ca_idx,
        bin_idx,
        interactions,
        basis_func="hs",
        scale=0.2,
        alpha=1.0,
        alpha_fixed=False,
        scale_fixed=False,
        vy_init=1.0,
        vy_fixed=False,
        p_drop=0.0,
        k=1,
        linear_decoded=False,
        non_negative=False,
        normalize_latent=None,
        normalize_weight=False,
        encode_y=False,
        **kwargs,
    ):
        super(DGBFGP, self).__init__()
        assert basis_func in [
            "regff",
            "hs",
        ]  # only support regular Fourier features and Hilbert space embeddings
        self.y_num_dim = y_num_dim
        self.x_num_dim = x_num_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.id_covariate = id_covariate
        self.linear_decoded = linear_decoded
        self.non_negative = non_negative
        self.normalize_latent = normalize_latent
        self.normalize_weight = normalize_weight
        self.encode_y = encode_y

        id_idx = []
        if id_handler == "onehot":
            id_embed_dim = P
            embed_model = OneHotEncoder(P)
            id_idx = [id_covariate]

        self.id_idx = id_idx
        self.id_handler = id_handler
        self.se_idx = se_idx
        self.ca_idx = ca_idx
        self.bin_idx = bin_idx
        self.interactions = interactions

        if isinstance(M, int):
            M = np.array([M] * self.x_num_dim)

        if ca_idx:
            M[ca_idx] = C

        M[bin_idx] = 2

        self.covariate_modules = nn.ModuleList()
        self.covar_mask_idx = []
        basis_func_dim = 1
        for i in id_idx + se_idx + ca_idx + bin_idx:
            covar_info = {"index": i}
            if i in se_idx:
                covar_info.update(
                    {
                        "type": "SE",
                        "basis": BasisFunction(
                            M[i],
                            basis_func,
                            "SE",
                            scale,
                            alpha,
                            alpha_fixed,
                            scale_fixed,
                            dim=basis_func_dim,
                            **kwargs,
                        ),
                        "A": BayesianLinear(
                            M[i] if basis_func == "hs" else 2 * M[i],
                            latent_dim,
                        ),
                    }
                )
            elif i in ca_idx:
                covar_info.update(
                    {
                        "type": "CA",
                        "basis": BasisFunction(
                            M[i],
                            basis_func,
                            "CA",
                            scale,
                            alpha,
                            alpha_fixed,
                            scale_fixed,
                            dim=basis_func_dim,
                            **kwargs,
                        ),
                        "A": BayesianLinear(
                            M[i] if basis_func == "hs" else 2 * M[i],
                            latent_dim,
                        ),
                    }
                )
            elif i in bin_idx:
                covar_info.update(
                    {
                        "type": "BIN",
                        "basis": BasisFunction(
                            M[i],
                            basis_func,
                            "BIN",
                            scale,
                            alpha,
                            alpha_fixed,
                            scale_fixed,
                            dim=basis_func_dim,
                            **kwargs,
                        ),
                        "A": BayesianLinear(M[i], latent_dim),
                    }
                )
            elif i in id_idx:
                covar_info.update(
                    {
                        "type": "ID",
                        "basis": BasisFunction(
                            id_embed_dim,
                            basis_func,
                            "ID",
                            scale,
                            alpha,
                            alpha_fixed,
                            scale_fixed,
                            dim=basis_func_dim,
                            **kwargs,
                        ),
                        "A": BayesianLinear(id_embed_dim, latent_dim),
                        "embed_model": embed_model,
                    }
                )
            self.covariate_modules.append(CovariateModule(covar_info))
            self.covar_mask_idx.append(i)

        for i, interaction in enumerate(interactions):
            cts_covar = interaction[0]
            cat_covar = interaction[1]
            interaction_info = {
                "type": "PROD",
                "basis": BasisFunction(
                    M[cts_covar],
                    basis_func,
                    "PROD",
                    scale,
                    alpha,
                    alpha_fixed,
                    scale_fixed,
                    C[i],
                    dim=basis_func_dim,
                    **kwargs,
                ),
                "A": BayesianLinear(
                    (
                        M[cts_covar] * C[i]
                        if basis_func == "hs"
                        else 2 * M[cts_covar] * C[i]
                    ),
                    latent_dim,
                ),
                "cts_covar": cts_covar,
                "cat_covar": cat_covar,
                "interaction_index": i,
            }
            self.covariate_modules.append(CovariateModule(interaction_info))
            self.covar_mask_idx.append(cat_covar)

        self.num_component = len(self.covariate_modules)

        self.p_drop = p_drop

        if linear_decoded:
            self.decoder = LinearDecoder(
                self.latent_dim,
                self.y_num_dim,
                non_negative,
                normalize_weight,
                kwargs.get("eta", 1),
                kwargs.get("b_init", None),
            )
        else:
            self.decoder = MLPDecoder(
                self.latent_dim,
                self.y_num_dim,
                self.hidden_dim,
                self.p_drop,
            )

        if self.encode_y:
            self.encoder = MLPEncoder(
                self.latent_dim,
                self.y_num_dim,
                self.hidden_dim,
                self.p_drop,
            )

        min_log_vy = torch.Tensor([-8.0])

        log_vy_init = torch.log(vy_init - torch.exp(min_log_vy))
        # log variance
        if isinstance(vy_init, float):
            self._log_vy = nn.Parameter(torch.Tensor(y_num_dim * [log_vy_init]))
        else:
            self._log_vy = nn.Parameter(torch.Tensor(log_vy_init))

        if vy_fixed:
            self._log_vy.requires_grad_(False)

        self.register_buffer("min_log_vy", min_log_vy * torch.ones(1))

        self.k = self.k_init = k

    @property
    def vy(self):
        log_vy = self.min_log_vy + F.softplus(self._log_vy - self.min_log_vy)
        return torch.exp(log_vy)

    @vy.setter
    def vy(self, vy):
        assert (
            torch.min(torch.tensor(vy)) >= 0.0005
        ), "Smallest allowed value for vy is 0.0005"
        with torch.no_grad():
            self._log_vy.copy_(torch.log(vy - torch.exp(self.min_log_vy)))

    def encode(self, x, y=None, stochastic_flag=True):
        f_lst = []
        densities = []
        A_samples = []
        for covar_module in self.covariate_modules:
            f_c, densities_c, A_c = covar_module(x, y, stochastic_flag, self.k)
            f_lst.append(f_c)
            densities.append(densities_c)
            A_samples.append(A_c)

        return f_lst, densities, A_samples

    def decode(self, z):
        y = self.decoder(z)
        return y

    def forward(self, x, y=None, x_mask=None, stochastic_flag=True):
        x_repeat = x.unsqueeze(1).repeat(1, self.k, 1)
        f_lst, densities, A_samples = self.encode(
            x_repeat, stochastic_flag=stochastic_flag
        )
        f_x = torch.stack(f_lst, dim=-1)  # B x K x L x R
        if x_mask is None:
            x_mask = torch.ones_like(x, dtype=torch.uint8)
        f_mask = x_mask[:, self.covar_mask_idx]

        if self.normalize_latent == "sum":
            f_mask = f_mask.view(-1, 1, 1, self.num_component)
            z_x = torch.sum(f_x * f_mask, dim=-1)
            theta_x = F.softmax(z_x, dim=-1)
            log_theta_x = F.log_softmax(z_x, dim=-1)
        elif self.normalize_latent == "mean":
            g_x = F.softmax(f_x, dim=-2)
            f_mask = f_mask.view(-1, 1, 1, self.num_component)
            f_weight = f_mask / torch.sum(f_mask, dim=-1, keepdim=True)
            theta_x = torch.sum(g_x * f_weight, dim=-1)
            log_theta_x = torch.log(theta_x)
        else:
            f_mask = f_mask.view(-1, 1, 1, self.num_component)
            z_x = torch.sum(f_x * f_mask, dim=-1)
            theta_x = z_x
            log_theta_x = torch.log(theta_x)
        logits_x = self.decode(theta_x)
        if self.normalize_latent and self.normalize_weight:
            pred_y = logits_x
        else:
            pred_y = F.softmax(logits_x, dim=-1)

        if self.encode_y and y is not None:
            z_y, _, _ = self.encoder(y, stochastic_flag, self.k)
            if self.normalize_latent:
                theta_y = F.softmax(z_y, dim=-1)
                log_theta_y = F.log_softmax(z_y, dim=-1)
            else:
                theta_y = z_y
            logits_y = self.decode(theta_y)
            if self.normalize_latent and self.normalize_weight:
                recon_y = logits_y
            else:
                recon_y = F.softmax(logits_y, dim=-1)
        else:
            logits_y = recon_y = log_theta_y = None

        return (
            logits_x,
            pred_y,
            log_theta_x,
            densities,
            A_samples,
            logits_y,
            recon_y,
            log_theta_y,
            f_x,
        )

    def pred_loss_clr(self, logits_x, y_clr, y_mask):
        logits_x = logits_x.view(-1, self.y_num_dim)
        y_clr = torch.repeat_interleave(y_clr, self.k, 0)
        y_mask = torch.repeat_interleave(y_mask, self.k, 0)

        y_ori = F.softmax(y_clr, dim=-1)
        cce = nn.functional.cross_entropy(logits_x, y_ori, reduction="none")

        loss = nn.MSELoss(reduction="none")
        se = torch.mul(loss(logits_x, y_clr), y_mask)
        mask_sum = torch.sum(y_mask, dim=1)
        mask_sum[mask_sum == 0] = 1
        mse = torch.sum(se, dim=1) / mask_sum

        return mse, cce

    def pred_loss(self, logits_x, pred_y, y, y_mask):
        logits_x = logits_x.view(-1, self.y_num_dim)
        pred_y = pred_y.view(-1, self.y_num_dim)
        y = torch.repeat_interleave(y, self.k, 0)
        y_mask = torch.repeat_interleave(y_mask, self.k, 0)

        if self.normalize_latent and self.normalize_weight:
            cce = -torch.sum(torch.log(pred_y) * y, dim=-1)
        else:
            cce = nn.functional.cross_entropy(logits_x, y, reduction="none")

        loss = nn.MSELoss(reduction="none")
        se = torch.mul(loss(pred_y, y), y_mask)
        mask_sum = torch.sum(y_mask, dim=1)
        mask_sum[mask_sum == 0] = 1
        mse = torch.sum(se, dim=1) / mask_sum

        return mse, cce

    def klx_loss(self, densities):
        kls = []
        for c, m in enumerate(self.covariate_modules):
            sigma_bar = torch.sqrt(densities[c])
            kls.append(m.loss(sigma_bar))
        if kls:
            KL = torch.stack(kls).sum()
        else:
            KL = torch.zeros(1, device=self.min_log_vy.device)
        return KL

    def recon_loss(self, logits_y, recon_y, y, y_mask):
        logits_y = logits_y.view(-1, self.y_num_dim)
        recon_y = recon_y.view(-1, self.y_num_dim)
        y = torch.repeat_interleave(y, self.k, 0)
        y_mask = torch.repeat_interleave(y_mask, self.k, 0)

        if self.normalize_latent and self.normalize_weight:
            cce = -torch.sum(torch.log(recon_y) * y, dim=-1)
        else:
            cce = nn.functional.cross_entropy(logits_y, y, reduction="none")

        loss = nn.MSELoss(reduction="none")
        se = torch.mul(loss(recon_y, y), y_mask)
        mask_sum = torch.sum(y_mask, dim=1)
        mask_sum[mask_sum == 0] = 1
        mse = torch.sum(se, dim=1) / mask_sum
        return mse, cce

    def kl_loss_qy_px(self, log_theta_x, log_theta_y):
        log_theta_x = log_theta_x.view(-1, self.latent_dim)
        log_theta_y = log_theta_y.view(-1, self.latent_dim)
        kl_qy_px = F.kl_div(
            log_theta_x,
            log_theta_y,
            reduction="batchmean",
            log_target=True,
        )
        return kl_qy_px

    def train(self, mode=True):
        super().train(mode)
        if mode:
            self.k = self.k_init
        else:
            self.k = 1

    def print_model_size(self):
        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(num_params)

    def get_beta(self):
        if not self.linear_decoded:
            raise ValueError("Not applicable.")
        return self.decoder.get_loadings()

    def get_theta(self, x, y=None, x_mask=None):
        (
            logits_x,
            pred_y,
            log_theta_x,
            densities,
            A_samples,
            logits_y,
            recon_y,
            log_theta_y,
            f_x,
        ) = self.forward(x, y=y, x_mask=x_mask, stochastic_flag=False)
        theta = log_theta_x.exp().squeeze().detach().cpu().numpy()
        if log_theta_y is not None:
            theta = log_theta_y.exp().squeeze().detach().cpu().numpy()
        return theta
