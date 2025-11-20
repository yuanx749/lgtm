import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class NonNegativeLinear(nn.Module):

    def __init__(self, in_features, out_features, normalize_weight=False, eta=1):
        super().__init__()
        self.eta = eta
        if normalize_weight:
            self.act = nn.Softmax(dim=0)
        else:
            self.act = F.relu
        self.std = np.sqrt(1 / eta * (1 - 1 / out_features))
        self.weight = nn.Parameter(
            torch.normal(0, self.std, size=(out_features, in_features))
        )

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
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.y_num_dim = y_num_dim
        self.non_negative = non_negative
        self.normalize_weight = normalize_weight
        self.eta = eta
        if non_negative:
            self.linear = NonNegativeLinear(
                latent_dim, y_num_dim, normalize_weight, eta
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

    def __init__(
        self,
        latent_dim,
        y_num_dim,
        hidden_dim,
        p_drop=0.0,
        batchnorm=False,
        alpha=1,
    ):
        super().__init__()
        self.batchnorm = batchnorm
        self.latent_dim = latent_dim
        self.y_num_dim = y_num_dim
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(y_num_dim, hidden_dim)
        self.dropout1 = nn.Dropout(p_drop)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        if batchnorm:
            self.mean_bn = nn.BatchNorm1d(latent_dim, affine=False)
            self.logvar_bn = nn.BatchNorm1d(latent_dim, affine=False)

        k = latent_dim
        a = self.alpha = alpha * np.ones((1, k)).astype(np.float32)
        self.mu1 = nn.Parameter(torch.as_tensor((np.log(a).T - np.mean(np.log(a), 1)).T))
        self.var1 = nn.Parameter(
            torch.as_tensor(
                (((1 / a) * (1 - (2 / k))).T + (1 / k**2) * np.sum(1 / a, 1)).T
            )
        )
        self.mu1.requires_grad = False
        self.var1.requires_grad = False

    def encode(self, y):
        h = self.fc1(y)
        h = F.relu(h)
        h = self.dropout1(h)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        if self.batchnorm:
            mu = self.mean_bn(mu)
            logvar = self.logvar_bn(logvar)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, y):
        mu, logvar = self.encode(y)
        if self.training:
            z = self.reparameterize(mu, logvar)
        else:
            z = mu
        return z, mu, logvar

    def loss(self, mu, logvar):
        kld = -0.5 * torch.sum(1 + logvar - mu**2 - logvar.exp(), dim=-1).mean()
        return kld

    def loss2(self, mu, logvar):
        var = logvar.exp()
        var_div = var / self.var1
        diff = mu - self.mu1
        diff_term = diff * diff / self.var1
        logvar_div = self.var1.log() - logvar
        kld = 0.5 * ((var_div + diff_term + logvar_div).sum(dim=-1) - self.latent_dim).mean()
        return kld


class SimpleVAE(nn.Module):

    def __init__(
        self,
        input_dim,
        latent_dim,
        hidden_dim,
        linear_decoded=True,
        non_negative=True,
        normalize_latent=True,
        normalize_weight=True,
        batchnorm=False,
        p_drop=0.0,
        alpha=1,
        eta=1,
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.linear_decoded = linear_decoded
        self.non_negative = non_negative
        self.normalize_latent = normalize_latent
        self.normalize_weight = normalize_weight
        self.batchnorm = batchnorm
        if linear_decoded:
            self.decoder = LinearDecoder(
                latent_dim,
                input_dim,
                non_negative,
                normalize_weight,
                eta,
            )
        else:
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, input_dim),
            )
        self.encoder = MLPEncoder(
            latent_dim,
            input_dim,
            hidden_dim,
            p_drop,
            batchnorm,
            alpha,
        )

    def encode(self, y):
        z, mu, logvar = self.encoder(y)
        return z, mu, logvar

    def decode(self, z):
        return self.decoder(z)

    def forward(self, y):
        z, mu, logvar = self.encode(y)
        if self.normalize_latent:
            theta = F.softmax(z, dim=-1)
        else:
            theta = z
        logits = self.decode(theta)
        if self.normalize_latent and self.normalize_weight:
            recon_y = logits
        else:
            recon_y = F.softmax(logits, dim=-1)
        return logits, recon_y, mu, logvar, theta

    def vae_loss(self, logits, recon_y, mu, logvar, y, b=1):
        mse_loss = nn.functional.mse_loss(recon_y, y, reduction="none")
        mse = mse_loss.mean(dim=1).mean()
        if self.normalize_latent and self.normalize_weight:
            cce = -torch.sum(torch.log(recon_y) * y, dim=-1).mean()
        else:
            cce = nn.functional.cross_entropy(logits, y, reduction="mean")
        kld = self.encoder.loss2(mu, logvar)
        loss = cce + b * kld
        return loss, mse, cce, kld
