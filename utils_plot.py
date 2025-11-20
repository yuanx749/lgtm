import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from scipy.special import softmax


def plot_si_heatmap(theta, sobol_result, sobol_agg, col_labels):
    colors = list(map(mpl.colors.to_hex, mpl.color_sequences["tab10"]))

    topic_props = np.mean(theta, axis=0)
    l_sorted = np.argsort(topic_props)[::-1]

    fig = plt.figure(figsize=(9, 5))
    gs = fig.add_gridspec(
        2,
        3,
        width_ratios=[0.6, 8, 0.5],
        height_ratios=[6, 0.8],
        hspace=0.15,
        wspace=0.15,
    )

    ax_stack = fig.add_subplot(gs[0, 0])
    ax_sobol = fig.add_subplot(gs[0, 1])
    ax_cbar = fig.add_subplot(gs[0, 2])
    ax_agg_left = fig.add_subplot(gs[1, 0])
    ax_agg_right = fig.add_subplot(gs[1, 1])

    topic_props_sorted = topic_props[l_sorted]
    topic_colors_sorted = [colors[i % len(colors)] for i in range(len(l_sorted))]

    n_topics = len(l_sorted)
    max_prop = float(np.max(topic_props_sorted)) if n_topics > 0 else 1.0
    for i, (prop, color) in enumerate(zip(topic_props_sorted, topic_colors_sorted)):
        y_bottom = (n_topics - 1 - i) / n_topics
        y_height = 1 / n_topics
        width = float(prop) / max_prop if max_prop > 0 else 0.0
        x_start = 1 - width
        rect = plt.Rectangle((x_start, y_bottom), width, y_height, facecolor=color)
        ax_stack.add_patch(rect)

    ax_stack.set_xlim(0, 1)
    ax_stack.set_ylim(0, 1)
    ax_stack.set_xticks([])
    ax_stack.set_yticks([])
    for spine in ax_stack.spines.values():
        spine.set_visible(False)

    sobol_ordered = sobol_result.total_order[l_sorted, :]
    m = ax_sobol.pcolormesh(sobol_ordered, cmap="Blues", vmin=0, vmax=1, shading="flat")
    ax_sobol.set_xlim(0, sobol_ordered.shape[1])
    ax_sobol.set_ylim(0, sobol_ordered.shape[0])
    ax_sobol.invert_yaxis()
    ax_sobol.set_xticks([])
    ax_sobol.set_xticklabels([])
    ax_sobol.set_yticks([])
    ax_sobol.set_yticklabels([])
    plt.colorbar(m, cax=ax_cbar)

    ax_agg_left.axis("off")
    sobol_agg_2d = np.asarray(sobol_agg).reshape(1, -1)
    ax_agg_right.pcolormesh(sobol_agg_2d, cmap="Blues", vmin=0, vmax=1, shading="flat")
    ax_agg_right.set_xlim(0, sobol_ordered.shape[1])
    ax_agg_right.set_ylim(0, 1)
    ax_agg_right.invert_yaxis()
    ax_agg_right.set_xticks(np.arange(sobol_ordered.shape[1]) + 0.5)
    if len(col_labels) > 0:
        ax_agg_right.set_xticklabels(
            col_labels, ha="right", rotation=45, rotation_mode="anchor"
        )
    ax_agg_right.set_yticks([])

    plt.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.15)

    return fig, (ax_stack, ax_sobol, ax_cbar, ax_agg_left, ax_agg_right)


def plot_topics(B, theta, taxa=None, top_k=5, threshold=0.03, num_plot=None):
    colors = list(map(mpl.colors.to_hex, mpl.color_sequences["tab10"]))
    if taxa is None:
        taxa = np.arange(B.shape[1])

    num_topics = B.shape[0]
    l_sorted = np.argsort(np.mean(theta, axis=0))[::-1]
    n_plot = num_topics if num_plot is None else min(num_plot, num_topics)

    fig, axes = plt.subplots(n_plot, 1, figsize=(2, 2 * n_plot), sharex=False)
    fig.subplots_adjust(wspace=0.1)
    if n_plot == 1:
        axes = [axes]

    for cnt, l in enumerate(l_sorted[:n_plot]):
        topic_weights = B[l]
        valid_indices = np.where(np.abs(topic_weights) > threshold)[0]
        if len(valid_indices) > 0:
            sorted_valid_indices = valid_indices[
                np.argsort(np.abs(topic_weights[valid_indices]))[::-1]
            ]
            top_indices = sorted_valid_indices[:top_k]
            top_k_taxa = [taxa[idx] for idx in top_indices]
            top_k_values = topic_weights[top_indices]
            axes[cnt].barh(top_k_taxa, top_k_values, color=colors[cnt % len(colors)])
        axes[cnt].invert_yaxis()
        axes[cnt].set_xlabel("")
        axes[cnt].set_ylabel("")

    return fig, axes


def plot_topic(
    B,
    theta,
    cnt,
    l,
    taxa=None,
    top_n=20,
    threshold=0.03,
    figsize=(24, 1),
):
    colors = list(map(mpl.colors.to_hex, mpl.color_sequences["tab10"]))
    if taxa is None:
        taxa = np.arange(B.shape[1])
    top_taxa_indices = np.unique(np.argsort(B, axis=1)[:, -top_n:])
    top_taxa = list(np.array(taxa)[top_taxa_indices])
    top_values = B[:, top_taxa_indices]

    top_values_for_topic = top_values[l]
    prop = np.mean(theta, axis=0)[l]

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(top_taxa, top_values_for_topic, color=colors[cnt % len(colors)])
    for bar, taxa_name in zip(bars, top_taxa):
        height = bar.get_height()
        if height > threshold:
            ax.annotate(
                taxa_name,
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="left",
                va="bottom",
                color="black",
                rotation=30,
                rotation_mode="anchor",
            )
    ax.set_xticks([])
    ax.set_xlabel("")
    ax.set_title(f"Topic {cnt + 1}\nProp={prop:.3f}", loc="left")
    return fig, ax


def plot_gp_id_subplot(
    ax,
    model,
    attrs,
    cnt,
    l,
    idx,
    plot_softmax=True,
):
    module = model.covariate_modules[idx]

    timepoints = getattr(attrs, "timepoints")
    x_num_dim = getattr(attrs, "n_covariates")
    ids = np.arange(getattr(attrs, "n_subjects"))
    x_points = torch.zeros(len(timepoints), x_num_dim)

    for id_ in ids:
        x_points[:, 1] = id_
        z_id, _, _ = module(x_points, stochastic_flag=False)
        z_id = z_id.squeeze().detach().cpu().numpy()
        if plot_softmax:
            z_id = softmax(z_id, axis=-1)
        ax.plot(timepoints, z_id[:, l])

    title = rf"$g_{{{cnt + 1}}}^{{({idx + 1})}}(\text{{id}})$"
    ax.set_title(title)


def plot_gp_se_subplot(
    ax,
    model,
    dataset,
    attrs,
    cnt,
    l,
    idx,
    plot_softmax=True,
    repeats=100,
    color=None,
):
    module = model.covariate_modules[idx]

    colors = list(map(mpl.colors.to_hex, mpl.color_sequences["tab10"]))
    if color is None:
        color = colors[0]

    timepoints = getattr(attrs, "timepoints")
    T = getattr(attrs, "n_steps")
    x_num_dim = getattr(attrs, "n_covariates")
    x_time = np.linspace(timepoints[0], timepoints[-1], num=10 * T)
    x_time_samples = np.tile(x_time, repeats)
    x_points = torch.zeros(T, x_num_dim)
    x_samples = torch.zeros(len(x_time), x_num_dim)

    x_points[:, 0] = torch.tensor(
        dataset.scaler.transform(timepoints[:, np.newaxis])
    ).squeeze()
    z_time, _, _ = module(x_points, stochastic_flag=False)
    z_time = z_time.squeeze().detach().cpu().numpy()
    if plot_softmax:
        z_time = softmax(z_time, axis=-1)
    ax.scatter(timepoints, z_time[:, l])

    x_samples[:, 0] = torch.tensor(
        dataset.scaler.transform(x_time[:, np.newaxis])
    ).squeeze()
    z_time, _, _ = module(x_samples, stochastic_flag=False)
    z_time = z_time.squeeze().detach().cpu().numpy()
    if plot_softmax:
        z_time = softmax(z_time, axis=-1)
    ax.plot(x_time, z_time[:, l])

    z_time_samples = []
    for _ in range(repeats):
        z_time, _, _ = module(x_samples.unsqueeze(1), stochastic_flag=True)
        z_time = z_time.squeeze().detach().cpu().numpy()
        if plot_softmax:
            z_time = softmax(z_time, axis=-1)
        z_time_samples.append(z_time)
    z_time_samples = np.concatenate(z_time_samples, axis=0)
    sns.lineplot(
        x=x_time_samples,
        y=z_time_samples[:, l],
        errorbar=("sd", 1),
        ax=ax,
        linestyle="",
        color=color,
    )

    title = rf"$g_{{{cnt + 1}}}^{{({idx + 1})}}(\text{{time}})$"
    ax.set_title(title)


def plot_gp_prod_subplot(
    ax,
    model,
    dataset,
    attrs,
    cnt,
    l,
    idx,
    plot_softmax=True,
    repeats=100,
    colors=None,
):
    module = model.covariate_modules[idx]

    if colors is None:
        colors = list(map(mpl.colors.to_hex, mpl.color_sequences["tab10"]))

    cat_covar = module.cat_covar
    cat_name = attrs.x_cols[cat_covar]
    labels = attrs.df[cat_name].cat.categories

    timepoints = getattr(attrs, "timepoints")
    T = getattr(attrs, "n_steps")
    x_num_dim = getattr(attrs, "n_covariates")
    x_time = np.linspace(timepoints[0], timepoints[-1], num=10 * T)
    x_time_samples = np.tile(x_time, repeats)
    x_points = torch.zeros(T, x_num_dim)
    x_samples = torch.zeros(len(x_time), x_num_dim)

    x_points[:, 0] = torch.tensor(
        dataset.scaler.transform(timepoints[:, np.newaxis])
    ).squeeze()
    x_samples[:, 0] = torch.tensor(
        dataset.scaler.transform(x_time[:, np.newaxis])
    ).squeeze()

    for code in range(len(labels)):
        x_points[:, cat_covar] = code
        z_time_cat, _, _ = module(x_points, stochastic_flag=False)
        z_time_cat = z_time_cat.squeeze().detach().cpu().numpy()
        if plot_softmax:
            z_time_cat = softmax(z_time_cat, axis=-1)
        ax.scatter(timepoints, z_time_cat[:, l], label=None, color=colors[code])

        x_samples[:, cat_covar] = code
        z_time_cat, _, _ = module(x_samples, stochastic_flag=False)
        z_time_cat = z_time_cat.squeeze().detach().cpu().numpy()
        if plot_softmax:
            z_time_cat = softmax(z_time_cat, axis=-1)
        ax.plot(x_time, z_time_cat[:, l], label=labels[code], color=colors[code])

        z_time_cat_samples = []
        for _ in range(repeats):
            z_time_cat, _, _ = module(x_samples.unsqueeze(1), stochastic_flag=True)
            z_time_cat = z_time_cat.squeeze().detach().cpu().numpy()
            if plot_softmax:
                z_time_cat = softmax(z_time_cat, axis=-1)
            z_time_cat_samples.append(z_time_cat)
        z_time_cat_samples = np.concatenate(z_time_cat_samples, axis=0)
        sns.lineplot(
            x=x_time_samples,
            y=z_time_cat_samples[:, l],
            errorbar=("sd", 1),
            ax=ax,
            linestyle="",
            color=colors[code],
        )

    title = (
        rf"$g_{{{cnt + 1}}}^{{({idx + 1})}}(\text{{time}}\times\text{{{cat_name}}})$"
    )
    ax.set_title(title)
    ax.legend()


def plot_gp_ca_subplot(
    ax,
    model,
    attrs,
    cnt,
    l,
    idx,
    plot_softmax=True,
    colors=None,
):
    module = model.covariate_modules[idx]

    if colors is None:
        colors = list(map(mpl.colors.to_hex, mpl.color_sequences["tab10"]))

    timepoints = getattr(attrs, "timepoints")
    T = getattr(attrs, "n_steps")
    x_num_dim = getattr(attrs, "n_covariates")
    x_points = torch.zeros(T, x_num_dim)

    cat_covar = module.index
    cat_name = attrs.x_cols[cat_covar]
    labels = attrs.df[cat_name].cat.categories

    for code in range(len(labels)):
        x_points[:, cat_covar] = code
        z_cat, _, _ = module(x_points, stochastic_flag=False)
        z_cat = z_cat.squeeze().detach().cpu().numpy()
        if plot_softmax:
            z_cat = softmax(z_cat, axis=-1)
        ax.plot(timepoints, z_cat[:, l], label=labels[code], color=colors[code])

    title = rf"$g_{{{cnt + 1}}}^{{({idx + 1})}}(\text{{{cat_name}}})$"
    ax.set_title(title)
    ax.legend()


def plot_gp(
    model,
    dataset,
    attrs,
    cnt,
    l,
    sobol_result=None,
    plot_softmax=True,
    repeats=100,
    figsize=(24, 3),
):
    colors = list(map(mpl.colors.to_hex, mpl.color_sequences["tab10"]))

    n_plot = len(model.covariate_modules)
    fig, ax = plt.subplots(
        1,
        n_plot,
        sharex=True,
        sharey=True,
        figsize=figsize,
        constrained_layout=True,
    )
    if n_plot == 1:
        ax = [ax]
    if plot_softmax and n_plot > 0:
        ax[0].set_ylim(-0.05, 1.05)

    for idx, module in enumerate(model.covariate_modules):
        if module.covar_type == "ID":
            id_si_idx = model.id_idx[0] if getattr(model, "id_idx", []) else None
            plot_gp_id_subplot(
                ax[idx], model, attrs, cnt, l, idx, plot_softmax=plot_softmax
            )
            if sobol_result is not None and id_si_idx is not None:
                try:
                    si = sobol_result.total_order[l, id_si_idx]
                    title = ax[idx].get_title()
                    ax[idx].set_title(f"{title}\nSI={si:.3f}")
                except Exception:
                    pass
        elif module.covar_type == "SE":
            se_si_idx = model.se_idx[0] if getattr(model, "se_idx", []) else None
            plot_gp_se_subplot(
                ax[idx],
                model,
                dataset,
                attrs,
                cnt,
                l,
                idx,
                plot_softmax=plot_softmax,
                repeats=repeats,
                color=colors[0],
            )
            if sobol_result is not None and se_si_idx is not None:
                try:
                    si = sobol_result.total_order[l, se_si_idx]
                    title = ax[idx].get_title()
                    ax[idx].set_title(f"{title}\nSI={si:.3f}")
                except Exception:
                    pass
        elif module.covar_type == "PROD":
            plot_gp_prod_subplot(
                ax[idx],
                model,
                dataset,
                attrs,
                cnt,
                l,
                idx,
                plot_softmax=plot_softmax,
                repeats=repeats,
                colors=colors,
            )
            if sobol_result is not None:
                try:
                    si = sobol_result.total_order[l, module.cat_covar]
                    title = ax[idx].get_title()
                    ax[idx].set_title(f"{title}\nSI={si:.3f}")
                except Exception:
                    pass
        elif module.covar_type == "CA":
            plot_gp_ca_subplot(
                ax[idx],
                model,
                attrs,
                cnt,
                l,
                idx,
                plot_softmax=plot_softmax,
                colors=colors,
            )
            if sobol_result is not None:
                try:
                    si = sobol_result.total_order[l, module.index]
                    title = ax[idx].get_title()
                    ax[idx].set_title(f"{title}\nSI={si:.3f}")
                except Exception:
                    pass
        elif module.covar_type == "BIN":
            pass

    return fig, ax


def plot_latent_se_subplot(
    ax,
    model,
    attrs,
    theta,
    l,
    idx,
):
    np.random.seed(0)
    sns.stripplot(data=attrs.df, x="time", y=theta[:, l], alpha=0.2, ax=ax)
    ax.set_xticklabels([])
    ax.set_xlabel("")


def plot_latent_prod_subplot(
    ax,
    model,
    attrs,
    theta,
    l,
    idx,
):
    module = model.covariate_modules[idx]
    cat_covar = module.cat_covar
    cat_name = attrs.x_cols[cat_covar]
    np.random.seed(0)
    sns.stripplot(
        data=attrs.df,
        x="time",
        y=theta[:, l],
        hue=cat_name,
        alpha=0.2,
        legend=False,
        ax=ax,
    )
    ax.set_xticklabels([])
    ax.set_xlabel("")


def plot_latent_ca_subplot(
    ax,
    model,
    attrs,
    theta,
    l,
    idx,
):
    module = model.covariate_modules[idx]
    cat_covar = module.index
    cat_name = attrs.x_cols[cat_covar]
    np.random.seed(0)
    sns.stripplot(
        data=attrs.df,
        x="time",
        y=theta[:, l],
        hue=cat_name,
        alpha=0.2,
        legend=False,
        ax=ax,
    )
    ax.set_xticklabels([])
    ax.set_xlabel("")


def plot_latent(model, attrs, theta, l, plot_softmax=True, figsize=(24, 3)):
    n_plot = len(model.covariate_modules)
    fig, ax = plt.subplots(
        1,
        n_plot,
        sharex=True,
        sharey=True,
        figsize=figsize,
        constrained_layout=True,
    )
    if n_plot == 1:
        ax = [ax]
    if plot_softmax and n_plot > 0:
        ax[0].set_ylim(-0.05, 1.05)

    for idx, module in enumerate(model.covariate_modules):
        if module.covar_type in ["ID", "BIN"]:
            continue
        if module.covar_type == "SE":
            plot_latent_se_subplot(ax[idx], model, attrs, theta, l, idx)
        elif module.covar_type == "PROD":
            plot_latent_prod_subplot(ax[idx], model, attrs, theta, l, idx)
        elif module.covar_type == "CA":
            plot_latent_ca_subplot(ax[idx], model, attrs, theta, l, idx)

    return fig, ax
