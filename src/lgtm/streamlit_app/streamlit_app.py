import io
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from lgtm.utils_plot import (
    plot_gp,
    plot_latent,
    plot_si_heatmap,
    plot_topic,
    plot_topics,
)
from lgtm.api.backend import (
    REQUIRED_METADATA_COLUMNS,
    prepare_data,
    read_table,
    sample_topic_to_csv,
    topic_taxon_to_csv,
    train_and_analyze,
)


def _fig_to_image_bytes(fig, fmt="svg", dpi=None):
    buf = io.BytesIO()
    savefig_kwargs = {"format": fmt, "bbox_inches": "tight"}
    if dpi is not None:
        savefig_kwargs["dpi"] = dpi
    fig.savefig(buf, **savefig_kwargs)
    buf.seek(0)
    data = buf.getvalue()
    buf.close()
    return data


def _build_overview_artifacts(output, prepared):
    artifacts = {}
    l_sorted = np.argsort(np.mean(output.theta, axis=0))[::-1]

    fig_si, _ = plot_si_heatmap(
        output.theta,
        output.sobol_result,
        output.var_y,
        prepared.attrs.x_cols,
    )
    artifacts["si.svg"] = _fig_to_image_bytes(fig_si)
    plt.close(fig_si)

    fig_topics, _ = plot_topics(
        output.beta,
        output.theta,
        taxa=prepared.taxa,
        top_k=5,
        threshold=0.01,
        num_plot=None,
    )
    artifacts["topics.svg"] = _fig_to_image_bytes(fig_topics)
    plt.close(fig_topics)

    return artifacts, l_sorted


def _build_gp_artifact(output, prepared, cnt, l):
    fig_gp, _ = plot_gp(
        output.model,
        prepared.dataset,
        prepared.attrs,
        cnt=cnt,
        l=l,
        sobol_result=output.sobol_result,
        repeats=5,
        figsize=(18, 3),
        plot_id=False,
    )
    gp_svg = _fig_to_image_bytes(fig_gp)
    plt.close(fig_gp)
    return gp_svg


def _build_all_plot_artifacts(output, prepared, topic_order):
    artifacts, _ = _build_overview_artifacts(output, prepared)

    for cnt, l in enumerate(topic_order):
        l_id = cnt + 1

        fig_topic, _ = plot_topic(
            output.beta,
            output.theta,
            cnt=cnt,
            l=l,
            taxa=prepared.taxa,
            top_n=20,
            threshold=0.01,
            figsize=(18, 2),
        )
        artifacts[f"topic-{l_id}.svg"] = _fig_to_image_bytes(fig_topic)
        plt.close(fig_topic)

        artifacts[f"gp-{l_id}.svg"] = _build_gp_artifact(output, prepared, cnt, l)

        fig_latent, _ = plot_latent(
            output.model,
            prepared.attrs,
            output.theta,
            l=l,
            figsize=(18, 3),
            plot_pointplot=False,
            plot_id=False,
        )
        artifacts[f"latent-{l_id}.svg"] = _fig_to_image_bytes(fig_latent)
        plt.close(fig_latent)

    return artifacts


def _render_overview(run_state=None):
    st.subheader("Overview")
    top_left, top_right = st.columns([2, 1])
    with top_left:
        st.markdown("#### SI Overview")
        si_placeholder = st.empty()
    with top_right:
        st.markdown("#### Topics Overview")
        topics_placeholder = st.empty()

    if run_state is not None and "overview_artifacts" in run_state:
        overview_artifacts = run_state["overview_artifacts"]
        si_placeholder.image(
            overview_artifacts["si.svg"].decode("utf-8"), width="stretch"
        )
        topics_placeholder.image(
            overview_artifacts["topics.svg"].decode("utf-8"),
            width="stretch",
        )

    if run_state is None:
        st.selectbox(
            "Select topic",
            [1],
            disabled=True,
            key="select_topic_placeholder",
        )
        st.markdown("#### Latent functions of topic")
        st.empty()
        return

    topic_order = run_state["topic_order"]
    gp_options = list(range(1, len(topic_order) + 1))
    topic_id = st.selectbox(
        "Select topic",
        gp_options,
    )
    gp_title = f"Latent functions of topic {topic_id}"
    st.markdown(f"#### {gp_title}")
    gp_placeholder = st.empty()
    if topic_id not in run_state["gp_preview_artifacts"]:
        with st.spinner(f"Rendering {gp_title}..."):
            l = topic_order[topic_id - 1]
            run_state["gp_preview_artifacts"][topic_id] = _build_gp_artifact(
                run_state["output"],
                run_state["prepared"],
                topic_id - 1,
                l,
            )
    if topic_id in run_state["gp_preview_artifacts"]:
        gp_placeholder.image(
            run_state["gp_preview_artifacts"][topic_id].decode("utf-8"),
            width="stretch",
        )


def _build_bundle_zip(sample_topic_bytes, topic_taxon_bytes, artifacts):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("sample-topic.csv", sample_topic_bytes)
        zf.writestr("topic-taxon.csv", topic_taxon_bytes)
        for filename, image_bytes in artifacts.items():
            zf.writestr(filename, image_bytes)
    buf.seek(0)
    return buf.getvalue()


def _render_downloads(run_state):
    st.subheader("Downloads")
    output = run_state["output"]
    prepared = run_state["prepared"]

    topic_order = run_state["topic_order"]
    sample_topic_bytes = sample_topic_to_csv(
        output.theta,
        prepared.sample_ids,
        topic_order=topic_order,
    )
    topic_taxon_bytes = topic_taxon_to_csv(
        output.beta,
        prepared.taxa,
        topic_order=topic_order,
    )

    col1, col2 = st.columns(2)
    col1.download_button(
        label="Download sample-topic.csv",
        data=sample_topic_bytes,
        file_name="sample-topic.csv",
        mime="text/csv",
    )
    col2.download_button(
        label="Download topic-taxon.csv",
        data=topic_taxon_bytes,
        file_name="topic-taxon.csv",
        mime="text/csv",
    )
    if "bundle_bytes" not in run_state:
        if st.button("Prepare results.zip"):
            with st.spinner("Preparing full results.zip..."):
                all_artifacts = _build_all_plot_artifacts(output, prepared, topic_order)
                run_state["bundle_bytes"] = _build_bundle_zip(
                    sample_topic_bytes, topic_taxon_bytes, all_artifacts
                )
            st.success("results.zip is ready.")
    if "bundle_bytes" in run_state:
        st.download_button(
            label="Download results.zip",
            data=run_state["bundle_bytes"],
            file_name="results.zip",
            mime="application/zip",
        )


def _render_result_block(run_state):
    st.success("Training completed.")
    render_status = st.empty()
    if "overview_artifacts" not in run_state:
        render_status.info("Rendering...")
        with st.spinner("Rendering..."):
            overview_artifacts, topic_order = _build_overview_artifacts(
                run_state["output"],
                run_state["prepared"],
            )
            run_state["overview_artifacts"] = overview_artifacts
            run_state["topic_order"] = topic_order
            run_state["gp_preview_artifacts"] = {}
        render_status.success("Rendering finished.")
    else:
        render_status.info("Rendering finished.")
    if "topic_order" not in run_state:
        run_state["topic_order"] = np.argsort(
            np.mean(run_state["output"].theta, axis=0)
        )[::-1]
    if "gp_preview_artifacts" not in run_state:
        run_state["gp_preview_artifacts"] = {}
    _render_overview(run_state)
    _render_downloads(run_state)


def main():
    st.set_page_config(page_title="LGTM Web App", layout="wide")
    st.title("LGTM Web App")
    st.caption(
        "Upload metadata and microbiome profile, train once, view figures, and download matrices."
    )

    with st.expander("Input requirements", expanded=False):
        required_cols_text = ", ".join(f"`{col}`" for col in REQUIRED_METADATA_COLUMNS)
        st.write(
            f"1. Metadata file: CSV/TSV with required columns: {required_cols_text}."
        )
        st.write(
            "2. Microbiome profile file: CSV/TSV with one row per sample. First column must be `sample_id` and remaining columns are taxa abundance."
        )

    metadata_file = st.file_uploader(
        "Metadata file (.csv/.tsv)", type=["csv", "tsv"], key="meta"
    )
    microbiome_file = st.file_uploader(
        "Microbiome profile file (.csv/.tsv)",
        type=["csv", "tsv"],
        key="micro",
    )
    latent_dim = st.number_input(
        "Latent dimension", min_value=2, max_value=20, value=6, step=1
    )

    run_clicked = st.button("Train model", type="primary")
    if run_clicked:
        if metadata_file is None or microbiome_file is None:
            st.error("Please upload both metadata and microbiome files.")
        else:
            with st.spinner("Reading files and training model..."):
                try:
                    metadata = read_table(metadata_file, metadata_file.name)
                    microbiome = read_table(microbiome_file, microbiome_file.name)
                    prepared = prepare_data(metadata, microbiome)
                    output = train_and_analyze(prepared, int(latent_dim))
                    st.session_state["run_state"] = {
                        "prepared": prepared,
                        "output": output,
                    }
                except Exception as exc:
                    st.exception(exc)

    if "run_state" in st.session_state:
        try:
            _render_result_block(st.session_state["run_state"])
        except Exception as exc:
            st.exception(exc)
    else:
        _render_overview()


if __name__ == "__main__":
    main()
