# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# %%
folder = Path("data/hmp")
metadata_file = folder / "hmp2_metadata_2018-08-20.csv"
metadata = pd.read_csv(metadata_file, low_memory=False)
df_meta = metadata.loc[
    metadata["data_type"] == "metagenomics",
    [
        "Participant ID",
        "External ID",
        "week_num",
        "diagnosis",
        "Diet soft drinks, tea or coffee with sugar (Stevia, Equal, Splenda etc)",
        "Yogurt or other foods containing active bacterial cultures (kefir, sauerkraut)",
        "Dairy (milk, cream, ice cream, cheese, cream cheese)",
        "Probiotic",
        "Vegetables (salad, tomatoes, onions, greens, carrots, peppers, green beans, etc)",
        "Beans (tofu, soy, soy burgers, lentils, Mexican beans, lima beans etc)",
        "Whole grains (wheat, oats, brown rice, rye, quinoa, wheat bread, wheat pasta)",
        "Red meat (beef, hamburger, pork, lamb)",
        "Fish (fish nuggets, breaded fish, fish cakes, salmon, tuna, etc.)",
        "Sweets (pies, jam, chocolate, cake, cookies, etc.)",
    ],
]
column_rename_dict = {
    "Participant ID": "Participant ID",
    "External ID": "External ID",
    "week_num": "week_num",
    "diagnosis": "diagnosis",
    "Diet soft drinks, tea or coffee with sugar (Stevia, Equal, Splenda etc)": "diet_drinks",
    "Yogurt or other foods containing active bacterial cultures (kefir, sauerkraut)": "yogurt",
    "Dairy (milk, cream, ice cream, cheese, cream cheese)": "dairy",
    "Probiotic": "probiotic",
    "Vegetables (salad, tomatoes, onions, greens, carrots, peppers, green beans, etc)": "vegetables",
    "Beans (tofu, soy, soy burgers, lentils, Mexican beans, lima beans etc)": "beans",
    "Whole grains (wheat, oats, brown rice, rye, quinoa, wheat bread, wheat pasta)": "whole_grains",
    "Red meat (beef, hamburger, pork, lamb)": "red_meat",
    "Fish (fish nuggets, breaded fish, fish cakes, salmon, tuna, etc.)": "fish",
    "Sweets (pies, jam, chocolate, cake, cookies, etc.)": "sweets",
}
df_meta = df_meta.rename(columns=column_rename_dict)

# %%
diet_columns = [
    "diet_drinks",
    "yogurt",
    "dairy",
    "probiotic",
    "vegetables",
    "beans",
    "whole_grains",
    "red_meat",
    "fish",
    "sweets",
]
df_meta[diet_columns] = df_meta[diet_columns].map(
    lambda x: "No" if "No" in str(x) else "Yes", na_action="ignore"
)

# %%
abundance_file = folder / "taxonomic_profiles_3.tsv"
abundance = pd.read_csv(abundance_file, sep="\t")
abundance = abundance.set_index("Feature\Sample").T
abundance.index = abundance.index.str.removesuffix("_profile")
species_abundance = abundance.loc[
    abundance["UNKNOWN"] == 0,
    abundance.columns.str.contains(r"s__[a-zA-Z0-9_]+$", regex=True),
]
df_ra = species_abundance.div(species_abundance.sum(axis=1), axis=0)
df_ra = df_ra.rename_axis("External ID", axis=0)
y_ori = df_ra.to_numpy()
y_ori_sample_ids = df_ra.index.to_numpy(copy=True)

# %%
dysbiosis_file = folder / "dysbiosis_scores.tsv"
df_dysbiosis = pd.read_csv(dysbiosis_file, sep="\t", header=None)
df_dysbiosis.columns = ["External ID", "dysbiosis_score", "dysbiosis"]

# %%
df = pd.merge(
    df_meta,
    df_ra,
    on="External ID",
)
df = pd.merge(df, df_dysbiosis, on="External ID", how="left")
df["time"] = df["week_num"]
df["subjectID"] = df["Participant ID"]

# %%
df = df[df["time"].isin(range(0, 47))]
df = df.drop_duplicates(subset=["subjectID", "time"])
df = df.groupby("subjectID").filter(lambda x: len(x) > 2)
df = df.set_index(["subjectID", "time"])
df = df.sort_index()
df = df.reset_index()
subjects = df["subjectID"].unique()
timepoints = df["time"].unique()
timepoints.sort()

# %%
df = df.set_index(["subjectID", "time"])
df["idx"] = range(len(df))
sample_ids = df["External ID"].to_numpy(copy=True)
samples = df.index
mi = pd.MultiIndex.from_product([subjects, timepoints], names=["subjectID", "time"])
df_full = df.reindex(mi)
df = df.reset_index()

# %%
n_samples = len(samples)
n_subjects = len(subjects)
n_steps = len(timepoints)
idx_2d = df_full["idx"].to_numpy().reshape(n_subjects, n_steps)
features = np.array([s.split("|")[-1] for s in df_ra.columns])

# %%
n_features = len(df_ra.columns)
df_y = df_full[df_ra.columns]
y_3d = df_y.to_numpy().reshape(n_subjects, n_steps, n_features)
y = df_y.loc[samples].to_numpy()

# %%
num_cols = ["time"]
x_num = df[num_cols].to_numpy()
cat_cols = [
    "subjectID",
    "diagnosis",
    "dysbiosis",
    "diet_drinks",
    "yogurt",
    "dairy",
    "probiotic",
    "vegetables",
    "beans",
    "whole_grains",
    "red_meat",
    "fish",
    "sweets",
]
df[cat_cols] = df[cat_cols].astype("category")
n_cat_lst = [len(df[cat_name].cat.categories) for cat_name in cat_cols]
x_cat = df[cat_cols].apply(lambda x: x.cat.codes).to_numpy()
x = np.hstack((x_num, x_cat))
x_cols = num_cols + cat_cols
n_covariates = x.shape[1]

# %%
se_idx = [x_cols.index(col) for col in num_cols]
id_covariate = x_cols.index("subjectID") if "subjectID" in x_cols else None
interactions = []
C = []
if num_cols:
    time_idx = x_cols.index("time") if "time" in x_cols else se_idx[0]
    for i, cat_col in enumerate(cat_cols):
        cat_idx = len(num_cols) + i
        if id_covariate is not None and cat_idx == id_covariate:
            continue
        n_cat = n_cat_lst[i] + int(df[cat_col].isna().any())
        interactions.append([time_idx, cat_idx])
        C.append(n_cat)
ca_idx = []
bin_idx = []

# %%
__all__ = [
    "df",
    "df_full",
    "df_y",
    "y",
    "y_ori",
    "y_ori_sample_ids",
    "y_3d",
    "samples",
    "sample_ids",
    "subjects",
    "timepoints",
    "n_samples",
    "n_subjects",
    "n_steps",
    "idx_2d",
    "features",
    "n_features",
    "x",
    "x_num",
    "x_cat",
    "x_cols",
    "n_covariates",
    "n_cat_lst",
    "se_idx",
    "ca_idx",
    "bin_idx",
    "interactions",
    "C",
    "id_covariate",
]
