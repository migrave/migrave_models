from pathlib import Path
import pandas as pd
import seaborn as sns
import os
from os import path
import matplotlib
import ptitprince as pt

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from utils import MIGRAVE_PALETTE

# sns.set(style="darkgrid")
# sns.set(style="whitegrid")
# sns.set_style("white")
sns.set(style="whitegrid", font_scale=2)


def export_fig(fig, fig_name, output_dir):
    """
    Saves figure to directory. Creates directory if it does not exist.

    Args:
        fig: plt figure
        fig_name: name of figure including suffix (.png, .svg)
        output_dir: directory to save figure

    Returns:

    """
    if not path.isdir(output_dir):
        os.makedirs(output_dir)
    fig_dir = path.join(output_dir, fig_name)
    fig.savefig(fig_dir, bbox_inches='tight')


def raincloud_target_features(study_0: Path, study_1: Path, output_dir: Path, feature_names):
    all_dfs = []
    for data_1 in study_1.iterdir():
        if data_1.is_file():
            dataset = data_1.stem
            data_0 = learnstudy.joinpath(data_1.name)
            df_1 = pd.read_csv(data_1, index_col=0)[feature_names]
            df_0 = pd.read_csv(data_0, index_col=0)[feature_names]
            df_0["study"] = data_0.parent.parent.stem
            df_1["study"] = data_1.parent.parent.stem
            df_0["camera"] = dataset
            df_1["camera"] = dataset
            all_dfs.append(df_0)
            all_dfs.append(df_1)

    concat_df = pd.concat(all_dfs, ignore_index=True, join="inner")
    fig_dir = output_dir.joinpath("raincloud")
    fig_dir.mkdir(parents=True, exist_ok=True)

    dx = "camera"
    ort = "h"
    # pal = "Set2"
    pal = sns.color_palette([[v / 255 for v in c] for c in MIGRAVE_PALETTE])
    sigma = .2
    dhue = "study"

    for feature_name in feature_names:
        dy = feature_name
        fig, ax = plt.subplots(figsize=(15, 10))
        ax = pt.RainCloud(x=dx, y=dy, hue=dhue, data=concat_df, palette=pal, bw=sigma, width_viol=.6, ax=ax, orient=ort,
                          move=.2, alpha=.65, dodge=True, box_medianprops={"zorder": 11})
        if not path.isdir(fig_dir):
            os.makedirs(fig_dir)
        plt.title("")
        export_fig(fig, fig_name=f"{feature_name}.png", output_dir=fig_dir)
        export_fig(fig, fig_name=f"{feature_name}.svg", output_dir=fig_dir)
        plt.close(fig)


# # merge datasets from both studies for each camera
# fieldstudy = Path("/home/rfh/Documents/MigrAVE/Feldstudie/Learning_Data")
# learnstudy = Path("/home/rfh/Documents/MigrAVE/Lerndatenerhebung/Learning_Data")
# output = fieldstudy.joinpath("Feldstudie_Lerndatenerhebung")
# output.mkdir(parents=True, exist_ok=True)
#
# for field_data in fieldstudy.iterdir():
#     if field_data.is_file():
#         learn_data = learnstudy.joinpath(field_data.name)
#         output_data = output.joinpath(field_data.name)
#         field_df = pd.read_csv(field_data, index_col=0)
#         learn_df = pd.read_csv(learn_data, index_col=0)
#
#         field_df["participant"] = field_df["participant"] + 100
#         concat_df = pd.concat([learn_df, field_df], ignore_index=True, join="inner")
#         concat_df.to_csv(output_data)

# plot distribution of single feature between both studies and for each camera
fieldstudy = Path("/home/rfh/Documents/MigrAVE/Feldstudie/Learning_Data")
learnstudy = Path("/home/rfh/Documents/MigrAVE/Lerndatenerhebung/Learning_Data")
features = ["of_confidence", "of_success", "of_pose_distance", "op_person_n_col"]
output = Path("/home/rfh/Documents/MigrAVE/Feldstudie/")

raincloud_target_features(learnstudy, fieldstudy, output_dir=output, feature_names=features)
