
import marimo as mo

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

import matplotlib.pyplot as plt
import seaborn as sn

from dataset import OxfordPetsDataset


def plot_counts(df: pd.DataFrame):
    counts_race = df.groupby(["animal", "race"], observed=True)\
        .agg(count = pd.NamedAgg(column="identifier", aggfunc="count") )

    counts_animal = counts_race.groupby("animal", observed=True)\
        .agg(count = pd.NamedAgg(column="count", aggfunc="sum") )

    fig, (ax1, ax2) = plt.subplots(1, 2, width_ratios=[1, 3], figsize=(13,7))

    ax1 = sn.barplot(counts_animal, y="count", x="animal", orient="x", hue="animal", ax=ax1)
    for i in range(2) :
        ax1.bar_label(ax1.containers[i], fontsize=9)

    ax1.set_xlabel(ax1.get_xlabel(), fontsize=14)
    ax1.set_ylabel(ax1.get_ylabel(), fontsize=14)
    ax1.tick_params(axis='x', labelsize=12)
    ax1.tick_params(axis='y', labelsize=12)

    per_genre_order = counts_race.index.get_level_values(1).values
    ax2 = sn.barplot(counts_race, y="race", x="count", orient="y", hue="animal", order=per_genre_order, ax=ax2)
    ax2.set(xlim=(0, 250))
    for i in range(2) :
        labels_ = ax2.bar_label(ax2.containers[i], fontsize=9)

        for txt in labels_:
            if txt.get_text() != "200":
                txt.set_weight("bold")

    ax2.set_xlabel(ax2.get_xlabel(), fontsize=14)
    ax2.set_ylabel(ax2.get_ylabel(), fontsize=14)
    ax2.tick_params(axis='x', labelsize=12)
    ax2.tick_params(axis='y', labelsize=12)

    fig.tight_layout()
    fig.supxlabel("Fig X. Image count per species and race", y=-0.02, size=15.5, style="italic")
    plt.show()
    
    
    
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches

colors = ["red", "black", "grey", "white"]   # 0 = background (unused but matplotlib needs it)
cmap = ListedColormap(colors)                # Define colors for levels 0, 1, 2, 3
norm = BoundaryNorm([0, 1, 2, 3, 4], cmap.N) # Define boundaries: [0,1), [1,2), [2,3), [3,4)

patches = [
    mpatches.Patch(color="black", label="1: pet"),
    mpatches.Patch(color="grey",  label="2: background"),
    mpatches.Patch(facecolor="white", label="3: border", edgecolor='black', linewidth=1.0),
]

def plot_masks(
    dataset,
    idx: list[int],
    caption: str = "Some images along with their segmentation mask",
    max_images:int = 15
):

    N_COLS = 5
    N_IMAGES = min(len(idx), max_images)
    N_ROWS = N_IMAGES // N_COLS
    FOOTER_SIZE = 0.015

    fig = plt.figure(figsize=(2.5*N_COLS, 5.8*N_ROWS + FOOTER_SIZE))
    grid_outer = fig.add_gridspec(2, 1, height_ratios=[5*N_ROWS, FOOTER_SIZE])
    grid_inner = grid_outer[0].subgridspec(N_ROWS*2, N_COLS, wspace=0.05, hspace=0.05)

    for k in range(N_IMAGES):
        image, animal_id, race_id, mask = dataset[idx[k]].values()
        image = OxfordPetsDataset.default_transform_unnormalize(image)

        row = k // 5
        col = k % 5

        ax1 = fig.add_subplot(grid_inner[2*row, col])
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.axis('off')
        ax1.imshow(image)
        ax1.text(
            0.05, 0.95, # Coordinates within the subplot (0.0 to 1.0)
            str(idx[k]),
            transform=ax1.transAxes,
            color="white",
            fontsize=12,
            fontweight="bold",
            ha="left",
            va="top",
            bbox=dict(facecolor="black", alpha=0.7, edgecolor="none", pad=2)
        )


        ax2 = fig.add_subplot(grid_inner[2*row + 1, col])
        ax2.imshow(mask.squeeze(), cmap=cmap, norm=norm)
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax2.axis("off")

    footer = fig.add_subplot(grid_outer[1])
    footer.axis('off')

    footer.legend(
        handles=patches,
        loc='center',
        ncol=len(patches), # Display all patches in a single row
        frameon=False, # No box around the legend
        fontsize='large'
    )

    fig.supxlabel(caption, size=15.5, style="italic")
    # fig.subplots_adjust(hspace=0.05)
    fig.tight_layout(rect=[0, 0, 1, 1])
    plt.show()