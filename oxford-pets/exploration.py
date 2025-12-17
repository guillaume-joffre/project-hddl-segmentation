import marimo

__generated_with = "0.18.4"
app = marimo.App()


@app.cell
def _(mo):
    mo.md(r"""
    # <span style="font-size:0.5em">High-Dimensional-Deep-Learning :<br></span> **Mini-Projet 1/3: Chats ou Chiens ?**
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ### Introduction

    L'objectif de ce notebook est d'effectuer une analyse préalable des données. Le jeu de données utilisé pour ce mini-projet est le Oxford-IIIT Pet Dataset, composé de photographies de chats et de chiens appartenant à 37 races différentes. Le jeu de données est disponible à l’adresse suivante :
    [www.robots.ox.ac.uk/~vgg/data/pets](www.robots.ox.ac.uk/~vgg/data/pets)

    le but de cette analyse est de répondre aux questions suivantes posées par le sujet:
    - Quelle est la répartition par espèce et par race dans le jeu de donnée.
    - Les masques de segmentations associés aux images sont-ils bons/cohérents.
    - Y-a t'il eventuellement des déséquilibres ou biais visuels.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ### 1. Imports et fonctions utilitaires
    """)
    return


@app.cell
def _():
    ### --- Imports principaux ----------------------------------- ###
    import marimo as mo

    import numpy as np
    import pandas as pd

    import torch
    from torch.utils.data import Dataset, DataLoader
    from PIL import Image

    import matplotlib.pyplot as plt
    import seaborn as sn

    from dataset import OxfordPetsDataset
    return OxfordPetsDataset, mo, np, pd, plt, sn, torch


@app.cell
def _():
    ### --- Fonctions pour l'affichage --------------------------- ###
    from display_tables import summary, summary_categorical
    from display_plots import  plot_counts, plot_masks
    return plot_counts, plot_masks, summary, summary_categorical


@app.cell
def _(pd, sn):
    ### --- Options d'affichage ---------------------------------- ###
    # pandas
    pd.set_option("display.max_columns", None)
    pd.set_option('display.max_colwidth', None)

    # seaborn
    sn.set_style("darkgrid")
    sn.set_palette("colorblind")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ### 2. Répartition des images entre espèces et races
    """)
    return


@app.cell
def _(OxfordPetsDataset):
    ### --- Récupération du Dataset ------------------------------ ###
    data = OxfordPetsDataset(
        fetch_masks = True,
        dataset_root = "../data/oxford-pets"
    )

    print(f"""Le dataset \"data\" contient {len(data)} membres, avec:
    - les images de chats/chiens
    - leurs labels espèce/race 
    - les masques de segmentation (arrière-plan, bordure, animal) associés
    """)
    return (data,)


@app.cell
def _(data):
    ### --- Informations sur le Dataset -------------------------- ###
    dframe__ = data.get_dataframe()
    dframe__.sample(5, random_state=53)
    return (dframe__,)


@app.cell
def _(mo):
    mo.md(r"""
    Toutes les images du Oxford-IIIT Pet Dataset ont un masque de segmentation associé, mais toutes n'ont pas un fichier describant une bounding box pour la tête de l'animal. C'est pourquoi la colonne "bboxfile" à des valeurs manquantes. Comme il s'agit d'un mini-projet portant sur la segmentation et non pas sur la detection d'objet nous allons ignorer la colonne et revenir à un dataset sans valeurs manquantes.
    """)
    return


@app.cell
def _(dframe__, summary):
    ### --- Détails du DataFrame --------------------------------- ###
    dframe = dframe__.drop(columns=["bboxfile"])
    del dframe__
    summary(dframe)
    return (dframe,)


@app.cell
def _(dframe, summary_categorical):
    ### --- Détails du DataFrame --------------------------------- ###
    summary_categorical(dframe["animal"])
    return


@app.cell
def _(dframe, summary_categorical):
    ### --- Détails du DataFrame --------------------------------- ###
    summary_categorical(dframe["race"])
    return


@app.cell
def _(dframe, plot_counts):
    ### --- Animal / Race bar plots ------------------------------ ###
    plot_counts(dframe)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ### 3. Coherence des masques de segmentation
    """)
    return


@app.cell
def _(data, dframe, np, plot_masks):
    ### --- Animal / Race bar plots ------------------------------ ###
    plot_masks(data, np.random.choice(dframe.index, size=5, replace=False))
    return


@app.cell
def _(dframe, mo, np, pd, torch):
    ### --- Detection d'anomalies pour les masques --------------- ###

    from torchvision.io import read_image

    def count_mask_levels():
        # might work with a One-hot trick for fast (batched?) bincount:       
        # oh = torch.nn.functional.one_hot(masks.long(), num_classes=4) # oh: (B, 1, H, W, 4)
        # counts[...] = oh.sum(dim=[1,2,3]).cpu().numpy() # Sum over spatial dims: (B, 4)

        counts = torch.zeros((len(dframe), 3), dtype=torch.int64)
        for row in dframe.itertuples():
            mask = read_image(row.maskfile)  # H, W - uint8
            counts[row.Index] = torch.bincount(mask.flatten(), minlength=4)[1:4]  # index 0 unused

        return pd.DataFrame(counts, columns=["mask_1","mask_2","mask_3"])

    levels = count_mask_levels()
    levels = levels.div(levels.sum(axis=1).replace(0, np.nan), axis=0)

    mo.ui.table(levels.iloc[130:140], selection=None)

    return (levels,)


@app.cell
def _(levels, plt, sn):
    sn.violinplot(data=levels)
    plt.title("Distribution of mask level ratios")
    plt.ylabel("Ratio")
    plt.show()

    sn.boxplot(data=levels)
    plt.title("Distribution of mask level ratios")
    plt.ylabel("Ratio")
    plt.show()
    return


@app.cell
def _(data, levels, plot_masks):
    mask_1_high = levels[levels["mask_1"] > 0.9].index.values
    plot_masks(data, mask_1_high)
    print(f"There are {len(mask_1_high)} images with more than 90% level 1 in their segmentation mask (big pets)")
    return


@app.cell
def _(data, levels, plot_masks):
    mask_2_full = levels[levels["mask_2"] > 0.99].index.values
    plot_masks(data, mask_2_full, max_images = 10)
    print(f"There are {len(mask_2_full)} images with more than 99% level 2 in their segmentation masks (only background)")
    return


@app.cell
def _(data, levels, plot_masks):
    mask_2_quasi = levels[(levels["mask_2"] <= 0.99) & (levels["mask_2"] > 0.95)].index.values
    plot_masks(data, mask_2_quasi, max_images = 10)
    print(f"There are {len(mask_2_quasi)} non empty (less than 99% of level 2) images with more than 95% level 2 in their segmentation masks (tiny pets)")
    return


@app.cell
def _(data, levels, plot_masks):
    mask_3_high = levels[levels["mask_3"] > 0.32].index.values
    plot_masks(data, mask_3_high, max_images = 15)
    print(f"There are {len(mask_3_high)} images with more than 32% level 3 in their segmentation masks (big borders / unknown zones)")
    return


@app.cell
def _(torch):

    from torchvision.io import decode_image
    import torch.nn.functional as F

    def touches_count(mask: torch.Tensor, crop:int = 0) -> int:

        PET = 1
        BACKGROUND = 2
        BORDER = 3

        mask_background = (mask == BACKGROUND).unsqueeze(0).float()
        kernel = torch.ones(1, 1, 3, 3, device=mask.device)
        proximity = F.conv2d(
            F.pad(mask_background, (1, 1, 1, 1), mode='constant', value=0), 
            kernel
        )

        proximity_mask = (proximity > 0).squeeze().float() # Squeeze back to HxW
        mask_pet = (mask == PET).float()

        # This matrix is 1.0 at every LEVEL_1 pixel that is adjacent to LEVEL_2
        touches_mask = mask_pet * proximity_mask
        touches_mask = touches_mask.squeeze()

        # Optionnally cropping the edge since there is quite a few errors here
        if crop > 0:
            touches_mask = touches_mask[crop:-crop, crop:-crop]        

        total_touches = touches_mask.sum().item()
        return int(total_touches)

    # ID = 3033
    # mask_t = decode_image(dframe.iloc[ID]["maskfile"])  # H, W - uint8
    # touches_count(mask_t)
    return decode_image, touches_count


@app.cell
def _(decode_image, dframe, np, pd, touches_count):
    #tests if mask level 1 touches directly mask level 2
    touching = np.zeros(len(dframe), dtype=int)

    for row in dframe.itertuples():
        mask_t = decode_image(row.maskfile)  # H, W - uint8
        touching[row.Index] = touches_count(mask_t)

    touching = pd.Series(touching, name="touching")
    touching
    return (touching,)


@app.cell
def _(data, plot_masks, touching):
    mask_touching = touching[touching > 10].index.values
    plot_masks(data, mask_touching, max_images = 15)
    print(f"There are {len(mask_touching)} images with more the level 1 and 2 touches directly without borders")
    return


@app.cell
def _(mo):
    mo.md(r"""
 
    """)
    return


if __name__ == "__main__":
    app.run()
