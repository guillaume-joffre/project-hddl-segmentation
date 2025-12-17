import marimo

__generated_with = "0.18.4"
app = marimo.App()


@app.cell
def _(mo):
    mo.md(r"""
    # <span style="font-size:0.45em">High-Dimensional-Deep-Learning :<br></span> **Mini-Projet 1/3: Chats ou Chiens ?**
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ### Introduction
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Le jeu de données utilisé pour ce mini-projet est le Oxford-IIIT Pet Dataset, composé de photographies de chats et de chiens appartenant à 37 races différentes. Le jeu de données est disponible à l’adresse suivante :
    [www.robots.ox.ac.uk/~vgg/data/pets](www.robots.ox.ac.uk/~vgg/data/pets)
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    <hr style="opacity: 0.33">

    #### Imports et fonctions utilitaires
    """)
    return


@app.cell
def _():
    ### Imports Principaux

    import torch
    from torch.utils.data import Dataset, DataLoader
    import torch.nn.functional as F

    from torchvision.io import decode_image
    from PIL import Image
    from dataset import OxfordPetsDataset

    import numpy as np
    import pandas as pd

    import marimo as mo
    import matplotlib.pyplot as plt
    import seaborn as sn
    return F, OxfordPetsDataset, decode_image, mo, np, pd, plt, sn, torch


@app.cell
def _():
    ### Imports pour l'affichage

    from matplotlib.colors import ListedColormap, BoundaryNorm
    import matplotlib.patches as mpatches
    return BoundaryNorm, ListedColormap, mpatches


@app.cell
def _(pd, sn):
    ### Options d'affichage

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
    ### I. Analyse Exploratoire
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    L'objectif de cette partie est d'effectuer une analyse exploratoire du jeu de données. Le jeu de données utilisé pour ce mini-projet est le `Oxford-IIIT Pet Dataset`, composé de photographies de chats et de chiens appartenant à 37 races différentes. Le jeu de données est disponible à l’adresse suivante :
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
    <hr style="opacity: 0.33">

    #### I.0 Récupération du Dataset
    """)
    return


@app.cell
def _(OxfordPetsDataset):
    ### Récupération du Dataset

    data = OxfordPetsDataset(
        fetch_masks = True,
        dataset_root = "../data/oxford-pets"
    )

    print(f"""Le dataset \"data\" contient {len(data)} membres, avec:
    - les images de chats/chiens
    - leurs labels espèce/race 
    - les masques de segmentation (animal, arrière-plan, bordure) associés
    """)
    return (data,)


@app.cell
def _(data):
    ### Récupération du DataFrame avec les informations sur le Dataset

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
def _(dframe__):
    ### On retire la colonne bboxfile qui est inutilisée

    dframe = dframe__.drop(columns=["bboxfile"])
    del dframe__
    return (dframe,)


@app.cell
def _(pd):
    ### UTILS: summary(df: DataFrame)

    def summary(df: pd.DataFrame) -> pd.DataFrame:
        over_total = " / "+str(len(df))
        infos = pd.DataFrame(data={
            "dtype": df.dtypes,
            "na/total": df.isnull().sum().map(str) + over_total,
            "unique": df.nunique()
        })

        infos.index.names = ["column"]
        return infos
    return (summary,)


@app.cell
def _(dframe, summary):
    ### Détails sur les colonnes du DataFrame

    summary(dframe)
    return


@app.cell
def _(mo):
    mo.md(r"""
    <hr style="opacity: 0.33">

    #### I.1 Répartition des images entre espèces et races
    """)
    return


@app.cell
def _(mo, pd):
    ### UTILS: summary_categorical(sr: pd.Series)

    def summary_categorical(sr: pd.Series):

        def gen_summary():

            # metadata
            name = sr.name
            ordered = sr.cat.ordered 
            missing = sr.isna().sum()
            count = sr.nunique()

            # display strings
            disp_missing: str = f"{missing} / {len(sr)}"
            disp_count: str = f"{count} / {len(sr.cat.categories)}"

            return mo.hstack([
                mo.stat(
                    value = f"{ordered}", 
                    label = "Ordered Column", 
                    caption = "column is ordered" if ordered else "column not ordered", 
                    direction = "increase" if ordered else "decrease"
                ),

                mo.stat(
                    value = disp_count, 
                    label = "Categories (observed/total)", 
                    caption = f"{count} observed categories"
                ),

                mo.stat(
                    value = disp_missing, 
                    label = "Missing Values", 
                    caption = "no missing values" if missing == 0 else "missing values present",
                    direction = "increase" if missing == 0  else "decrease"
                )
            ], justify="center", gap="1rem")

        def gen_mostleast():

            ncategories = sr.nunique()
            freqs = sr.value_counts(dropna=True)
            total = freqs.sum()

            mapping = {
                "count": lambda n: f"{n} / {total}",
                "frequency": lambda x: f"{x*100:.2f} %"
            }

            if ncategories > 6:

                top = freqs.head(3).reset_index()
                top.columns = [sr.name, "count"]
                top["frequency"] = (top["count"] / total)

                bottom = freqs.tail(3).reset_index()
                bottom.columns = [sr.name, "count"]
                bottom["frequency"] = (bottom["count"] / total)

                return mo.hstack([
                    mo.vstack([
                        mo.md("**Top 3 - Most frequent categories**"), 
                        mo.ui.table(top, 
                            selection = None,
                            format_mapping = mapping
                        )
                    ]),

                    mo.vstack([
                        mo.md("**Bottom 3 - Least frequent categories**"), 
                        mo.ui.table(bottom, 
                            selection = None,
                            format_mapping = mapping
                        )
                    ])
                ], wrap=True)

            else:

                freqs = freqs.reset_index()
                freqs.columns = [sr.name, "count"]
                freqs["frequency"] = (freqs["count"] / total)

                return mo.vstack([
                    mo.md("**Categories per frequency**"), 
                    mo.ui.table(freqs, 
                        selection = None,
                        format_mapping = mapping
                    )
                ])

        def gen_imbalance():
            freqs = sr.value_counts(dropna=True)
            total = freqs.sum()

            max_cat, max_count = freqs.index[0], freqs.iloc[0]
            min_cat, min_count = freqs.index[-1], freqs.iloc[-1]

            mapping = {
                "count": lambda n: f"{n} / {total}",
                "frequency": lambda x: f"{x*100:.2f} %"
            }

            cumulative = freqs.reset_index()
            cumulative.columns = [sr.name, "count"]
            cumulative["count"] = cumulative["count"].cumsum()
            cumulative["frequency"] = (cumulative["count"] / total)

            return mo.vstack([
                mo.md("**Imbalances Diagnostics**"),
                mo.md(f"""
                    - most frequent / least frequent ratio: `{max_count / min_count:.2f}`
                    - in rare categories (with .<10 samples): `{int((freqs < 10).sum())}`
                """),
                mo.md("**Cumulative top frequencies**"),
                mo.ui.table(cumulative.head(6),
                    selection = None,
                    format_mapping = mapping
                )
            ])

        ncategories = sr.nunique()
        if ncategories > 6:
            return mo.vstack([
                mo.md(f"### **Summary of categorical column: \"{sr.name}\"**"),
                gen_summary(),
                gen_mostleast(),
                gen_imbalance()
            ])  
        else:
            return mo.vstack([
                mo.md(f"### **Summary of categorical column: \"{sr.name}\"**"),
                mo.hstack([
                    mo.vstack([
                        gen_summary(),
                        gen_mostleast()
                    ]),
                    gen_imbalance()
                ], widths="equal", align="end", wrap=True)
            ])
    return (summary_categorical,)


@app.cell
def _(dframe, summary_categorical):
    ### Détails sur la colonne catégorique "animal" 

    summary_categorical(dframe["animal"])
    return


@app.cell
def _(dframe, summary_categorical):
    ### Détails sur la colonne catégorique "race" 

    summary_categorical(dframe["race"])
    return


@app.cell
def _(dframe, pd, plt, sn):
    ### Répartition Animal / Race

    def plot_repartition(df):

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

    plot_repartition(dframe)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Nous avons donc environ deux fois plus de chiens que de chats dans le jeu de données, avec 12 races de chats et 25 races de chiens. Chaque race avec environ 200 images dans le dataset.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    <hr style="opacity: 0.33">

    #### I.2 Cohérence des masques de segmentation
    """)
    return


@app.cell
def _(BoundaryNorm, ListedColormap, OxfordPetsDataset, data, mpatches, plt):
    ### UTILS: plot_masks(sr: pd.Series)

    def plot_masks(
        idx: list[int],
        caption: str = "Images along with their segmentation mask",
        max_images: int = 15
    ):

        N_IMAGES = min(len(idx), max_images)
        N_COLS = N_IMAGES if N_IMAGES < 5 else 5
        N_ROWS = 0 if N_IMAGES == 0 else (N_IMAGES - 1) // N_COLS + 1
        FOOTER_SIZE = 0.015

        colors = ["violet", "black", "grey", "white"]    # 0 is unused but matplotlib needs it
        cmap = ListedColormap(colors)                    # Define colors for levels 0, 1, 2, 3
        norm = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap.N) # set boundaries at half-steps

        patches = [
            mpatches.Patch(color="black", label="1: pet"),
            mpatches.Patch(color="grey",  label="2: background"),
            mpatches.Patch(facecolor="white", label="3: border", edgecolor='black', linewidth=1.0),
        ]

        fig = plt.figure(figsize=(2.5*N_COLS, 5.8*N_ROWS + FOOTER_SIZE))
        grid_outer = fig.add_gridspec(2, 1, height_ratios=[5*N_ROWS, FOOTER_SIZE])
        grid_inner = grid_outer[0].subgridspec(N_ROWS*2, N_COLS, wspace=0.05, hspace=0.05)

        for k in range(N_IMAGES):
            image, animal_id, race_id, mask = data[idx[k]].values()
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
            ax2.set_xticks([])
            ax2.set_yticks([])
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
    return (plot_masks,)


@app.cell
def _(dframe, np, plot_masks):
    ### Échantillon aléatoire d'images avec leurs masques

    plot_masks(np.random.choice(dframe.index, size=5, replace=False))
    return


@app.cell
def _(mo):
    mo.md(r"""
    Les masques de segmentation du jeu de données contiennent 3 classes, encodées numériquement par des valeurs discrètes 1, 2, 3:
    1. l'animal (pet)
    2. le fond (background)
    3. une troisième classe de bordure (border)

    On s'intéresse par la suite à la fréquence (ratio) de chaque classe au sein de chaque masque, afin de detecter des anomalies.
    On regarde en particulier si certains masques ne contiennent pas l'animal (la classe 1: animal est absente du masque).
    """)
    return


@app.cell
def _(decode_image, dframe, np, pd, torch):
    ### On calcule pour chaque masque la proportion de chaque classe (1, 2, 3)

    def __class_ratios():
        # might work with a One-hot trick for fast (batched?) bincount:       
        # oh = torch.nn.functional.one_hot(masks.long(), num_classes=4) # oh: (B, 1, H, W, 4)
        # counts[...] = oh.sum(dim=[1,2,3]).cpu().numpy() # Sum over spatial dims: (B, 4)

        counts = torch.zeros((len(dframe), 3), dtype=torch.int64)
        for row in dframe.itertuples():
            mask = decode_image(row.maskfile)  # H, W - uint8
            counts[row.Index] = torch.bincount(mask.flatten(), minlength=4)[1:4]  # index 0 unused

        ratios = pd.DataFrame(counts, columns=["mask_class1","mask_class2","mask_class3"])
        ratios = ratios.div(ratios.sum(axis=1).replace(0, np.nan), axis=0)
        return ratios

    ratios = __class_ratios()
    ratios.iloc[130:140]
    return (ratios,)


@app.cell
def _(mo):
    mo.md(r"""
    La 136ième image a un masque de segmentation avec seulement la classe 2 (background) ce qui semble être une anomalie.
    Regardons les valeurs que prennent ces ratios de plus près:
    """)
    return


@app.cell
def _(plt, ratios, sn):
    ### On regarde les ratios sur les classes de segmentation de plus près

    def __plot(ratios):
        fig, (ax1, ax2) = plt.subplots(1, 2, width_ratios=[1, 1], figsize=(13,7))

        ax1 = sn.violinplot(data=ratios, ax=ax1)
        #ax1.set_title("Distribution of mask level ratios")
        ax1.set_ylabel("Ratio")

        ax2 = sn.boxplot(data=ratios, ax=ax2)
        #ax2.set_title("Distribution of mask level ratios")
        ax2.set_ylabel("Ratio")

        fig.tight_layout()
        fig.supxlabel("Fig X. Presence ratio for each segmentation mask class", y=-0.02, size=15.5, style="italic")
        plt.show()

    __plot(ratios)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Regardons les extrèmes, donc les masques avec presque exclusivement la classe 1 et 2 et ceux avec un ratio haut de classe 3 (plus de 30%)
    """)
    return


@app.cell
def _(plot_masks, ratios):
    __mask_1_high = ratios[ratios["mask_class1"] > 0.9].index.values
    plot_masks(__mask_1_high)
    print(f"There are {len(__mask_1_high)} images with more than 90% level 1 (pet) in their segmentation mask (big pets)")
    return


@app.cell
def _(mo):
    mo.md(r"""
    Les masques comportant plus de 90% de la classe 1 (animal) correspondent juste aux images avec un zoom sur l'animal, qui prend la majorité de l'image. Ce ne sont pas des anomalies.
    """)
    return


@app.cell
def _(plot_masks, ratios):
    __mask_2_full = ratios[ratios["mask_class2"] > 0.99].index.values
    plot_masks(__mask_2_full, max_images = 10)
    print(f"There are {len(__mask_2_full)} images with more than 99% level 2 (background) in their segmentation masks (only background)")
    return


@app.cell
def _(mo):
    mo.md(r"""
    Les masques avec exclusivement de la classe 2 (background) sont erronés. Le problème semble être principalement sur une seule race de chat: les Egyptian Mau (id 1033 et plus).
    """)
    return


@app.cell
def _(plot_masks, ratios):
    __mask_2_quasi = ratios[(ratios["mask_class2"] <= 0.99) & (ratios["mask_class2"] > 0.95)].index.values
    plot_masks(__mask_2_quasi, max_images = 10)
    print(f"There are {len(__mask_2_quasi)} non empty (less than 99% of level 2) images with more than 95% level 2 (background) in their segmentation masks (tiny pets)")
    return


@app.cell
def _(mo):
    mo.md(r"""
    Les masques avec beaucoup de classe 2 (background) mais avec tout de même une présence des autres classes correspondent juste à des images ou les animaux sont petits/loin. On remarque par contre que le masque de l'image 3033 est bizarre. Il ne suit pas l'organisation des autres masques où la classe 3 (border) sépare la classe 1 (animal) de la classe 2 (background).
    """)
    return


@app.cell
def _(plot_masks, ratios):
    __mask_3_high = ratios[ratios["mask_class3"] > 0.32].index.values
    plot_masks(__mask_3_high, max_images = 10)
    print(f"There are {len(__mask_3_high)} images with more than 32% level 3 in their segmentation masks (big borders / unknown zones)")
    return


@app.cell
def _(mo):
    mo.md(r"""
    Enfin, les masques avec un ratio comparativement élevé de classe 3 (border) correspondent à des photos prises en extérieur avec de l'herbe, de la boue ou de la neige qui n'est pas comptée comme faisant partie de l'animal. Cela risque de biaiser les photos prises en extérieur ou avec ces obstacles en particulier.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Regardons à présent si d'autre masques ont un contact direct entre la classe 1 (animal) et la classe 2 (background) comme celui de l'image ID:3033.
    """)
    return


@app.cell
def _(F, torch):
    ### UTILS: count_unusual_touches(mask: torch.Tensor [1, H, W])

    def count_unusual_touches(
        mask: torch.Tensor, 
        crop: int = 0, 
        return_touches: bool = False
    ):

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

        # Optionnally cropping the edge touches
        if crop > 0:
            total_touches = touches_mask[crop:-crop, crop:-crop].sum().item()
        else:
            total_touches = touches_mask.sum().item()

        if return_touches:
            return int(round(total_touches)), touches_mask
        else:
            return int(round(total_touches)), None
    return (count_unusual_touches,)


@app.cell
def _(
    BoundaryNorm,
    ListedColormap,
    OxfordPetsDataset,
    count_unusual_touches,
    data,
    mpatches,
    plt,
):
    ### UTILS: plot_masks_and_touches(sr: pd.Series)

    def plot_masks_and_touches(
        idx: list[int],
        caption: str = "Images with class 1 touching class 2 directly",
        max_images: int = 10
    ):

        N_IMAGES = min(len(idx), max_images)
        N_COLS = N_IMAGES if N_IMAGES < 5 else 5
        N_ROWS = 0 if N_IMAGES == 0 else (N_IMAGES - 1) // N_COLS + 1
        FOOTER_SIZE = 0.015

        colors1 = ["violet", "black", "grey", "white"]    # 0 is unused but matplotlib needs it
        cmap1 = ListedColormap(colors1)                    # Define colors for levels 0, 1, 2, 3
        norm1 = BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5], cmap1.N) # set boundaries at half-steps

        colors2 = ["lightgrey", "red"]
        cmap2 = ListedColormap(colors2)
        norm2 = BoundaryNorm([-0.5, 0.5, 1.5], cmap2.N)

        patches = [
            mpatches.Patch(color="red", label="direct contact"),
            mpatches.Patch(color="black", label="1: pet"),
            mpatches.Patch(color="grey",  label="2: background"),
            mpatches.Patch(facecolor="white", label="3: border", edgecolor='black', linewidth=1.0)
        ]

        fig = plt.figure(figsize=(2.5*N_COLS, 7.95*N_ROWS + FOOTER_SIZE))
        grid_outer = fig.add_gridspec(2, 1, height_ratios=[5*N_ROWS, FOOTER_SIZE])
        grid_inner = grid_outer[0].subgridspec(N_ROWS*3, N_COLS, wspace=0.05, hspace=0.05)

        for k in range(N_IMAGES):
            image, animal_id, race_id, mask = data[idx[k]].values()
            image = OxfordPetsDataset.default_transform_unnormalize(image)
            nb_touches, touches = count_unusual_touches(mask, return_touches=True)
            # touches = touches

            row = k // 5
            col = k % 5

            ax1 = fig.add_subplot(grid_inner[3*row, col])
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

            ax2 = fig.add_subplot(grid_inner[3*row + 1, col])
            ax2.imshow(mask.squeeze(), cmap=cmap1, norm=norm1)
            ax2.set_xticks([])
            ax2.set_yticks([])
            ax2.axis("off")

            ax3 = fig.add_subplot(grid_inner[3*row + 2, col])
            ax3.imshow(touches.squeeze(), cmap=cmap2, norm=norm2)
            ax3.set_xticks([])
            ax3.set_yticks([])
            ax3.axis("off")
            ax3.text(
                0.95, 0.05, # Coordinates within the subplot (0.0 to 1.0)
                "nb="+str(nb_touches),
                transform=ax3.transAxes,
                color="red",
                fontsize=12,
                fontweight="bold",
                ha="right",
                va="bottom",
                bbox=dict(facecolor="black", alpha=0.7, edgecolor="none", pad=2)
            )

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
    return (plot_masks_and_touches,)


@app.cell
def _(count_unusual_touches, data, dframe, np, pd):
    # PREND 2-3 MINUTES SUR MON ORDI !

    COMPUTE_TOUCHES = False

    if COMPUTE_TOUCHES:
        touches = np.zeros(len(dframe), dtype=int)

        for row in dframe.itertuples():
            _, _, _, mask_t = data[row.Index].values()
            touches[row.Index] = count_unusual_touches(mask_t, crop=0)[0]

        touches = pd.Series(touches, name="touches")

    else:
        touches = None 
    return (touches,)


@app.cell
def _(plot_masks_and_touches, touches):

    if touches is None:
        __idx = [395, 1246, 2404, 2440, 3033, 3034, 3035, 3037, 3041, 3043, 3044, 3045, 3047, 3048, 3050, 3051, 3052, 3055, 3056, 3057, 3058, 3059, 3090, 3105, 3112, 3275, 3417, 3507, 3712, 4104, 4225, 4637, 4883, 5271, 6191, 6945, 7066]
    else:
        __idx = touches[touches > 10].index.values

    plot_masks_and_touches(__idx, max_images=10)
    print(f"There are {len(__idx)} images with more then 10 pixels of direct contact between class 1 (animal) and class 2 (background) in their segmentation masks")
    return


@app.cell
def _(mo):
    mo.md(r"""
    This seems to mostly affect dog images (2 cats - 35 dogs) but appart from image 3033 doesn't look too bad. We will keep everything in the dataset for now.

    This count was done on the transformed (scaled down to 224x224) masks.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    <hr style="opacity: 0.33">

    #### I.3 On enlève les anomalies du Dataset
    """)
    return


@app.cell
def _(dframe, np, ratios):
    __idx_empty = ratios[ratios["mask_class2"] > 0.99].index.to_numpy()
    __idx_non_empty = np.setdiff1d(dframe.index.to_numpy(), __idx_empty)
    identifiers = dframe["identifier"].iloc[__idx_non_empty]
    identifiers # identifiers to keep

    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ### II. Classification Binaire
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
 
    """)
    return


if __name__ == "__main__":
    app.run()
