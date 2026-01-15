import marimo

__generated_with = "0.18.4"
app = marimo.App()


@app.cell
def title(mo):
    mo.md(r"""
    # <span style="font-size:0.45em">High-Dimensional-Deep-Learning :<br></span> **Mini-Projet 1/3: Chats ou Chiens ?**
    """)
    return


@app.cell
def introduction(mo):
    mo.md(r"""
    ---
    ## TODO Introduction & Installation
    """)
    return


@app.cell
def introduction_content(mo):
    mo.md(r"""
    <a id="installation"></a>
    Pour faire tourner ce notebook un environnement python avec certains packages spécifiques est nécessaire.
    Vous trouverez dans `install` le fichier `install/environment.yml` qui permet de creer un environnement
    fonctionnel pour ce notebook avec conda:
    ```bash
        # pour créer l'environement et installer les packages
        conda env create -f install/environment.yml --verbose

        # pour activer l'environement
        conda activate HDDL.1

        # pour rendre l'environement visible pour le notebook
        python -m ipykernel install --user --name HDDL.1 --display-name "HDDL.1"
    ```

    Le jeu de données utilisé pour ce mini-projet est le Oxford-IIIT Pet Dataset, composé de photographies de chats et de chiens appartenant à 37 races différentes. Le jeu de données est disponible à l’adresse suivante :
    [www.robots.ox.ac.uk/~vgg/data/pets](www.robots.ox.ac.uk/~vgg/data/pets).

    Il peut être téléchargé en faisant tourner le script `downloader.py` avec la commande :
    ```bash
        python oxford-pets/downloader.py
    ```
    dans un terminal depuis le dossier *root* du project (project-hddl). Vous pouvez aussi le télécharger manuellement depuis l’adresse :
    [www.robots.ox.ac.uk/~vgg/data/pets](www.robots.ox.ac.uk/~vgg/data/pets)
    et placer les images et annotations dans ``data/oxford-pets`.

    La structure des dossiers/fichiers doit être:
    ```
    .
    ├── + data/
    │   └── + oxford-pets/
    │       ├── images/ ...
    │       └── annotations/ ...
    │
    ├── install/ ...
    ├── oxford-pets/ ...
    :
    └── readme.md
    ```
    """)
    return


@app.cell
def imports(mo):
    mo.md(r"""
    <hr style="opacity: 0.33">

    #### Imports et fonctions utilitaires
    """)
    return


@app.cell
def cl_main_imports():
    ### Imports Principaux

    import torch
    from torch.utils.data import Dataset, DataLoader
    import torchvision.models as models
    import torch.nn.functional as F

    from torchvision.io import decode_image
    from PIL import Image
    from dataset import OxfordPetsDataset

    import numpy as np
    import pandas as pd

    import marimo as mo
    import matplotlib.pyplot as plt
    import seaborn as sn

    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    from tqdm.auto import tqdm
    from statistics import mean
    return (
        A,
        DataLoader,
        F,
        OxfordPetsDataset,
        ToTensorV2,
        decode_image,
        mean,
        mo,
        models,
        np,
        pd,
        plt,
        sn,
        torch,
        tqdm,
    )


@app.cell
def _(torch):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cpu":
        print("!!!") # todo better message
    return (device,)


@app.cell
def cl_visual_imports():
    ### Imports pour l'affichage

    from matplotlib.colors import ListedColormap, BoundaryNorm
    import matplotlib.patches as mpatches
    return BoundaryNorm, ListedColormap, mpatches


@app.cell
def cl_visual_options(pd, sn):
    ### Options d'affichage

    # pandas
    pd.set_option("display.max_columns", None)
    pd.set_option('display.max_colwidth', None)

    # seaborn
    sn.set_style("darkgrid")
    sn.set_palette("colorblind")
    return


@app.cell
def exploratory(mo):
    mo.md(r"""
    ---
    ## I. Analyse Exploratoire
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Le jeu de données utilisé pour ce mini-projet est le `Oxford-IIIT Pet Dataset`, composé de photographies de chats et de chiens appartenant à 37 races différentes.

    L'objectif de cette partie est d'effectuer une analyse exploratoire. Nous souhaitons comprendre la structure du jeu de données Oxford-IIIT Pet et évaluer la qualité de ses annotations (labels et masques) avant de procéder à l'entraînement des modèles.

    En particulier nous explorerons les questions suivantes posées par le sujet:
    - Quelle est la répartition par espèce et par race dans le jeu de donnée.
    - Les masques de segmentations associés aux images sont-ils bons/cohérents.
    - Y-a t'il eventuellement des déséquilibres ou biais visuels.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    <hr style="opacity: 0.33">

    #### I.1 Récupération du Dataset
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Si une erreur survient lors de cette partie ou que le dataset est vide assurez-vous d'avoir téléchargé le dataset comme indiqué dans la partie [**Introduction**](#installation).
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
    Toutes les images du Oxford-IIIT Pet Dataset ont un masque de segmentation associé, mais toutes n'ont pas un fichier décrivant une bounding box pour la tête de l'animal. C'est pourquoi la colonne "bboxfile" à des valeurs manquantes. Comme il s'agit d'un mini-projet portant sur la segmentation et non pas sur la detection d'objet nous allons ignorer la colonne et revenir à un dataset sans ces valeurs manquantes.
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

    #### I.2 Analyse de la répartition des images entre espèces et races
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Le jeu de données comporte 7390 images avec un seul animal par image, il y a deux animaux possibles (chien ou chat) et 37 races distinctes. Analysons la répartition de ces images parmi ces catégories.
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
    return (plot_repartition,)


@app.cell
def _(mo):
    mo.md(r"""
    Nous avons donc environ deux fois plus de chiens que de chats dans le jeu de données, avec 12 races de chats et 25 races de chiens. Nous remarquons les points suivants:
    - Un déséquilibre des espèces: avec environ 2/3 de chiens pour 1/3 de chats présents dans le jeu de donnée.
    - Un équilibre des races: Malgré le déséquilibre entre espèces, la répartition par race est très homogène. Chaque race dispose d'environ 200 images, ce qui garantit une base d'apprentissage stable pour la classification fine.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    <hr style="opacity: 0.33">

    #### I.3 Exploration de la cohérence des masques de segmentation
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Les masques de segmentation du jeu de données contiennent 3 classes, encodées numériquement par des valeurs discrètes 1, 2 et 3 qui correspondent à :
    1. l'animal (pet)
    2. le fond (background)
    3. une troisième classe de bordure (border)
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
    sampled_images = np.random.choice(dframe.index, size=5, replace=False)
    plot_masks(sampled_images)
    return (sampled_images,)


@app.cell
def _(
    BoundaryNorm,
    ListedColormap,
    OxfordPetsDataset,
    data,
    mpatches,
    np,
    plt,
    sampled_images,
):
    def plot_masks_border_overlap(
        idx: list[int],
        caption: str = "Images along with the border (mask class 3)",
        max_images: int = 20
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

        fig = plt.figure(figsize=(2.5*N_COLS, 3.2*N_ROWS + FOOTER_SIZE))
        grid_outer = fig.add_gridspec(2, 1, height_ratios=[2.5*N_ROWS, FOOTER_SIZE])
        grid_inner = grid_outer[0].subgridspec(N_ROWS, N_COLS, wspace=0.05, hspace=0.05)

        for k in range(N_IMAGES):
            image, animal_id, race_id, mask = data[idx[k]].values()
            image = OxfordPetsDataset.default_transform_unnormalize(image)
            mask = mask.squeeze().numpy()

            border_mask = np.zeros((*mask.shape, 4)) # RGBA
            border_mask[mask == 3] = [1, 0, 0, 0.5]  # Rouge avec 50% d'opacité

            row = k // 5
            col = k % 5

            ax = fig.add_subplot(grid_inner[row, col])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.axis('off')
            ax.imshow(image)
            ax.imshow(border_mask) # Overlap
            ax.text(
                0.05, 0.95, # Coordinates within the subplot (0.0 to 1.0)
                str(idx[k]),
                transform=ax.transAxes,
                color="white",
                fontsize=12,
                fontweight="bold",
                ha="left",
                va="top",
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

    plot_masks_border_overlap(sampled_images)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Cette troisième classe de bordure est très intéressante, elle semble dénoter soit une bordure entre les deux classes que l'on souhaite trouver lors de la segmentation (pet et background) ou bien une zone d'incertitude/complexité sémantique (comme par exemple un collier qui peut selon l'interprétation faire partie du chien ou non).

    Il y a plusieurs possibilités pour cette troisième classe:
    - on peut l'ignorer lors de la segmentation, c'est-à-dire que l'on considère que notre réseau pourra assigner la classe qu'il souhaite à ces régions sans impact sur la loss.
    - on peut la considérer comme faisant partie de l'animal, voire même lui assigner un poids plus fort afin de mieux préciser les contours.

    Comme ici la classe 3 ne superpose pas *que* l'animal en son bord mais est bien une zone de bordure (qui couvre autant l'animal que le fond), la consiérer comme faisant partie de l'animal risque surtout d'entraîner notre modèle à prendre une marge autour des contours de l'animal. Il n'y aurait vraisemblablement pas de gain de précision des contours.

    Les pixels avec la classe 3 seront donc ignorés dans la loss pour la segmentation.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
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
    Les masques avec exclusivement de la classe 2 (background) sont erronés.
    Le problème visible semble être principalement sur la race de chat: les Egyptian Mau (id 1033 et plus) mais d'autres images anormales ne sont pas affichées ici.

    Après avoir augmenté `max_images` le problème touche plusieurs races.
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
    Les masques avec beaucoup de classe 2 (background) mais avec tout de même une présence des autres classes correspondent juste à des images ou les animaux sont petits/loin.

    On remarque par contre que le masque de l'image 3033 est bizarre. Il ne suit pas l'organisation des autres masques où la classe 3 (border) sépare la classe 1 (animal) de la classe 2 (background). La classe 3 (border) sert de classe de transition ou de marge si l'on est incertain de la classe à donner.
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
    Enfin, les masques avec un ratio comparativement élevé de classe 3 (border) correspondent à des photos prises en extérieur avec de l'herbe, de la boue ou de la neige qui n'est pas juste comptée comme faisant partie de l'animal. Il s'agit surtout de photos présentant une potentielle difficulté pour la segmentation.

    Cela risque aussi de biaiser notre modèle si les photos prises en extérieur ou avec ces obstacles sont plus présentes avec une certaine espèce/race.
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

    plot_masks_and_touches(__idx, max_images=15)
    print(f"There are {len(__idx)} images with more then 10 pixels of direct contact between class 1 (animal) and class 2 (background) in their segmentation masks")
    return


@app.cell
def _(mo):
    mo.md(r"""
    This seems to mostly affect dog images (2 cats - 35 dogs). Appart for image 3033 where the organisation of the classes wasn't respected the masks don't look too bad, the direct touches are small errors and imprecisions. We will keep everything except the image 3033 in the dataset during the segmentation part of the project.

    TODO : This count was done on the transformed (scaled down to 224x224) masks.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    <hr style="opacity: 0.33">

    #### I.4 Exclusion des anomalies vues précédemment
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Les anomalies détectées touchent les masques de segmentation, elles seront donc enlevées du dataset lors de la section sur la segmentation.
    """)
    return


@app.cell
def _(OxfordPetsDataset, dframe, np, ratios):

    # empty masks with only background
    __idx_to_remove = ratios[ratios["mask_class2"] > 0.99].index.to_numpy()

    # weird masks
    __idx_to_remove = np.concatenate((__idx_to_remove, [3033]))


    __idx_to_keep = np.setdiff1d(dframe.index.to_numpy(), __idx_to_remove)
    identifiers = dframe["identifier"].iloc[__idx_to_keep]

    dframe_clean = dframe.iloc[__idx_to_keep]
    data_clean = OxfordPetsDataset(
        fetch_masks = True,
        dataset_root = "../data/oxford-pets",
        identifiers = identifiers
    )

    print(f"""Le dataset sans anomalies \"data_clean\" contient {len(data_clean)} membres, avec:
    - les images de chats/chiens
    - leurs labels espèce/race 
    - les masques de segmentation (animal, arrière-plan, bordure) associés
    """)
    return dframe_clean, identifiers


@app.cell
def _(dframe_clean, plot_repartition):
    plot_repartition(dframe_clean)
    return


@app.cell
def _(mo):
    mo.md(r"""
    <hr style="opacity: 0.33">

    #### I.5 Exploration des biais potentiels
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Regardons si sur les photos du dataset les chats ont tendance à être plus petits que sur les photos de chien, une différence dans le zoom pourrait introduire un biai dans l'apprentissage.
    """)
    return


@app.cell
def _(dframe, plt, ratios, sn):

    def plot_ratio_bias(
        col:str, 
        group:str, 
        selection:list|None=None,
        title: str = "",
        xlabel: str = ""

    ):
        df_ratios_bias = dframe[["animal", "race"]].join(ratios)

        if selection is not None:
            df_ratios_bias = df_ratios_bias[df_ratios_bias[group].isin(selection)]
            df_ratios_bias[group] = df_ratios_bias[group].cat.remove_unused_categories()


        fig, ax = plt.subplots(figsize=(12, 8))
        sn.kdeplot(data=df_ratios_bias, x=col, hue=group, common_norm=False, alpha=0.5, ax=ax)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        sn.move_legend(ax, "upper left", bbox_to_anchor=(1, 1), ncol=1, title=f"{group}s")
        plt.show()
    return (plot_ratio_bias,)


@app.cell
def _(plot_ratio_bias):
    plot_ratio_bias("mask_class1", "animal",
        title = f"Distribution de la taille relative de l'animal dans l'image, par animal.",
        xlabel = "Ratio de la classe 1 (Pet) dans le masque"
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    Regardons aussi si il y a une différence pour les races individuellement.
    """)
    return


@app.cell
def _(plot_ratio_bias):
    plot_ratio_bias("mask_class1", "race",
        title = f"Distribution de la taille relative de l'animal dans l'image, par race.",
        xlabel = "Ratio de la classe 1 (Pet) dans le masque"
    )
    return


@app.cell
def _(plot_ratio_bias):
    plot_ratio_bias("mask_class1", "race", selection=[
            "persian", "ragdoll", "egyptian_mau", "miniature_pinscher"
        ],
        title = f"Distribution de la taille relative de l'animal dans l'image, par race.",
        xlabel = "Ratio de la classe 1 (Pet) dans le masque"
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    L'herbe et la neige rendent la classe 3 plus présente/rajoute de la complexité sur la classe animal/background à donner. Il y a donc potentiellement un biai environnemental. On peut ainsi regarder la différence chien/chat, avec l'idée que les chiens sont peut-être plus pris en photo à l'extérieur que les chats.
    """)
    return


@app.cell
def _(plot_ratio_bias):
    plot_ratio_bias("mask_class3", "animal",
        title = f"Complexité de segmentation (Ratio classe 3) par espèce.",
        xlabel = "Ratio de la classe 3 (Pet) dans le masque"
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    Regardons aussi si il y a une différence pour les races individuellement.
    """)
    return


@app.cell
def _(plot_ratio_bias):
    plot_ratio_bias("mask_class3", "race",
        title = f"Complexité de segmentation (Ratio classe 3) par race.",
        xlabel = "Ratio de la classe 3 (Pet) dans le masque"
    )
    return


@app.cell
def _(plot_ratio_bias):
    plot_ratio_bias("mask_class3", "race", selection=[
            "persian", "ragdoll", "egyptian_mau", "miniature_pinscher"
        ],
        title = f"Complexité de segmentation (Ratio classe 3) par race.",
        xlabel = "Ratio de la classe 3 (Pet) dans le masque"
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    TODO conclure sur les masques classe 3 ici
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    IDEES/TODO MAIS POUR PLUS TARD: TODO A FAIRE TRES INTERESSANT
    1. **Différence de couleur de l'environement**: différentier les chats/chiens et leur race depuis leur couleur c'est ok/normal mais si l'environnement à tendence à être différent (plus sombre par exemple) pour une espèce/race en particulier dans le dataset cela risque d'introduire un biai.

    2. **Format de l'image** Comme on rescale les images vers 224x224, si les images de chiens sont en mode portrait et de chat en mode paysage, le réseau risque d'apprendre depuis la distortion.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    <hr style="opacity: 0.33">

    #### I.6 Bilan de l'analyse exploratoire
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Cette analyse a permis de valider la qualité du jeu de données et d'identifier les stratégies nécessaires pour l'entraînement des modèles de segmentation et de classification.

    **1. Qualité des données et nettoyage :**

    - **Anomalies détectées :** Environ 1% du dataset présentait des masques de segmentation vides (uniquement de l'arrière-plan) ou corrompus (contact direct anormal entre l'animal et le fond sans bordure de transition). Ces images ont été exclues pour garantir que le modèle ne reçoive pas de signaux contradictoires.

    - **Diversité des échelles :** La part occupée par l'animal varie de **5% à 95%** de l'image. Cette grande variabilité nous dirige vers l'utilisation d'augmentations de type `RandomResizedCrop` pour rendre le modèle robuste aux changements sur le zoom de l'animal.


    **2. Déséquilibres :**

    - Le dataset présente un **déséquilibre au niveau des espèces** (~2/3 de chiens pour ~1/3 de chats). Cependant, la répartition par race est extrêmement stable (~200 images par race), ce qui limite le risque qu'une race spécifique ne domine l'apprentissage au sein d'une espèce.

    - **Biais :** Les distributions de la proportion de classe 1 chez les chats et les chiens sont proches, mais ce n'est pas le cas entre les races de façon plus individuelles. Cela pourrait amener le modèle à classifier dans une race plutôt qu'une autre à partir du zoom sur la photo. Cela confirme la nécéssité d'utiliser des augmeentations de type `RandomResizedCrop`.


    Robustesse de la classification fine (Race) : Bien qu'il y ait plus de chiens que de chats au total, chaque race individuelle dispose d'un effectif quasi identique (~200 images). Cela signifie que le modèle ne sera pas "biaisé" en faveur d'une race spécifique au détriment d'une autre au sein d'une même espèce. C'est un point positif pour la classification à 37 classes.

    Risque de biais de prédiction majoritaire (Espèce) : Le déséquilibre 2/3 chiens (25 races) contre 1/3 chats (12 races) est un point de vigilance pour la classification binaire. Sans précaution, un modèle pourrait obtenir une précision de ~67% en prédisant systématiquement "chien". Cela justifie l'utilisation de métriques plus précises que la simple accuracy, comme le F1-score ou une matrice de confusion, pour valider les performances réelles sur les chats.

    Stratégie d'échantillonnage : Cette observation valide la méthode de partitionnement des données (Split Train/Val). Il sera crucial d'utiliser un échantillonnage stratifié (ou de vérifier la répartition après le split aléatoire effectué dans la cellule 53) pour s'assurer que la proportion 2:1 est conservée dans les ensembles d'entraînement et de validation.


    **3. Rôle critique de la Classe 3 (Bordure/Incertitude) :**

    - Plutôt qu'une simple "bordure", la classe 3 agit comme une **zone d'incertitude sémantique**. Elle capture l'aliasing, les textures complexes (poils, herbe, neige) et les séparations floues entre l'animal et son environnement.




    **Implications pour la suite (Segmentation) :**
    La gestion de la classe 3 sera déterminante. Lors de l'entraînement, il faudra décider si nous traitons cette classe comme une catégorie à part entière ou si nous utilisons une fonction de coût (Loss) qui ignore ces pixels ambigus pour se concentrer sur les zones de certitude (Pet vs Background).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    TODO A mettre plus tôt: J'ai noté que la classe 3 (border) est une bordure quand en réalité il s'agit plus d'une classe "incertaine" ou le modèle peut donner l'une ou l'autre classe (animal/background). Elle agit comme bordure car assez naturellement la séparation entre l'object/animal et le fond peut être compliquée (aliasing).
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    A PLACER QUELQUE PART:

    Les masques avec que la classe 2 sont des anomalies et les enlever c'est carrément ok. Ca veut aussi dire que le modèle s'attend à trouver un animal sur l'image on a pas de données sans ou avec plusieurs animaux. (évident mais ça vaut le coup de le dire)

    La classe 3 (border) n'est pas juste une bordure mais représente plutôt une complexité (aliasing, non sharp separation between animal/background, texture différente (neige, herbe, superposition avec l'animal)). A DIRE: la présence de la classe 3 peut indiquer une difficulté de segmentation/classification.

    Normalisation du zoom : L'utilisation de RandomResizedCrop (échelle 0.7 à 1.0) permet de rendre le modèle robuste à la taille relative de l'animal dans l'image, compensant ainsi la forte variabilité (5% à 95%) observée précédemment.

    Data Augmentation : Des retournements horizontaux (HorizontalFlip) et une normalisation standard (ImageNet) sont appliqués pour améliorer la généralisation.

    Gestion du déséquilibre : Bien que les chiens soient deux fois plus nombreux, nous avons choisi de conserver cette répartition tout en surveillant la matrice de confusion pour détecter un éventuel biais prédictif envers les chiens.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Ce pose la question de la gestion de la classe 3: Est-elle ignorée complètement ou partiellement (avec poids) ? (notre réseau peut assigner n'importe quelle classe) ou considérée comme partie de l'animal ou du background ?
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---
    ## II. Classification
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    L'objectif de cette section est de développer un classifieur capable de distinguer les espèces (binaire : chat vs chien) puis dans un deuxième temps un classifieur pouvant distinguer les races (classification fine : 37 catégories). Cette étape sert de fondation pour comprendre comment un réseau extrait des caractéristiques globales avant de passer à la segmentation locale.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    <hr style="opacity: 0.5">

    #### Préparation des données pour la classification.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    En nous basant sur l'analyse exploratoire, nous allons :
    - Utiliser un **RandomResizedCrop** pour normaliser les différences de zoom (biais de taille).
    - Garder les images avec des anomalies puisque celles-ci ne concèrnent que les masques de segmentation.
    - Gérer le déséquilibre des classes (2/3 chiens, 1/3 chats) via une fonction de perte pondérée ou un échantillonnage adapté.

    TODO Pas sur pour le deuxième (weight=WEIGHT pas mis en place)
    TODO Gestion du Biais d'Espèce : nous pourrins calculer des poids `WEIGHTS` inversement proportionnels à la fréquence des classes et les utiliser dans le critère `CrossEntropyLoss(weights=WEIGHTS)`. Cela empêcherait le modèle de simplement prédire "chien" systématiquement pour obtenir une précision de 66%.
    """)
    return


@app.cell
def _(A, ToTensorV2):
    augmentation_transform = A.Compose([
        A.RandomResizedCrop(size=(224, 224), scale=(0.7, 1.0), p=1.0),
        A.HorizontalFlip(p=0.5),
        A.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),

        # Shift the colors but I don't think it's a good idea.
        # A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),

        # askip ça aide si pour cetains animaux les photos sont prises avec des 
        # angles bizarres, pas parfaitement droites.
        # A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=15, p=0.5),

        ToTensorV2()
    ]) 
    return (augmentation_transform,)


@app.cell
def _(OxfordPetsDataset, augmentation_transform):
    data_classif = OxfordPetsDataset(
        fetch_masks = False,
        joint_transform = augmentation_transform,
        dataset_root = "../data/oxford-pets"
    )
    return (data_classif,)


@app.cell
def _(data_classif, torch):
    # can be set to None for random seed
    _generator = torch.Generator().manual_seed(42)

    # Split Train/Val (80% / 20%)
    _train_size = int(0.8 * len(data_classif))
    _val_size = len(data_classif) - _train_size

    subset_classif_train, subset_classif_val = torch.utils.data.random_split(
        data_classif, [_train_size, _val_size], 
        generator = _generator
    )
    return subset_classif_train, subset_classif_val


@app.cell
def _(DataLoader, subset_classif_train, subset_classif_val):
    loader_classif_train = DataLoader(subset_classif_train, batch_size=32, shuffle=True)
    loader_classif_val = DataLoader(subset_classif_val, batch_size=32, shuffle=False)
    return loader_classif_train, loader_classif_val


@app.cell
def _(mo):
    mo.md(r"""
    <hr style="opacity: 0.5">

    ### II.A Classification binaire (chien/chat)
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Dans cette partie, nous comparons quatre architectures pour valider l'efficacité du Transfer Learning par rapport à une approche from scratch sur le classifieur bianire.
    Les modèles utilisés seront:
    - Un "VGG-8" In-House : Un modèle simplifié avec 3 blocs de convolution et un classifieur dense de 1024 neurones. Ce modèle sert de base de référence (baseline) pour évaluer la complexité nécessaire à la tâche.
    - Les modèles VGG-16, ResNet18 & MobileNetV3 : Ces modèles pré-entraînés bénéficient de connaissances acquises sur ImageNet. Nous allons remplacer leurs couches finales pour une sortie binaire et les fine-tuner sur nos données.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    <hr style="opacity: 0.33">

    #### II.A.1 Classification binaire \[in-house \"VGG-8\"\].
    """)
    return


@app.cell
def _(torch):
    ### Model definition: simple in-house VGG (VGG-8))
    import torch.nn as nn

    class Model_VGG8_InHouse(nn.Module):

            def __init__(self, num_classes=2, input_channels=3):
                super(Model_VGG8_InHouse, self).__init__()

                self.features = nn.Sequential(
                    # Block 1: 2 convs
                    nn.Conv2d(3, 64, kernel_size=3, padding=1), 
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 64, kernel_size=3, padding=1), 
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2),

                    # Block 2: 2 convs
                    nn.Conv2d(64, 128, kernel_size=3, padding=1), 
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, 128, kernel_size=3, padding=1), 
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2),

                    # Block 3: 2 convs
                    nn.Conv2d(128, 256, kernel_size=3, padding=1), 
                    nn.ReLU(inplace=True),
                    nn.Conv2d(256, 256, kernel_size=3, padding=1), 
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2, stride=2),
                )

                self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

                self.classifier = nn.Sequential(
                    nn.Linear(256 * 7 * 7, 1024), 
                    nn.ReLU(inplace=True), 
                    nn.Dropout(0.5),
                    nn.Linear(1024, num_classes),
                )

            def forward(self, x):
                x = self.features(x)
                x = self.avgpool(x)
                x = torch.flatten(x, 1)
                x = self.classifier(x)
                return x
    return Model_VGG8_InHouse, nn


@app.cell
def _(Model_VGG8_InHouse):
    from torchinfo import summary as model_summary

    model_2A1 = Model_VGG8_InHouse(num_classes=2)
    _ = model_summary(model_2A1, input_size=(1, 3, 224, 224))
    return model_2A1, model_summary


@app.cell
def _(device, mean, torch, tqdm):
    def train_model_classif(model, train_loader, val_loader, criterion, optimizer, epochs=5, label_key="animal_id"):

            history = {"train_loss": [], "validation_accuracy": []}
            model.to(device)

            for epoch in range(epochs):

                # Phase d'entraînement
                model.train()
                running_loss = []
                pbar = tqdm(train_loader, desc=f"[train] epoch {epoch+1}/{epochs}")

                for batch in pbar:
                    images = batch["image"].to(device)
                    labels = batch[label_key].to(device).long()

                    optimizer.zero_grad()
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()

                    running_loss.append(loss.item())
                    pbar.set_postfix({"mean loss": f"{mean(running_loss):.04f}"})

                # Phase de validation
                model.eval()
                correct, total = 0, 0
                pbar2 = tqdm(val_loader, desc=f"[validation] on validation data")

                with torch.no_grad():
                    for batch in pbar2:
                        images = batch["image"].to(device)
                        labels = batch[label_key].to(device).long()

                        outputs = model(images)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()

                        accuracy = 100 * correct / total
                        pbar2.set_postfix({"accuracy": f"{accuracy:.03f}% "})

                accuracy = 100 * correct / total
                history["train_loss"].append(mean(running_loss))
                history["validation_accuracy"].append(accuracy)

            print(f"training step completed.")
            return history
    return (train_model_classif,)


@app.cell
def _():
    training_results = {}
    return (training_results,)


@app.cell
def _(
    device,
    loader_classif_train,
    loader_classif_val,
    model_2A1,
    torch,
    train_model_classif,
    training_results,
):
    def _train_model_2A1():

        print()
        print(f"using device: {device}")
        model_2A1.to(device)

        training_results["2A1"] = train_model_classif(
            model_2A1, loader_classif_train, loader_classif_val, 
            criterion = torch.nn.CrossEntropyLoss(),
            optimizer = torch.optim.Adam(model_2A1.parameters(), lr=0.001),
            label_key="animal_id",
            epochs=10
        )


    _train_model_2A1()
    torch.save(model_2A1.state_dict(), 'weights/model_2A1.pth')
    print("Model saved to weights/model_2A1.pth")
    return


@app.cell
def _(mo):
    mo.md(r"""
    <hr style="opacity: 0.33">

    #### II.A.2 Classification binaire depuis un modèle pré-entrainé \[VGG-16\].
    """)
    return


@app.cell
def _(model_summary, models, torch):
    # On utilise un ResNet18 pré-entraîné
    model_2A2 = models.vgg16(weights=models.VGG16_Weights.DEFAULT)

    # On remplace la toute dernière couche du bloc 'classifier' (index 6)
    # pour une classification binaire (2 sorties)
    # in_features est généralement 4096 pour VGG16
    model_2A2.classifier[6] = torch.nn.Linear(model_2A2.classifier[6].in_features, 2)

    _ = model_summary(model_2A2, input_size=(1, 3, 224, 224))
    return (model_2A2,)


@app.cell
def _(
    device,
    loader_classif_train,
    loader_classif_val,
    model_2A2,
    torch,
    train_model_classif,
    training_results,
):
    def _train_model_2A2():

        print()
        print(f"using device: {device}")
        model_2A2.to(device)

        ### Classifier training
        for param in model_2A2.parameters(): # 1. On gèle tous les paramètres
            param.requires_grad = False

        for param in model_2A2.classifier.parameters(): # 2. On dégèle la couche de classification
            param.requires_grad = True

        training_results["2A2"] = train_model_classif(
            model_2A2, loader_classif_train, loader_classif_val, 
            criterion = torch.nn.CrossEntropyLoss(),
            optimizer = torch.optim.Adam(model_2A2.parameters(), lr=0.001),
            label_key="animal_id",
            epochs=5
        )

        ### Fine-tuning
        for param in model_2A2.parameters():  # 2. On débloque tous les params
            param.requires_grad = True

        training_results["2A2_ft"] = train_model_classif(
            model_2A2, loader_classif_train, loader_classif_val, 
            criterion = torch.nn.CrossEntropyLoss(),
            optimizer = torch.optim.Adam(model_2A2.parameters(), lr=0.00001),
            label_key="animal_id",
            epochs=5
        )


    _train_model_2A2()
    torch.save(model_2A2.state_dict(), 'weights/model_2A2.pth')
    print("Model saved to weights/model_2A2.pth")
    return


@app.cell
def _(mo):
    mo.md(r"""
    <hr style="opacity: 0.33">

    #### II.A.3 Classification binaire depuis un modèle pré-entrainé \[ResNet18\].
    """)
    return


@app.cell
def _(model_summary, models, torch):
    # On utilise un ResNet18 pré-entraîné
    model_2A3 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # On remplace la dernière couche (FC) pour une classification chien/chat (2 sorties)
    model_2A3.fc = torch.nn.Linear(model_2A3.fc.in_features, 2)

    _ = model_summary(model_2A3, input_size=(1, 3, 224, 224))
    return (model_2A3,)


@app.cell
def _(
    device,
    loader_classif_train,
    loader_classif_val,
    model_2A3,
    torch,
    train_model_classif,
    training_results,
):
    def _train_model_2A3():

        print()
        print(f"using device: {device}")
        model_2A3.to(device)

        ### Classifier training
        for param in model_2A3.parameters(): # 1. On gèle tous les paramètres
            param.requires_grad = False

        for param in model_2A3.fc.parameters(): # 2. On dégèle la couche de classification
            param.requires_grad = True

        training_results["2A3"] = train_model_classif(
            model_2A3, loader_classif_train, loader_classif_val, 
            criterion = torch.nn.CrossEntropyLoss(),
            optimizer = torch.optim.Adam(model_2A3.parameters(), lr=0.001),
            label_key="animal_id",
            epochs=5
        )

        ### Fine-tuning
        for param in model_2A3.parameters():  # 2. On débloque tous les params
            param.requires_grad = True

        training_results["2A3_ft"] = train_model_classif(
            model_2A3, loader_classif_train, loader_classif_val, 
            criterion = torch.nn.CrossEntropyLoss(),
            optimizer = torch.optim.Adam(model_2A3.parameters(), lr=0.00001),
            label_key="animal_id",
            epochs=5
        )


    _train_model_2A3()
    torch.save(model_2A3.state_dict(), 'weights/model_2A3.pth')
    print("Model saved to weights/model_2A3.pth")
    return


@app.cell
def _(mo):
    mo.md(r"""
    <hr style="opacity: 0.33">

    #### II.A.4 Classification binaire depuis un modèle pré-entrainé \[MobileNetV3\].
    """)
    return


@app.cell
def _(model_summary, models, torch):
    # On utilise un MobileNetV3 pré-entraîné
    model_2A4 = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)

    # On remplace la dernière couche (FC) pour une classification chien/chat (2 sorties)
    model_2A4.classifier[3] = torch.nn.Linear(model_2A4.classifier[3].in_features, 2)

    _ = model_summary(model_2A4, input_size=(1, 3, 224, 224))
    return (model_2A4,)


@app.cell
def _(
    device,
    loader_classif_train,
    loader_classif_val,
    model_2A4,
    torch,
    train_model_classif,
    training_results,
):
    def _train_model_2A4():

        print()
        print(f"using device: {device}")
        model_2A4.to(device)

        ### Classifier training
        for param in model_2A4.parameters(): # 1. On gèle tous les paramètres
            param.requires_grad = False

        for param in model_2A4.classifier.parameters(): # 2. On dégèle la couche de classification
            param.requires_grad = True

        training_results["2A4"] = train_model_classif(
            model_2A4, loader_classif_train, loader_classif_val, 
            criterion = torch.nn.CrossEntropyLoss(),
            optimizer = torch.optim.Adam(model_2A4.parameters(), lr=0.001),
            label_key="animal_id",
            epochs=5
        )

        ### Fine-tuning
        for param in model_2A4.parameters():  # 2. On débloque tous les params
            param.requires_grad = True

        training_results["2A4_ft"] = train_model_classif(
            model_2A4, loader_classif_train, loader_classif_val, 
            criterion = torch.nn.CrossEntropyLoss(),
            optimizer = torch.optim.Adam(model_2A4.parameters(), lr=0.00001),
            label_key="animal_id",
            epochs=5
        )


    _train_model_2A4()
    torch.save(model_2A4.state_dict(), 'weights/model_2A4.pth')
    print("Model saved to weights/model_2A4.pth")
    return


@app.cell
def _(mo):
    mo.md(r"""
    <hr style="opacity: 0.33">

    #### II.A.5 Comparaison des modèles.
    """)
    return


@app.cell
def _(model_2A1, model_2A2, model_2A3, model_2A4):
    models_map = {
        "2A1": (model_2A1, "In-House VGG-8"),
        "2A2": (model_2A2, "Pre-Trained VGG-16"),
        "2A3": (model_2A3, "Pre-Trained ResNet18"),
        "2A4": (model_2A4, "Pre-Trained MobileNetV3"),
    }
    return (models_map,)


@app.cell
def _(models_map, plt, training_results):
    def plot_comparison(results):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

        for name, hist in results.items():
            ax1.plot(hist["train_loss"], label=f"{models_map[name][2]}")
            ax2.plot(hist["validation_accuracy"], label=f"{models_map[name][2]}")

        ax1.set_title("Perte d'Entraînement (loss)")
        ax1.set_xlabel("Époque")
        ax1.legend()

        ax2.set_title("Précision de Validation (accuracy)")
        ax2.set_xlabel("Époque")
        ax2.set_ylabel("%")
        ax2.legend()

        plt.show()

    plot_comparison(training_results)
    return (plot_comparison,)


@app.cell
def _(model_2A1, model_2A2, model_2A3, model_2A4, pd, training_results):
    pd.DataFrame({
        "model": [
            "In-House VGG-8", 
            "Pre-Trained VGG-16", 
            "Pre-Trained ResNet18", 
            "Pre-Trained MobileNetV3"
        ],
        "params": [
            sum(p.numel() for p in model_2A1.parameters()),
            sum(p.numel() for p in model_2A2.parameters()),
            sum(p.numel() for p in model_2A3.parameters()),
            sum(p.numel() for p in model_2A4.parameters())
        ],
        "training loss": [
            training_results["2A1"]["train_loss"][-1],
            training_results["2A2"]["train_loss"][-1],
            training_results["2A3"]["train_loss"][-1],
            training_results["2A4"]["train_loss"][-1]
        ],
        "validation accuracy": [
            training_results["2A1"]["validation_accuracy"][-1],
            training_results["2A2"]["validation_accuracy"][-1],
            training_results["2A3"]["validation_accuracy"][-1],
            training_results["2A4"]["validation_accuracy"][-1]
        ]
    })
    return


@app.cell
def _(DataLoader, device, plt, races, sn, subset_classif_val, torch):
    import sklearn.metrics as metrics

    def plot_classification_results(model, label_key="animal_id"):

        model.eval()
        all_preds = []
        all_labels = []

        ticks = ["cat", "dog"] if label_key=="animal_id" else races

        with torch.no_grad():
            for res in DataLoader(subset_classif_val, batch_size=32):
                images = res["image"].to(device)
                labels = res[label_key].to(device)
                outputs = model(images)

                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        cm = metrics.confusion_matrix(all_labels, all_preds)

        plt.figure(figsize=(12, 10))
        sn.heatmap(cm, xticklabels=ticks, yticklabels=ticks, annot=False, cmap="Blues") 
        plt.title("Matrice de Confusion - Classification")
        plt.show()
    return (plot_classification_results,)


@app.cell
def _(model_2A1, plot_classification_results):
    plot_classification_results(model_2A1, label_key="animal_id")
    return


@app.cell
def _(model_2A2, plot_classification_results):
    plot_classification_results(model_2A2, label_key="animal_id")
    return


@app.cell
def _(model_2A3, plot_classification_results):
    plot_classification_results(model_2A3, label_key="animal_id")
    return


@app.cell
def _(model_2A4, plot_classification_results):
    plot_classification_results(model_2A4, label_key="animal_id")
    return


@app.cell
def _(mo):
    mo.md(r"""
    <hr style="opacity: 0.5">

    ### II.B Classification fine (par race)
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    La classification par race est une tâche plus difficile en raison de la forte similarité inter-classes (ex: deux races de terriers, ou Abyssinian vs. Bengal pour les chats).

    Dans cette partie, nous comparons les même modèles que durant la partie précédents, avec cette fois des couches finales avec 37 logits en sorties.
    Les modèles utilisés sont donc toujours:
    - Un "VGG-8" In-House : Un modèle simplifié avec 3 blocs de convolution et un classifieur dense de 1024 neurones.
    - Les modèles VGG-16, ResNet18 & MobileNetV3 : Les modèles pré-entraînés.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    <hr style="opacity: 0.33">

    #### II.B.1 Classification fine (37 races) \[in-house \"VGG-8\"\].
    """)
    return


@app.cell
def _(Model_VGG8_InHouse, dframe_clean, model_summary):
    races = dframe_clean["race"].cat.categories.to_numpy()

    model_2B1 = Model_VGG8_InHouse(num_classes=len(races))
    _ = model_summary(model_2B1, input_size=(1, 3, 224, 224))
    return model_2B1, races


@app.cell
def _(
    device,
    loader_classif_train,
    loader_classif_val,
    model_2B1,
    torch,
    train_model_classif,
    training_results,
):
    def _train_model_2B1():

        print()
        print(f"using device: {device}")
        model_2B1.to(device)

        training_results["2B1"] = train_model_classif(
            model_2B1, loader_classif_train, loader_classif_val, 
            criterion = torch.nn.CrossEntropyLoss(),
            optimizer = torch.optim.Adam(model_2B1.parameters(), lr=0.001),
            label_key="race_id",
            epochs=10
        )


    _train_model_2B1()
    torch.save(model_2B1.state_dict(), 'weights/model_2B1.pth')
    print("Model saved to weights/model_2B1.pth")
    return


@app.cell
def _(mo):
    mo.md(r"""
    <hr style="opacity: 0.33">

    #### II.B.2 Classification fine (37 races) depuis un modèle pré-entrainé \[VGG-16\].
    """)
    return


@app.cell
def _(model_summary, models, races, torch):
    # On utilise un VGG16 pré-entraîné
    model_2B2 = models.vgg16(weights=models.VGG16_Weights.DEFAULT)

    # On remplace la toute dernière couche du bloc 'classifier' (index 6)
    # pour une classification fine (len(races) sorties)
    # in_features est généralement 4096 pour VGG16
    model_2B2.classifier[6] = torch.nn.Linear(model_2B2.classifier[6].in_features, len(races))

    _ = model_summary(model_2B2, input_size=(1, 3, 224, 224))
    return (model_2B2,)


@app.cell
def _(
    device,
    loader_classif_train,
    loader_classif_val,
    model_2B2,
    torch,
    train_model_classif,
    training_results,
):
    def _train_model_2B2():

        print()
        print(f"using device: {device}")
        model_2B2.to(device)

        ### Classifier training
        for param in model_2B2.parameters(): # 1. On gèle tous les paramètres
            param.requires_grad = False

        for param in model_2B2.classifier.parameters(): # 2. On dégèle la couche de classification
            param.requires_grad = True

        training_results["2B2"] = train_model_classif(
            model_2B2, loader_classif_train, loader_classif_val, 
            criterion = torch.nn.CrossEntropyLoss(),
            optimizer = torch.optim.Adam(model_2B2.parameters(), lr=0.001),
            label_key="race_id",
            epochs=5
        )

        ### Fine-tuning
        for param in model_2B2.parameters():  # 2. On débloque tous les params
            param.requires_grad = True

        training_results["2B2_ft"] = train_model_classif(
            model_2B2, loader_classif_train, loader_classif_val, 
            criterion = torch.nn.CrossEntropyLoss(),
            optimizer = torch.optim.Adam(model_2B2.parameters(), lr=0.00001),
            label_key="race_id",
            epochs=5
        )


    _train_model_2B2()
    torch.save(model_2B2.state_dict(), 'weights/model_2B2.pth')
    print("Model saved to weights/model_2B2.pth")
    return


@app.cell
def _(mo):
    mo.md(r"""
    <hr style="opacity: 0.33">

    #### II.B.3 Classification fine (37 races) depuis un modèle pré-entrainé \[ResNet18\].
    """)
    return


@app.cell
def _(model_summary, models, torch):
    # On utilise un ResNet18 pré-entraîné
    model_2B3 = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # On remplace la dernière couche (FC) pour une classification chien/chat (2 sorties)
    model_2B3.fc = torch.nn.Linear(model_2B3.fc.in_features, 2)

    _ = model_summary(model_2B3, input_size=(1, 3, 224, 224))
    return (model_2B3,)


@app.cell
def _(
    device,
    loader_classif_train,
    loader_classif_val,
    model_2B3,
    torch,
    train_model_classif,
    training_results,
):
    def _train_model_2B3():

        print()
        print(f"using device: {device}")
        model_2B3.to(device)

        ### Classifier training
        for param in model_2B3.parameters(): # 1. On gèle tous les paramètres
            param.requires_grad = False

        for param in model_2B3.fc.parameters(): # 2. On dégèle la couche de classification
            param.requires_grad = True

        training_results["2B3"] = train_model_classif(
            model_2B3, loader_classif_train, loader_classif_val, 
            criterion = torch.nn.CrossEntropyLoss(),
            optimizer = torch.optim.Adam(model_2B3.parameters(), lr=0.001),
            label_key="race_id",
            epochs=5
        )

        ### Fine-tuning
        for param in model_2B3.parameters():  # 2. On débloque tous les params
            param.requires_grad = True

        training_results["2B3_ft"] = train_model_classif(
            model_2B3, loader_classif_train, loader_classif_val, 
            criterion = torch.nn.CrossEntropyLoss(),
            optimizer = torch.optim.Adam(model_2B3.parameters(), lr=0.00001),
            label_key="race_id",
            epochs=5
        )


    _train_model_2B3()
    torch.save(model_2B3.state_dict(), 'weights/model_2B3.pth')
    print("Model saved to weights/model_2B3.pth")
    return


@app.cell
def _(mo):
    mo.md(r"""
    <hr style="opacity: 0.33">

    #### II.B.4 Classification fine (37 races) depuis un modèle pré-entrainé \[MobileNetV3\].
    """)
    return


@app.cell
def _(model_summary, models, torch):
    # On utilise un MobileNetV3 pré-entraîné
    model_2B4 = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)

    # On remplace la dernière couche (FC) pour une classification chien/chat (2 sorties)
    model_2B4.classifier[3] = torch.nn.Linear(model_2B4.classifier[3].in_features, 2)

    _ = model_summary(model_2B4, input_size=(1, 3, 224, 224))
    return (model_2B4,)


@app.cell
def _(
    device,
    loader_classif_train,
    loader_classif_val,
    model_2B4,
    torch,
    train_model_classif,
    training_results,
):
    def _train_model_2B4():

        print()
        print(f"using device: {device}")
        model_2B4.to(device)

        ### Classifier training
        for param in model_2B4.parameters(): # 1. On gèle tous les paramètres
            param.requires_grad = False

        for param in model_2B4.classifier.parameters(): # 2. On dégèle la couche de classification
            param.requires_grad = True

        training_results["2B4"] = train_model_classif(
            model_2B4, loader_classif_train, loader_classif_val, 
            criterion = torch.nn.CrossEntropyLoss(),
            optimizer = torch.optim.Adam(model_2B4.parameters(), lr=0.001),
            label_key="animal_id",
            epochs=5
        )

        ### Fine-tuning
        for param in model_2B4.parameters():  # 2. On débloque tous les params
            param.requires_grad = True

        training_results["2B4_ft"] = train_model_classif(
            model_2B4, loader_classif_train, loader_classif_val, 
            criterion = torch.nn.CrossEntropyLoss(),
            optimizer = torch.optim.Adam(model_2B4.parameters(), lr=0.00001),
            label_key="animal_id",
            epochs=5
        )


    _train_model_2B4()
    torch.save(model_2B4.state_dict(), 'weights/model_2B4.pth')
    print("Model saved to weights/model_2B4.pth")
    return


@app.cell
def _(mo):
    mo.md(r"""
    <hr style="opacity: 0.33">

    #### II.B.5 Comparaison des modèles.
    """)
    return


@app.cell
def _(plot_comparison, training_results):
    plot_comparison(training_results)
    return


@app.cell
def _(model_2B1, model_2B2, model_2B3, model_2B4, pd, training_results):
    pd.DataFrame({
        "model": [
            "In-House VGG-8", 
            "Pre-Trained VGG-16", 
            "Pre-Trained ResNet18", 
            "Pre-Trained MobileNetV3"
        ],
        "params": [
            sum(p.numel() for p in model_2B1.parameters()),
            sum(p.numel() for p in model_2B2.parameters()),
            sum(p.numel() for p in model_2B3.parameters()),
            sum(p.numel() for p in model_2B4.parameters())
        ],
        "training loss": [
            training_results["2B1"]["train_loss"][-1],
            training_results["2B2"]["train_loss"][-1],
            training_results["2B3"]["train_loss"][-1],
            training_results["2B4"]["train_loss"][-1]
        ],
        "validation accuracy": [
            training_results["2B1"]["validation_accuracy"][-1],
            training_results["2B2"]["validation_accuracy"][-1],
            training_results["2B3"]["validation_accuracy"][-1],
            training_results["2B4"]["validation_accuracy"][-1]
        ]
    })
    return


@app.cell
def _(model_2A1, plot_classification_results):
    plot_classification_results(model_2A1, label_key="race_id")
    return


@app.cell
def _(model_2A2, plot_classification_results):
    plot_classification_results(model_2A2, label_key="race_id")
    return


@app.cell
def _(model_2A3, plot_classification_results):
    plot_classification_results(model_2A3, label_key="race_id")
    return


@app.cell
def _(model_2A4, plot_classification_results):
    plot_classification_results(model_2A4, label_key="race_id")
    return


@app.cell
def _(mo):
    mo.md(r"""
    TODO: EXPLIQUER pourquoi faire un modèle qui fait les deux serait interressant --> explicabilité de la classification avec la segmentation (non LIME en vrai, ou SHAP) ?? (revoir interpretabilité !)
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## III. Segmentation
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    <hr style="opacity: 0.5">

    #### Préparation des données pour la segmentation.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    En nous basant sur l'analyse exploratoire, nous allons :
    - Utiliser un **RandomResizedCrop** pour normaliser les différences de zoom (biais de taille).
    - Retirer les images avec des anomalies sur les masques de segmentation.
    - Gérer le déséquilibre des classes (2/3 chiens, 1/3 chats) via une fonction de perte pondérée ou un échantillonnage adapté.

    TODO Pas sur pour le deuxième (weight=WEIGHT pas mis en place)
    TODO Gestion du Biais d'Espèce : je sais pas comment on peut faire
    """)
    return


@app.cell
def _(OxfordPetsDataset, augmentation_transform, identifiers):
    data_segment = OxfordPetsDataset(
        fetch_masks = True,
        joint_transform = augmentation_transform,
        dataset_root = "../data/oxford-pets",
        identifiers = identifiers
    )
    return (data_segment,)


@app.cell
def _(data_segment, torch):
    # can be set to None
    _generator = torch.Generator().manual_seed(42)

    # Split Train/Val (80% / 20%)
    _train_size = int(0.8 * len(data_segment))
    _val_size = len(data_segment) - _train_size

    subset_segment_train, subset_segment_val = torch.utils.data.random_split(
        data_segment, [_train_size, _val_size], 
        generator = _generator
    )
    return subset_segment_train, subset_segment_val


@app.cell
def _(DataLoader, subset_segment_train, subset_segment_val):
    loader_segment_train = DataLoader(subset_segment_train, batch_size=32, shuffle=True)
    loader_segment_val = DataLoader(subset_segment_val, batch_size=32, shuffle=False)
    return loader_segment_train, loader_segment_val


@app.cell
def _(mo):
    mo.md(r"""
    <hr style="opacity: 0.5">

    ### III.A Segmentation des animaux
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    Scores:
    -   accuracy                EZ
    -   iou                     oui
    -   dice score              oui
    -   mean pixel accuracy     oui
    -   precision / recall
    -   boundary iou
    """)
    return


@app.cell
def _(np, torch):
    def compute_iou(preds, masks, num_classes=3):
        # preds: (B, C, H, W), masks: (B, H, W)
        preds = torch.argmax(preds, dim=1)
        ious = []

        for cls in range(num_classes):
            intersection = ((preds == cls) & (masks == cls)).sum().float().item()
            union = ((preds == cls) | (masks == cls)).sum().float().item()

            if union == 0:
                ious.append(np.nan) # Pas de pixels de cette classe dans le batch
            else:
                ious.append(intersection / union)
        return ious
    return (compute_iou,)


@app.cell
def _(mo):
    mo.md(r"""
    <hr style="opacity: 0.33">

    #### III.A.1 UNet simple
    """)
    return


@app.cell
def _(nn, torch):
    ### UNet class (Ronneberger et al., 2015)

    # in the original paper (Ronneberger et al., 2015) the UNet 
    # has 4 convolution blocks and deals with 572*572 images
    # In our case we might need less convolution blocks to avoid overfitting
    # which is why the parameter conv_blocks exists
    # The original paper also doesn't uses padding in the convolutions
    # But it makes it harder to get the segmentation masks to the same size
    # as the images.

    class UNet(nn.Module):

        def __init__(self, conv_blocks=4, in_channels=3, out_channels=3):
            super(UNet, self).__init__()

            def ConvBlock(in_c, out_c):
                return nn.Sequential(
                    nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True)
                )

            # Listes pour stocker les modules
            self.encoder_convs = nn.ModuleList()
            self.decoder_ups = nn.ModuleList()
            self.decoder_convs = nn.ModuleList()

            self.nb_blocks = conv_blocks

            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

            # Encodeur
            in_c = in_channels
            out_c = 64
            for i in range(conv_blocks):
                self.encoder_convs.append(
                    ConvBlock(in_c, out_c)
                )

                in_c = out_c
                out_c *= 2

            # Bottleneck
            self.bottleneck = ConvBlock(in_c, out_c)

            # Décodeur
            up_in = out_c
            up_out = out_c // 2
            for i in range(conv_blocks):

                self.decoder_ups.append(
                    nn.ConvTranspose2d(up_in, up_out, kernel_size=2, stride=2)
                )

                # concatenate up_out + skip_connection_channels (qui est aussi up_out)
                self.decoder_convs.append(
                    ConvBlock(up_in, up_out)
                )

                up_in = up_out
                up_out //= 2

            # Conv finale
            self.final_conv = nn.Conv2d(up_in, out_channels, kernel_size=1)

        def forward(self, x):
            skip_list = []

            # Encodeur
            for conv in self.encoder_convs:
                x = conv(x)
                skip_list.append(x)
                x = self.pool(x)

            # Bottleneck
            x = self.bottleneck(x)

            # Décodeur
            skip_list_reversed = skip_list[::-1] # Inverser pour le décodeur
            for i in range(self.nb_blocks):
                x = self.decoder_ups[i](x)
                skip_connection = skip_list_reversed[i]

                x = torch.cat((skip_connection, x), dim=1)
                x = self.decoder_convs[i](x)

            return self.final_conv(x)
    return (UNet,)


@app.cell
def _(UNet, model_summary):
    model_UNet = UNet(conv_blocks=2)
    _ = model_summary(model_UNet, input_size=(1, 3, 224, 224))
    return (model_UNet,)


@app.cell
def _(compute_iou, device, mean, np, torch, tqdm, train_losses, val_ious):
    def train_model_segment(model, train_loader, val_loader, criterion, optimizer, epochs=10):

        history = {
            "train_loss": [], 
            "validation_iou_pet": [],
            "validation_iou_back": []
        }
        model.to(device)

        for epoch in range(epochs):

            # Phase d'entraînement
            model.train()
            running_loss = []
            pbar = tqdm(train_loader, desc=f"[train] epoch {epoch+1}/{epochs}")

            for batch in pbar:
                images = batch["image"].to(device)

                # On soustrait 1 pour passer de [1, 2, 3] à [0, 1, 2] (alignement avec CrossEntropy)
                masks = batch["mask"].to(device).long().squeeze(1) - 1 # .long().squeeze(1) ?

                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, masks)
                loss.backward()
                optimizer.step()

                running_loss.append(loss.item())
                pbar.set_postfix({"mean loss": f"{mean(running_loss):.04f}"})

            # Phase de validation
            model.eval()
            validation_ious = []
            pbar2 = tqdm(val_loader, desc=f"[validation] on validation data")

            with torch.no_grad():
                for batch in pbar2:
                    images = batch["image"].to(device)
                    masks = (batch["mask"].to(device).long().squeeze(1)) - 1

                    outputs = model(images)
                    batch_ious = compute_iou(outputs, masks)
                    validation_ious.append(batch_ious)

                    pbar2.set_postfix({
                        "pet iou": f"{batch_ious[0]:.03f}",
                        "back iou": f"{batch_ious[1]:.03f}"
                    })

            # Moyenne des IoU sur toutes les images (en ignorant les NaN)
            mean_ious = np.nanmean(np.array(val_ious), axis=0)

            history["train_loss"].append(mean(train_losses))
            history["val_iou_pet"].append(mean_ious[0]) # Classe 0 : Animal
            history["val_iou_bg"].append(mean_ious[1])  # Classe 1 : Fond

        print(f"training step completed.")
        return history
    return (train_model_segment,)


@app.cell
def _(
    device,
    loader_segment_train,
    loader_segment_val,
    model_UNet,
    torch,
    train_model_segment,
    training_results,
):
    def _train_model_UNet():

        print()
        print(f"using device: {device}")
        model_UNet.to(device)

        training_results["UNet"] = train_model_segment(
            model_UNet, loader_segment_train, loader_segment_val, 
            criterion = torch.nn.CrossEntropyLoss(ignore_index=2),
            optimizer = torch.optim.Adam(model_UNet.parameters(), lr=0.001),
            epochs=10
        )


    _train_model_UNet()
    torch.save(model_UNet.state_dict(), 'weights/model_UNet.pth')
    print("Model saved to weights/model_UNet.pth")
    return


@app.cell
def _(mo):
    mo.md(r"""
    <hr style="opacity: 0.33">

    #### III.A.2 Attention UNet.
    """)
    return


@app.cell
def _(nn):
    class AttentionGate(nn.Module):
        def __init__(self, F_g, F_l, F_int):
            super(AttentionGate, self).__init__()

            # F_g: nombre de canaux du décodeur (gate)
            # F_l: nombre de canaux de la skip connection
            # F_int: nombre de canaux intermédiaires

            self.W_g = nn.Sequential(
                nn.Conv2d(F_g, F_int, kernel_size=1, padding=0),
                nn.BatchNorm2d(F_int)
            )
            self.W_x = nn.Sequential(
                nn.Conv2d(F_l, F_int, kernel_size=1, padding=0),
                nn.BatchNorm2d(F_int)
            )
            self.psi = nn.Sequential(
                nn.Conv2d(F_int, 1, kernel_size=1, padding=0),
                nn.BatchNorm2d(1),
                nn.Sigmoid()
            )
            self.relu = nn.ReLU(inplace=True)

        def forward(self, g, x):
            g1 = self.W_g(g)
            x1 = self.W_x(x)
            psi = self.relu(g1 + x1)
            psi = self.psi(psi)
            return x * psi # On filtre la skip connection par l'attention
    return (AttentionGate,)


@app.cell
def _(AttentionGate, nn, torch):
    class AttentionUNet(nn.Module):

        def __init__(self, conv_blocks=4, in_channels=3, out_channels=3):
            super(AttentionUNet, self).__init__()

            def ConvBlock(in_c, out_c):
                return nn.Sequential(
                    nn.Conv2d(in_c, out_c, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_c, out_c, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True)
                )

            # Listes pour stocker les modules
            self.encoder_convs = nn.ModuleList()
            self.attention_gates = nn.ModuleList()
            self.decoder_ups = nn.ModuleList()
            self.decoder_convs = nn.ModuleList()

            self.nb_blocks = conv_blocks
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

            # Encodeur
            in_c = in_channels
            out_c = 64
            for i in range(conv_blocks):
                self.encoder_convs.append(
                    ConvBlock(in_c, out_c)
                )

                in_c = out_c
                out_c *= 2

            # Bottleneck
            self.bottleneck = ConvBlock(in_c, out_c)

            # Décodeur
            up_in = out_c
            up_out = out_c // 2
            for i in range(conv_blocks):

                self.decoder_ups.append(
                    nn.ConvTranspose2d(up_in, up_out, kernel_size=2, stride=2)
                )

                # Attention Gate: 
                # g (gate signal) vient du décodeur
                # x (skip) vient de l'encodeur
                # F_int est souvent la moitié de up_out askip
                self.attention_gates.append(
                    AttentionGate(F_g=up_out, F_l=up_out, F_int=up_out // 2)
                )

                self.decoder_convs.append(
                    ConvBlock(up_in, up_out)
                )

                up_in = up_out
                up_out //= 2

            # Conv finale
            self.final_conv = nn.Conv2d(up_in, out_channels, kernel_size=1)

        def forward(self, x):
            skip_list = []

            # Encodeur
            for conv in self.encoder_convs:
                x = conv(x)
                skip_list.append(x)
                x = self.pool(x)

            # Bottleneck
            x = self.bottleneck(x)

            # Décodeur
            skip_list_reversed = skip_list[::-1] # Inverser pour le décodeur
            for i in range(self.nb_blocks):
                g = self.decoder_ups[i](x)
                x_skipped = skip_list_reversed[i]
                x_attentioned = self.attention_gates[i](g, x_skipped)

                x = torch.cat((x_attentioned, g), dim=1)
                x = self.decoder_convs[i](x)

            return self.final_conv(x)
    return (AttentionUNet,)


@app.cell
def _(AttentionUNet, model_summary):
    model_AUNet = AttentionUNet(conv_blocks=2)
    _ = model_summary(model_AUNet, input_size=(1, 3, 224, 224))
    return (model_AUNet,)


@app.cell
def _(
    device,
    loader_segment_train,
    loader_segment_val,
    model_AUNet,
    torch,
    train_model_segment,
    training_results,
):
    def _train_model_AUNet():

        print()
        print(f"using device: {device}")
        model_AUNet.to(device)

        training_results["AUNet"] = train_model_segment(
            model_AUNet, loader_segment_train, loader_segment_val, 
            criterion = torch.nn.CrossEntropyLoss(ignore_index=2),
            optimizer = torch.optim.Adam(model_AUNet.parameters(), lr=0.001),
            epochs=10
        )


    _train_model_AUNet()
    torch.save(model_AUNet.state_dict(), 'weights/model_AUNet.pth')
    print("Model saved to weights/model_AUNet.pth")
    return


@app.cell
def _(mo):
    mo.md(r"""
    <hr style="opacity: 0.5">

    ### III.B Analyse comparative
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ---

    ## Conclusion
    """)
    return


if __name__ == "__main__":
    app.run()
