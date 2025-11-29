import marimo

__generated_with = "0.18.1"
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
    return DataLoader, Image, OxfordPetsDataset, mo, np, pd, plt, sn, torch


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
def _():
    ### --- Fonctions pour l'affichage de tables ----------------- ###

    # potentiellement
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
    ### --- Récupération des DataFrames -------------------------- ###

    dframe = data.get_dataframe()
    to_display = dframe.sample(5, random_state=46)
    dframe = dframe.drop(columns=["bboxfile"])
    to_display
    return (dframe,)


@app.cell
def _(mo):
    mo.md(r"""
    Toutes les images du Oxford-IIIT Pet Dataset ont un masque de segmentation associé, mais toutes n'ont pas un fichier describant une bounding box pour la tête de l'animal. C'est pourquoi la colonne "bboxfile" à des valeurs manquantes. Comme il s'agit d'un mini-projet portant sur la segmentation et non pas sur la detection d'objet nous allons ignorer la colonne et revenir à un dataset sans valeurs manquantes.
    """)
    return


@app.cell
def _(mo, pd, plt, sn):
    ### --- Fonctions utiles pour la suite ----------------------- ###

    def summary(df: pd.DataFrame) -> pd.DataFrame:
        over_total = " / "+str(len(df))
        infos = pd.DataFrame(data={
            "dtype": df.dtypes,
            "na/total": df.isnull().sum().map(str) + over_total,
            "unique": df.nunique()
        })

        infos.index.names = ["column"]
        return infos


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
    
    return plot_counts, summary, summary_categorical


@app.cell
def _(dframe, summary):
    ### --- Détails du DataFrame --------------------------------- ###

    summary(dframe)

    return


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
def _(DataLoader, OxfordPetsDataset, data, plt):
    ### --- Animal / Race bar plots ------------------------------ ###

    def fig_2():

        loader = DataLoader(data, batch_size=5, shuffle=True)
        images, animals, races, masks = next(iter(loader)).values()
        fig, axes = plt.subplots(2, 5, figsize=(13,7))

        for idx in range(5):

            img = OxfordPetsDataset.default_transform_unnormalize(images[idx])
        
            # ax1: image
            ax1 = axes[0][idx]
            ax1.imshow(img)
            ax1.axis("off")
            ax1.set_title(f"{data.index2animal(animals[idx])} / {data.index2race(races[idx])}")

            # ax2: segmentation mask
            ax2 = axes[2][idx]
            ax2.imshow(masks[idx].squeeze(), cmap="gray")
            ax2.axis("off")

        fig.supxlabel("Test: Some images along with their segmentation mask", size=15.5, style="italic")
        fig.subplots_adjust(hspace=0.05)
        fig.tight_layout()
        plt.show()

    fig_2()
    return


@app.cell
def _(Image, dframe, dset, np, pd, torch):
    ### LES MASQUES SONT-ILS EQUILIBRES ###
    def count_mask_levels():
        counts = np.zeros((len(dframe), 3))
        for row in dframe.itertuples():
            mask = dset.mask_transform(Image.open(row.maskfile)).squeeze(0)
            flat = mask.view(-1)

            # bincount on values {1,2,3}
            counts[row.Index,:] = torch.bincount(flat, minlength=4)[1:]  # index 0 unused

        return pd.DataFrame(counts, columns=["mask_1","mask_2","mask_3"])

    levels = pd.merge(
        dframe[["identifier"]], 
        count_mask_levels(), 
        left_index=True, 
        right_index=True
    )

    levels
    return (levels,)


@app.cell
def _(dset, plt):
    def fig_3(idx: int):

        image, mask, animal, race = dset[idx]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.5,5))

        img = image.permute(1, 2, 0).numpy()
        img = (img - img.min()) / (img.max() - img.min())   # quick unnormalize
        ax1.imshow(img)
        ax1.axis("off")

        ax2.imshow(mask, cmap="gray")
        ax2.axis("off")

        fig.supxlabel("Fig X. An example of wrong/bizarre mask", y=-0.02, size=15.5, style="italic")
        fig.tight_layout(rect=[0, 0, 1, 0.8])
        fig.subplots_adjust(hspace=0.05)
        plt.show()

    fig_3(136)
    return


@app.cell
def _(levels):
    levels[(levels == 0).any(axis=1)]
    return


@app.cell
def _(mo):
    mo.md(r"""
 
    """)
    return


if __name__ == "__main__":
    app.run()
