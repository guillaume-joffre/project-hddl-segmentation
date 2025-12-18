import marimo

__generated_with = "0.18.4"
app = marimo.App()


@app.cell
def _(mo):
    mo.md(r"""
    # Mini-projet 3 : ViT vs CNN
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Import des packages
    """)
    return


@app.cell
def _():
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, Dataset
    import torchvision
    import torchvision.transforms as transforms
    import torchvision.datasets as datasets

    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd

    from tqdm import tqdm
    import time
    import random
    import os
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Choix des datasets et justifications
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Explication de l'architecture Vision Transformers
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Implémentation d'une architecure ViT
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Implémentation d'une architecture CNN
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Comparaison des modèles selon les datasets
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Conclusion
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
 
    """)
    return


if __name__ == "__main__":
    app.run()
