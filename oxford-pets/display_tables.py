
import marimo as mo

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

import matplotlib.pyplot as plt
import seaborn as sn

from dataset import OxfordPetsDataset


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



