import os

import polars as pl
from .utils import normalize_for_nlp


def cargar_datos(path: str):
    df = pl.read_excel(path).rename({
        "Código actividad económica": "codigo",
        "Descripción actividad económica": "descripcion",
        "Nivel": "categoria"
    })

    df = df.with_columns([
        pl.col("descripcion").map_elements(normalize_for_nlp,
                                           return_dtype=pl.Utf8).alias("descripcion_limpia")
    ])

    return df
