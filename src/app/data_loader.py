import polars as pl
from .utils import limpiar_texto
from .config import EXCEL_PATH

def cargar_datos():
    df = pl.read_excel(EXCEL_PATH).rename({
        "Código actividad económica": "codigo",
        "Descripción actividad económica": "descripcion",
        "Nivel": "categoria"
    })

    df = df.with_columns([
        pl.col("descripcion").map_elements(limpiar_texto, return_dtype=pl.Utf8).alias("descripcion_limpia")
    ])

    return df
