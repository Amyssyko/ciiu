import os
from typing import Optional

import polars as pl
from .utils import normalize_for_nlp


def cargar_datos(path: str) -> pl.DataFrame:
    """Carga datos oficiales CIIU desde un Excel y normaliza descripciones.

    Espera columnas con encabezados originales: "Código actividad económica",
    "Descripción actividad económica" y "Nivel". Renombra a
    (codigo, descripcion, categoria) y agrega "descripcion_limpia".

    Args:
        path: Ruta absoluta del archivo Excel de CIIU.

    Returns:
        DataFrame de Polars con columnas: codigo, descripcion,
        categoria, descripcion_limpia.
    """
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


def cargar_descripciones_adicionales(path: Optional[str]) -> pl.DataFrame:
    """Carga un archivo de descripciones adicionales opcional.

    Soporta cabeceras flexibles: si existe columna "descripcion" se usa tal cual;
    si existe "DESCRIPCION" (mayúsculas) se renombra. No requiere "codigo" ni "categoria".

    Args:
        path: Ruta del Excel con descripciones extra o None.

    Returns:
        DataFrame con columnas: descripcion, descripcion_limpia.
        Si el archivo no existe o las columnas son inesperadas, retorna vacío.
    """
    if not path or not os.path.exists(path):
        return pl.DataFrame({"descripcion": [], "descripcion_limpia": []})

    df = pl.read_excel(path)
    if "descripcion" not in df.columns and "DESCRIPCION" in df.columns:
        df = df.rename({"DESCRIPCION": "descripcion"})
    if "descripcion" not in df.columns:
        # Estructura inesperada: devolver vacío para evitar romper la app
        return pl.DataFrame({"descripcion": [], "descripcion_limpia": []})

    df = df.select([pl.col("descripcion")])
    df = df.with_columns([
        pl.col("descripcion").map_elements(normalize_for_nlp,
                                           return_dtype=pl.Utf8).alias("descripcion_limpia")
    ])
    return df
