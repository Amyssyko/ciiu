import faiss
import numpy as np
import polars as pl
from typing import Any


def construir_index(embeddings: np.ndarray, df: pl.DataFrame, ids: np.ndarray | None = None):
    """Construye un índice FAISS (IP) con mapeo de IDs por código.

    Normaliza L2 los embeddings (float32) para que el producto interno equivalga
    a la similitud coseno. Envuelve el índice con IDMap y asigna IDs por
    hash(codigo) salvo que se suministren explícitamente.

    Args:
        embeddings: Matriz (N, dim) float32.
        df: DataFrame con columna "codigo" para derivar IDs si no se pasan.
        ids: Opcional, arreglo int64 de IDs a utilizar.

    Returns:
        Índice FAISS listo para búsqueda con search(x, k).
    """
    # Asegurar dtype float32 antes de normalizar
    if embeddings.dtype != np.float32:
        embeddings = embeddings.astype(np.float32)
    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]

    # Usar FAISS CPU con producto interno (coseno tras normalización)
    index = faiss.IndexFlatIP(dim)

    index = faiss.IndexIDMap(index)
    if ids is None:
        ids = np.array(
            df.select("codigo").to_series().map_elements(
                hash, return_dtype=pl.Int64)
        ).astype("int64").reshape(-1)
    else:
        ids = np.asarray(ids, dtype="int64").reshape(-1)
    # Compatibilidad con distintas firmas de FAISS Python

    def _add_with_ids(idx: Any, x: np.ndarray, xids: np.ndarray) -> None:
        try:
            idx.add_with_ids(x, xids)  # pybind signature
        except TypeError:
            idx.add_with_ids(x.shape[0], x, xids)  # swig signature

    _add_with_ids(index, embeddings, ids)

    return index
