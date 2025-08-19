import os
from typing import Optional, cast

import faiss
import polars as pl
from fastapi import HTTPException
import numpy as np

from .config import EXCEL_CIIU_V4_PATH, EXCEL_CIIU_V2_PATH, DESCRIPCIONES_PATH
from .data_loader import cargar_datos, cargar_descripciones_adicionales
from .embeddings import cargar_modelo, generar_embeddings
from .faiss_index import construir_index
from .models import CiiuRequest, CiiuResults, CiiuResponse
from .utils import normalize_for_nlp

# Estado global (inicializado al importar y recargable)
model = cargar_modelo()
ciiu_v4: pl.DataFrame | None = None
ciiu_v2: pl.DataFrame | None = None
index_ciiu_v4 = None
index_ciiu_v2 = None
hashes_ciiu: list[int] = []
hashes_ciiu_v2: list[int] = []

# Corpus adicional para expansión de consulta
extra_df: pl.DataFrame | None = None
extra_texts: list[str] = []
emb_extra_mat: np.ndarray | None = None


def _construir_indices() -> None:
    """(Re)construye índices e embeddings de datos oficiales y del corpus extra.

    Efectos:
        - Carga CIIU v4 y v2 desde Excel y normaliza.
        - Genera embeddings y construye índices FAISS (IP + L2 normalize).
        - Calcula lista de hashes por código para mapear resultados FAISS.
        - Carga descripciones adicionales (si existen) y precalcula embeddings
          normalizados para expansión de consulta.
    """
    global ciiu_v4, ciiu_v2, index_ciiu_v4, index_ciiu_v2, hashes_ciiu, hashes_ciiu_v2
    global extra_df, extra_texts, emb_extra_mat

    # Cargar datos base
    ciiu_v4 = cargar_datos(path=EXCEL_CIIU_V4_PATH)
    ciiu_v2 = cargar_datos(path=EXCEL_CIIU_V2_PATH)

    # Embeddings e índices principales SOLO con datos oficiales
    emb_v4 = generar_embeddings(model, ciiu_v4["descripcion_limpia"].to_list())
    emb_v2 = generar_embeddings(model, ciiu_v2["descripcion_limpia"].to_list())
    index_ciiu_v4 = construir_index(emb_v4, ciiu_v4)
    index_ciiu_v2 = construir_index(emb_v2, ciiu_v2)

    hashes_ciiu = ciiu_v4.select(pl.col("codigo")).to_series().map_elements(
        hash, return_dtype=pl.Int64).to_list()
    hashes_ciiu_v2 = ciiu_v2.select(pl.col("codigo")).to_series().map_elements(
        hash, return_dtype=pl.Int64).to_list()

    # Cargar corpus extra para enriquecimiento de consulta (opcional)
    extra_df = cargar_descripciones_adicionales(DESCRIPCIONES_PATH)
    extra_texts = extra_df["descripcion_limpia"].to_list(
    ) if extra_df.height > 0 else []
    if extra_texts:
        emb_extra = generar_embeddings(model, extra_texts)
        if emb_extra.dtype != np.float32:
            emb_extra = emb_extra.astype(np.float32)
        faiss.normalize_L2(emb_extra)
        emb_extra_mat = emb_extra
    else:
        emb_extra_mat = None


def recargar_indices() -> dict:
    """Recarga datasets, corpus extra e índices en caliente.

    Returns:
        Resumen con contadores de registros en v4, v2 y corpus extra.
    """
    _construir_indices()
    return {
        "v4_registros": int(ciiu_v4.height) if ciiu_v4 is not None else 0,
        "v2_registros": int(ciiu_v2.height) if ciiu_v2 is not None else 0,
        "extra_registros": int(extra_df.height) if extra_df is not None else 0,
    }


# Construcción inicial al importar
_construir_indices()


def _buscar(req: CiiuRequest, df: pl.DataFrame, index, hashes) -> CiiuResults:
    """Ejecución de búsqueda sobre un índice y DataFrame dados.

    Flujo:
        1) Validación y normalización de la descripción.
        2) Expansión de consulta con corpus extra (opcional) usando similitud coseno.
        3) Embeddings de la consulta y búsqueda FAISS (k=top_n*5).
        4) Mapeo de IDs a filas por hash(codigo), filtrado por umbral y categoría.

    Raises:
        HTTPException 400: parámetros inválidos.
        HTTPException 404: sin resultados sobre el umbral.
    """
    texto = normalize_for_nlp(req.descripcion)

    # Validaciones
    if not texto:
        raise HTTPException(400, "La descripción no puede estar vacía.")
    if len(texto) < 3:
        raise HTTPException(
            400, "La descripción debe tener al menos 3 caracteres.")
    if req.top_n <= 0:
        raise HTTPException(
            400, "El número de resultados debe ser mayor que 0.")
    if not (0 <= req.umbral_similitud <= 1):
        raise HTTPException(400, "El umbral debe estar entre 0 y 1.")
    if req.categoria not in ["TODOS"] + df["categoria"].unique().to_list():
        raise HTTPException(400, "Categoría no válida.")

    # Enriquecimiento opcional de consulta con corpus extra
    consulta = texto
    if emb_extra_mat is not None and extra_texts:
        emb_q = model.encode([texto], convert_to_numpy=True)
        if emb_q.dtype != np.float32:
            emb_q = emb_q.astype(np.float32)
        faiss.normalize_L2(emb_q)
        # Similaridades por producto interno (coseno)
        sims = np.dot(emb_q, emb_extra_mat.T)[0]  # shape (N_extra,)
        # Top-k aproximado
        k = min(5, sims.shape[0])
        if k > 0:
            top_idx = np.argpartition(-sims, kth=k-1)[:k]
            orden = top_idx[np.argsort(-sims[top_idx])]
            seleccion = [extra_texts[i] for i in orden if sims[i] >= 0.4][:3]
        else:
            seleccion = []
        if seleccion:
            consulta = " ".join([texto] + seleccion)

    # Embedding + búsqueda FAISS (coseno via L2-normalización + IP)
    emb = model.encode([consulta], convert_to_numpy=True)
    faiss.normalize_L2(emb)
    D, I = index.search(emb, req.top_n * 5)

    resultados = []
    for dist, idx_hash in zip(D[0], I[0]):
        if dist < req.umbral_similitud:
            continue
        try:
            idx = hashes.index(idx_hash)
        except ValueError:
            continue

        fila = df.row(idx)
        if req.categoria != "TODOS" and fila[df.columns.index("categoria")] != req.categoria:
            continue

        resultados.append(CiiuResponse(
            codigo=fila[0],
            descripcion=fila[1],
            categoria=fila[2],
            similitud=round(float(dist), 4)
        ))

        if len(resultados) >= req.top_n:
            break

    if not resultados:
        raise HTTPException(404, "No se encontraron resultados relevantes.")

    return CiiuResults(resultados=resultados)


def buscar_ciiu_v4(req: CiiuRequest) -> CiiuResults:
    """Busca en el dataset CIIU v4 usando el índice FAISS correspondiente."""
    if ciiu_v4 is None or index_ciiu_v4 is None:
        _construir_indices()
    assert ciiu_v4 is not None and index_ciiu_v4 is not None
    return _buscar(req, cast(pl.DataFrame, ciiu_v4), index_ciiu_v4, hashes_ciiu)


def buscar_ciiu_v2(req: CiiuRequest) -> CiiuResults:
    """Busca en el dataset CIIU v2 usando el índice FAISS correspondiente."""
    if ciiu_v2 is None or index_ciiu_v2 is None:
        _construir_indices()
    assert ciiu_v2 is not None and index_ciiu_v2 is not None
    return _buscar(req, cast(pl.DataFrame, ciiu_v2), index_ciiu_v2, hashes_ciiu_v2)
