from .faiss_index import construir_index
from .embeddings import cargar_modelo, generar_embeddings
from .data_loader import cargar_datos
from fastapi import APIRouter, HTTPException
from .models import CiiuRequest, CiiuResults, ErrorResponse, CiiuResponse
from .utils import limpiar_texto

import faiss
import numpy as np
import polars as pl

router = APIRouter()

# Globales (podrías mover a un estado compartido en memoria si escalas)

df = cargar_datos()
model = cargar_modelo()
embeddings = generar_embeddings(model, df["descripcion_limpia"].to_list())
index = construir_index(embeddings, df)
hashes = df.select(pl.col("codigo")).to_series().map_elements(
    hash, return_dtype=pl.Int64).to_list()


@router.post("/buscar_ciiu", response_model=CiiuResults, responses={404: {"model": ErrorResponse}})
def buscar_ciiu(req: CiiuRequest):
    texto = limpiar_texto(req.descripcion)

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
    if req.categoria and req.categoria not in df["categoria"].unique().to_list():
        raise HTTPException(400, "Categoría no válida.")

    emb = model.encode([texto], convert_to_numpy=True)
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
        if req.categoria and fila[df.columns.index("categoria")] != req.categoria:
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
