from .faiss_index import construir_index
from .embeddings import cargar_modelo, generar_embeddings
from .data_loader import cargar_datos
from fastapi import APIRouter, HTTPException
from .models import Categoria, CiiuRequest, CiiuResults, ErrorResponse, CiiuResponse
from .utils import limpiar_texto
from .config import EXCEL_CIIU_V4_PATH, EXCEL_CIIU_V2_PATH
import faiss
import polars as pl

router = APIRouter()

# Globales (podrías mover a un estado compartido en memoria si escalas)

ciiu_v4 = cargar_datos(path=EXCEL_CIIU_V4_PATH)
ciiu_v2 = cargar_datos(path=EXCEL_CIIU_V2_PATH)

model = cargar_modelo()

embeddings_ciiu_v4 = generar_embeddings(model, ciiu_v4["descripcion_limpia"].to_list())
embeddings_ciiu_v2 = generar_embeddings(model, ciiu_v2["descripcion_limpia"].to_list())

index_ciiu_v4 = construir_index(embeddings_ciiu_v4, ciiu_v4)
index_ciiu_v2 = construir_index(embeddings_ciiu_v2, ciiu_v2)

hashes_ciiu = ciiu_v4.select(pl.col("codigo")).to_series().map_elements(
    hash, return_dtype=pl.Int64).to_list()
hashes_ciiu_v2 = ciiu_v2.select(pl.col("codigo")).to_series().map_elements(
    hash, return_dtype=pl.Int64).to_list()


@router.post("/buscar_ciiu_v4", response_model=CiiuResults, responses={404:{"model":ErrorResponse}})
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
    if req.categoria not in ["TODOS"] + ciiu_v4["categoria"].unique().to_list():
        raise HTTPException(400, "Categoría no válida.")

    emb = model.encode([texto], convert_to_numpy=True)
    faiss.normalize_L2(emb)
    D, I = index_ciiu_v4.search(emb, req.top_n * 5)

    resultados = []
    for dist, idx_hash in zip(D[0], I[0]):
        if dist < req.umbral_similitud:
            continue
        try:
            idx = hashes_ciiu.index(idx_hash)
        except ValueError:
            continue

        fila = ciiu_v4.row(idx)
        # Filtro por categoría solo si no es "TODOS"
        if req.categoria != "TODOS" and fila[ciiu_v4.columns.index("categoria")] != req.categoria:
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


@router.post("/buscar_ciiu_v2", response_model=CiiuResults, responses={404:{"model":ErrorResponse}})
def buscar_ciiu_v2(req: CiiuRequest):
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
    if req.categoria not in ["TODOS"] + ciiu_v2["categoria"].unique().to_list():
        raise HTTPException(400, "Categoría no válida.")

    emb = model.encode([texto], convert_to_numpy=True)
    faiss.normalize_L2(emb)
    D, I = index_ciiu_v2.search(emb, req.top_n * 5)

    resultados = []
    for dist, idx_hash in zip(D[0], I[0]):
        if dist < req.umbral_similitud:
            continue
        try:
            idx = hashes_ciiu_v2.index(idx_hash)
        except ValueError:
            continue

        fila = ciiu_v2.row(idx)
        # Filtro por categoría solo si no es "TODOS"
        if req.categoria != "TODOS" and fila[ciiu_v2.columns.index("categoria")] != req.categoria:
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
