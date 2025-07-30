from .faiss_index import construir_index
from .embeddings import cargar_modelo, generar_embeddings
from .data_loader import cargar_datos
from fastapi import APIRouter, HTTPException
from .models import Categoria, CiiuRequest, CiiuResults, ErrorResponse, CiiuResponse
from .utils import limpiar_texto
from .config import EXCEL_CIIU_CACPE_PATH, EXCEL_CIIU_PATH
import faiss
import polars as pl

router = APIRouter()

# Globales (podrías mover a un estado compartido en memoria si escalas)

ciiu = cargar_datos(path=EXCEL_CIIU_PATH)
ciiu_cacpe = cargar_datos(path=EXCEL_CIIU_CACPE_PATH)

model = cargar_modelo()

embeddings_ciiu = generar_embeddings(model, ciiu["descripcion_limpia"].to_list())
embeddings_ciiu_cacpe = generar_embeddings(model, ciiu_cacpe["descripcion_limpia"].to_list())

index_ciiu = construir_index(embeddings_ciiu, ciiu)
index_ciiu_cacpe = construir_index(embeddings_ciiu_cacpe, ciiu_cacpe)

hashes_ciiu = ciiu.select(pl.col("codigo")).to_series().map_elements(
    hash, return_dtype=pl.Int64).to_list()
hashes_ciiu_cacpe = ciiu_cacpe.select(pl.col("codigo")).to_series().map_elements(
    hash, return_dtype=pl.Int64).to_list()


@router.post("/buscar_ciiu", response_model=CiiuResults, responses={404:{"model":ErrorResponse}})
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
    if req.categoria not in ["TODOS"] + ciiu["categoria"].unique().to_list():
        raise HTTPException(400, "Categoría no válida.")

    emb = model.encode([texto], convert_to_numpy=True)
    faiss.normalize_L2(emb)
    D, I = index_ciiu.search(emb, req.top_n * 5)

    resultados = []
    for dist, idx_hash in zip(D[0], I[0]):
        if dist < req.umbral_similitud:
            continue
        try:
            idx = hashes_ciiu.index(idx_hash)
        except ValueError:
            continue

        fila = ciiu.row(idx)
        # Filtro por categoría solo si no es "TODOS"
        if req.categoria != "TODOS" and fila[ciiu.columns.index("categoria")] != req.categoria:
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


@router.post("/buscar_ciiu_cacpe", response_model=CiiuResults, responses={404:{"model":ErrorResponse}})
def buscar_ciiu_cacpe(req: CiiuRequest):
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
    if req.categoria not in ["TODOS"] + ciiu_cacpe["categoria"].unique().to_list():
        raise HTTPException(400, "Categoría no válida.")

    emb = model.encode([texto], convert_to_numpy=True)
    faiss.normalize_L2(emb)
    D, I = index_ciiu_cacpe.search(emb, req.top_n * 5)

    resultados = []
    for dist, idx_hash in zip(D[0], I[0]):
        if dist < req.umbral_similitud:
            continue
        try:
            idx = hashes_ciiu_cacpe.index(idx_hash)
        except ValueError:
            continue

        fila = ciiu_cacpe.row(idx)
        # Filtro por categoría solo si no es "TODOS"
        if req.categoria != "TODOS" and fila[ciiu_cacpe.columns.index("categoria")] != req.categoria:
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
