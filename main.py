from enum import Enum
import logging
import numpy as np
import re
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import polars as pl
import torch
from sentence_transformers import SentenceTransformer
import faiss

# ----------------- CONFIGURACIÓN -----------------

app = FastAPI(title="API CIIU",
              description="API para buscar códigos CIIU basados en descripciones de actividades económicas.",
              summary="API para búsqueda de CIIU",
              version="1.0.0")

# Detectar si hay GPU disponible para usar en el modelo
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Usando dispositivo: {device}")


# ----------------- FUNCIONES AUXILIARES -----------------

def limpiar_texto(texto: str) -> str:
    if not isinstance(texto, str):
        return ""
    texto = texto.lower()  # Convertir a minúsculas
    texto = re.sub(r'[^\w\s]', '', texto)  # Eliminar caracteres especiales
    texto = re.sub(r'\s+', ' ', texto)  # Eliminar espacios extra
    return texto.strip()


# ----------------- CARGAR Y PROCESAR DATOS -----------------

df = pl.read_excel("ciiu.xlsx")
df = df.rename({
    "Código actividad económica": "codigo",
    "Descripción actividad económica": "descripcion",
    "Nivel": "categoria"
})

# Crear columna "descripcion_limpia"
df = df.with_columns([
    pl.col("descripcion").map_elements(limpiar_texto,
                                       return_dtype=pl.Utf8).alias("descripcion_limpia")
])

# Extraer descripciones limpias como lista
descripciones = df["descripcion_limpia"].to_list()

# ----------------- MODELO EMBEDDINGS -----------------

model = SentenceTransformer(
    "sentence-transformers/paraphrase-multilingual-mpnet-base-v2", device=device)
cii_embeddings = model.encode(
    descripciones, convert_to_numpy=True, show_progress_bar=True)

# ----------------- FAISS GPU -----------------

dim = cii_embeddings.shape[1]

# Normalizar embeddings para similitud coseno
faiss.normalize_L2(cii_embeddings)

# Crear índice FAISS en GPU si es posible
if faiss.get_num_gpus() > 0:
    res = faiss.StandardGpuResources()
    index_flat = faiss.IndexFlatIP(dim)
    index = faiss.index_cpu_to_gpus_list(res, 0, index_flat)
else:
    index = faiss.IndexFlatIP(dim)

# Asociar índices con IDs únicos (por ejemplo, posición)
index = faiss.IndexIDMap(index)
ids = np.array(df.select("codigo").to_series().map_elements(
    hash, return_dtype=pl.Int64)).astype("int64")
index.add_with_ids(cii_embeddings, ids)


# ----------------- API MODELS -----------------

class Categoria(str, Enum):
    SECCION = "SECCION"
    GRUPO = "GRUPO"
    SUBGRUPO = "SUBGRUPO"
    CLASE = "CLASE"
    SUBCLASE = "SUBCLASE"
    ACTIVIDAD = "ACTIVIDAD"
    SUBNIVEL = "SUBNIVEL"


class CiiuRequest(BaseModel):
    descripcion: str
    top_n: int = 3
    categoria:  Categoria | None = None
    umbral_similitud: float = 0.6


class CiiuResponse(BaseModel):
    codigo: str
    descripcion: str
    categoria: Categoria | None = None
    similitud: float


class CiiuResults(BaseModel):
    resultados: list[CiiuResponse]


class ErrorResponse(BaseModel):
    detail: str
# ----------------- ENDPOINT -----------------


@app.post("/buscar_ciiu", response_model=CiiuResults, responses={404: {"model": ErrorResponse}})
def buscar_ciiu(req: CiiuRequest):

    texto_limpio = limpiar_texto(req.descripcion)
    if not texto_limpio:
        raise HTTPException(
            status_code=400, detail="La descripción no puede estar vacía.")
    if len(texto_limpio) < 3:
        raise HTTPException(
            status_code=400, detail="La descripción debe tener al menos 3 caracteres.")
    if req.top_n <= 0:
        raise HTTPException(
            status_code=400, detail="El número de resultados debe ser mayor que 0.")
    if req.umbral_similitud < 0 or req.umbral_similitud > 1:
        raise HTTPException(
            status_code=400, detail="El umbral de similitud debe estar entre 0 y 1.")
    if req.categoria and req.categoria not in df["categoria"].unique().to_list():
        raise HTTPException(
            status_code=400, detail="La categoría especificada no es válida.")

    consulta_emb = model.encode([texto_limpio], convert_to_numpy=True)
    faiss.normalize_L2(consulta_emb)

    # Buscar más resultados para filtrar después
    D, I = index.search(consulta_emb, req.top_n * 5)

    resultados = []
    codigos_hash = df.select(pl.col("codigo")).to_series(
    ).map_elements(hash, return_dtype=pl.Int64).to_list()

    for dist, idx_hash in zip(D[0], I[0]):
        if dist < req.umbral_similitud:
            continue

        try:
            idx_df = codigos_hash.index(idx_hash)
        except ValueError:
            continue  # hash no encontrado

        fila = df.row(idx_df)

        if req.categoria and fila[df.columns.index("categoria")] != req.categoria:
            continue

        resultado = CiiuResponse(
            codigo=fila[df.columns.index("codigo")],
            descripcion=fila[df.columns.index("descripcion")],
            categoria=fila[df.columns.index("categoria")],
            similitud=round(float(dist), 4)
        )
        resultados.append(resultado)

        if len(resultados) >= req.top_n:
            break

    if not resultados:
        raise HTTPException(
            status_code=404, detail="No se encontraron resultados relevantes.")

    return CiiuResults(resultados=resultados)
