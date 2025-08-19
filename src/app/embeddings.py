from typing import List
import numpy as np
from sentence_transformers import SentenceTransformer
from .config import EMBEDDING_MODEL, DEVICE


def cargar_modelo() -> SentenceTransformer:
    """Carga el modelo de embeddings configurado.

    Returns:
        Instancia de SentenceTransformer lista para usar en el dispositivo seleccionado.
    """
    return SentenceTransformer(EMBEDDING_MODEL, device=DEVICE)


def generar_embeddings(model: SentenceTransformer, textos: List[str]) -> np.ndarray:
    """Genera embeddings para una lista de textos.

    Args:
        model: Modelo de Sentence-Transformers.
        textos: Secuencia de textos ya normalizados.

    Returns:
        Matriz numpy de shape (N, dim) con dtype float32.
    """
    emb = model.encode(textos, convert_to_numpy=True)
    if emb.dtype != np.float32:
        emb = emb.astype(np.float32)
    return emb
