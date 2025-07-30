from sentence_transformers import SentenceTransformer
from .config import EMBEDDING_MODEL, DEVICE

def cargar_modelo():
    return SentenceTransformer(EMBEDDING_MODEL, device=DEVICE)

def generar_embeddings(model, textos):
    return model.encode(textos, convert_to_numpy=True, show_progress_bar=True)
