import os
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BASE_DIR = os.path.join(os.path.dirname(__file__), "..", "..")
EXCEL_CIIU_V4_PATH = os.path.join(BASE_DIR, "ciiu.xlsx")
EXCEL_CIIU_V2_PATH = os.path.join(BASE_DIR, "ciiu_2.0.xlsx")

# Ruta opcional para archivo de descripciones adicionales.
DESCRIPCIONES_PATH = os.path.join(BASE_DIR, "descripciones.xlsx")

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
