import os
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EXCEL_CIIU_V4_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "ciiu.xlsx")
EXCEL_CIIU_V2_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "ciiu_2.0.xlsx")
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
