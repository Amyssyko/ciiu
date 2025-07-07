import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
EXCEL_PATH = "ciiu.xlsx"
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
