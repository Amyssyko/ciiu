import faiss
import numpy as np
import polars as pl


def construir_index(embeddings, df: pl.DataFrame):
    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]

    if faiss.get_num_gpus() > 0:
        res = faiss.StandardGpuResources()
        index_flat = faiss.IndexFlatIP(dim)
        index = faiss.index_cpu_to_gpus_list(res, 0, index_flat)
    else:
        index = faiss.IndexFlatIP(dim)

    index = faiss.IndexIDMap(index)
    ids = np.array(df.select("codigo").to_series().map_elements(
        hash, return_dtype=pl.Int64)).astype("int64")
    index.add_with_ids(embeddings, ids)

    return index
