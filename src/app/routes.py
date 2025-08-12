from .faiss_index import construir_index
from .embeddings import cargar_modelo, generar_embeddings
from .data_loader import cargar_datos
from fastapi import APIRouter, HTTPException
from .models import Categoria, CiiuRequest, CiiuResults, ErrorResponse, CiiuResponse
from .utils import normalize_for_nlp
from .search_service import buscar_ciiu_v4, buscar_ciiu_v2

router = APIRouter()


@router.post("/buscar_ciiu_v4", response_model=CiiuResults, responses={404: {"model": ErrorResponse}})
def buscar_ciiu(req: CiiuRequest):
    return buscar_ciiu_v4(req)


@router.post("/buscar_ciiu_v2", response_model=CiiuResults, responses={404: {"model": ErrorResponse}})
def buscar_ciiu_v2(req: CiiuRequest):
    return buscar_ciiu_v2(req)
