from fastapi import APIRouter
from .models import CiiuRequest, CiiuResults, ErrorResponse
from .search_service import (
    buscar_ciiu_v4 as service_buscar_ciiu_v4,
    buscar_ciiu_v2 as service_buscar_ciiu_v2,
    recargar_indices as service_recargar_indices,
)

router = APIRouter()


@router.post("/buscar_ciiu_v4", response_model=CiiuResults, responses={404: {"model": ErrorResponse}})
def buscar_ciiu(req: CiiuRequest):
    """Endpoint para buscar en CIIU v4 por descripción."""
    return service_buscar_ciiu_v4(req)


@router.post("/buscar_ciiu_v2", response_model=CiiuResults, responses={404: {"model": ErrorResponse}})
def buscar_ciiu_v2(req: CiiuRequest):
    """Endpoint para buscar en CIIU v2 por descripción."""
    return service_buscar_ciiu_v2(req)


@router.post("/recargar_indices")
def recargar():
    """Reconstruye índices tras cambios en archivos fuente o descripciones."""
    return service_recargar_indices()
