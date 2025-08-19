from pydantic import BaseModel
from typing import Optional, List
from enum import Enum


# ----------------- API MODELS -----------------

# ENUM Categorias

class Categoria(str, Enum):
    """Categorías válidas del sistema CIIU."""
    TODOS = "TODOS"
    SECCION = "SECCION"
    GRUPO = "GRUPO"
    SUBGRUPO = "SUBGRUPO"
    CLASE = "CLASE"
    SUBCLASE = "SUBCLASE"
    ACTIVIDAD = "ACTIVIDAD"
    SUBNIVEL = "SUBNIVEL ACTIVIDAD"


# Modelos de solicitud y respuesta para la API categoria default ACTIVIDAD


class CiiuRequest(BaseModel):
    """Solicitud de búsqueda CIIU.

    Atributos:
        descripcion: Texto a buscar.
        top_n: Número máximo de resultados (por defecto 7).
        categoria: Filtro de categoría CIIU; por defecto ACTIVIDAD.
        umbral_similitud: Umbral de similitud [0,1] para filtrar resultados.
    """
    descripcion: str
    top_n: int = 7
    categoria: Categoria = Categoria.ACTIVIDAD
    umbral_similitud: float = 0.6


class CiiuResponse(BaseModel):
    """Elemento de respuesta con código, descripción, categoría y similitud."""
    codigo: str
    descripcion: str
    categoria: None | Categoria = None
    similitud: float


class CiiuResults(BaseModel):
    """Contenedor de resultados de búsqueda CIIU."""
    resultados: list[CiiuResponse]


class ErrorResponse(BaseModel):
    """Modelo de error estándar para respuestas 4xx/5xx."""
    detail: str
