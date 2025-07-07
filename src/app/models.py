from pydantic import BaseModel
from typing import Optional, List
from enum import Enum


# ----------------- API MODELS -----------------

# ENUM Categorias

class Categoria(str, Enum):
    SECCION = "SECCION"
    GRUPO = "GRUPO"
    SUBGRUPO = "SUBGRUPO"
    CLASE = "CLASE"
    SUBCLASE = "SUBCLASE"
    ACTIVIDAD = "ACTIVIDAD"
    SUBNIVEL = "SUBNIVEL"

# Modelos de solicitud y respuesta para la API categoria default ACTIVIDAD


class CiiuRequest(BaseModel):
    descripcion: str
    top_n: int = 7
    categoria:  Optional[Categoria] = Categoria.ACTIVIDAD
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
