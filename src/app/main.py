from fastapi import FastAPI
from .routes import router

app = FastAPI(title="API CIIU",
              description="API para buscar códigos CIIU basados en descripciones de actividades económicas.",
              summary="API para búsqueda de CIIU",
              version="1.1.0")
app.include_router(router)
