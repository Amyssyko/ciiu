# 📊 Proyecto CIIU

Este repositorio contiene el proyecto **CIIU**, una API para búsqueda semántica de actividades económicas CIIU (rev 4.0 y 2.0) usando embeddings de Sentence-Transformers y un índice FAISS. La solución está optimizada para consultas robustas (por ejemplo, maneja tildes: "maíz" ≈ "maiz").

## 📁 Estructura del Proyecto

```
ciiu.iml                # Archivo de configuración del proyecto (IDE)
ciiu.sql                # Script o base de datos SQL relacionada con CIIU
ciiu.xlsx               # Archivo Excel con datos CIIU versión 4.0
main.py                 # Script principal, ejemplo en un solo archivo
requirements.txt        # Dependencias del proyecto
src/                    # Código fuente principal
  └─ run.py             # Script para ejecutar la aplicación
   └─ app/               # Módulo principal de la aplicación
      ├─ __init__.py    # Inicialización del módulo
      ├─ config.py      # Configuración de la aplicación
      ├─ data_loader.py # Carga y procesamiento de datos
      ├─ embeddings.py  # Generación y manejo de embeddings
      ├─ faiss_index.py # Indexación y búsqueda con FAISS
      ├─ search_service.py # Servicio central con datos, modelo, índices y lógica de búsqueda
      ├─ main.py        # Punto de entrada de la app
      ├─ models.py      # Definición de modelos de datos
      ├─ routes.py      # Definición de rutas/endpoints
      ├─ utils.py       # Utilidades y funciones auxiliares
```

## 🚀 ¿Cómo empezar?

1. **Clonar el repositorio:**
   ```bash
   git clone <url-del-repositorio>
   ```
2. **Instalar las dependencias:**
   ```bash
   pip install -r requirements.txt
   ```
   Nota: Para instalar las dependencias de PyTorch con soporte para CUDA, se puede usar:
   ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```
3. **Ejecutar la aplicación:**
   ```bash
   python main.py
   # o
   python src/run.py
   # o
   uvicorn src.app.main:app --reload
   ```
4. **Acceder a la API:**
   Abrir un navegador y acceder a `http://localhost:8000/docs` para revisar la documentación interactiva de la API.

Notas:

- Compatible con GPU y CPU; el dispositivo se selecciona automáticamente en `src/app/config.py` (`DEVICE`).
- Si se cambia el modelo de embeddings en `config.EMBEDDING_MODEL`, se deben regenerar los embeddings y reconstruir el índice (reiniciar el servidor es suficiente en el diseño actual, ya que se inicializa al importar).

## 🛠️ Arquitectura y funcionalidades

- FastAPI en `src/app`, con endpoints en `routes.py`.
- Servicio central `search_service.py` que al importarse:
  - Carga datos CIIU v4/v2 desde Excel (`data_loader.py` con Polars).
  - Normaliza descripciones con `utils.normalize_for_nlp` (minúsculas, acentos manejados).
  - Carga el modelo de embeddings (`embeddings.py`, por defecto `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`).
  - Genera embeddings y construye índices FAISS (`IndexFlatIP` + `faiss.normalize_L2`).
- Métrica: producto interno sobre vectores L2-normalizados (≈ coseno).
- IDs en FAISS: `hash(codigo)` mapeados a filas del DataFrame.

## 📦 Dependencias

Las dependencias principales se encuentran en `requirements.txt`. Incluyen librerías para procesamiento de datos, machine learning y APIs.

Relevantes:

- polars
- sentence-transformers (requiere torch)
- faiss-cpu (1.11.x)
- fastapi, uvicorn

## 🔌 API

Endpoints:

- `POST /buscar_ciiu_v4`
- `POST /buscar_ciiu_v2`

Request (CiiuRequest):

```json
{
	"descripcion": "Elaboración de productos de panadería",
	"top_n": 7,
	"categoria": "ACTIVIDAD",
	"umbral_similitud": 0.6
}
```

Response (CiiuResults):

```json
{
	"resultados": [
		{
			"codigo": "C106122",
			"descripcion": "ELABORACIÓN DE HARINAS O MASAS MEZCLADAS PREPARADAS PARA LA FABRICACIÓN DE PAN, PASTELES, BIZCOCHOS O PANQUEQUES.",
			"categoria": "ACTIVIDAD",
			"similitud": 0.8235
		},
		{
			"codigo": "G463093",
			"descripcion": "VENTA AL POR MAYOR DE PRODUCTOS DE PANADERÍA Y REPOSTERÍA.",
			"categoria": "ACTIVIDAD",
			"similitud": 0.7588
		},
		{
			"codigo": "C107101",
			"descripcion": "ELABORACIÓN DE PAN Y OTROS PRODUCTOS DE PANADERÍA SECOS: PAN DE TODO TIPO, PANECILLOS, BIZCOCHOS, TOSTADAS, GALLETAS, ETCÉTERA, INCLUSO ENVASADOS.",
			"categoria": "ACTIVIDAD",
			"similitud": 0.7559
		},
		{
			"codigo": "C107102",
			"descripcion": "ELABORACIÓN DE PASTELES Y OTROS PRODUCTOS DE PASTELERÍA: PASTELES DE FRUTAS, TORTAS, PASTELES, TARTAS, ETCÉTERA, CHURROS, BUÑUELOS, APERITIVOS (BOCADILLOS), ETCÉTERA.",
			"categoria": "ACTIVIDAD",
			"similitud": 0.7289
		}
	]
}
```

Patrón de búsqueda (resumen):

1. Normalizar: `texto = normalize_for_nlp(descripcion)`
2. Embeddings: `emb = model.encode([texto], convert_to_numpy=True)` y `faiss.normalize_L2(emb)`
3. Buscar: `D, I = index.search(emb, k)` con `k = top_n * 5`
4. Mapear `I` (ids hash) a filas y filtrar por `umbral_similitud` y `categoria`.

## 🧪 Prueba rápida

Con el servidor levantado (`uvicorn src.app.main:app --reload`), acceder a `http://localhost:8000/docs` y utilizar el try-out de Swagger. Se debe tener en cuenta que "maíz" y "maiz" se comportan de forma similar gracias a la normalización.

## ⚠️ Advertencias

- Asegúrate de que los embeddings de consulta sean `float32` y con shape `(1, dim)` antes de llamar a FAISS; el código ya aplica `faiss.normalize_L2`.
- Si cambias `EMBEDDING_MODEL` o la normalización en `utils.normalize_for_nlp`, reinicia para regenerar embeddings e índices.
- Los DataFrames deben tener columnas: `codigo`, `descripcion`, `descripcion_limpia`, `categoria`.
- La carga inicial es pesada (modelo + embeddings + índices); se realiza una vez al importar el servicio.

## 🤝 Contribuciones

Las contribuciones son bienvenidas. Se invita a abrir issues o pull requests para sugerencias o mejoras.

## 📝 Licencia

Este proyecto está licenciado bajo los términos de la licencia MIT. Se permite usar, copiar, modificar, fusionar, publicar, distribuir, sublicenciar y/o vender copias del software, siempre que se incluya el aviso de copyright y la nota de permiso.

El software se proporciona "tal cual", sin garantía de ningún tipo. Consultar el archivo `LICENSE` para más detalles.
