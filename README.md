# 📊 Proyecto CIIU

Bienvenido al repositorio del proyecto **CIIU**. Este proyecto está diseñado para el procesamiento, análisis y consulta de datos relacionados con la Clasificación Industrial Internacional Uniforme (CIIU) rev 4.0. Utiliza Python y diversas herramientas modernas para el manejo eficiente de datos y la construcción de servicios inteligentes.

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
      ├─ main.py        # Punto de entrada de la app
      ├─ models.py      # Definición de modelos de datos
      ├─ routes.py      # Definición de rutas/endpoints
      ├─ utils.py       # Utilidades y funciones auxiliares
```

## 🚀 ¿Cómo empezar?

1. **Clona el repositorio:**
   ```bash
   git clone <url-del-repositorio>
   ```
2. **Instala las dependencias:**
   ```bash
   pip install -r requirements.txt
   ```
   Nota: Para instalar las dependencias de PyTorch con soporte para CUDA, puedes usar:
   ```bash
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```
3. **Ejecuta la aplicación:**
   ```bash
   python main.py
   # o
   python src/run.py
   # o
   uvicorn src.app.main:app --reload
   ```
4. **Accede a la API:**
   Abre tu navegador y ve a `http://localhost:8000/docs` para ver la documentación interactiva de la API.

Nota: Si usas `uvicorn`, asegúrate de tenerlo instalado y configurado correctamente. Compatible con GPU y CPU.
Para hacer uso de GPU, asegúrate de tener las dependencias adecuadas instaladas y configuradas.

## 🛠️ Funcionalidades principales

- Procesamiento y consulta de datos CIIU desde archivos SQL y Excel.
- Generación de embeddings para búsqueda semántica.
- Indexación eficiente usando FAISS.
- API para exponer funcionalidades de consulta y análisis.

## 📦 Dependencias

Las dependencias principales se encuentran en `requirements.txt`. Incluyen librerías para procesamiento de datos, machine learning y APIs.

## 🤝 Contribuciones

¡Las contribuciones son bienvenidas! Por favor, abre un issue o pull request para sugerencias o mejoras.

## 📝 Licencia

Este proyecto está licenciado bajo los términos de la licencia MIT. Puedes usar, copiar, modificar, fusionar, publicar, distribuir, sublicenciar y/o vender copias del software, siempre que incluyas el aviso de copyright y la nota de permiso en todas las copias o partes sustanciales del software.

El software se proporciona "tal cual", sin garantía de ningún tipo. Consulta el archivo `LICENSE` para más detalles.
Obtener año actual del copyright y nombre del autor.
