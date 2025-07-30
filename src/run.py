import sys
import os
import uvicorn

# Asegurar que 'src/' esté en sys.path, para ejecutar desde cualquier lugar
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)


if __name__ == "__main__":
    uvicorn.run("app.main:app", port=8000, host="localhost", reload=True, workers=1)
