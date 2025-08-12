import re
import unicodedata


def limpiar_texto(texto: str) -> str:
    """
    Limpia el texto eliminando caracteres no deseados.

    Args:
        texto (str): Texto original.

    Returns:
        str: Texto limpio.
    """
    if not isinstance(texto, str):
        return ""
    texto = texto.lower()
    texto = re.sub(r'[^\w\s]', '', texto)
    texto = re.sub(r'\s+', ' ', texto)
    return texto.strip()


def normalize_text(text: str) -> str:
    """
    Normaliza un texto eliminando acentos, caracteres especiales y espacios innecesarios.
    Devuelve el texto en minúsculas.

    Args:
        text (str): Texto original.

    Returns:
        str: Texto limpio, legible y en minúsculas.
    """
    if not isinstance(text, str):
        raise ValueError("El texto debe ser una cadena de caracteres.")

    # 1. Eliminar espacios al inicio y al final
    text = text.strip()

    # 2. Reemplazar múltiples espacios por uno solo
    text = re.sub(r"\s+", " ", text)

    # 3. Eliminar acentos y diacríticos
    text = unicodedata.normalize("NFKD", text)
    text = "".join([c for c in text if not unicodedata.combining(c)])

    # 4. Eliminar caracteres especiales, conservar letras, números y espacios
    text = re.sub(r"[^a-zA-Z0-9 ]", "", text)

    # 5. Convertir a minúsculas
    return text.lower()


def normalize_for_nlp(text: str) -> str:
    # Minúsculas
    text = text.lower()

    # Reemplazar acentos manualmente
    replacements = {
        'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u',
        'ü': 'u', 'ñ': 'ñ'
    }
    for accented, plain in replacements.items():
        text = text.replace(accented, plain)

    # Eliminar caracteres especiales pero conservar puntuación básica
    text = re.sub(r"[^\w\s.,!?¿¡]", "", text)

    # Reemplazar múltiples espacios
    text = re.sub(r"\s+", " ", text).strip()

    return text
