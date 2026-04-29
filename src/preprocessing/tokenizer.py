# Se encarga de limpiar el texto retornando una lista de palabras strings sin articulos ni signos de puntuacion
# Input: "Clima muy caluroso, pero nublado!"
# Output: ["clima", "muy", "caluroso", "pero", "nublado"]

import re
import logging

logger = logging.getLogger(__name__)

STOPWORDS = {"el", "la", "los", "las", "un", "una", "unos", "unas"}

def tokenizer(text: str, remove_stopwords=True):
    if not isinstance(text, str) or not text:
        raise ValueError("Input text inválido")

    text = text.lower()
    text = re.sub(r'[^a-záéíóúñ\s]', '', text)

    tokens = text.split()

    if remove_stopwords:
        tokens = [t for t in tokens if t not in STOPWORDS]

    logger.info(f"Tokens generados: {len(tokens)}")

    return tokens