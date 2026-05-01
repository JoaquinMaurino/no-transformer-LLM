import re
import logging

logger = logging.getLogger(__name__)

STOPWORDS = {"el", "la", "los", "las", "un", "una", "unos", "unas"}


def _is_page_number_line(line: str) -> bool:
    stripped = line.strip().lower()
    if not stripped:
        return False
    return bool(re.fullmatch(r"(?:\d{1,4}|[ivxlcdm]{1,10})", stripped))


def _is_heading_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False

    words = stripped.split()
    if len(words) > 10:
        return False

    if stripped.endswith(":"):
        return True

    if stripped.isupper():
        return True

    return bool(re.match(r"^(cap[ií]tulo|chapter|parte|secci[oó]n)\b", stripped.lower()))


def clean_corpus_text(text: str) -> str:
    if not isinstance(text, str):
        raise ValueError("Input text inválido: se esperaba un string")

    if not text.strip():
        return ""

    raw_lines = text.splitlines()
    cleaned_lines = []

    skip_acknowledgements = False
    for line in raw_lines:
        line = line.strip()

        if not line:
            continue

        if _is_page_number_line(line):
            continue

        line_lower = line.lower()
        if line_lower.startswith("agradecimientos"):
            skip_acknowledgements = True
            continue

        if skip_acknowledgements:
            if _is_heading_line(line) and not line_lower.startswith("agradecimientos"):
                skip_acknowledgements = False
            else:
                continue

        if _is_heading_line(line):
            continue

        cleaned_lines.append(line_lower)

    text = "\n".join(cleaned_lines)

    # Separa signos para que queden como tokens independientes.
    text = re.sub(r'([\.,;:!\?\(\)"«»])', r' \1 ', text)

    # Conserva letras, acentos, numeros y la puntuacion ya separada.
    text = re.sub(r'[^0-9a-záéíóúüñ\s\.,;:!\?\(\)"«»]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def tokenize_text(text: str, remove_stopwords: bool = False):
    cleaned_text = clean_corpus_text(text)
    if not cleaned_text:
        return []

    tokens = cleaned_text.split()

    if remove_stopwords:
        tokens = [token for token in tokens if token not in STOPWORDS]

    logger.info("Tokens generados: %s", len(tokens))
    return tokens


def tokenizer(text: str, remove_stopwords: bool = False):
    return tokenize_text(text, remove_stopwords=remove_stopwords)