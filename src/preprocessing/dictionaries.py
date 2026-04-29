#Se encarga de generar dos diccionarios de token a index y viceversa
#Input:  tokens: list[str]
#Output: word_to_idx: dict[str, int]
#        idx_to_word: dict[int, str]

#Ejemplo:
#tokens = ["clima", "muy", "caluroso", "pero", "nublado"]
# word_to_idx = {
#   "caluroso": 0,
#   "clima": 1,
#   "muy": 2,
#   "nublado": 3,
#   "pero": 4
# }

import logging

logger = logging.getLogger(__name__)

def build_dictionaries(tokens):
    if not tokens:
        raise ValueError("tokens vacío")

    vocab = sorted(set(tokens))

    word_to_idx = {w: i for i, w in enumerate(vocab)}
    idx_to_word = {i: w for w, i in word_to_idx.items()}

    logger.info(f"Vocab size: {len(vocab)}")

    return word_to_idx, idx_to_word