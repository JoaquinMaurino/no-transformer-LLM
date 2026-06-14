import os
import logging
import numpy as np

logger = logging.getLogger(__name__)


def generate_lm_dataset(tokens, tokenizer, context_size=5):
    """
    Genera dataset causal (Language Model) en formato arrays.
    
    Args:
        tokens (list[str]): Lista de tokens (o strings).
        tokenizer: Tokenizador BPE (para convertir strings a IDs).
        context_size (int): N palabras previas para predecir la siguiente.
        
    Returns:
        contexts (np.ndarray) shape (N, context_size)
        targets  (np.ndarray) shape (N,)
    """
    if not tokens:
        raise ValueError("tokens vacío")
        
    # Convertimos los tokens a IDs usando el tokenizer
    # Asumimos que tokens ya es una lista continua de strings del corpus.
    # En BPE, lo ideal es pasarle el corpus entero para que devuelva los IDs.
    # Aquí unimos los tokens de vuelta en un texto y los encodeamos de una.
    text = " ".join(tokens)
    encoded = tokenizer.encode(text)
    token_ids = encoded.ids
    
    contexts = []
    targets = []
    
    for i in range(len(token_ids) - context_size):
        context = token_ids[i : i + context_size]
        target = token_ids[i + context_size]
        
        contexts.append(context)
        targets.append(target)
        
    contexts = np.array(contexts, dtype=np.int32)
    targets = np.array(targets, dtype=np.int32)
    
    logger.info(f"Dataset generado: {contexts.shape[0]} ejemplos (context_size={context_size})")
    
    return contexts, targets


def save_dataset(contexts, targets, output_path):
    """
    Guarda dataset en formato NPZ.
    """
    if contexts is None or targets is None:
        raise ValueError("contexts/targets inválidos")

    if contexts.shape[0] != targets.shape[0]:
        raise ValueError("Mismatch entre contexts y targets")

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    np.savez_compressed(output_path, contexts=contexts, targets=targets)
    logger.info(f"Dataset guardado en: {output_path}")


def load_dataset(input_path):
    """
    Carga dataset desde archivo NPZ.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"No existe: {input_path}")

    data = np.load(input_path)
    contexts = data["contexts"]
    targets = data["targets"]

    logger.info(f"Dataset cargado: {contexts.shape[0]} ejemplos")
    return contexts, targets
