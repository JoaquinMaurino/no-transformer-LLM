# Genera el dataset de indices para CBOW
# 

import os
import logging
import numpy as np

logger = logging.getLogger(__name__)


def generate_cbow_dataset(tokens, word_to_idx, window_size=2):
    """
    Genera dataset CBOW en formato arrays.

    Args:
        tokens (list[str])
        word_to_idx (dict[str, int])
        window_size (int)

    Returns:
        contexts (np.ndarray) shape (N, 2*window_size)
        targets  (np.ndarray) shape (N,)
    """

    if not tokens:
        raise ValueError("tokens vacío")

    if not word_to_idx:
        raise ValueError("word_to_idx vacío")

    C = 2 * window_size

    contexts = []
    targets = []

    for i in range(window_size, len(tokens) - window_size):
        try:
            context = []
            for j in range(i - window_size, i + window_size + 1):
                if j == i:
                    continue
                context.append(word_to_idx[tokens[j]])

            if len(context) != C:
                continue

            target = word_to_idx[tokens[i]]

            contexts.append(context)
            targets.append(target)

        except KeyError as e:
            logger.warning(f"Token fuera de vocabulario: {e}")
            continue

    contexts = np.array(contexts, dtype=np.int32)
    targets = np.array(targets, dtype=np.int32)

    logger.info(f"Dataset generado: {contexts.shape[0]} ejemplos")

    return contexts, targets


def save_dataset(contexts, targets, output_path):
    """
    Guarda dataset en formato NPZ.

    Args:
        contexts (np.ndarray)
        targets  (np.ndarray)
        output_path (str)
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

    Args:
        input_path (str)

    Returns:
        contexts, targets
    """

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"No existe: {input_path}")

    data = np.load(input_path)

    contexts = data["contexts"]
    targets = data["targets"]

    logger.info(f"Dataset cargado: {contexts.shape[0]} ejemplos")

    return contexts, targets


def build_and_save_dataset(tokens, word_to_idx, output_path, window_size=2):
    """
    Pipeline completo:
    tokens → dataset → guardado

    Returns:
        contexts, targets
    """

    contexts, targets = generate_cbow_dataset(
        tokens=tokens,
        word_to_idx=word_to_idx,
        window_size=window_size
    )

    save_dataset(contexts, targets, output_path)

    return contexts, targets