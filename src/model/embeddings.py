import logging
import numpy as np

logger = logging.getLogger(__name__)


def init_embeddings(vocab_size: int, embed_dim: int, seed: int = 42):
    """
    Inicializa matrices de embeddings en CPU.

    Args:
        vocab_size (int): tamaño del vocabulario (V)
        embed_dim (int): dimensión del embedding (D)
        seed (int): semilla para reproducibilidad

    Returns:
        W_in  (np.ndarray): shape (V, D)
        W_out (np.ndarray): shape (V, D)
    """

    if vocab_size <= 0 or embed_dim <= 0:
        raise ValueError("vocab_size y embed_dim deben ser positivos")

    rng = np.random.default_rng(seed)

    # Inicialización pequeña para evitar saturacion de softmax
    scale = 0.01

    W_in = (rng.standard_normal((vocab_size, embed_dim)).astype(np.float32)) * scale
    W_out = (rng.standard_normal((vocab_size, embed_dim)).astype(np.float32)) * scale

    logger.info(f"Embeddings inicializados: V={vocab_size}, D={embed_dim}")

    return W_in, W_out


def save_embeddings(W_in, W_out, path: str):
    """
    Guarda embeddings en disco (.npz)

    Args:
        W_in, W_out (np.ndarray)
        path (str)
    """

    if W_in is None or W_out is None:
        raise ValueError("Embeddings inválidos")

    np.savez_compressed(path, W_in=W_in.astype(np.float32), W_out=W_out.astype(np.float32))

    logger.info(f"Embeddings guardados en: {path}")


def load_embeddings(path: str):
    """
    Carga embeddings desde disco

    Returns:
        W_in, W_out (np.ndarray)
    """

    data = np.load(path)

    W_in = data["W_in"].astype(np.float32)
    W_out = data["W_out"].astype(np.float32)

    logger.info(f"Embeddings cargados desde: {path}")

    return W_in, W_out


def update_embeddings(W_in, W_out, grad_W_in, grad_W_out, learning_rate: float):
    if learning_rate <= 0:
        raise ValueError("learning_rate debe ser positivo")

    W_in -= learning_rate * grad_W_in
    W_out -= learning_rate * grad_W_out

    return W_in, W_out