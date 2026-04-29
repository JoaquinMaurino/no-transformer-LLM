# Inicializacion de y manejo de matrices de embeddings (W y W')

import logging
import cupy as cp

logger = logging.getLogger(__name__)


def init_embeddings(vocab_size: int, embed_dim: int, seed: int = 42):
    """
    Inicializa matrices de embeddings en GPU.

    Args:
        vocab_size (int): tamaño del vocabulario (V)
        embed_dim (int): dimensión del embedding (D)
        seed (int): semilla para reproducibilidad

    Returns:
        W_in  (cp.ndarray): shape (V, D)
        W_out (cp.ndarray): shape (V, D)
    """

    if vocab_size <= 0 or embed_dim <= 0:
        raise ValueError("vocab_size y embed_dim deben ser positivos")

    cp.random.seed(seed)

    # Inicialización pequeña para evitar saturacion de softmax
    scale = 0.01

    W_in = (cp.random.randn(vocab_size, embed_dim).astype(cp.float32)) * scale
    W_out = (cp.random.randn(vocab_size, embed_dim).astype(cp.float32)) * scale

    logger.info(f"Embeddings inicializados: V={vocab_size}, D={embed_dim}")

    return W_in, W_out


def save_embeddings(W_in, W_out, path: str):
    """
    Guarda embeddings en disco (.npz)

    Args:
        W_in, W_out (cp.ndarray)
        path (str)
    """

    if W_in is None or W_out is None:
        raise ValueError("Embeddings inválidos")

    # mover a CPU antes de guardar
    W_in_cpu = cp.asnumpy(W_in)
    W_out_cpu = cp.asnumpy(W_out)

    cp.savez(path, W_in=W_in_cpu, W_out=W_out_cpu)

    logger.info(f"Embeddings guardados en: {path}")


def load_embeddings(path: str):
    """
    Carga embeddings desde disco

    Returns:
        W_in, W_out (cp.ndarray)
    """

    data = cp.load(path)

    W_in = cp.array(data["W_in"], dtype=cp.float32)
    W_out = cp.array(data["W_out"], dtype=cp.float32)

    logger.info(f"Embeddings cargados desde: {path}")

    return W_in, W_out