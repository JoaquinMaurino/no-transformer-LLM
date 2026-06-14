import logging
import numpy as np

logger = logging.getLogger(__name__)


def init_mlp_weights(vocab_size: int, embed_dim: int, context_size: int, hidden_dim: int, seed: int = 42):
    """
    Inicializa los pesos del MLP Language Model.

    Args:
        vocab_size (int): V
        embed_dim (int): D (dimensión del embedding por palabra)
        context_size (int): N (tamaño del contexto)
        hidden_dim (int): H (unidades de la capa oculta)
        seed (int): Semilla aleatoria

    Returns:
        W_in, W_hidden, b_hidden, W_out
    """
    if vocab_size <= 0 or embed_dim <= 0 or context_size <= 0 or hidden_dim <= 0:
        raise ValueError("Las dimensiones deben ser positivas")

    rng = np.random.default_rng(seed)

    # Escalado recomendado para inicialización
    scale_embed = 0.01
    scale_hidden = np.sqrt(2.0 / (context_size * embed_dim + hidden_dim))
    scale_out = np.sqrt(2.0 / (hidden_dim + vocab_size))

    W_in = (rng.standard_normal((vocab_size, embed_dim)).astype(np.float32)) * scale_embed
    W_hidden = (rng.standard_normal((context_size * embed_dim, hidden_dim)).astype(np.float32)) * scale_hidden
    b_hidden = np.zeros(hidden_dim, dtype=np.float32)
    W_out = (rng.standard_normal((vocab_size, hidden_dim)).astype(np.float32)) * scale_out

    logger.info(f"Pesos inicializados: V={vocab_size}, D={embed_dim}, N={context_size}, H={hidden_dim}")

    return W_in, W_hidden, b_hidden, W_out


def save_mlp_weights(W_in, W_hidden, b_hidden, W_out, path: str):
    """
    Guarda pesos en disco (.npz)
    """
    np.savez_compressed(
        path,
        W_in=W_in.astype(np.float32),
        W_hidden=W_hidden.astype(np.float32),
        b_hidden=b_hidden.astype(np.float32),
        W_out=W_out.astype(np.float32)
    )
    logger.info(f"Pesos guardados en: {path}")


def load_mlp_weights(path: str):
    """
    Carga pesos desde disco
    """
    data = np.load(path)
    W_in = data["W_in"].astype(np.float32)
    W_hidden = data["W_hidden"].astype(np.float32)
    b_hidden = data["b_hidden"].astype(np.float32)
    W_out = data["W_out"].astype(np.float32)

    logger.info(f"Pesos cargados desde: {path}")
    return W_in, W_hidden, b_hidden, W_out


def update_mlp_weights(W_in, W_hidden, b_hidden, W_out, 
                       grad_W_in, grad_W_hidden, grad_b_hidden, grad_W_out, 
                       learning_rate: float):
    """
    Actualiza los pesos usando descenso por gradiente.
    """
    if learning_rate <= 0:
        raise ValueError("learning_rate debe ser positivo")

    W_in -= learning_rate * grad_W_in
    W_hidden -= learning_rate * grad_W_hidden
    b_hidden -= learning_rate * grad_b_hidden
    W_out -= learning_rate * grad_W_out

    return W_in, W_hidden, b_hidden, W_out