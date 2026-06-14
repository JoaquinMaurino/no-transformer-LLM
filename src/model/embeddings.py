import logging
import backend

logger = logging.getLogger(__name__)


def init_mlp_weights(vocab_size: int, embed_dim: int, context_size: int, hidden_dim: int, seed: int = 42):
    """
    Inicializa los pesos del MLP Language Model.
    """
    if vocab_size <= 0 or embed_dim <= 0 or context_size <= 0 or hidden_dim <= 0:
        raise ValueError("Las dimensiones deben ser positivas")

    xp = backend.xp()
    backend.set_seed(seed)

    # Escalado recomendado para inicialización
    scale_embed = 0.01
    scale_hidden = xp.sqrt(2.0 / (context_size * embed_dim + hidden_dim))
    scale_out = xp.sqrt(2.0 / (hidden_dim + vocab_size))

    W_in = backend.as_float32(xp.random.standard_normal((vocab_size, embed_dim))) * scale_embed
    W_hidden = backend.as_float32(xp.random.standard_normal((context_size * embed_dim, hidden_dim))) * scale_hidden
    b_hidden = backend.zeros(hidden_dim, dtype=xp.float32)
    W_out = backend.as_float32(xp.random.standard_normal((vocab_size, hidden_dim))) * scale_out

    logger.info(f"Pesos inicializados: V={vocab_size}, D={embed_dim}, N={context_size}, H={hidden_dim}")

    return W_in, W_hidden, b_hidden, W_out


def save_mlp_weights(W_in, W_hidden, b_hidden, W_out, path: str):
    import numpy as np
    
    W_in_cpu = backend.to_cpu(W_in)
    W_hidden_cpu = backend.to_cpu(W_hidden)
    b_hidden_cpu = backend.to_cpu(b_hidden)
    W_out_cpu = backend.to_cpu(W_out)
    
    np.savez_compressed(
        path,
        W_in=W_in_cpu.astype(np.float32),
        W_hidden=W_hidden_cpu.astype(np.float32),
        b_hidden=b_hidden_cpu.astype(np.float32),
        W_out=W_out_cpu.astype(np.float32)
    )
    logger.info(f"Pesos guardados en: {path}")


def load_mlp_weights(path: str):
    import numpy as np
    data = np.load(path)
    
    W_in = backend.to_backend(data["W_in"], dtype=np.float32)
    W_hidden = backend.to_backend(data["W_hidden"], dtype=np.float32)
    b_hidden = backend.to_backend(data["b_hidden"], dtype=np.float32)
    W_out = backend.to_backend(data["W_out"], dtype=np.float32)

    logger.info(f"Pesos cargados desde: {path}")
    return W_in, W_hidden, b_hidden, W_out


def update_mlp_weights(W_in, W_hidden, b_hidden, W_out, 
                       grad_W_in, grad_W_hidden, grad_b_hidden, grad_W_out, 
                       learning_rate: float):
    if learning_rate <= 0:
        raise ValueError("learning_rate debe ser positivo")

    W_in -= learning_rate * grad_W_in
    W_hidden -= learning_rate * grad_W_hidden
    b_hidden -= learning_rate * grad_b_hidden
    W_out -= learning_rate * grad_W_out

    return W_in, W_hidden, b_hidden, W_out