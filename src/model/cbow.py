import logging
import numpy as np

from model.loss import stable_softmax

logger = logging.getLogger(__name__)


def forward(contexts, W_in, W_out):
    """
    Forward pass para CBOW.

    Args:
        contexts (np.ndarray):
            shape (B, C) o (C,)
            índices de palabras de contexto

        W_in (np.ndarray):
            shape (V, D)

        W_out (np.ndarray):
            shape (V, D)

    Returns:
        h (np.ndarray):
            shape (B, D) o (D,)
            vector oculto (promedio embeddings)

        scores (np.ndarray):
            shape (B, V) o (V,)
            logits antes de softmax
    """

    if contexts is None:
        raise ValueError("contexts es None")

    if W_in is None or W_out is None:
        raise ValueError("Embeddings no inicializados")

    # Caso 1: un solo ejemplo → (C,)
    if contexts.ndim == 1:
        # (C, D)
        context_vecs = W_in[contexts]

        # (D,)
        h = np.mean(context_vecs, axis=0)

        # (V,)
        scores = W_out @ h

        return h, scores

    # Caso 2: batch → (B, C)
    elif contexts.ndim == 2:
        # (B, C, D)
        context_vecs = W_in[contexts]

        # (B, D)
        h = np.mean(context_vecs, axis=1)

        # (B, V)
        scores = h @ W_out.T

        return h, scores

    else:
        raise ValueError(f"Dimensión inválida para contexts: {contexts.ndim}")


def backward(h, scores, targets, contexts, W_out):
    """
    Backward pass de CBOW para gradientes de W_in y W_out.
    """

    if contexts.ndim == 1:
        contexts = contexts[None, :]
        h = h[None, :]
        scores = scores[None, :]
        targets = np.array([targets], dtype=np.int32)

    batch_size = contexts.shape[0]
    context_size = contexts.shape[1]
    vocab_size, embed_dim = W_out.shape

    grad_scores = stable_softmax(scores)
    grad_scores[np.arange(batch_size), targets] -= 1.0
    grad_scores /= float(batch_size)

    # scores = h @ W_out.T, por eso grad_W_out queda (V, D)
    grad_W_out = grad_scores.T @ h

    grad_h = grad_scores @ W_out
    grad_each_context = grad_h / float(context_size)

    grad_W_in = np.zeros((vocab_size, embed_dim), dtype=np.float32)
    for pos in range(context_size):
        np.add.at(grad_W_in, contexts[:, pos], grad_each_context)

    return grad_W_in, grad_W_out.astype(np.float32)