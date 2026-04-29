# Implementar el forward pass

import logging
import cupy as cp

logger = logging.getLogger(__name__)


def forward(contexts, W_in, W_out):
    """
    Forward pass para CBOW.

    Args:
        contexts (cp.ndarray):
            shape (B, C) o (C,)
            índices de palabras de contexto

        W_in (cp.ndarray):
            shape (V, D)

        W_out (cp.ndarray):
            shape (V, D)

    Returns:
        h (cp.ndarray):
            shape (B, D) o (D,)
            vector oculto (promedio embeddings)

        scores (cp.ndarray):
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
        h = cp.mean(context_vecs, axis=0)

        # (V,)
        scores = W_out @ h

        return h, scores

    # Caso 2: batch → (B, C)
    elif contexts.ndim == 2:
        # (B, C, D)
        context_vecs = W_in[contexts]

        # (B, D)
        h = cp.mean(context_vecs, axis=1)

        # (B, V)
        scores = h @ W_out.T

        return h, scores

    else:
        raise ValueError(f"Dimensión inválida para contexts: {contexts.ndim}")