import logging
import numpy as np
from model.loss import stable_softmax

logger = logging.getLogger(__name__)

def forward(contexts, W_in, W_hidden, b_hidden, W_out):
    """
    Forward pass para MLP Language Model.
    """
    contexts = contexts.astype(np.int32)
    
    if contexts.ndim == 1:
        contexts = contexts[None, :]
        
    B, C = contexts.shape
    V, D = W_in.shape
    
    # 1. Look up embeddings (B, C, D)
    context_vecs = W_in[contexts]
    
    # 2. Flatten/Concatenate embeddings (B, C * D)
    x = context_vecs.reshape(B, C * D)
    
    # 3. Hidden layer (B, H)
    h_pre = x @ W_hidden + b_hidden
    h = np.tanh(h_pre)
    
    # 4. Output layer (B, V)
    scores = h @ W_out.T
    
    return x, h, scores


def backward(x, h, scores, targets, contexts, W_in, W_hidden, W_out):
    """
    Backward pass del MLP LM.
    """
    targets = np.array(targets, dtype=np.int32)
    if targets.ndim == 0:
        targets = targets[None]
        
    B, C = contexts.shape
    V, D = W_in.shape
    H = W_hidden.shape[1]
    
    # Derivada del Loss respecto a scores (z) -> (B, V)
    grad_scores = stable_softmax(scores)
    grad_scores[np.arange(B), targets] -= 1.0
    grad_scores /= float(B)
    
    # Gradientes de W_out -> (V, H)
    grad_W_out = grad_scores.T @ h
    
    # Derivada respecto a h -> (B, H)
    grad_h = grad_scores @ W_out
    
    # Derivada a través de Tanh
    grad_h_pre = grad_h * (1.0 - h**2)  # (B, H)
    
    # Gradientes de W_hidden y b_hidden
    grad_W_hidden = x.T @ grad_h_pre
    grad_b_hidden = np.sum(grad_h_pre, axis=0)
    
    # Derivada respecto a x (embeddings concatenados) -> (B, C*D)
    grad_x = grad_h_pre @ W_hidden.T
    
    # Derivada respecto a W_in (Embedding layer)
    grad_context_vecs = grad_x.reshape(B, C, D)
    
    # Acumular gradientes en W_in usando np.add.at
    grad_W_in = np.zeros((V, D), dtype=np.float32)
    for pos in range(C):
        np.add.at(grad_W_in, contexts[:, pos], grad_context_vecs[:, pos, :])
    
    return grad_W_in, grad_W_hidden, grad_b_hidden, grad_W_out
