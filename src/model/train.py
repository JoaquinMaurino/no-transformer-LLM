import math
import time
import backend

from model.lm_mlp import forward, backward
from model.embeddings import update_mlp_weights
from model.loss import cross_entropy_loss


def train(W_in, W_hidden, b_hidden, W_out, contexts, targets, config):
    contexts = backend.as_int32(contexts)
    targets = backend.as_int32(targets)
    W_in = backend.as_float32(W_in)
    W_hidden = backend.as_float32(W_hidden)
    b_hidden = backend.as_float32(b_hidden)
    W_out = backend.as_float32(W_out)

    epochs = int(config.get("epochs", 5))
    batch_size = int(config.get("batch_size", 32))
    learning_rate = float(config.get("learning_rate", 0.01))
    seed = int(config.get("seed", 42))
    checkpoint_every = int(config.get("checkpoint_every", 0))
    checkpoint_callback = config.get("checkpoint_callback")

    if epochs <= 0 or batch_size <= 0 or learning_rate <= 0:
        raise ValueError("epochs, batch_size y learning_rate deben ser positivos")

    num_samples = contexts.shape[0]
    if num_samples == 0:
        raise ValueError("Dataset vacío")

    backend.set_seed(seed)
    history = []
    total_batches = math.ceil(num_samples / batch_size)

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        permutation = backend.random_permutation(num_samples)
        accumulated_loss = 0.0

        for batch_idx in range(total_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, num_samples)
            batch_indices = permutation[start:end]

            batch_contexts = contexts[batch_indices]
            batch_targets = targets[batch_indices]

            # Forward pass
            x, h, scores = forward(
                contexts=batch_contexts, 
                W_in=W_in, 
                W_hidden=W_hidden, 
                b_hidden=b_hidden, 
                W_out=W_out
            )
            
            # Loss computation
            loss = cross_entropy_loss(scores, batch_targets)

            if backend.contains_nan(loss):
                raise ValueError("El loss se volvió NaN")

            # Backward pass
            grad_W_in, grad_W_hidden, grad_b_hidden, grad_W_out = backward(
                x=x,
                h=h,
                scores=scores,
                targets=batch_targets,
                contexts=batch_contexts,
                W_in=W_in,
                W_hidden=W_hidden,
                W_out=W_out,
            )

            # Update weights
            W_in, W_hidden, b_hidden, W_out = update_mlp_weights(
                W_in=W_in,
                W_hidden=W_hidden,
                b_hidden=b_hidden,
                W_out=W_out,
                grad_W_in=grad_W_in,
                grad_W_hidden=grad_W_hidden,
                grad_b_hidden=grad_b_hidden,
                grad_W_out=grad_W_out,
                learning_rate=learning_rate,
            )

            accumulated_loss += backend.to_float(loss) * (end - start)

        avg_loss = accumulated_loss / float(num_samples)
        elapsed = time.time() - start_time
        gpu_mem = backend.gpu_memory_info_mb()
        
        if gpu_mem is None:
            print(f"Epoch {epoch}/{epochs} - loss: {avg_loss:.6f} - time: {elapsed:.2f}s")
        else:
            used_mb, total_mb = gpu_mem
            print(f"Epoch {epoch}/{epochs} - loss: {avg_loss:.6f} - time: {elapsed:.2f}s - gpu_mem: {used_mb:.1f}/{total_mb:.1f} MB")
            
        history.append(avg_loss)

        if checkpoint_every > 0 and checkpoint_callback and (epoch % checkpoint_every == 0):
            checkpoint_callback(
                epoch=epoch,
                W_in=W_in,
                W_hidden=W_hidden,
                b_hidden=b_hidden,
                W_out=W_out,
                avg_loss=avg_loss,
                history=history,
            )

    return W_in, W_hidden, b_hidden, W_out, history
