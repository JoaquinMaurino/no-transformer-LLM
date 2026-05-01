import math
import time

import numpy as np

from model.cbow import backward, forward
from model.embeddings import update_embeddings
from model.loss import cross_entropy_loss


def train(W_in, W_out, contexts, targets, config):
	epochs = int(config.get("epochs", 5))
	batch_size = int(config.get("batch_size", 32))
	learning_rate = float(config.get("learning_rate", 0.01))
	seed = int(config.get("seed", 42))

	if epochs <= 0 or batch_size <= 0 or learning_rate <= 0:
		raise ValueError("epochs, batch_size y learning_rate deben ser positivos")

	num_samples = contexts.shape[0]
	if num_samples == 0:
		raise ValueError("Dataset vacío")

	rng = np.random.default_rng(seed)
	history = []
	total_batches = math.ceil(num_samples / batch_size)

	for epoch in range(1, epochs + 1):
		start_time = time.time()

		permutation = rng.permutation(num_samples)
		contexts_shuffled = contexts[permutation]
		targets_shuffled = targets[permutation]

		accumulated_loss = 0.0

		for batch_idx in range(total_batches):
			start = batch_idx * batch_size
			end = min(start + batch_size, num_samples)

			batch_contexts = contexts_shuffled[start:end]
			batch_targets = targets_shuffled[start:end]

			h, scores = forward(batch_contexts, W_in, W_out)
			loss = cross_entropy_loss(scores, batch_targets)

			if np.isnan(loss):
				raise ValueError("El loss se volvió NaN")

			grad_W_in, grad_W_out = backward(
				h=h,
				scores=scores,
				targets=batch_targets,
				contexts=batch_contexts,
				W_out=W_out,
			)

			W_in, W_out = update_embeddings(
				W_in=W_in,
				W_out=W_out,
				grad_W_in=grad_W_in,
				grad_W_out=grad_W_out,
				learning_rate=learning_rate,
			)

			accumulated_loss += loss * (end - start)

		avg_loss = accumulated_loss / float(num_samples)
		elapsed = time.time() - start_time
		print(f"Epoch {epoch}/{epochs} - loss: {avg_loss:.6f} - time: {elapsed:.2f}s")
		history.append(avg_loss)

	return W_in, W_out, history
