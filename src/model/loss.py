import numpy as np


def stable_softmax(logits: np.ndarray) -> np.ndarray:
	if logits.ndim == 1:
		logits = logits[None, :]

	shifted = logits - np.max(logits, axis=1, keepdims=True)
	exp_scores = np.exp(shifted)
	probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
	return probs.astype(np.float32)


def cross_entropy_loss(logits: np.ndarray, targets: np.ndarray, eps: float = 1e-10) -> float:
	if logits.ndim == 1:
		logits = logits[None, :]
		targets = np.array([targets], dtype=np.int32)

	probs = stable_softmax(logits)
	batch_indices = np.arange(logits.shape[0])
	selected = probs[batch_indices, targets]
	loss = -np.mean(np.log(selected + eps))
	return float(loss)
