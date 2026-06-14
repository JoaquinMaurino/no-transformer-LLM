import backend

def stable_softmax(logits):
	xp = backend.xp()
	if logits.ndim == 1:
		logits = logits[None, :]

	shifted = logits - xp.max(logits, axis=1, keepdims=True)
	exp_scores = xp.exp(shifted)
	probs = exp_scores / xp.sum(exp_scores, axis=1, keepdims=True)
	return backend.as_float32(probs)


def cross_entropy_loss(logits, targets, eps: float = 1e-10) -> float:
	xp = backend.xp()
	if logits.ndim == 1:
		logits = logits[None, :]
		targets = backend.as_int32([targets])

	probs = stable_softmax(logits)
	batch_indices = xp.arange(logits.shape[0])
	selected = probs[batch_indices, targets]
	loss = -xp.mean(xp.log(selected + eps))
	return backend.to_float(loss)
