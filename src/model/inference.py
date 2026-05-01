import numpy as np

from model.cbow import forward
from model.loss import stable_softmax
from utils.io import load_model_artifact


class CBOWPredictor:
    def __init__(self, artifact_path: str):
        W_in, W_out, word_to_idx, idx_to_word, metadata = load_model_artifact(artifact_path)
        self.W_in = W_in
        self.W_out = W_out
        self.word_to_idx = word_to_idx
        self.idx_to_word = idx_to_word
        self.metadata = metadata

    def predict(self, context_words, top_k: int = 5):
        if len(context_words) != 4:
            raise ValueError("Se requieren exactamente 4 palabras de contexto (2 izquierda + 2 derecha)")

        unknown = [word for word in context_words if word not in self.word_to_idx]
        if unknown:
            raise ValueError(f"Palabras fuera de vocabulario: {unknown}")

        context_indices = np.array([[self.word_to_idx[word] for word in context_words]], dtype=np.int32)
        _, scores = forward(context_indices, self.W_in, self.W_out)
        probs = stable_softmax(scores)[0]

        sorted_indices = np.argsort(probs)[::-1]
        top_indices = sorted_indices[:top_k]

        best_idx = int(top_indices[0])
        predicted_word = self.idx_to_word[best_idx]

        top_predictions = [(self.idx_to_word[int(idx)], float(probs[int(idx)])) for idx in top_indices]
        return predicted_word, top_predictions