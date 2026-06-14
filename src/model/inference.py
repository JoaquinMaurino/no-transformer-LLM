import backend
from tokenizers import Tokenizer
from model.lm_mlp import forward
from model.loss import stable_softmax
from model.embeddings import load_mlp_weights

class LMPredictor:
    def __init__(self, weights_path: str, tokenizer_path: str, context_size: int = None):
        self.W_in, self.W_hidden, self.b_hidden, self.W_out = load_mlp_weights(weights_path)
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        
        # Calcular dinámicamente el context size si no se provee
        if context_size is None:
            self.context_size = self.W_hidden.shape[0] // self.W_in.shape[1]
        else:
            self.context_size = context_size

    def generate_next_token(self, prompt_text: str, top_k: int = 5):
        # 1. Tokenizar el texto de entrada
        encoded = self.tokenizer.encode(prompt_text)
        token_ids = encoded.ids

        # 2. Rellenar si es menor que el context_size, o truncar si es mayor
        if len(token_ids) < self.context_size:
            # Rellenar con ID 0 (o algún pad token, aunque en este modelo simple podemos usar 0)
            token_ids = [0] * (self.context_size - len(token_ids)) + token_ids
        else:
            token_ids = token_ids[-self.context_size:]

        # 3. Preparar input
        xp = backend.xp()
        context_indices = xp.array([token_ids], dtype=xp.int32)

        # 4. Forward Pass
        _, _, scores = forward(context_indices, self.W_in, self.W_hidden, self.b_hidden, self.W_out)
        
        # 5. Obtener probabilidades
        probs = stable_softmax(scores)[0]
        
        # Convertir a CPU para top_k
        probs_cpu = backend.to_cpu(probs)

        # 6. Ordenar y devolver los K tokens más probables
        sorted_indices = probs_cpu.argsort()[::-1]
        top_indices = sorted_indices[:top_k]

        best_idx = int(top_indices[0])
        predicted_word = self.tokenizer.decode([best_idx])

        top_predictions = []
        for idx in top_indices:
            decoded = self.tokenizer.decode([int(idx)])
            # Si el token BPE no se puede decodificar solo, mostrar el ID
            if not decoded.strip():
                decoded = f"[Token ID: {int(idx)}]"
            top_predictions.append((decoded, float(probs_cpu[int(idx)])))

        return predicted_word, top_predictions

    def generate_text(self, prompt_text: str, max_tokens: int = 10, top_k: int = 1):
        """Genera una secuencia de texto repitiendo el predictor."""
        generated = prompt_text
        for _ in range(max_tokens):
            next_word, _ = self.generate_next_token(generated, top_k=top_k)
            generated += " " + next_word
        return generated