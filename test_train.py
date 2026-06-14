import os
import sys

# Agregar src al path
sys.path.insert(0, os.path.abspath("src"))

from preprocessing.bpe_tokenizer import train_bpe_tokenizer
from preprocessing.lm_dataset import generate_lm_dataset
from model.embeddings import init_mlp_weights
from model.train import train

def run_sanity_check():
    print("--- INICIANDO SANITY CHECK ---")
    
    # 1. Crear corpus de prueba pequeño (comentado para usar el corpus largo)
    os.makedirs("data/raw", exist_ok=True)
    # with open("data/raw/test_corpus.txt", "w", encoding="utf-8") as f:
    #     f.write("el gato persigue al raton bajo la mesa . el raton corre muy rapido . el perro ladra fuerte .\n")
        
    # 2. Entrenar tokenizador BPE
    vocab_size = 20
    tokenizer = train_bpe_tokenizer("data/raw/test_corpus.txt", vocab_size=vocab_size)
    actual_vocab_size = tokenizer.get_vocab_size()
    print(f"Tokenizador entrenado. Vocab Size: {actual_vocab_size}")
    
    # 3. Leer texto y generar dataset
    with open("data/raw/test_corpus.txt", "r", encoding="utf-8") as f:
        tokens = f.read().split()
        
    context_size = 10
    contexts, targets = generate_lm_dataset(tokens, tokenizer, context_size=context_size)
    print(f"Dataset generado. Contexts shape: {contexts.shape}, Targets shape: {targets.shape}")
    
    # 4. Inicializar pesos del MLP
    embed_dim = 10
    hidden_dim = 20
    
    W_in, W_hidden, b_hidden, W_out = init_mlp_weights(
        vocab_size=actual_vocab_size, 
        embed_dim=embed_dim, 
        context_size=context_size, 
        hidden_dim=hidden_dim
    )
    
    print(f"Pesos inicializados.")
    print(f"W_in: {W_in.shape}, W_hidden: {W_hidden.shape}, b_hidden: {b_hidden.shape}, W_out: {W_out.shape}")
    
    # 5. Entrenar
    config = {
        "epochs": 60,
        "batch_size": 2,
        "learning_rate": 0.01,
        "seed": 42
    }
    
    print("Comenzando entrenamiento...")
    W_in, W_hidden, b_hidden, W_out, history = train(
        W_in, W_hidden, b_hidden, W_out, contexts, targets, config
    )
    
    print("--- SANITY CHECK COMPLETADO ---")
    print(f"Loss inicial: {history[0]:.4f} -> Loss final: {history[-1]:.4f}")
    
    # 6. Guardar el modelo y tokenizador para la UI
    os.makedirs("outputs/model", exist_ok=True)
    from model.embeddings import save_mlp_weights
    save_mlp_weights(W_in, W_hidden, b_hidden, W_out, "outputs/model/lm_weights.npz")
    tokenizer.save("outputs/model/bpe_tokenizer.json")
    print("Modelo y tokenizador guardados en outputs/model/ para usarlos en la UI.")

if __name__ == "__main__":
    run_sanity_check()
