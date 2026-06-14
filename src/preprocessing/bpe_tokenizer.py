import os
import logging
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

logger = logging.getLogger(__name__)

def train_bpe_tokenizer(corpus_path, vocab_size=5000, min_frequency=2, save_path=None):
    """
    Entrena un tokenizador BPE sobre un archivo de texto.
    
    Args:
        corpus_path (str): Ruta al archivo de texto (corpus).
        vocab_size (int): Tamaño máximo del vocabulario.
        min_frequency (int): Frecuencia mínima para agrupar caracteres.
        save_path (str): Ruta donde guardar el tokenizador entrenado (.json).
        
    Returns:
        Tokenizer: El tokenizador BPE entrenado.
    """
    if not os.path.exists(corpus_path):
        raise FileNotFoundError(f"No se encontró el corpus en {corpus_path}")

    logger.info(f"Entrenando BPE tokenizer con vocab_size={vocab_size}...")
    
    # Inicializar el fragmentador con el método BPE
    tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))
    
    # Indicar que separe el texto en palabras primero
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    
    # Crear un entrenador BPE
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=["[UNK]", "[PAD]", "[BOS]", "[EOS]"]
    )
    
    # Entrenar sobre el corpus
    tokenizer.train([corpus_path], trainer=trainer)
    
    logger.info(f"Tamaño del vocabulario obtenido: {tokenizer.get_vocab_size()}")
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        tokenizer.save(save_path)
        logger.info(f"Tokenizador guardado en {save_path}")
        
    return tokenizer

def load_bpe_tokenizer(load_path):
    """
    Carga un tokenizador BPE desde un archivo JSON.
    """
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"No se encontró el tokenizador en {load_path}")
    
    tokenizer = Tokenizer.from_file(load_path)
    logger.info(f"Tokenizador cargado con vocab_size={tokenizer.get_vocab_size()}")
    return tokenizer
