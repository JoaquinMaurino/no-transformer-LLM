import argparse
import os

from model.embeddings import init_embeddings
from model.train import train
from preprocessing.cbow_dataset import build_and_save_dataset, load_dataset
from preprocessing.dictionaries import build_dictionaries
from preprocessing.tokenizer import tokenize_text
from utils.io import ensure_dir, load_corpus, load_multiple_corpus, save_model_artifact


def parse_args():
	parser = argparse.ArgumentParser(description="Entrenamiento CBOW minimalista")
	parser.add_argument("--corpus", type=str, default="data/raw/corpus.txt")
	parser.add_argument("--window-size", type=int, default=2)
	parser.add_argument("--embedding-dim", type=int, default=50)
	parser.add_argument("--epochs", type=int, default=10)
	parser.add_argument("--batch-size", type=int, default=32)
	parser.add_argument("--learning-rate", type=float, default=0.05)
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--processed-path", type=str, default="")
	parser.add_argument("--model-path", type=str, default="outputs/model/model_artifact.npz")
	parser.add_argument("--force-rebuild-dataset", action="store_true")
	parser.add_argument("--remove-stopwords", action="store_true")
	return parser.parse_args()


def _load_raw_text(corpus_path: str) -> str:
	if os.path.isdir(corpus_path):
		return load_multiple_corpus(corpus_path)
	return load_corpus(corpus_path)


def main():
	args = parse_args()

	if args.window_size <= 0:
		raise ValueError("window-size debe ser positivo")

	processed_path = args.processed_path or f"data/processed/cbow_w{args.window_size}.npz"

	ensure_dir(os.path.dirname(processed_path))
	ensure_dir(os.path.dirname(args.model_path))

	print("1) Cargando corpus...")
	raw_text = _load_raw_text(args.corpus)
	if not raw_text.strip():
		raise ValueError(
			f"Corpus vacío en '{args.corpus}'. Agrega texto al archivo o usa otro corpus con --corpus."
		)

	print("2) Limpiando y tokenizando...")
	tokens = tokenize_text(raw_text, remove_stopwords=args.remove_stopwords)
	if not tokens:
		raise ValueError(
			"El corpus quedó vacío después de la limpieza/tokenización. "
			"Revisa reglas de limpieza o usa un texto más largo."
		)
	if len(tokens) < (2 * args.window_size + 1):
		raise ValueError("Corpus demasiado chico para el window size elegido")

	print("3) Construyendo vocabulario...")
	word_to_idx, idx_to_word = build_dictionaries(tokens)
	print(f"   Tokens: {len(tokens)} | Vocabulario: {len(word_to_idx)}")

	print("4) Generando/cargando dataset CBOW...")
	if os.path.exists(processed_path) and not args.force_rebuild_dataset:
		contexts, targets = load_dataset(processed_path)
	else:
		contexts, targets = build_and_save_dataset(
			tokens=tokens,
			word_to_idx=word_to_idx,
			output_path=processed_path,
			window_size=args.window_size,
		)

	print("5) Inicializando embeddings...")
	W_in, W_out = init_embeddings(
		vocab_size=len(word_to_idx),
		embed_dim=args.embedding_dim,
		seed=args.seed,
	)

	print("6) Entrenando...")
	config = {
		"epochs": args.epochs,
		"batch_size": args.batch_size,
		"learning_rate": args.learning_rate,
		"seed": args.seed,
	}
	W_in, W_out, history = train(
		W_in=W_in,
		W_out=W_out,
		contexts=contexts,
		targets=targets,
		config=config,
	)

	print("7) Guardando artefacto final...")
	metadata = {
		"window_size": args.window_size,
		"embedding_dim": args.embedding_dim,
		"epochs": args.epochs,
		"batch_size": args.batch_size,
		"learning_rate": args.learning_rate,
		"seed": args.seed,
		"num_tokens": len(tokens),
		"vocab_size": len(word_to_idx),
		"final_loss": history[-1] if history else None,
	}
	save_model_artifact(
		path=args.model_path,
		W_in=W_in,
		W_out=W_out,
		word_to_idx=word_to_idx,
		idx_to_word=idx_to_word,
		metadata=metadata,
	)

	print(f"Listo. Modelo guardado en: {args.model_path}")


if __name__ == "__main__":
	main()
