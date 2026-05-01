import json
import os
from typing import Any

import numpy as np


def ensure_dir(path: str) -> None:
	if not path:
		return
	os.makedirs(path, exist_ok=True)


def load_corpus(filepath: str) -> str:
	if not os.path.exists(filepath):
		raise FileNotFoundError(f"No existe el archivo: {filepath}")

	encodings = ["utf-8", "latin-1"]
	for encoding in encodings:
		try:
			with open(filepath, "r", encoding=encoding) as file:
				return file.read()
		except UnicodeDecodeError:
			continue

	raise UnicodeDecodeError("codec", b"", 0, 1, "No se pudo decodificar el corpus")


def load_multiple_corpus(directory: str) -> str:
	if not os.path.isdir(directory):
		raise NotADirectoryError(f"No existe el directorio: {directory}")

	text_chunks = []
	for filename in sorted(os.listdir(directory)):
		if not filename.lower().endswith(".txt"):
			continue
		filepath = os.path.join(directory, filename)
		text_chunks.append(load_corpus(filepath))

	if not text_chunks:
		raise ValueError("No se encontraron archivos .txt en el directorio")

	return "\n\n".join(text_chunks)


def save_model_artifact(
	path: str,
	W_in: np.ndarray,
	W_out: np.ndarray,
	word_to_idx: dict[str, int],
	idx_to_word: dict[int, str],
	metadata: dict[str, Any],
) -> None:
	ensure_dir(os.path.dirname(path))

	idx_to_word_str = {str(k): v for k, v in idx_to_word.items()}

	np.savez_compressed(
		path,
		W_in=W_in.astype(np.float32),
		W_out=W_out.astype(np.float32),
		word_to_idx_json=np.array(json.dumps(word_to_idx, ensure_ascii=False), dtype=object),
		idx_to_word_json=np.array(json.dumps(idx_to_word_str, ensure_ascii=False), dtype=object),
		metadata_json=np.array(json.dumps(metadata, ensure_ascii=False), dtype=object),
	)


def load_model_artifact(path: str):
	if not os.path.exists(path):
		raise FileNotFoundError(f"No existe el artefacto: {path}")

	data = np.load(path, allow_pickle=True)

	W_in = data["W_in"].astype(np.float32)
	W_out = data["W_out"].astype(np.float32)

	word_to_idx = json.loads(str(data["word_to_idx_json"].item()))
	idx_to_word_raw = json.loads(str(data["idx_to_word_json"].item()))
	metadata = json.loads(str(data["metadata_json"].item()))

	idx_to_word = {int(k): v for k, v in idx_to_word_raw.items()}

	return W_in, W_out, word_to_idx, idx_to_word, metadata
