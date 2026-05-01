# word2vec CBOW (minimal)

Implementacion simple de CBOW en CPU con NumPy.

## Requisitos

```bash
pip install -r requirements.txt
```

## Entrenar modelo

```bash
python src/main.py --corpus data/raw/corpus.txt --window-size 2 --embedding-dim 50 --epochs 10 --batch-size 32 --learning-rate 0.05
```

Si aparece el error de corpus vacio, revisa que `data/raw/corpus.txt` tenga contenido real.
Tambien puedes probar con un corpus alternativo:

```bash
python src/main.py --corpus data/raw/corpus_demo.txt --window-size 2 --embedding-dim 50 --epochs 10 --batch-size 32 --learning-rate 0.05
```

Artefacto generado por defecto:

- `outputs/model/model_artifact.npz`

## UI con Streamlit

```bash
streamlit run streamlit_app.py
```

En la UI:

1. Cargar el artefacto del modelo.
2. Escribir 2 palabras a la izquierda y 2 a la derecha.
3. Ejecutar la prediccion de la palabra central.
