# No-Transformer LLM (MLP-based)

Este proyecto implementa un Modelo de Lenguaje (Language Model) causal y autoregresivo desde cero, utilizando únicamente un Perceptrón Multicapa (MLP) y NumPy, sin depender de arquitecturas de Transformadores ni frameworks de Deep Learning (como PyTorch o TensorFlow). 

Fue desarrollado como parte del Trabajo Práctico 2 de Aprendizaje Automático Avanzado.

## Características Principales

1. **Tokenización BPE**: Utiliza la librería `tokenizers` de HuggingFace para construir un vocabulario basado en Byte Pair Encoding (BPE), solucionando el problema de las palabras fuera de vocabulario (OOV) y permitiendo un tamaño de vocabulario controlado.
2. **Dataset Causal**: Dada una secuencia de texto, genera contextos de tamaño $N$ (por defecto 3 a 5 palabras previas) para predecir la palabra exactamente siguiente ($N+1$).
3. **MLP y Backpropagation Manual**: La arquitectura de red neuronal consiste en una capa de embeddings, una capa oculta con función de activación `Tanh`, y una capa de salida Softmax. El _backward pass_ (derivadas y regla de la cadena) está implementado enteramente a mano con NumPy usando *Cross-Entropy Loss*.

## Requisitos

Instala las dependencias necesarias ejecutando:

```bash
pip install -r requirements.txt
```

## Entrenar el Modelo

Actualmente, el ciclo completo de entrenamiento (desde la carga de datos hasta la tokenización y la optimización de pesos) se puede verificar y ejecutar mediante el script de prueba integrado. 

Para ejecutar una demostración (Sanity Check) del modelo aprendiendo sobre un corpus local:

```bash
python test_train.py
```

Este script:
1. Crea un pequeño corpus de prueba en `data/raw/test_corpus.txt`.
2. Entrena el tokenizador BPE en base a ese corpus.
3. Genera el dataset causal con el tamaño de contexto especificado.
4. Inicializa los pesos del MLP (Embeddings, W_hidden, b_hidden, W_out).
5. Ejecuta 20 épocas de entrenamiento reportando cómo disminuye la pérdida (Loss) iteración a iteración.

Si deseas entrenar el modelo en un entorno de **Google Cloud Platform (GCP)** con un corpus real y pesado, simplemente asegúrate de descargar tu archivo `corpus.txt` en `data/raw/` y modificar `test_train.py` (o el script principal) para que apunte a tu archivo y vocabulario deseado.
