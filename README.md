# NumPy-only text AI demos

Two runnable educational demos written with **NumPy and the Python standard library**. No PyTorch, TensorFlow, scikit-learn, embedding API, or external vector database.

## 1. NumPy retrieval + extractive question answering (`numpy_rag.py`)

Uses TF-IDF vectors and cosine similarity to retrieve documents. Selects a relevant sentence as an *extractive* answer, with its source filename. This is the retrieval / context-selection side of a RAG system, **not full generative RAG**. It does not have a language generator, nor does it perform semantic embedding search.

```bash
pip install numpy
python numpy_rag.py --query "What frequencies are used for EEG alpha waves?"
```

To query your own knowledge base, put `.txt` and `.md` files in a folder and run:

```bash
python numpy_rag.py --docs ./my_notes
```

## 2. Tiny neural language model (`numpy_lm.py`)

A character-level feed-forward neural language model with learned character embeddings, an 8-character context, one hidden layer, a softmax output, manually implemented backpropagation and Adam optimization. **This is a very small learning exercise, not an LLM in the modern sense**. It has an extremely short context, it overfits small training datasets, and it does not do instruction following.

```bash
pip install numpy
python numpy_lm.py
python numpy_lm.py --corpus my_corpus.txt --steps 1200 --prompt "the "
```

The built-in toy corpus only demonstrates learning and generation. Supply more text for experiments; don't expect useful general-purpose completions.

## True generative RAG

Combine a retriever like `numpy_rag.py` with a pretrained generator (a model whose weights were trained separately). NumPy *can* implement inference for a tiny transformer if the needed weights, tokenizer, layers and operations are implemented, but useful generative RAG is normally built with a separate pretrained LLM. This demo deliberately does not claim to be one.
