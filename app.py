import streamlit as st
from llama_index.core import SimpleDirectoryReader, SentenceSplitter, MetadataMode
from llama_index.finetuning import EmbeddingAdapterFinetuneEngine, generate_qa_embedding_pairs
from llama_index.core.embeddings import resolve_embed_model, LinearAdapterEmbeddingModel
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import VectorStoreIndex, TextNode
from llama_index.core.evaluation import EmbeddingQAFinetuneDataset
from tqdm import tqdm
import pandas as pd
import json
import torch

# Configuração da página com mais opções de personalização
st.set_page_config(page_icon="💬", layout="wide", page_title="Finetuning de Embedding Models")

st.header("Finetuning de Adapters em Modelos de Embedding")

# Código para carregar os documentos
def load_corpus(files, verbose=False):
    if verbose:
        st.write(f"Carregando arquivos {files}")
    reader = SimpleDirectoryReader(input_files=files)
    docs = reader.load_data()
    if verbose:
        st.write(f"{len(docs)} documentos carregados")
    parser = SentenceSplitter()
    nodes = parser.get_nodes_from_documents(docs, show_progress=verbose)
    if verbose:
        st.write(f"{len(nodes)} nodes parseados")
    return nodes

# Treinamento e validação de arquivos
TRAIN_FILES = ["./data/10k/lyft_2021.pdf"]
VAL_FILES = ["./data/10k/uber_2021.pdf"]

if st.button("Carregar e Processar Datasets"):
    train_nodes = load_corpus(TRAIN_FILES, verbose=True)
    val_nodes = load_corpus(VAL_FILES, verbose=True)
    # Gerar pares de treinamento e validação
    train_dataset = generate_qa_embedding_pairs(train_nodes)
    val_dataset = generate_qa_embedding_pairs(val_nodes)
    train_dataset.save_json("train_dataset.json")
    val_dataset.save_json("val_dataset.json")
    st.write("Datasets de treinamento e validação gerados e salvos.")

# Seção para o fine-tuning
st.subheader("Fine-tuning do Adapter")
base_embed_model = resolve_embed_model("local:BAAI/bge-small-en")

if st.button("Iniciar Fine-tuning"):
    finetune_engine = EmbeddingAdapterFinetuneEngine(
        train_dataset,
        base_embed_model,
        model_output_path="model_output_test",
        epochs=4,
        verbose=True
    )
    finetune_engine.finetune()
    embed_model = finetune_engine.get_finetuned_model()

    st.write("Modelo fine-tuned disponível.")

# Seção para avaliação do modelo
st.subheader("Avaliação do Modelo Fine-tuned")
from eval_utils import evaluate, display_results

if st.button("Avaliar Modelo"):
    ada = OpenAIEmbedding()
    ada_val_results = evaluate(val_dataset, ada)
    bge_val_results = evaluate(val_dataset, embed_model)
    ft_val_results = evaluate(val_dataset, embed_model)
    
    results_df = pd.DataFrame({
        "retrievers": ["ada", "bge", "ft"],
        "hit_rate": [ada_val_results["hit_rate"], bge_val_results["hit_rate"], ft_val_results["hit_rate"]],
        "mrr": [ada_val_results["mrr"], bge_val_results["mrr"], ft_val_results["mrr"]]
    })
    st.write(results_df)
