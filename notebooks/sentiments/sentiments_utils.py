import os
import random
from tqdm import tqdm
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any
import sentencepiece as spm

import evaluate
import numpy as np
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, DatasetDict, Dataset, concatenate_datasets, disable_progress_bar
from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    set_seed,
    Trainer,
    TrainingArguments
)
from transformers.trainer_utils import get_last_checkpoint
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, balanced_accuracy_score

def clear_gpu_memory():
    """Clear GPU memory cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        import gc
        gc.collect()
        print("🧹 GPU memory cleared")
        
        
def load_full_dataset(dataset_path: str, dataset_splits: Dict[str, str]) -> DatasetDict:
    """
    Load Shipibo-Spanish parallel corpus with Sentiment labels from device
    
    Args:
        dataset_path: Path to the dataset file (e.g., "data/train.parquet")

    Returns:
        DatasetDict with train/validation/test splits
    """
    print(f"\n📥 Loading dataset: {dataset_path}")

    # Load the dataset
    dataset = load_dataset("parquet", data_files=dataset_splits, split=None)

    # Display statistics
    print("\n📊 Dataset Statistics:")
    for split in dataset.keys():
        print(f"   {split}: {len(dataset[split])} examples")
        if len(dataset[split]) > 0:
            # Show first example
            example = dataset[split][0]
            print(f"   Example Shipibo: {example.get('shp', example.get('shipibo', 'N/A'))[:50]}...")
            print(f"   Example Spanish: {example.get('spa', example.get('spanish', 'N/A'))[:50]}...")
    
    return dataset

def compute_metrics(preds, labels):
    return {
        "accuracy": accuracy_score(labels, preds),
        "balanced_accuracy": balanced_accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, average="weighted"),
        "recall": recall_score(labels, preds, average="weighted"),
        "f1": f1_score(labels, preds, average="weighted"),
    }
    
def prepare_tokenizer(tokenizer_path: str) -> AutoTokenizer:
    """
    Load and configure tokenizer for Shipibo→Spanish
    """
    print(f"\n🔧 Loading tokenizer: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    print(f"   Tokenizer loaded from: {tokenizer_path}")
    return tokenizer

def prepare_model(model_path: str, num_labels: int):
    """
    Get the pre-trained model locally.
    
    Args:
        model_path: Directory where the model is saved.
    """
    print(f"\n⬇️  Loading model: {model_path}")
    model = AutoModelForSequenceClassification.from_pretrained(model_path, num_labels=num_labels, ignore_mismatched_sizes=True)
    model.config.label2id = {'NEG': 0, 'NEU': 1, 'POS': 2}
    model.config.id2label = {0: 'NEG', 1: 'NEU', 2: 'POS'}
    print(f"   Model loaded from: {model_path}")
    return model

def download_model(model_path: str, output_dir: str):
    """
    Download and save the pre-trained model locally.
    
    Args:
        model_path: Directory where the model is saved.
        output_dir: Directory to save the downloaded model.
    """
    print(f"\n⬇️  Downloading model: {model_path}")
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.save_pretrained(output_dir)
    print(f"   Model saved to: {output_dir}")
    
def download_tokenizer(tokenizer_path: str, output_dir: str):
    """
    Download and save the tokenizer locally.
    
    Args:
        tokenizer_name: Name of the tokenizer to download.
        tokenizer_path: Directory where the tokenizer is saved.
        output_dir: Directory to save the downloaded tokenizer.
    """
    print(f"\n⬇️  Downloading tokenizer: {tokenizer_path}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    tokenizer.save_pretrained(output_dir)
    print(f"   Tokenizer saved to: {output_dir}")
 
 
def norm(x):
    return x.strip().lower()    
    
def dedupe_bilingual(dataset, langs=("spa", "shp")) -> Dataset:
    """
    Remove duplicate sentence pairs in a bilingual dataset.
    
    Args:
        dataset: Hugging Face Dataset containing bilingual sentence pairs.
        langs: Tuple of language column names (default: ("spa", "shp")).
    """
    seen = set()
    rows = []
    for row in dataset:
        pair = (norm(row[langs[0]]), norm(row[langs[1]]))
        if pair not in seen:
            seen.add(pair)
            rows.append(row)
    return Dataset.from_list(rows)

def overlap_percentage(a, b):
    """
    a, b are Python sets
    Returns: % of elements in a that also appear in b
    """
    if len(a) == 0:
        return 0
    return len(a & b) / len(a) * 100

def remove_overlaps_bilingual(dataset_from, dataset_against, langs=("spa", "shp")):
    """
    dataset_from, dataset_against: HuggingFace Datasets objects
    Returns: a new Dataset object with rows from dataset_from that do not
             appear in dataset_against (based on bilingual pairs)
    """
    pairs_against = {
        (norm(r[langs[0]]), norm(r[langs[1]])) for r in dataset_against
    }
    return dataset_from.filter(
        lambda x: (norm(x[langs[0]]), norm(x[langs[1]])) not in pairs_against
    )